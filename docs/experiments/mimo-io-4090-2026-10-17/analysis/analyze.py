#!/usr/bin/env python3
"""Analyze an IOLOG (scratch instrumentation of c/mimo.c): reads, drives, stalls, prefetch outcomes."""
import struct, sys, collections, math

EV = dict(SUBMIT=1, START=2, END=3, PROMOTE=4, DEMAND=5, DEMAND_END=6, MOE_BEGIN=7, WAIT_BEGIN=8,
          WAIT_END=9, MOE_END=10, PRED=11, FWD=12, REQ=13, USED=14, WASTE=15, EVICT=16, WAVE_BEGIN=17,
          WAVE_END=18, ATTN_BEGIN=19, ATTN_END=20, HIT=21, FWD_END=22, PRED_RANK=23, MTP_BEGIN=24, MTP_END=25)
N = {v: k for k, v in EV.items()}
EB = 20.0e6 / 1e9  # GB an expert (approx; refined below)


def load(path):
    raw = open(path, 'rb').read()
    n = len(raw) // 32
    out = []
    for i in range(n):
        t, k, a, b, c, d, e = struct.unpack_from('<d6i', raw, i * 32)
        if k == 0:
            continue
        out.append((t, k, a, b, c, d, e))
    out.sort(key=lambda x: x[0])
    return out


def f32(i):
    return struct.unpack('<f', struct.pack('<i', i))[0]


def main():
    evs = load(sys.argv[1])
    which = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    # requests: REQ 0 start, 1 prefilled, 2 report
    reqs = []
    cur = None
    for x in evs:
        if x[1] == EV['REQ']:
            if x[2] == 0: cur = [x[0], None, None]
            elif x[2] == 1 and cur: cur[1] = x[0]
            elif x[2] == 2 and cur: cur[2] = x[0]; reqs.append(cur); cur = None
    print(f"{len(evs)} events, {len(reqs)} requests")
    for i, r in enumerate(reqs):
        print(f"  req {i}: prompt {r[1]-r[0]:.2f}s generation {r[2]-r[1]:.2f}s")
    r = reqs[which]
    g0, g1 = r[1], r[2]
    # reads: rid -> dict
    reads = {}
    for x in evs:
        t, k, a, b, c, d, e = x
        if k == EV['SUBMIT']:
            reads[a] = dict(sub=t, li=b, eid=c, pf=d, drive=e, start=None, end=None, q=None, out=None, reader=None)
        elif k == EV['START'] and a in reads:
            reads[a]['start'] = t; reads[a]['q'] = c; reads[a]['reader'] = b
        elif k == EV['END'] and a in reads:
            reads[a]['end'] = t
        elif k == EV['USED'] and a in reads:
            reads[a]['out'] = ('used', d, t)
        elif k == EV['WASTE'] and a in reads:
            reads[a]['out'] = ('waste', d, t)
        elif k == EV['EVICT'] and a in reads:
            if reads[a]['out'] is None: reads[a]['out'] = ('evict', 0, t)
        elif k == EV['PROMOTE'] and a in reads:
            reads[a]['promoted'] = (t, b)
    gen = {rid: v for rid, v in reads.items() if g0 <= v['sub'] < g1}
    print(f"\n== request {which if which >= 0 else len(reqs)+which}, generation {g1-g0:.2f}s: {len(gen)} reads submitted")
    dem = [v for v in gen.values() if v['pf'] == 0]
    pf = [v for v in gen.values() if v['pf'] == 1]
    outc = collections.Counter()
    for v in pf:
        o = v['out']
        outc[(o[0], o[1]) if o else ('none', 0)] += 1
    print(f"  demand {len(dem)}, prefetch {len(pf)}")
    print("  prefetch outcomes (used: 0 ready/1 loading at demand; waste: 0 ready/1 queued/2 in flight):")
    for key, n in sorted(outc.items()):
        print(f"    {key}: {n}")
    # per drive
    ndrive = max(v['drive'] for v in gen.values()) + 1
    for dv in range(ndrive):
        rs = [v for v in gen.values() if v['drive'] == dv and v['start'] is not None and v['end'] is not None]
        if not rs: continue
        # concurrency timeline within [g0, g1]
        pts = []
        for v in rs:
            pts.append((max(v['start'], g0), 1)); pts.append((min(v['end'], g1), -1))
        pts.sort()
        occ = collections.Counter(); level = 0; last = g0
        for t, dl in pts:
            if t > last:
                occ[level] += t - last
            last = max(last, t); level += dl
        if g1 > last: occ[level] += g1 - last
        dur = [v['end'] - v['start'] for v in rs]
        print(f"  drive {dv}: {len(rs)} reads ({sum(1 for v in rs if v['pf']==0)} demand), "
              f"{len(rs)*20.97e6/1e9/(g1-g0):.2f} GB/s avg; busy levels " +
              ", ".join(f"{l}: {occ[l]:.2f}s" for l in sorted(occ)) +
              f"; read time mean {1e3*sum(dur)/len(dur):.2f} ms, p50 {1e3*sorted(dur)[len(dur)//2]:.2f}")
        # throughput when 1 vs 2 readers busy: bytes/time per level is not separable; report per-read times by overlap
    # queue wait (submit->start) by kind
    for name, group in (('demand', dem), ('prefetch', pf)):
        qs = sorted(v['start'] - v['sub'] for v in group if v['start'] is not None)
        if qs:
            print(f"  {name}: queue wait mean {1e3*sum(qs)/len(qs):.2f} ms, p50 {1e3*qs[len(qs)//2]:.2f}, p90 {1e3*qs[int(len(qs)*.9)]:.2f}")
    # stalls per layer and per step
    waits = []
    wb = None
    for x in evs:
        if not (g0 <= x[0] < g1): continue
        if x[1] == EV['WAIT_BEGIN']: wb = x
        elif x[1] == EV['WAIT_END'] and wb: waits.append((wb[0], x[0], wb[2])); wb = None
    tot = sum(b - a for a, b, _ in waits)
    print(f"  waits: {len(waits)} totalling {tot:.2f}s")
    # what ended each wait: the END events in (a, b]
    ends = sorted((v['end'], rid) for rid, v in gen.items() if v['end'])
    import bisect
    et = [e[0] for e in ends]
    cat = collections.Counter(); catn = collections.Counter()
    for a, b, li in waits:
        i = bisect.bisect_left(et, b - 1e-4)
        # the read finishing nearest b (within 0.3ms before b)
        j = i
        best = None
        for jj in range(max(0, i - 3), min(len(ends), i + 3)):
            if abs(ends[jj][0] - b) < 5e-4 and (best is None or abs(ends[jj][0] - b) < abs(ends[best][0] - b)):
                best = jj
        if best is None:
            cat['?'] += b - a; catn['?'] += 1; continue
        v = gen[ends[best][1]]
        if v['pf'] == 0: kind = 'demand'
        elif v.get('promoted'): kind = 'prefetch-promoted' + ('-queued' if v['promoted'][1] else '-inflight')
        else: kind = 'prefetch'
        cat[kind] += b - a; catn[kind] += 1
    print("  wait time by the read that ended it: " + ", ".join(f"{k} {cat[k]:.2f}s ({catn[k]})" for k in sorted(cat)))
    # phases per layer: attention, moe
    att = 0.0; moe = 0.0; ab = None; mb = None; mtp = 0.0; tb = None; wave = 0.0; wvb = None
    for x in evs:
        if not (g0 <= x[0] < g1): continue
        k = x[1]
        if k == EV['ATTN_BEGIN']: ab = x[0]
        elif k == EV['ATTN_END'] and ab: att += x[0] - ab; ab = None
        elif k == EV['DEMAND']: mb = x[0]
        elif k == EV['MOE_END'] and mb: moe += x[0] - mb; mb = None
        elif k == EV['MTP_BEGIN']: tb = x[0]
        elif k == EV['MTP_END'] and tb: mtp += x[0] - tb; tb = None
        elif k == EV['WAVE_BEGIN']: wvb = x[0]
        elif k == EV['WAVE_END'] and wvb: wave += x[0] - wvb; wvb = None
    print(f"  attention {att:.2f}s, moe (demand..end) {moe:.2f}s of which waves {wave:.2f}s, mtp {mtp:.2f}s")
    # disk idle: both drives idle
    ivs = sorted((max(v['start'], g0), min(v['end'], g1), v['drive']) for v in gen.values() if v['start'] and v['end'])
    for dv in range(ndrive):
        busy = 0.0; cs = None; ce = None
        for s, e, d in ivs:
            if d != dv: continue
            if cs is None: cs, ce = s, e
            elif s <= ce: ce = max(ce, e)
            else: busy += ce - cs; cs, ce = s, e
        if cs is not None: busy += ce - cs
        print(f"  drive {dv}: some read active {busy:.2f}s of {g1-g0:.2f}s ({100*busy/(g1-g0):.1f}%)")
    # per-drive split of waste
    wq = collections.Counter()
    for v in pf:
        if v['out'] and v['out'][0] == 'waste': wq[(v['drive'], v['out'][1])] += 1
    print("  waste by (drive, state):", dict(sorted(wq.items())))


if __name__ == '__main__':
    main()
