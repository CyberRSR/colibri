#!/usr/bin/env python3
"""Where the drives idle: each drive's no-read intervals against the main thread's phase."""
import struct, sys, collections, bisect

E = dict(SUBMIT=1, START=2, END=3, DEMAND=5, DEMAND_END=6, MOE_BEGIN=7, WAIT_BEGIN=8, WAIT_END=9, MOE_END=10,
         FWD=12, REQ=13, WAVE_BEGIN=17, WAVE_END=18, ATTN_BEGIN=19, ATTN_END=20, FWD_END=22, MTP_BEGIN=24, MTP_END=25)


def load(path):
    raw = open(path, 'rb').read()
    out = []
    for i in range(len(raw) // 32):
        t, k, a, b, c, d, e = struct.unpack_from('<d6i', raw, i * 32)
        if k: out.append((t, k, a, b, c, d, e))
    out.sort(key=lambda x: x[0])
    return out


def main():
    evs = load(sys.argv[1])
    which = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    reqs, cur = [], None
    for x in evs:
        if x[1] == E['REQ']:
            if x[2] == 0: cur = [x[0], None, None]
            elif x[2] == 1 and cur: cur[1] = x[0]
            elif x[2] == 2 and cur: cur[2] = x[0]; reqs.append(cur); cur = None
    g0, g1 = reqs[which][1], reqs[which][2]
    # main-thread phases: list of (t_start, label)
    ph = []
    state = 'other'
    for x in evs:
        t, k, a = x[0], x[1], x[2]
        if not (g0 <= t < g1): continue
        lab = None
        if k == E['ATTN_BEGIN']: lab = 'attn' + ('0' if a == 0 else '')
        elif k == E['ATTN_END']: lab = 'post-attn'
        elif k == E['DEMAND']: lab = 'route'
        elif k == E['MOE_BEGIN']: lab = 'wave'
        elif k == E['WAIT_BEGIN']: lab = 'wait'
        elif k == E['WAIT_END']: lab = 'wave'
        elif k == E['MOE_END']: lab = 'post-moe'
        elif k == E['MTP_BEGIN']: lab = 'mtp'
        elif k == E['MTP_END']: lab = 'post-mtp'
        elif k == E['FWD']: lab = 'fwd-start'
        elif k == E['FWD_END']: lab = 'head/sample'
        if lab: ph.append((t, lab))
    pt = [p[0] for p in ph]
    reads = {}
    for x in evs:
        t, k, a, b, c, d, e = x
        if k == E['SUBMIT']: reads[a] = [t, None, None, e, d]
        elif k == E['START'] and a in reads: reads[a][1] = t
        elif k == E['END'] and a in reads: reads[a][2] = t
    ndrive = 1 + max(r[3] for r in reads.values())
    for dv in range(ndrive):
        ivs = sorted((r[1], r[2]) for r in reads.values() if r[3] == dv and r[1] and r[2] and r[2] > g0 and r[1] < g1)
        idle = []
        last = g0
        for s, e in ivs:
            if s > last: idle.append((last, s))
            last = max(last, e)
        if last < g1: idle.append((last, g1))
        by = collections.Counter()
        tot = 0
        for a, b in idle:
            tot += b - a
            # split [a,b) by phases
            i = bisect.bisect_right(pt, a) - 1
            t = a
            while t < b:
                lab = ph[i][1] if i >= 0 else 'start'
                nxt = pt[i + 1] if i + 1 < len(pt) else g1
                end = min(b, nxt)
                by[lab] += end - t
                t = end; i += 1
        print(f"drive {dv}: idle {tot:.2f}s: " + ", ".join(f"{k} {v:.2f}" for k, v in by.most_common()))
    # main thread time by phase
    tot = collections.Counter()
    for i, (t, lab) in enumerate(ph):
        nxt = pt[i + 1] if i + 1 < len(pt) else g1
        tot[lab] += nxt - t
    print("main thread: " + ", ".join(f"{k} {v:.2f}" for k, v in tot.most_common()))


if __name__ == '__main__':
    main()
