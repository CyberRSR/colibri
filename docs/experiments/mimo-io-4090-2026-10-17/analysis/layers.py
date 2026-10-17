#!/usr/bin/env python3
"""Per-layer and per-region view of an IOLOG's last (or given) request's generation: attention, expert
waves, stall, reads per step; and each drive's idle share by layer region. layers.py LOG [request]"""
import struct, sys, collections, bisect


def load(path):
    raw = open(path, 'rb').read()
    out = []
    for i in range(len(raw) // 32):
        t, k, a, b, c, d, e = struct.unpack_from('<d6i', raw, i * 32)
        if k: out.append((t, k, a, b, c, d, e))
    out.sort(key=lambda x: x[0])
    return out


def region(lab):
    if isinstance(lab, str): return lab
    return 'L1-16' if lab <= 16 else 'L17-32' if lab <= 32 else 'L33-50' if lab <= 50 else 'L51-69'


def main():
    evs = load(sys.argv[1])
    which = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    reqs, cur = [], None
    for x in evs:
        if x[1] == 13:
            if x[2] == 0: cur = [x[0], None, None]
            elif x[2] == 1 and cur: cur[1] = x[0]
            elif x[2] == 2 and cur: cur[2] = x[0]; reqs.append(cur); cur = None
    g0, g1 = reqs[which][1], reqs[which][2]
    att, wav, wt, rd, un = (collections.Counter() for _ in range(5))
    ab = wb = vb = None; ns = 0; canc = collections.Counter()
    for x in evs:
        if not (g0 <= x[0] < g1): continue
        k = x[1]
        if k == 12: ns += 1
        elif k == 19: ab = x
        elif k == 20 and ab: att[ab[2]] += x[0] - ab[0]
        elif k == 17: vb = x
        elif k == 18 and vb: wav[vb[2]] += x[0] - vb[0]
        elif k == 8: wb = x
        elif k == 9 and wb: wt[wb[2]] += x[0] - wb[0]
        elif k == 1: rd[x[3]] += 1
        elif k == 7: un[x[2]] += x[4]
        elif k == 15 and x[5] == 3: canc[x[3]] += 1
    print(f"{ns} decode steps; per step: attention, expert waves, stall (ms), reads read (net of cancelled), union")
    for lo, hi in ((1, 16), (17, 32), (33, 50), (51, 69)):
        L = range(lo, hi + 1)
        f = lambda c: sum(c[l] for l in L) / ns / len(L)
        print(f"  L{lo}-{hi}: attention {1e3 * f(att):.2f} waves {1e3 * f(wav):.2f} stall {1e3 * f(wt):.2f} ms a layer; "
              f"reads {f(rd) - f(canc):.2f} of a union of {f(un):.1f} a layer")
    marks = []
    for x in evs:
        if not (g0 <= x[0] < g1): continue
        if x[1] == 19: marks.append((x[0], x[2]))
        elif x[1] == 24: marks.append((x[0], 'mtp'))
        elif x[1] == 22: marks.append((x[0], 'head'))
    mt = [m[0] for m in marks]
    tot = collections.Counter()
    for i, (t, lab) in enumerate(marks):
        tot[region(lab)] += (mt[i + 1] if i + 1 < len(mt) else g1) - t
    reads = {}
    for x in evs:
        t, k, a, b, c, d, e = x
        if k == 1: reads[a] = [t, None, None, e]
        elif k == 2 and a in reads: reads[a][1] = t
        elif k == 3 and a in reads: reads[a][2] = t
    for dv in (0, 1):
        ivs = sorted((r[1], r[2]) for r in reads.values() if r[3] == dv and r[1] and r[2] and r[2] > g0 and r[1] < g1)
        idle, last = [], g0
        for s, e in ivs:
            if s > last: idle.append((last, s))
            last = max(last, e)
        if last < g1: idle.append((last, g1))
        by = collections.Counter()
        for a, b in idle:
            i = bisect.bisect_right(mt, a) - 1; t = a
            while t < b:
                lab = marks[i][1] if i >= 0 else 'start'
                nxt = mt[i + 1] if i + 1 < len(mt) else g1
                end = min(b, nxt); by[region(lab)] += end - t; t = end; i += 1
        print(f"  drive {dv} idle: " + ", ".join(f"{k} {100 * by[k] / tot[k]:.0f}% of {tot[k]:.2f}s"
                                              for k in ('L1-16', 'L17-32', 'L33-50', 'L51-69', 'mtp') if tot[k]))


if __name__ == '__main__':
    main()
