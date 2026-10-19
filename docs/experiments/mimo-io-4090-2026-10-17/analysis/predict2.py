#!/usr/bin/env python3
"""Hit rates of the logged predictors against the union each layer's next step demanded.
kinds: 23 the prefetcher (after li-1's attention), 26 two ahead (after li-2's attention),
27 the chosen token's embedding (before the drafts), 28 two ahead after li-2's MLP,
29 one ahead after li-1's MLP. Usage: predict2.py LOG [request]"""
import struct, sys, collections

SUBMIT, DEMAND, DEMAND_END, REQ, USED, HIT = 1, 5, 6, 13, 14, 21
KINDS = {23: 'prefetch(attn,1)', 26: 'attn,2-ahead', 27: 'embedding', 28: 'mlp,2-ahead', 29: 'mlp,1-ahead'}


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
    which = sys.argv[2] if len(sys.argv) > 2 else '-1'
    reqs, cur = [], None
    for x in evs:
        if x[1] == REQ:
            if x[2] == 0: cur = [x[0], None, None]
            elif x[2] == 1 and cur: cur[1] = x[0]
            elif x[2] == 2 and cur: cur[2] = x[0]; reqs.append(cur); cur = None
    if which == 'all':
        spans = [(r[1], r[2]) for r in reqs[1:]] if len(reqs) > 1 else [(reqs[0][1], reqs[0][2])]
    else:
        r = reqs[int(which)]; spans = [(r[1], r[2])]
    inspan = lambda t: any(a <= t < b for a, b in spans)
    pending = {}   # (kind, li) -> [cands]
    lastt = {}
    n = collections.Counter(); h = collections.Counter()
    nd = collections.Counter(); hd = collections.Counter()  # distinct top-8 per prediction
    cur = None
    for x in evs:
        t, k, a, b, c, d, e = x
        if k in KINDS:
            key = (k, a)
            if key not in lastt or lastt[key] is None or t - lastt[key] > 2e-3:
                pending[key] = []
            lastt[key] = t
            pending[key].append((c // 100, c % 100, b, e))
        elif k == DEMAND:
            cur = (a, set(), t)
        elif k in (USED, HIT) and cur:
            cur[1].add(c)
        elif k == SUBMIT and cur and d == 0:
            cur[1].add(c)
        elif k == DEMAND_END and cur:
            li, U, td = cur; cur = None
            for kind in KINDS:
                cands = pending.pop((kind, li), None)
                lastt[(kind, li)] = None
                if not cands or not inspan(td): continue
                rows = cands[0][3]
                for row, rank, ex, _ in cands:
                    n[(kind, rows, row, rank)] += 1; h[(kind, rows, row, rank)] += ex in U
                top = set(ex for row, rank, ex, _ in cands if rank < 8)
                nd[(kind, rows)] += len(top); hd[(kind, rows)] += len(top & U)
    for kind, name in KINDS.items():
        for rows in (1, 2, 3):
            if not nd[(kind, rows)]: continue
            line = []
            for rank in range(16):
                tot = sum(n[(kind, rows, row, rank)] for row in range(rows))
                hit = sum(h[(kind, rows, row, rank)] for row in range(rows))
                if tot: line.append(f"{hit / tot:.2f}")
            r0 = []
            for rank in range(8):
                tot = n[(kind, rows, 0, rank)]
                if tot: r0.append(f"{h[(kind, rows, 0, rank)] / tot:.2f}")
            print(f"{name:18s} rows={rows}: top-8 distinct in union {hd[(kind, rows)]}/{nd[(kind, rows)]} = "
                  f"{hd[(kind, rows)] / nd[(kind, rows)]:.3f}; by rank " + " ".join(line))
            print(f"{'':18s}         row 0 by rank " + " ".join(r0))


if __name__ == '__main__':
    main()
