#!/usr/bin/env python3
"""Main-thread phase times from an IOLOG: per measured request (generation part)."""
import struct, sys, collections
E = dict(SUBMIT=1, START=2, END=3, PROMOTE=4, DEMAND=5, DEMAND_END=6, MOE_BEGIN=7, WAIT_BEGIN=8, WAIT_END=9, MOE_END=10,
         PRED=11, FWD=12, REQ=13, USED=14, WASTE=15, EVICT=16, WAVE_BEGIN=17, WAVE_END=18, ATTN_BEGIN=19, ATTN_END=20,
         HIT=21, FWD_END=22, PRED_RANK=23, MTP_BEGIN=24, MTP_END=25)
N = {v: k for k, v in E.items()}
def load(path):
    raw = open(path, 'rb').read()
    out = []
    for i in range(len(raw) // 32):
        t, k, a, b, c, d, e = struct.unpack_from('<d6i', raw, i * 32)
        if k: out.append((t, k, a, b, c, d, e))
    out.sort(key=lambda x: x[0])
    return out
evs = load(sys.argv[1])
reqs, cur = [], None
for x in evs:
    if x[1] == E['REQ']:
        if x[2] == 0: cur = [x[0], None, None]
        elif x[2] == 1 and cur: cur[1] = x[0]
        elif x[2] == 2 and cur: cur[2] = x[0]; reqs.append(cur); cur = None
MAIN = {E[k] for k in ('ATTN_BEGIN','ATTN_END','DEMAND','DEMAND_END','MOE_BEGIN','WAIT_BEGIN','WAIT_END','MOE_END','WAVE_BEGIN','WAVE_END','FWD','FWD_END','MTP_BEGIN','MTP_END')}
tot = collections.Counter(); cnt = collections.Counter()
for (r0, g0, g1) in reqs[1:]:
    seq = [x for x in evs if g0 <= x[0] < g1 and x[1] in MAIN]
    # PRED end-of-ranking marker (n>0 & e==-1)
    for i in range(len(seq) - 1):
        a, b = seq[i], seq[i + 1]
        key = N[a[1]] + '>' + N[b[1]]
        tot[key] += b[0] - a[0]; cnt[key] += 1
nreq = len(reqs) - 1
gen = sum(r[2] - r[1] for r in reqs[1:]) / nreq
print(f"{sys.argv[1]}: {nreq} measured requests, generation {gen:.3f}s a request")
for k, v in sorted(tot.items(), key=lambda kv: -kv[1])[:22]:
    print(f"  {k:28s} {v / nreq:8.3f}s/req  {cnt[k] / nreq:8.0f} calls  {1e3 * v / cnt[k]:8.4f} ms/call")
