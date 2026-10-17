#!/usr/bin/env python3
"""Offline: from MIMO_MTP_TRACE lines (a run that verified every draft, MIMO_MTP_MIN_P=0:
"P N K a p1 .. pN" from the committed engine, "P N a p1 .. pN" from the build that wrote
g.trace and t07.trace), how acceptance tracks the blocks' own draft probabilities, and
what a policy that verifies only the leading drafts whose cumulative probability stays
>= theta would yield.

Steps are treated as samples (a policy moves the step boundaries; the positions it would
draft are statistically like these). Cost per step = trunk(K) + N * block, K drafts verified,
from measured step times (ms) given on the command line."""
import argparse, statistics

ap = argparse.ArgumentParser()
ap.add_argument("traces", nargs="+")
ap.add_argument("--trunk", default="423,655,874,1100", help="trunk ms per step for K = 0..3 drafts verified")
ap.add_argument("--block", type=float, default=17.0, help="ms per MTP block per step")
a = ap.parse_args()
trunk = [float(x) for x in a.trunk.split(",")]
steps = []
for path in a.traces:
    for line in open(path):
        f = line.split()
        if len(f) < 4: continue
        P, N = int(f[0]), int(f[1])
        if len(f) == 4 + N:   # "P N K a p1..pN" (K verified); a cut trace: only K's outcome is known
            K, acc, ps = int(f[2]), int(f[3]), [float(x) for x in f[4:4 + N]]
            if K < N: N = K; ps = ps[:K]
        else:                 # "P N a p1..pN", every draft verified
            acc, ps = int(f[2]), [float(x) for x in f[3:3 + N]]
        steps.append((N, acc, ps))
Nmax = max(s[0] for s in steps)
full = [s for s in steps if s[0] == Nmax]
print(f"{len(steps)} steps, {len(full)} with all {Nmax} drafts verified")

# calibration: P(draft k kept | drafts before it kept), by the block's own probability
for k in range(Nmax):
    rows = [(s[2][k], s[1] > k) for s in full if s[1] >= k]   # draft k was tried
    if not rows: continue
    print(f"depth {k+1}: tried {len(rows)}, kept {sum(r[1] for r in rows)} ({100*sum(r[1] for r in rows)/len(rows):.1f}%)")
    for lo, hi in ((0, .3), (.3, .5), (.5, .7), (.7, .8), (.8, .9), (.9, .95), (.95, .99), (.99, 1.01)):
        b = [r for r in rows if lo <= r[0] < hi]
        if b: print(f"   p in [{lo:.2f},{hi:.2f}): {len(b):4d} tried, {100*sum(r[1] for r in b)/len(b):5.1f}% kept")

def simulate(K_of, blocks=None):
    blocks = Nmax if blocks is None else blocks
    tok = cost = 0.0
    for N, acc, ps in full:
        K = min(K_of(ps[:blocks]), blocks)
        tok += min(acc, K) + 1
        cost += trunk[K] + blocks * a.block
    return tok / len(full), cost / len(full), 1000 * tok / cost

print("\nfixed depth K, K blocks (MIMO_MTP=K):")
for K in range(Nmax + 1):
    t, c, r = simulate(lambda ps, K=K: K, blocks=K)
    print(f"  K={K}: {t:.3f} tokens/step, {c:.0f} ms/step -> {r:.3f} tok/s")
for blocks in range(1, Nmax + 1):
    print(f"\n{blocks} blocks, drafts verified while their probabilities multiply to >= theta:")
    for th in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        def K_of(ps, th=th):
            q, K = 1.0, 0
            for p in ps:
                q *= p
                if q < th: break
                K += 1
            return K
        t, c, r = simulate(K_of, blocks)
        ks = [min(K_of(s[2][:blocks]), blocks) for s in full]
        print(f"  theta {th:.1f}: {t:.3f} tokens/step, {c:.0f} ms/step -> {r:.3f} tok/s; K mean {statistics.mean(ks):.2f}")
