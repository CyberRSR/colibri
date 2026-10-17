"""Compare a new MIMO_MTP_TRACE ("P N K a p1..pN") with a recorded one ("P N a p1..pN" or the
new format): every step's position, drafts, kept count and printed probabilities, and the
cut a MIMO_MTP_MIN_P of 0.4 / 0.6 would make from the printed probabilities (float32
products, as mtp_step multiplies them), with how close any product came to the threshold.

usage: compare_traces.py recorded.trace new.trace"""
import sys
import numpy as np


def load(path):
    steps = []
    for line in open(path):
        f = line.split()
        if len(f) < 4:
            continue
        P, N = int(f[0]), int(f[1])
        if len(f) == 4 + N:
            K, a, ps = int(f[2]), int(f[3]), f[4:4 + N]
        else:
            K, a, ps = None, int(f[2]), f[3:3 + N]
        steps.append((P, N, K, a, ps))
    return steps


def cut(ps, theta):
    q, K = np.float32(1), 0
    margin = float("inf")
    for p in ps:
        q = np.float32(q * np.float32(float(p)))
        margin = min(margin, abs(float(q) - theta))
        if q < np.float32(theta):
            break
        K += 1
    return K, margin


rec, new = load(sys.argv[1]), load(sys.argv[2])
print(f"recorded {len(rec)} steps, new {len(new)} steps")
same_lines = sum(1 for r, n in zip(rec, new) if (r[0], r[1], r[3], r[4]) == (n[0], n[1], n[3], n[4]))
first_diff = next((i for i, (r, n) in enumerate(zip(rec, new)) if (r[0], r[1], r[3], r[4]) != (n[0], n[1], n[3], n[4])), None)
print(f"steps identical (P, N, kept, printed p): {same_lines} of {min(len(rec), len(new))}"
      + ("" if first_diff is None else f"; first difference at step {first_diff}: {rec[first_diff]} vs {new[first_diff]}"))
for theta in (0.4, 0.6):
    kr = [cut(s[4], theta) for s in rec]
    kn = [cut(s[4], theta) for s in new]
    same = sum(1 for a, b in zip(kr, kn) if a[0] == b[0])
    print(f"cut at {theta}: same verified count in {same} of {min(len(kr), len(kn))} steps; "
          f"closest product to {theta}: recorded {min(m for _, m in kr):.2e}, new {min(m for _, m in kn):.2e}")
