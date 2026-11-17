"""econ3.py <dump prefix> <anchors.npz> <eval.npz> <plain s> <row share> <overhead s>
The gate the engine has (the leading drafts whose own probability is tau or more, at most cap), over
the replayed sessions, with the block times the engine measured: a step of m rows costs
plain * (1 + share * (m - 1)) + overhead (the draft's pass and the dense matrices it displaced)."""
import collections, sys
import numpy as np

DUMP, ANCH, EVAL = sys.argv[1], sys.argv[2], sys.argv[3]
PLAIN, SHARE, OVER = float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
steps = [tuple(map(int, l.split())) for l in open(DUMP + '.steps.txt') if l.strip()]
ids_all = np.fromfile(DUMP + '.ids.i32', dtype=np.int32)
top_all = np.fromfile(DUMP + '.top1.i32', dtype=np.int32)
trans, row = [], 0
for pos0, C in steps:
    if pos0 == 0: trans.append([row, 0])
    trans[-1][1] += C; row += C
A, E = np.load(ANCH), np.load(EVAL)
meta = {(int(a), int(b)): (int(e), str(c)) for a, b, e, c in zip(A['tid'], A['pos'], A['end'], A['chan'])}
B = 7
look = {}
for tid, pos, d, pm in zip(E['tid'], E['pos'], E['draft'], E['pmax']):
    tid, pos = int(tid), int(pos)
    end, chan = meta[(tid, pos)]
    r0, N = trans[tid]
    ids, top = ids_all[r0:r0 + N], top_all[r0:r0 + N]
    nav = min(B, end - pos)
    if nav <= 0: continue
    kp = 0
    for i in range(nav):
        if d[i] != top[pos + i]: break
        kp += 1
        if top[pos + i] != ids[pos + i + 1]: break
    look[(tid, pos)] = (nav, kp, pm, chan)
msgs = collections.defaultdict(list)
for (tid, pos), (end, chan) in meta.items(): msgs[(tid, end)].append(pos)
t = lambda m: PLAIN * (1 + SHARE * (m - 1))


def run(tau, cap, only=None):
    secs = plain = toks = drafts = kept = 0
    for (tid, end), plist in msgs.items():
        a = min(plist)
        secs += t(1); plain += PLAIN; toks += 1
        while a < end:
            r = look.get((tid, a))
            if r is None or (only and r[3] != only):
                secs += t(1) + (OVER if r is not None else 0); plain += PLAIN; toks += 1; a += 1
                continue
            nav, kp, pm, _ = r
            n = 0
            while n < min(cap, nav) and pm[n] >= tau: n += 1
            k = min(kp, n)
            got = min(k + 1, end - a)
            secs += t(1 + n) + OVER; plain += PLAIN * got; toks += got; a += got
            drafts += n; kept += k
    return plain / secs, drafts, kept


print(f'a plain step {PLAIN:.3f} s, a draft row {SHARE:.2f} of it, {1e3 * OVER:.0f} ms a step for the draft model')
print('tau   ' + ''.join(f'  cap {c}' for c in (1, 2, 3, 4, 5, 7)) + '    (cap 3: drafts kept)')
for tau in (0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995):
    line = f'{tau:<6.3f}'
    for cap in (1, 2, 3, 4, 5, 7):
        s, dr, kp = run(tau, cap)
        line += f'  {s:5.3f}'
        if cap == 3: keep = f'{100 * kp / max(dr, 1):.0f}% of {dr}'
    print(line + '    ' + keep)
for chan in ('argument', 'think', 'response', 'call'):
    s, dr, kp = run(0.98, 3, chan)
    print(f'  drafts only in {chan:9s}: x{s:.3f} over the whole replay, {100 * kp / max(dr, 1):.0f}% of {dr} drafts kept')
