#!/usr/bin/env python3
"""winsample3 output: the main thread's samples in __expl_internal, by offset (the two FLDCW
end at +0x3a and +0x44) and by RSP mod 4 KiB. rsp_cw.py SAMPLES EXE"""
import bisect, collections, subprocess, sys
syms = []
for line in subprocess.run(["nm", "-n", sys.argv[2]], capture_output=True, text=True).stdout.splitlines():
    p = line.split()
    if len(p) == 3 and p[1] in "tT" and not p[2].startswith("."):
        syms.append((int(p[0], 16), p[2]))
addrs = [a for a, _ in syms]
base, tot, ins = None, 0, []
for line in open(sys.argv[1]):
    p = line.split()
    if not p: continue
    if p[0] == "mod":
        base = base or int(p[1], 16)
        continue
    if p[0] != "0": continue
    tot += 1
    rip, rsp = int(p[1], 16) - base + 0x140000000, int(p[2], 16)
    i = bisect.bisect_right(addrs, rip) - 1
    if syms[i][1] == "__expl_internal": ins.append((rip - addrs[i], rsp & 0xfff))
cw = [r for o, r in ins if o in (0x3a, 0x44)]
print(f"{sys.argv[1]}: {tot} main-thread samples, {len(ins)} in __expl_internal ({100 * len(ins) / tot:.2f}%), "
      f"{len(cw)} right after its two FLDCW (+0x3a, +0x44)")
print("  RSP mod 4096 of the samples in __expl_internal: " +
      ", ".join(f"{r:#05x} {n}" for r, n in collections.Counter(r for _, r in ins).most_common(8)))
print("  RSP mod 4096 right after the FLDCW: " + ", ".join(f"{r:#05x} {n}" for r, n in collections.Counter(cw).most_common(6)))
