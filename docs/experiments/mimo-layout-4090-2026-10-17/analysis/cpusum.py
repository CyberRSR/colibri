#!/usr/bin/env python3
"""Summarize cpumon output: mean busy % and effective perf % per logical processor over the window."""
import sys, collections
busy = collections.defaultdict(list); perf = collections.defaultdict(list); util = collections.defaultdict(list)
for line in open(sys.argv[1]):
    p = line.split()
    if len(p) != 4: continue
    t, kind, inst, v = float(p[0]), p[1], p[2], float(p[3])
    if '_Total' in inst: inst = 'T'
    else: inst = int(inst.split(',')[1])
    {'busy': busy, 'perf': perf, 'util': util}[kind][inst].append(v)
def m(x): return sum(x) / len(x) if x else 0
print("lp : busy%  perf%  util%")
rows = []
for lp in sorted(k for k in busy if k != 'T'):
    rows.append(f"{lp:2d}: {m(busy[lp]):5.1f} {m(perf[lp]):6.1f} {m(util[lp]):6.1f}")
for i in range(0, len(rows), 2):
    print("   ".join(rows[i:i + 2]))
print(f"total busy {m(busy['T']):.1f}%  perf {m(perf['T']):.1f}%  util {m(util['T']):.1f}%")
c0 = [lp for lp in busy if lp != 'T' and lp < 16]; c1 = [lp for lp in busy if lp != 'T' and lp >= 16]
print(f"CCD0 (lp 0-15) busy {m([m(busy[l]) for l in c0]):.1f}%  perf {m([m(perf[l]) for l in c0]):.1f}%;  CCD1 (16-31) busy {m([m(busy[l]) for l in c1]):.1f}%  perf {m([m(perf[l]) for l in c1]):.1f}%")
