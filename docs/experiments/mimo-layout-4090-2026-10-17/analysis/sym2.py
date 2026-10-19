#!/usr/bin/env python3
"""Attribute winsample2 output: exe functions (nm), DLL exports, per-thread CPU time.
sym2.py SAMPLES EXE [--threads 0] [--top 30] [--blocks 20] [--group]"""
import argparse, bisect, collections, os, struct, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument('samples'); ap.add_argument('exe')
ap.add_argument('--threads', default='0', help='comma list, ranges a-b, or all')
ap.add_argument('--top', type=int, default=30); ap.add_argument('--blocks', type=int, default=20)
ap.add_argument('--block', type=int, default=64)
ap.add_argument('--table', action='store_true', help='print the thread table')
ap.add_argument('--func', default=None, help='histogram of offsets inside this function')
a = ap.parse_args()
nm = os.environ.get('NM', 'nm')
syms = []
for line in subprocess.run([nm, '-n', a.exe], capture_output=True, text=True).stdout.splitlines():
    p = line.split()
    if len(p) == 3 and p[1] in 'tT' and not p[2].startswith('.'):
        syms.append((int(p[0], 16), p[2]))
addrs = [x for x, _ in syms]
IMAGE = 0x140000000

def pe_exports(path):
    try:
        d = open(path, 'rb').read()
    except OSError:
        return []
    pe = struct.unpack_from('<I', d, 0x3c)[0]
    nsec = struct.unpack_from('<H', d, pe + 6)[0]; optsz = struct.unpack_from('<H', d, pe + 20)[0]
    opt = pe + 24
    magic = struct.unpack_from('<H', d, opt)[0]
    ddir = opt + (112 if magic == 0x20b else 96)
    erva, esz = struct.unpack_from('<II', d, ddir)
    secs = []
    for i in range(nsec):
        s = pe + 24 + optsz + 40 * i
        vsz, va, rsz, rp = struct.unpack_from('<IIII', d, s + 8)
        secs.append((va, max(vsz, rsz), rp))
    def off(rva):
        for va, sz, rp in secs:
            if va <= rva < va + sz: return rva - va + rp
        return None
    if not erva: return []
    o = off(erva)
    if o is None: return []
    nfun, nnames, afun, anames, aord = struct.unpack_from('<IIIII', d, o + 20)
    fo, no, oo = off(afun), off(anames), off(aord)
    out = []
    for i in range(nnames):
        nrva = struct.unpack_from('<I', d, no + 4 * i)[0]
        ordi = struct.unpack_from('<H', d, oo + 2 * i)[0]
        frva = struct.unpack_from('<I', d, fo + 4 * ordi)[0]
        if erva <= frva < erva + esz: continue  # forwarder
        no2 = off(nrva); e = d.index(b'\0', no2)
        out.append((frva, d[no2:e].decode(errors='replace')))
    out.sort()
    return out

mods, thr, ends, samples, window = [], {}, {}, [], None
with open(a.samples) as f:
    for line in f:
        p = line.split()
        if not p: continue
        if p[0] == 'mod': mods.append((int(p[1], 16), int(p[2], 16), ' '.join(p[3:])))
        elif p[0] == 'thr': thr[int(p[1])] = (int(p[2]), int(p[3]), int(p[4], 16), int(p[5]))
        elif p[0] == 'end': ends[int(p[1])] = tuple(int(x) for x in p[2:5])
        elif p[0] == 'window': window = float(p[1])
        elif p[0] == 'base': mods.append((int(p[1], 16), 0x1000000, a.exe))
        else: samples.append((int(p[0]), int(p[1], 16)))
exe_base = mods[0][0]
mods_sorted = sorted(mods)
mbases = [m[0] for m in mods_sorted]
exp_cache = {}

def where(rip):
    """(group, name, block-key)"""
    if exe_base <= rip < exe_base + mods[0][1]:
        r = rip - exe_base + IMAGE
        i = bisect.bisect_right(addrs, r) - 1
        if i < 0: return ('exe', '?', None)
        return ('exe', syms[i][1], (syms[i][1], (r - addrs[i]) // a.block * a.block))
    i = bisect.bisect_right(mbases, rip) - 1
    if i >= 0 and rip < mods_sorted[i][0] + mods_sorted[i][1]:
        base, size, path = mods_sorted[i]
        mn = os.path.basename(path)
        if path not in exp_cache: exp_cache[path] = pe_exports(path)
        ex = exp_cache[path]
        rva = rip - base
        j = bisect.bisect_right([x for x, _ in ex], rva) - 1
        fn = ex[j][1] if j >= 0 else '?'
        return (mn, f'{mn}!{fn}', None)
    return ('?', '(unknown)', None)

def thread_kind(i):
    if i not in thr: return '?'
    g, n, _ = where(thr[i][2])
    return n

def parse_threads(s, n):
    if s == 'all': return set(range(n))
    out = set()
    for part in s.split(','):
        if '-' in part:
            x, y = part.split('-'); out |= set(range(int(x), int(y) + 1))
        else: out.add(int(part))
    return out

nthr = len(thr) if thr else 1 + max(t for t, _ in samples)
if a.table:
    print(f"window {window}s; {len(thr)} threads")
    for i in sorted(thr):
        tid, ideal, start, prio = thr[i]
        cyc, k, u = ends.get(i, (0, 0, 0))
        ns = sum(1 for t, _ in samples if t == i)
        print(f"  {i:3d} tid {tid:6d} ideal {ideal:2d} prio {prio:3d} cpu {(k + u) / 1e7:7.2f}s (k {k / 1e7:6.2f}) cyc {cyc / 1e9:7.2f}G  samples {ns:6d}  {thread_kind(i)}")
want = parse_threads(a.threads, nthr)
cnt = collections.Counter(); blocks = collections.Counter(); grp = collections.Counter(); total = 0
fo = collections.Counter()
for t, rip in samples:
    if t not in want: continue
    g, n, b = where(rip)
    total += 1; cnt[n] += 1; grp[g] += 1
    if b: blocks[b] += 1
    if a.func and n == a.func:
        r = rip - exe_base + IMAGE
        fo[r - addrs[bisect.bisect_right(addrs, r) - 1]] += 1
print(f"{total} samples of threads {a.threads}")
print("by module: " + ", ".join(f"{g} {100 * v / total:.1f}%" for g, v in grp.most_common(8)))
for n, v in cnt.most_common(a.top):
    print(f"  {100 * v / total:5.1f}%  {v:7d}  {n}")
print("hot blocks:")
for (n, off), v in blocks.most_common(a.blocks):
    print(f"  {100 * v / total:5.1f}%  {n}+0x{off:x}")
if a.func:
    print(f"offsets in {a.func}:")
    for off, v in sorted(fo.items()):
        print(f"  +0x{off:x} {v}")
