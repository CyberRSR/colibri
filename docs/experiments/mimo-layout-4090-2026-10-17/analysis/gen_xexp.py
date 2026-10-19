#!/usr/bin/env python3
"""Copies of an exp core at many code addresses, for xbench.c.

gen_xexp.py mingw EXE N STEP OUT.S   libmingwex's __expl_internal, its bytes as linked in EXE
                                     (MinGW-built, with nm on PATH), its c0/c1 copied too
gen_xexp.py fisttp N STEP OUT.S      exp_f.h's sequence (FISTTP + FILD, no FLDCW), the same
                                     (long double *ret, const long double *x) interface
Copy k starts at offset k*STEP of its own 4 KiB page (STEP 16 over 256 copies: every
16-byte slot of a page; STEP 1 over 2048: every byte of the first half)."""
import struct, subprocess, sys

FISTTP = r'''
  push %rbp
  push %rbx
  sub $0x28,%rsp
  fldt (%rdx)
  fldl2e
  fmul %st(1),%st
  fisttpll 8(%rsp)
  fildll 8(%rsp)
  fld %st(1)
  fisttpll 8(%rsp)
  fildll 8(%rsp)
  fld %st(1)
  fldt c0x(%rip)
  fld %st(2)
  fmul %st(1),%st
  fsubp %st,%st(2)
  fld %st(4)
  fsub %st(3),%st
  fmulp %st,%st(1)
  faddp %st,%st(1)
  fldt c1x(%rip)
  fmul %st(4),%st
  faddp %st,%st(1)
  f2xm1
  fld1
  faddp
  fstp %st(1)
  fscale
  fstp %st(1)
  fstp %st(1)
  fstpt (%rcx)
  add $0x28,%rsp
  pop %rbx
  pop %rbp
  ret
'''
# libmingwex's 1.44268798828125L and 7.05260771340735992468e-6L as 80-bit values
C0 = bytes.fromhex("000000000000aab8ff3f")
C1 = bytes.fromhex("20faeec25f70a5eced3f")


def image(exe):
    d = open(exe, "rb").read()
    pe = struct.unpack_from("<I", d, 0x3c)[0]
    nsec, optsz = struct.unpack_from("<H", d, pe + 6)[0], struct.unpack_from("<H", d, pe + 20)[0]
    base = struct.unpack_from("<Q", d, pe + 24 + 24)[0]
    secs = []
    for i in range(nsec):
        s = pe + 24 + optsz + 40 * i
        vsz, va, rsz, rp = struct.unpack_from("<IIII", d, s + 8)
        secs.append((va, max(vsz, rsz), rp))

    def read(va, n):
        rva = va - base
        for sva, sz, rp in secs:
            if sva <= rva < sva + sz:
                return d[rp + rva - sva: rp + rva - sva + n]
        raise KeyError(hex(va))
    return read


def mingw_body(exe):
    syms = {}
    for line in subprocess.run(["nm", exe], capture_output=True, text=True, check=True).stdout.splitlines():
        p = line.split()
        if len(p) == 3:
            syms[p[2]] = int(p[0], 16)
    read = image(exe)
    start, end = syms["__expl_internal"], syms["exp"]
    code = read(start, end - start)
    # its two RIP-relative loads (fldt c0, fldt c1): db 2d disp32
    fix = [i for i in range(len(code) - 6) if code[i:i + 2] == b"\xdb\x2d"]
    assert len(fix) == 2, fix
    for i, want in zip(fix, (C0, C1)):
        disp = struct.unpack_from("<i", code, i + 2)[0]
        assert read(start + i + 6 + disp, 10) == want, "c0/c1 not where the code reads them"
    out, at = [], 0
    for i, name in zip(fix, ("c0x", "c1x")):
        out.append(".byte " + ",".join(f"0x{b:02x}" for b in code[at:i + 2]))
        out.append(f".long {name} - (. + 4)")
        at = i + 6
    out.append(".byte " + ",".join(f"0x{b:02x}" for b in code[at:]))
    return "\n".join(out)


def main():
    if sys.argv[1] == "mingw":
        body, (n, step, dst) = mingw_body(sys.argv[2]), sys.argv[3:6]
    else:
        body, (n, step, dst) = FISTTP, sys.argv[2:5]
    n, step = int(n), int(step)
    lines = [".data", ".p2align 4",
             "c0x: .byte " + ",".join(f"0x{b:02x}" for b in C0 + bytes(6)),
             "c1x: .byte " + ",".join(f"0x{b:02x}" for b in C1 + bytes(6)),
             ".globl xexp_ptrs", ".globl xexp_n", f"xexp_n: .long {n}", ".p2align 3", "xexp_ptrs:"]
    lines += [f".quad xexp_{k}" for k in range(n)]
    lines.append(".text")
    for k in range(n):
        lines += [".p2align 12"] + ([f".skip {k * step}, 0x90"] if k else []) + [f"xexp_{k}:", body]
    open(dst, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
