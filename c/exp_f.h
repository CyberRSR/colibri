/* exp_f.h -- expf with the bits MinGW's gives, without its two FLDCW.
 *
 * A MinGW-w64 (msvcrt) expf is (float)exp((double)x), and libmingwex's exp truncates
 * twice on the x87 under a control word switched to chopping and back. On a Zen 4 those
 * FLDCW make the call 5.7x slower at a few code addresses (__expl_internal starting at
 * 0x3c0 or 0xbc0 of a 4 KiB page: two of the 64 places its 64-byte-aligned object lands),
 * so a change anywhere in an engine could move the library there and slow every softmax,
 * silu and router step: MiMo's CPU phases ran 3.5% slower in such a build
 * (docs/experiments/mimo-layout-4090-2026-10-17). Here FISTTP + FILD truncate (the same
 * integers; a truncated zero's sign never reaches the result) and the rest is the
 * library's own x87 sequence, inline: every float gives the library's bits
 * (tests/test_exp_f.py tries all 2^32), about 17 ns a value against 32. Elsewhere (UCRT,
 * other C libraries, no SSE3) exp_f is expf itself. */
#ifndef EXP_F_H
#define EXP_F_H

#include <math.h>

#if defined(__MINGW32__) && defined(__x86_64__) && defined(__SSE3__) && !defined(_UCRT) && !defined(__CRT__NO_INLINE)
#define EXP_F_X87 1
static const long double exp_f_c0 = 1.44268798828125L, exp_f_c1 = 7.05260771340735992468e-6L;
static inline float exp_f(float x) {
    double d = x;
    /* libmingwex's limits: above, HUGE_VAL; below, 0; a NaN back as it came */
    if (__builtin_expect(!(d >= -0x1.74910d52d3051p+9 && d <= 0x1.62e42fefa39efp+9), 0)) {
        if (d != d) { volatile double q = d; return (float)q; }   /* quieted, as the call's conversions do */
        return d > 0 ? (float)HUGE_VAL : 0.0f;
    }
    long double r;
    long long t;
    __asm__("fldl2e\n\t"
            "fmul %%st(1),%%st\n\t"     /* x log2(e) */
            "fisttpll %1\n\t"
            "fildll %1\n\t"             /* i, its integer part */
            "fld %%st(1)\n\t"
            "fisttpll %1\n\t"
            "fildll %1\n\t"             /* xi, x's */
            "fld %%st(1)\n\t"
            "fldt %2\n\t"
            "fld %%st(2)\n\t"
            "fmul %%st(1),%%st\n\t"
            "fsubp %%st,%%st(2)\n\t"    /* f = c0 xi - i */
            "fld %%st(4)\n\t"
            "fsub %%st(3),%%st\n\t"
            "fmulp %%st,%%st(1)\n\t"
            "faddp %%st,%%st(1)\n\t"    /* f += c0 (x - xi) */
            "fldt %3\n\t"
            "fmul %%st(4),%%st\n\t"
            "faddp %%st,%%st(1)\n\t"    /* f += c1 x */
            "f2xm1\n\t"
            "fld1\n\t"
            "faddp\n\t"
            "fstp %%st(1)\n\t"
            "fscale\n\t"                /* 2^f 2^i */
            "fstp %%st(1)\n\t"
            "fstp %%st(1)\n\t"
            : "=t"(r), "=m"(t) : "m"(exp_f_c0), "m"(exp_f_c1), "0"((long double)d));
    return (float)(double)r;
}
#else
#define EXP_F_X87 0
static inline float exp_f(float x) { return expf(x); }
#endif

/* exp_f behind a call, for the few sites where GCC arranged the arithmetic around the
 * library's call: inline there, exp_f (its range branches, its asm) changed which
 * multiply-adds GCC fused, and so the bits -- qwen38's sigmoid, qwen36's shared-expert
 * gate, DeepSeek V4's DSpark gates. The sites come from GCC's final GIMPLE, where the
 * .FMA calls are, compared with the build that called the library; a call to this is
 * not the library's call either, so each site was checked, not assumed. Off the x87
 * path it is expf, called as before. */
#if EXP_F_X87
static __attribute__((noinline, unused)) float exp_f_call(float x) { return exp_f(x); }
#else
static inline float exp_f_call(float x) { return expf(x); }
#endif

#endif
