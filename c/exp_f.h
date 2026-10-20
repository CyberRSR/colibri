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

#if EXP_F_X87 && defined(__AVX2__) && defined(__FMA__)
/* exp_f8: exp_f of eight floats at once, the same bits, 1.0 ns a value against 17. The x87
 * result is within about 2^-62 of e^x, so exp_f is e^x rounded to float -- through double,
 * which can only matter within 2^-53 of a point halfway between two floats. Here e^x in
 * double (2^n times a degree-12 polynomial on |r| <= ln2/2: within 2^-50) is rounded to float
 * from 2^-44 below it and from 2^-44 above: where the two agree, every number between them
 * rounds the same way, the library's among them. The rest -- a halfway point within 2^-44 of
 * e^x, about one value in 700,000, or a NaN -- goes to exp_f itself. Arguments outside
 * [-150, 100] give what -150 and 100 give, 0 and infinity, as the library's do.
 * tests/exp_f_check.c tries all 2^32 floats. */
#include <immintrin.h>
#define EXP_F8 1
static inline __m256d exp_f8_pd(__m256d x) {
    const __m256d shift = _mm256_set1_pd(0x1.8p52);
    __m256d t = _mm256_fmadd_pd(x, _mm256_set1_pd(0x1.71547652b82fep0), shift);   /* n = round(x log2 e) in its low bits */
    __m256d n = _mm256_sub_pd(t, shift);
    __m256d r = _mm256_fnmadd_pd(n, _mm256_set1_pd(0x1.62e42fefa39efp-1), x);      /* x - n ln2, ln2 in two parts */
    r = _mm256_fnmadd_pd(n, _mm256_set1_pd(0x1.abc9e3b39803fp-56), r);
    __m256d p = _mm256_set1_pd(1.0 / 479001600.0);
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0 / 39916800.0));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0 / 3628800.0));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0 / 362880.0));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0 / 40320.0));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0 / 5040.0));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0 / 720.0));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0 / 120.0));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0 / 24.0));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0 / 6.0));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(0.5));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0));
    __m256i e = _mm256_slli_epi64(_mm256_add_epi64(_mm256_castpd_si256(t), _mm256_set1_epi64x(1023)), 52);   /* 2^n */
    return _mm256_mul_pd(p, _mm256_castsi256_pd(e));
}
static inline __m256 exp_f8(__m256 x) {
    __m256 c = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-150.0f)), _mm256_set1_ps(100.0f));
    __m256d y0 = exp_f8_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(c)));
    __m256d y1 = exp_f8_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(c, 1)));
    const __m256d d = _mm256_set1_pd(0x1p-44);
    __m256 lo = _mm256_set_m128(_mm256_cvtpd_ps(_mm256_fnmadd_pd(y1, d, y1)), _mm256_cvtpd_ps(_mm256_fnmadd_pd(y0, d, y0)));
    __m256 hi = _mm256_set_m128(_mm256_cvtpd_ps(_mm256_fmadd_pd(y1, d, y1)), _mm256_cvtpd_ps(_mm256_fmadd_pd(y0, d, y0)));
    int same = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_castps_si256(lo), _mm256_castps_si256(hi))))
             & ~_mm256_movemask_ps(_mm256_cmp_ps(x, x, _CMP_UNORD_Q));
    if (__builtin_expect(same != 0xff, 0)) {
        float v[8], a[8];
        _mm256_storeu_ps(v, lo);
        _mm256_storeu_ps(a, x);
        for (int i = 0; i < 8; i++)
            if (!(same >> i & 1)) v[i] = exp_f(a[i]);
        return _mm256_loadu_ps(v);
    }
    return lo;
}
#endif

#endif
