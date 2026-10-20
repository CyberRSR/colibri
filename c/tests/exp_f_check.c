/* exp_f (exp_f.h) against the C library's expf, bit for bit: every float (or every STEP-th
 * low half with an argument), over the OpenMP workers, then the time a value of each in a
 * silu loop. Prints which exp_f compiled ("x87": exp_f.h's own; "expf": the library's) and
 * with which C runtime, and "ok" when no float differs. Where exp_f8 compiles (the x87
 * exp_f with AVX2 and FMA), the same for it against exp_f, eight floats a call, with how
 * many it handed to exp_f and its time a value. exp_f_check [STEP] */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "../exp_f.h"

#if defined(_UCRT)
#define CRT "ucrt"
#elif defined(__MINGW32__)
#define CRT "msvcrt"
#else
#define CRT "other"
#endif

__attribute__((noinline)) static float lib(float x) { return expf(x); }
__attribute__((noinline)) static float ours(float x) { return exp_f(x); }

int main(int argc, char **argv) {
    unsigned step = argc > 1 ? (unsigned)strtoul(argv[1], NULL, 0) : 1;
    if (!step) step = 1;
    long long bad = 0, n = 0;
    #pragma omp parallel for schedule(dynamic, 16) reduction(+:bad, n)
    for (long long hi = 0; hi < 65536; hi++)
        for (unsigned lo = 0; lo < 65536; lo += step) {
            uint32_t u = (uint32_t)hi << 16 | lo, ua, ub;
            float x, a, b;
            memcpy(&x, &u, 4);
            a = lib(x);
            b = ours(x);
            memcpy(&ua, &a, 4);
            memcpy(&ub, &b, 4);
            n++;
            if (ua != ub && bad++ < 4) {
                #pragma omp critical
                printf("x %08x: expf %08x, exp_f %08x\n", u, ua, ub);
            }
        }
    printf("%s (%s): %lld floats, %lld differ\n", EXP_F_X87 ? "x87" : "expf", CRT, n, bad);
    long long bad8 = 0;
#ifdef EXP_F8
    long long n8 = 0, handed = 0;
    #pragma omp parallel for schedule(dynamic, 16) reduction(+:bad8, n8, handed)
    for (long long hi = 0; hi < 65536; hi++)
        for (unsigned lo = 0; lo < 65536; lo += 8 * step) {
            uint32_t u[8], ua, ub;
            float x[8], r[8];
            for (int i = 0; i < 8; i++) u[i] = (uint32_t)hi << 16 | (lo + i);
            memcpy(x, u, sizeof x);
            __m256 v = _mm256_loadu_ps(x);
            _mm256_storeu_ps(r, exp_f8(v));
            for (int i = 0; i < 8; i++) {
                float b = ours(x[i]);
                memcpy(&ua, &r[i], 4);
                memcpy(&ub, &b, 4);
                if (ua != ub && bad8++ < 4) {
                    #pragma omp critical
                    printf("x %08x: exp_f8 %08x, exp_f %08x\n", u[i], ua, ub);
                }
            }
            n8 += 8;
            /* the lanes exp_f8 hands to exp_f: its two roundings apart (NaNs aside) */
            __m256 c = _mm256_min_ps(_mm256_max_ps(v, _mm256_set1_ps(-150.0f)), _mm256_set1_ps(100.0f));
            __m256d y0 = exp_f8_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(c)));
            __m256d y1 = exp_f8_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(c, 1)));
            const __m256d d = _mm256_set1_pd(0x1p-44);
            __m256 l8 = _mm256_set_m128(_mm256_cvtpd_ps(_mm256_fnmadd_pd(y1, d, y1)), _mm256_cvtpd_ps(_mm256_fnmadd_pd(y0, d, y0)));
            __m256 h8 = _mm256_set_m128(_mm256_cvtpd_ps(_mm256_fmadd_pd(y1, d, y1)), _mm256_cvtpd_ps(_mm256_fmadd_pd(y0, d, y0)));
            int same = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_castps_si256(l8), _mm256_castps_si256(h8))))
                     | _mm256_movemask_ps(_mm256_cmp_ps(v, v, _CMP_UNORD_Q));
            handed += 8 - __builtin_popcount(same);
        }
    printf("exp_f8: %lld floats, %lld differ, %lld (NaNs aside) handed to exp_f\n", n8, bad8, handed);
#endif
    /* the time a value, silu over a buffer (inline, as the engine's loops call it) */
    static float xs[4096];
    for (int i = 0; i < 4096; i++) xs[i] = 8.0f * (float)sin(i * 0.37);
    enum { N = 4000000 };
    float sa = 0, sb = 0;
    double t0 = omp_get_wtime();
    for (int i = 0; i < N; i++) { float v = xs[i & 4095]; __asm__ volatile("" : "+x"(v)); sa += v / (1.0f + expf(-v)); }
    double t1 = omp_get_wtime();
    for (int i = 0; i < N; i++) { float v = xs[i & 4095]; __asm__ volatile("" : "+x"(v)); sb += v / (1.0f + exp_f(-v)); }
    double t2 = omp_get_wtime();
    printf("silu: expf %.1f ns, exp_f %.1f ns a value (%s)\n", (t1 - t0) / N * 1e9, (t2 - t1) / N * 1e9,
           sa == sb ? "same sums" : "sums differ");
#ifdef EXP_F8
    /* a softmax's exps: e^(s - max) over a buffer, eight a call, against exp_f one by one */
    static float e1[4096], e8[4096];
    double t3 = omp_get_wtime();
    for (int k = 0; k < N / 4096; k++) {
        for (int i = 0; i < 4096; i++) e1[i] = exp_f(xs[i] - 8.0f);
        __asm__ volatile("" ::: "memory");
    }
    double t4 = omp_get_wtime();
    for (int k = 0; k < N / 4096; k++) {
        for (int i = 0; i < 4096; i += 8)
            _mm256_storeu_ps(e8 + i, exp_f8(_mm256_sub_ps(_mm256_loadu_ps(xs + i), _mm256_set1_ps(8.0f))));
        __asm__ volatile("" ::: "memory");
    }
    double t5 = omp_get_wtime();
    int same8 = !memcmp(e1, e8, sizeof e1);
    printf("softmax exps: exp_f %.2f ns, exp_f8 %.2f ns a value (%s)\n", (t4 - t3) / N * 1e9, (t5 - t4) / N * 1e9,
           same8 ? "same bits" : "bits differ");
    if (!same8) bad8++;
#endif
    puts(bad || bad8 || sa != sb ? "FAIL" : "ok");
    return bad || bad8 || sa != sb;
}
