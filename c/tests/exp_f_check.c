/* exp_f (exp_f.h) against the C library's expf, bit for bit: every float (or every STEP-th
 * low half with an argument), over the OpenMP workers, then the time a value of each in a
 * silu loop. Prints which exp_f compiled ("x87": exp_f.h's own; "expf": the library's) and
 * with which C runtime, and "ok" when no float differs. exp_f_check [STEP] */
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
    puts(bad || sa != sb ? "FAIL" : "ok");
    return bad || sa != sb;
}
