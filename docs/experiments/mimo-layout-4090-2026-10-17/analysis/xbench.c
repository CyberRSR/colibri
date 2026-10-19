/* The time a call of an exp core at each code address gen_xexp.py placed it, on one pinned CPU.
 * xbench [CPU] [CALLS_PER_COPY]: every copy first checked against exp() on 1024 inputs. */
#include <windows.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef void (*xf)(long double *r, const long double *x);
extern xf xexp_ptrs[];
extern int xexp_n;
static double xs[1024];

__attribute__((noinline)) static double run(xf f, long n) {
    double s = 0;
    for (long i = 0; i < n; i++) { long double a = xs[i & 1023], r; f(&r, &a); s += (double)r; }
    return s;
}
__attribute__((noinline)) static double run_exp(long n) {
    double s = 0;
    for (long i = 0; i < n; i++) s += exp(xs[i & 1023]);
    return s;
}
static double now(void) { LARGE_INTEGER f, t; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t); return (double)t.QuadPart / f.QuadPart; }

int main(int argc, char **argv) {
    int cpu = argc > 1 ? atoi(argv[1]) : 4;
    long n = argc > 2 ? atol(argv[2]) : 100000;
    SetThreadAffinityMask(GetCurrentThread(), 1ull << cpu);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    for (int i = 0; i < 1024; i++) xs[i] = -8.0 * (i * 0.61803398875 - (int)(i * 0.61803398875));
    for (int k = 0; k < xexp_n; k++)
        for (int i = 0; i < 1024; i++) {
            long double a = xs[i], r;
            xexp_ptrs[k](&r, &a);
            if ((double)r != exp(xs[i])) { printf("copy %d differs from exp() at %g\n", k, xs[i]); return 1; }
        }
    run_exp(n);
    double t0 = now(), s = run_exp(n), te = (now() - t0) / n * 1e9;
    printf("cpu %d: exp() %.2f ns a call (%g), __expl_internal at %p\n", cpu, te, s, (void *)exp);
    for (int k = 0; k < xexp_n; k++) {
        run(xexp_ptrs[k], n / 10);
        double t1 = now();
        run(xexp_ptrs[k], n);
        double t = (now() - t1) / n * 1e9;
        printf("%4d off %#05x %7.2f ns%s\n", k, (unsigned)((size_t)xexp_ptrs[k] & 0xfff), t, t > 1.5 * te ? "  SLOW" : "");
    }
    return 0;
}
