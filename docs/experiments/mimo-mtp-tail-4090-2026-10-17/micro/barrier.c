#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
/* OpenMP sync costs on this toolchain: a fork and join (each thread writes a word), two
 * barriers in a region, a parallel argmax over the vocabulary against the scalar pass. */
static volatile int touch[64];
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(void) {
    int n = 152576, CH = 1024, nc = (n + CH - 1) / CH;
    float *x = malloc(n * 4); int *best = malloc(nc * 4);
    for (int i = 0; i < n; i++) x[i] = (float)((i * 2654435761u) % 100000) * 0.001f;
    for (int rep = 0; rep < 4; rep++) {
        double t0 = now_s();
        #pragma omp parallel
        { touch[omp_get_thread_num() & 63] = rep; }
        double t1 = now_s();
        #pragma omp parallel
        {
            #pragma omp barrier
            #pragma omp barrier
        }
        double t2 = now_s();
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int c = 0; c < nc; c++) {
                int lo = c * CH, hi = lo + CH < n ? lo + CH : n, b = lo;
                for (int i = lo + 1; i < hi; i++) if (x[i] > x[b]) b = i;
                best[c] = b;
            }
        }
        double t3 = now_s();
        int b = 0; for (int i = 1; i < n; i++) if (x[i] > x[b]) b = i;
        double t4 = now_s();
        printf("T %d: fork+join %.1f us, +2 barriers %.1f us, argmax region %.1f us, serial argmax %.1f us (%d %d)\n",
               omp_get_max_threads(), 1e6 * (t1 - t0), 1e6 * (t2 - t1), 1e6 * (t3 - t2), 1e6 * (t4 - t3), b, best[0]);
    }
    return 0;
}
