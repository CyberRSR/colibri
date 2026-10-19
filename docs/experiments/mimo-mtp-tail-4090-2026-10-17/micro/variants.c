/* The MTP tail and dense_mlp's silu: the serial loops against threaded variants
 * (static / dynamic schedules, chunk sizes), on head-like logits, with a 1.4 ms idle
 * gap before each call as the GPU head leaves the workers. */
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void spin_ms(double ms) { double e = now_s() + ms * 1e-3; while (now_s() < e) {} }
static inline float silu(float x) { return x / (1.0f + expf(-x)); }
static uint64_t rng = 88172645463325252ull;
static double urand(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (rng >> 11) * (1.0 / 9007199254740992.0); }
static double nrand(void) { double u = urand() + 1e-300, v = urand(); return sqrt(-2 * log(u)) * cos(6.283185307179586 * v); }

static int old_tail(const float *lg, int vocab, float *p) {
    int d = 0;
    for (int i = 1; i < vocab; i++) if (lg[i] > lg[d]) d = i;
    double s = 0;
    for (int i = 0; i < vocab; i++) s += expf(lg[i] - lg[d]);
    *p = (float)(1.0 / s);
    return d;
}

static double sums[4096];
static int new_tail(const float *x, int n, float *p, int CH, int dyn) {
    int d = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[d]) d = i;
    int nc = (n + CH - 1) / CH;
    float top = x[d];
    if (dyn) {
        #pragma omp parallel for schedule(dynamic, 1)
        for (int c = 0; c < nc; c++) {
            int lo = c * CH, hi = lo + CH < n ? lo + CH : n;
            double s = 0;
            for (int i = lo; i < hi; i++) s += expf(x[i] - top);
            sums[c] = s;
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < nc; c++) {
            int lo = c * CH, hi = lo + CH < n ? lo + CH : n;
            double s = 0;
            for (int i = lo; i < hi; i++) s += expf(x[i] - top);
            sums[c] = s;
        }
    }
    double s = 0;
    for (int c = 0; c < nc; c++) s += sums[c];
    *p = (float)(1.0 / s);
    return d;
}

int main(int argc, char **argv) {
    int vocab = 152576, trials = argc > 1 ? atoi(argv[1]) : 100;
    float *lg = malloc(sizeof(float) * vocab);
    int CHs[] = {256, 512, 1024, 2048};
    double t_old = 0, t_new[4][2] = {{0}};
    int pdiff[4][2] = {{0}}, ddiff = 0;
    for (int t = 0; t < trials; t++) {
        double scale = 1.5 + 3 * urand();
        for (int i = 0; i < vocab; i++) lg[i] = (float)(scale * nrand());
        int top = (int)(urand() * vocab);
        lg[top] = (float)(scale * 5 + 25 * urand());
        if (t % 3 == 0) lg[(top + 7) % vocab] = lg[top] - (float)(2 * urand());
        if (t % 17 == 0) lg[(top + 99) % vocab] = lg[top];
        float po, pn;
        spin_ms(1.4);
        double a = now_s();
        int dold = old_tail(lg, vocab, &po);
        t_old += now_s() - a;
        for (int k = 0; k < 4; k++)
            for (int dyn = 0; dyn < 2; dyn++) {
                spin_ms(1.4);
                double b = now_s();
                int dnew = new_tail(lg, vocab, &pn, CHs[k], dyn);
                t_new[k][dyn] += now_s() - b;
                if (dnew != dold) ddiff++;
                if (memcmp(&po, &pn, 4)) pdiff[k][dyn]++;
            }
    }
    printf("threads %d, %d drafts: old %.3f ms; d differs %d\n", omp_get_max_threads(), trials, 1e3 * t_old / trials, ddiff);
    for (int k = 0; k < 4; k++)
        printf("  CH %4d: static %.3f ms (p differs %d), dynamic %.3f ms (p differs %d)\n", CHs[k],
               1e3 * t_new[k][0] / trials, pdiff[k][0], 1e3 * t_new[k][1] / trials, pdiff[k][1]);

    int F = 16384, ns[] = {1, 2, 3, 4, 50};
    for (unsigned k = 0; k < sizeof ns / sizeof *ns; k++) {
        int n = ns[k];
        size_t N = (size_t)n * F;
        float *g0 = malloc(N * 4), *u = malloc(N * 4), *g1 = malloc(N * 4), *g2 = malloc(N * 4), *g3 = malloc(N * 4);
        for (size_t i = 0; i < N; i++) { g0[i] = (float)(2 * nrand()); u[i] = (float)nrand(); }
        int reps = n >= 50 ? 10 : 100;
        double so = 0, ss = 0, sd = 0;
        for (int r = 0; r < reps; r++) {
            memcpy(g1, g0, N * 4); memcpy(g2, g0, N * 4); memcpy(g3, g0, N * 4);
            spin_ms(1.4);
            double a = now_s();
            for (size_t i = 0; i < N; i++) g1[i] = silu(g1[i]) * u[i];
            so += now_s() - a;
            spin_ms(1.4);
            a = now_s();
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < (int64_t)N; i++) g2[i] = silu(g2[i]) * u[i];
            ss += now_s() - a;
            spin_ms(1.4);
            a = now_s();
            #pragma omp parallel for schedule(dynamic, 512)
            for (int64_t i = 0; i < (int64_t)N; i++) g3[i] = silu(g3[i]) * u[i];
            sd += now_s() - a;
        }
        printf("silu n=%2d: old %.3f ms, static %.3f ms, dynamic(512) %.3f ms; bits %s\n", n, 1e3 * so / reps,
               1e3 * ss / reps, 1e3 * sd / reps, memcmp(g1, g2, N * 4) || memcmp(g1, g3, N * 4) ? "DIFFER" : "same");
        free(g0); free(u); free(g1); free(g2); free(g3);
    }
    return 0;
}
