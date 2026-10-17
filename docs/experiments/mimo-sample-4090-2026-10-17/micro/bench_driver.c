/* sample()/sample_draft(): ede829c7's (old_*) against the working tree's, on rows a timed
 * engine dumped (MIMO_SAMPLE_DUMP) or synthetic ones with many ties.
 *   bench [rows.bin|-] [seeds] [gap_ms]
 * For each row and seed: srand(seed), the old call, rand(); srand(seed), the new call,
 * rand() -- token, kept and the next rand() must match. Then each once more timed, after
 * gap_ms of the master spinning alone (the workers asleep, as between engine steps). */
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void spin_ms(double ms) { double e = now_s() + ms * 1e-3; while (now_s() < e) {} }
static void *xmalloc(size_t n, const char *what) { void *p = malloc(n); if (!p) { fprintf(stderr, "OOM %s\n", what); exit(1); } return p; }

@SHARED@
@OLD@
@NEW@

typedef struct { int32_t kind, vocab, d, y, kept, pad; float T, top_p; double t; } Rec;

static uint64_t rng = 88172645463325252ull;
static double urand(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (rng >> 11) * (1.0 / 9007199254740992.0); }
static double nrand(void) { double u = urand() + 1e-300, v = urand(); return sqrt(-2 * log(u)) * cos(6.283185307179586 * v); }

static int cmp_d(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

int main(int argc, char **argv) {
    const char *file = argc > 1 ? argv[1] : "-";
    int seeds = argc > 2 ? atoi(argv[2]) : 16;
    double gap = argc > 3 ? atof(argv[3]) : 1.0;
    /* the workers' x87 control word against the master's, and expf's bits on them */
    {
        unsigned short cw0, cw[256] = {0};
        __asm__ volatile("fnstcw %0" : "=m"(cw0));
        int nt = 0;
        #pragma omp parallel
        {
            unsigned short c;
            __asm__ volatile("fnstcw %0" : "=m"(c));
            cw[omp_get_thread_num()] = c;
            #pragma omp single
            nt = omp_get_num_threads();
        }
        int differ = 0;
        for (int t = 0; t < nt; t++) differ += cw[t] != cw0;
        printf("threads %d, master x87 cw %#x, workers differing %d\n", nt, cw0, differ);
        int N = 1 << 22;
        float *x = xmalloc((size_t)N * 4, "x"), *a = xmalloc((size_t)N * 4, "a"), *b = xmalloc((size_t)N * 4, "b");
        for (int i = 0; i < N; i++) x[i] = (float)(-30.0 * urand());
        for (int i = 0; i < N; i++) x[i] = i % 3 ? x[i] : (float)(-88.0 * urand());
        for (int i = 0; i < N; i++) a[i] = expf(x[i]);
        #pragma omp parallel for schedule(dynamic, 512)
        for (int i = 0; i < N; i++) b[i] = expf(x[i]);
        printf("expf over %d inputs: serial and threaded bits %s\n", N, memcmp(a, b, (size_t)N * 4) ? "DIFFER" : "the same");
        free(x); free(a); free(b);
    }
    FILE *f = strcmp(file, "-") ? fopen(file, "rb") : NULL;
    if (strcmp(file, "-") && !f) { perror(file); return 1; }
    int V = 152576, rows = 0, bad = 0;
    float *lg = xmalloc((size_t)V * 4, "logits");
    double *t_old = xmalloc(100000 * sizeof(double), "t"), *t_new = xmalloc(100000 * sizeof(double), "t"), *t_eng = xmalloc(100000 * sizeof(double), "t");
    long long checks = 0, ties_rows = 0;
    long long full0 = g_full, sorts0 = g_sorts;
    long long sumn = 0;
    for (;;) {
        Rec h;
        if (f) {
            if (fread(&h, sizeof h, 1, f) != 1) break;
            if (h.vocab > V) { V = h.vocab; lg = realloc(lg, (size_t)V * 4); }
            if (fread(lg, 4, (size_t)h.vocab, f) != (size_t)h.vocab) break;
        } else {
            if (rows >= 240) break;
            memset(&h, 0, sizeof h);
            h.vocab = V;
            float Ts[3] = {0.7f, 1.0f, 0.3f}, Ps[4] = {0.9f, 0.95f, 0.5f, 1.0f};
            h.T = Ts[rows % 3]; h.top_p = Ps[(rows / 3) % 4];
            h.kind = rows % 2;
            double scale = 1.5 + 3 * urand(), q = rows % 5 == 0 ? 0 : rows % 5 == 1 ? 8 : rows % 5 == 2 ? 64 : rows % 5 == 3 ? 1024 : 4;
            for (int i = 0; i < V; i++) {
                double v = scale * nrand();
                lg[i] = (float)(q > 0 ? round(v * q) / q : v);
            }
            int top = (int)(urand() * V);
            lg[top] = (float)(scale * (rows % 7 == 0 ? 1.0 : 3 + 4 * urand()));
            if (rows % 4 == 0) for (int k = 1; k < 6; k++) lg[(top + 97 * k) % V] = lg[top];   /* ties at the top */
            h.d = (top + 97) % V;
            h.t = 0;
        }
        int nV = h.vocab;
        /* drafts: the recorded one, the argmax, the second, an entry tied with another if any, a random one */
        int dr[5], nd = 0;
        if (h.kind == 1) {
            dr[nd++] = h.d;
            int a = argmax(lg, nV);
            dr[nd++] = a;
            int b = a == 0 ? 1 : 0;
            for (int i = 0; i < nV; i++) if (i != a && lg[i] > lg[b]) b = i;
            dr[nd++] = b;
            dr[nd++] = (int)(urand() * nV);
        }
        int any_tie = 0;
        for (int s = 0; s < seeds; s++) {
            unsigned seed = 1000u * (unsigned)rows + (unsigned)s;
            if (h.kind == 0) {
                srand(seed); int a = old_sample(lg, nV, h.T, h.top_p); int ra = rand();
                long long f0 = g_full;
                srand(seed); int b = sample(lg, nV, h.T, h.top_p); int rb = rand();
                if (g_full > f0) any_tie = 1;
                checks++;
                if (a != b || ra != rb) { bad++; if (bad < 20) printf("row %d seed %u sample: old %d new %d (rand %d %d)\n", rows, seed, a, b, ra, rb); }
            } else {
                for (int q = 0; q < nd; q++) {
                    int ka, kb;
                    srand(seed); int a = old_sample_draft(lg, nV, h.T, h.top_p, dr[q], &ka); int ra = rand();
                    long long f0 = g_full;
                    srand(seed); int b = sample_draft(lg, nV, h.T, h.top_p, dr[q], &kb); int rb = rand();
                    if (g_full > f0) any_tie = 1;
                    checks++;
                    if (a != b || ka != kb || ra != rb) { bad++; if (bad < 20) printf("row %d seed %u draft %d: old %d/%d new %d/%d (rand %d %d)\n", rows, seed, dr[q], a, ka, b, kb, ra, rb); }
                }
            }
        }
        ties_rows += any_tie;
        /* timing: each once, the master alone for `gap` before */
        int ko, kn;
        spin_ms(gap);
        double t0 = now_s();
        if (h.kind == 0) old_sample(lg, nV, h.T, h.top_p); else old_sample_draft(lg, nV, h.T, h.top_p, dr[0], &ko);
        t_old[rows] = now_s() - t0;
        spin_ms(gap);
        long long cand0 = g_cand, s0 = g_sorts;
        t0 = now_s();
        if (h.kind == 0) sample(lg, nV, h.T, h.top_p); else sample_draft(lg, nV, h.T, h.top_p, dr[0], &kn);
        t_new[rows] = now_s() - t0;
        (void)cand0; (void)s0;
        t_eng[rows] = h.t;
        rows++;
    }
    /* candidates and nucleus sizes over the rows, from one more pass */
    printf("rows %d, checks %lld, mismatches %d; rows where a check fell back to the full sort: %lld\n", rows, checks, bad, ties_rows);
    printf("full sorts %lld of %lld sample_sort calls (incl. timing, top_p >= 1 rows), mean candidates of the partial ones %.1f\n",
           g_full - full0, g_sorts - sorts0, (g_sorts - g_full) ? (double)g_cand / (double)(g_sorts - g_full) : 0.0);
    (void)sumn;
    double so = 0, sn = 0, se = 0;
    for (int i = 0; i < rows; i++) { so += t_old[i]; sn += t_new[i]; se += t_eng[i]; }
    qsort(t_old, rows, sizeof(double), cmp_d); qsort(t_new, rows, sizeof(double), cmp_d);
    printf("per call: old mean %.3f ms (median %.3f, max %.3f); new mean %.3f ms (median %.3f, max %.3f)",
           1e3 * so / rows, 1e3 * t_old[rows / 2], 1e3 * t_old[rows - 1], 1e3 * sn / rows, 1e3 * t_new[rows / 2], 1e3 * t_new[rows - 1]);
    if (se > 0) printf("; in the engine (old) mean %.3f ms", 1e3 * se / rows);
    printf("\n%s\n", bad ? "FAIL" : "ok");
    return bad != 0;
}
