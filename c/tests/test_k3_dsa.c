/* Kimi K3.1's lightning indexer on the CPU (kimi_k3.c): k3_dsa_scores against its arithmetic
 * spelled out in double (each head's dot, relu, weighted sum), k3_dsa_pick against a stable
 * sort (the index_topk highest, ties to the lower position, listed ascending) on random rows,
 * rows with many ties and rows shorter than K, and k3_dsa_ln against LayerNorm in double.
 *   make tests/test_k3_dsa && tests/test_k3_dsa
 *   tests/test_k3_dsa bench [positions]   the real indexer's widths (32 heads of 128): the
 *                                         scores and the pick of one row, and a prompt step's */
#define main coli_k3_main_unused
#include "../kimi_k3.c"
#undef main

#include <stdio.h>

static uint32_t rng = 20261104u;
static float rnd(void){ rng = rng * 1664525u + 1013904223u; return (float)(rng >> 9) / 4194304.0f - 1.0f; }

typedef struct { float s; int t; } Ent;
static int ent_cmp(const void *a, const void *b){
    const Ent *x = a, *y = b;
    if(x->s != y->s) return x->s > y->s ? -1 : 1;
    return x->t - y->t;
}
static int int_cmp(const void *a, const void *b){ return *(const int*)a - *(const int*)b; }

/* the K highest of sc, ties to the lower position, ascending */
static int naive_pick(int *sel, const float *sc, int nt, int K){
    Ent *e = malloc((size_t)nt * sizeof *e);
    for(int t = 0; t < nt; t++) e[t] = (Ent){ sc[t], t };
    qsort(e, (size_t)nt, sizeof *e, ent_cmp);
    int n = nt < K ? nt : K;
    for(int i = 0; i < n; i++) sel[i] = e[i].t;
    qsort(sel, (size_t)n, sizeof *sel, int_cmp);
    free(e);
    return n;
}

static int bench(int nt){
    const int nh = 32, hd = 128, K = 2048, rows = 64;
    float *Ic = malloc((size_t)nt * hd * sizeof(float)), *q = malloc((size_t)rows * nh * hd * sizeof(float));
    float *w = malloc((size_t)rows * nh * sizeof(float));
    for(int64_t i = 0; i < (int64_t)nt * hd; i++) Ic[i] = rnd();
    for(int i = 0; i < rows * nh * hd; i++) q[i] = rnd();
    for(int i = 0; i < rows * nh; i++) w[i] = rnd();
    float *sc = malloc((size_t)nt * sizeof(float)), *tmp = malloc((size_t)nt * sizeof(float));
    int *sel = malloc((size_t)rows * K * sizeof(int));
    double t0 = now_s();
    const int reps = 5;
    for(int r = 0; r < reps; r++) k3_dsa_row(sel, q, w, Ic, nt, nh, hd, K, sc, tmp, 1);
    double t1 = (now_s() - t0) / reps;
    printf("decode row: %d positions, %d threads: %.2f ms (%.1f GMAC/s), x24 layers %.1f ms a token\n", nt,
           omp_get_max_threads(), t1 * 1e3, (double)nt * nh * hd / t1 / 1e9, t1 * 24e3);
    /* a prompt step's rows: each row over the team as in k3_dsa_select */
    t0 = now_s();
    #pragma omp parallel
    {
        float *s2 = malloc((size_t)nt * sizeof(float)), *t2 = malloc((size_t)nt * sizeof(float));
        #pragma omp for schedule(dynamic,1)
        for(int r = 0; r < rows; r++)
            k3_dsa_row(sel + (int64_t)r * K, q + (int64_t)r * nh * hd, w + (int64_t)r * nh, Ic, nt, nh, hd, K, s2, t2, 0);
        free(s2); free(t2);
    }
    double t2 = now_s() - t0;
    printf("prompt step: %d rows of %d positions: %.1f ms, %.1f GMAC/s\n", rows, nt, t2 * 1e3,
           (double)rows * nt * nh * hd / t2 / 1e9);
    free(Ic); free(q); free(w); free(sc); free(tmp); free(sel);
    return 0;
}

int main(int argc, char **argv){
    if(argc > 1 && !strcmp(argv[1], "bench")) return bench(argc > 2 ? atoi(argv[2]) : 131072);
    int bad = 0, checked = 0;
    /* scores: widths of the real indexer (32 x 128), the tiny fixture's (4 x 16), and odd ones */
    struct { int nh, hd; } shapes[] = { {32, 128}, {4, 16}, {3, 8}, {5, 12} };
    for(size_t c = 0; c < sizeof shapes / sizeof *shapes; c++){
        const int nh = shapes[c].nh, hd = shapes[c].hd, T = 45;
        float *K = malloc((size_t)T * hd * sizeof(float)), *q = malloc((size_t)nh * hd * sizeof(float));
        float *w = malloc((size_t)nh * sizeof(float)), *sc = malloc((size_t)T * sizeof(float));
        for(int i = 0; i < T * hd; i++) K[i] = rnd();
        for(int i = 0; i < nh * hd; i++) q[i] = rnd();
        for(int i = 0; i < nh; i++) w[i] = rnd();
        for(int t0 = 0; t0 < 3; t0++){
            k3_dsa_scores(sc, q, w, K, t0, T, nh, hd);
            for(int t = t0; t < T; t++){
                double s = 0, mag = 0;
                for(int h = 0; h < nh; h++){
                    double d = 0, m = 0;
                    for(int i = 0; i < hd; i++){ d += (double)q[h * hd + i] * K[t * hd + i]; m += fabs((double)q[h * hd + i] * K[t * hd + i]); }
                    s += w[h] * (d > 0 ? d : 0); mag += fabs(w[h]) * m;
                }
                checked++;
                if(fabs(sc[t - t0] - s) > 1e-5 * mag + 1e-6){
                    if(bad++ < 10) printf("scores %dx%d t0 %d t %d: %.9g, want %.9g\n", nh, hd, t0, t, sc[t - t0], s);
                }
            }
        }
        free(K); free(q); free(w); free(sc);
    }
    /* picks: random rows, tied rows (a few levels, many zeros as relu leaves them), short rows */
    for(int trial = 0; trial < 300; trial++){
        const int nt = 1 + (int)((rng = rng * 1664525u + 1013904223u) >> 20) % 3000;
        const int K = 1 + (int)((rng = rng * 1664525u + 1013904223u) >> 21) % 600;
        float *sc = malloc((size_t)nt * sizeof(float)), *tmp = malloc((size_t)nt * sizeof(float));
        int *a = malloc((size_t)(nt < K ? nt : K) * sizeof(int)), *b = malloc((size_t)(nt < K ? nt : K) * sizeof(int));
        for(int t = 0; t < nt; t++){
            float v = rnd();
            if(trial % 3 == 1) v = (float)((int)(v * 4));          /* few levels: many ties */
            if(trial % 3 == 2 && v < 0.5f) v = 0;                   /* relu's zeros */
            if(trial % 17 == 0 && t % 13 == 0) v = NAN;              /* sanitized to -inf by the scores */
            sc[t] = v != v ? -INFINITY : v;
        }
        int na = k3_dsa_pick(a, sc, tmp, nt, K), nb = naive_pick(b, sc, nt, K);
        checked++;
        if(na != nb || memcmp(a, b, (size_t)na * sizeof(int))){
            if(bad++ < 10) printf("pick trial %d nt %d K %d: %d vs %d positions, differ\n", trial, nt, K, na, nb);
        }
        free(sc); free(tmp); free(a); free(b);
    }
    /* LayerNorm */
    {
        const int n = 128; float x[128], w[128], bb[128], y[128];
        for(int i = 0; i < n; i++){ x[i] = rnd() * 5 + 2; w[i] = rnd(); bb[i] = rnd(); }
        k3_dsa_ln(y, x, w, bb, n);
        double mu = 0, var = 0; for(int i = 0; i < n; i++) mu += x[i]; mu /= n;
        for(int i = 0; i < n; i++) var += (x[i] - mu) * (x[i] - mu); var /= n;
        for(int i = 0; i < n; i++){
            double want = (x[i] - mu) / sqrt(var + 1e-6) * w[i] + bb[i];
            checked++;
            if(fabs(y[i] - want) > 1e-5 * (fabs(want) + 1)){ if(bad++ < 10) printf("ln %d: %.9g want %.9g\n", i, y[i], want); }
        }
    }
    printf("k3 dsa: %d checks, %d bad\n", checked, bad);
    return bad ? 1 : 0;
}
