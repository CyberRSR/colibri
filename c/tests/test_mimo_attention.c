/* attend_keys (a few rows over a long full-attention cache) against attend_heads, the loop it
 * stands in for, bit for bit: MiMo-V2.6-Pro's full-layer shape (128 heads, 8 key/value heads,
 * dims 192 and 128) for 1-4 rows at several positions, the tiny fixture's (2 heads a key head,
 * spare lanes), a 12-head group with sinks and a 24-head one (three groups of eight lanes).
 * The data has tied scores (repeated key rows) and heads whose scores spread far enough for
 * exps that round to zero.
 *   test_mimo_attention                 the checks
 *   test_mimo_attention bench KEYS ROWS the Pro shape at KEYS keys, ROWS rows: ms a call of each */
#define main mimo_main_unused
#include "../mimo.c"
#undef main

#if !defined(__AVX2__)
int main(void) {
    puts("skipped: attend_keys and attend_heads are the AVX2 path");
    return 0;
}
#else
static uint64_t g_rng = 0x9e3779b97f4a7c15ull;
static float frand(void) {   /* about N(0, 1): a sum of four uniforms */
    float s = 0;
    for (int k = 0; k < 4; k++) {
        g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
        s += (float)(g_rng >> 40) / (float)(1 << 24) - 0.5f;
    }
    return s * 1.7320508f;
}

typedef struct { int nh, kvh, hd, vd, sink; } Shape;

/* both paths on one random case; 0 when every output byte agrees */
static int one(Shape sh, int n, int p0, int reps, double *ms_heads, double *ms_keys) {
    int nk = p0 + n, qd = sh.nh * sh.hd, kd = sh.kvh * sh.hd, vdd = sh.kvh * sh.vd, rows = qd + kd + vdd;
    float *qkv = xmalloc((size_t)n * rows * sizeof(float), "qkv");
    float *K = xmalloc((size_t)nk * kd * sizeof(float), "K"), *V = xmalloc((size_t)nk * vdd * sizeof(float), "V");
    float *sinks = sh.sink ? xmalloc((size_t)sh.nh * sizeof(float), "sinks") : NULL;
    const float **Kr = xmalloc((size_t)nk * sizeof *Kr, "Kr"), **Vr = xmalloc((size_t)nk * sizeof *Vr, "Vr");
    for (size_t i = 0; i < (size_t)n * rows; i++) qkv[i] = frand();
    for (int t = 0; t < n; t++)   /* every fifth head sharp: scores far apart, exps that round to zero */
        for (int h = 0; h < sh.nh; h += 5)
            for (int i = 0; i < sh.hd; i++) qkv[(size_t)t * rows + (size_t)h * sh.hd + i] *= 12.0f;
    for (int j = 0; j < nk; j++) {
        if (j % 7 == 3 && j > 3) {   /* a repeated key row: tied scores */
            memcpy(K + (size_t)j * kd, K + (size_t)(j - 3) * kd, (size_t)kd * sizeof(float));
        } else
            for (int i = 0; i < kd; i++) K[(size_t)j * kd + i] = frand();
        for (int i = 0; i < vdd; i++) V[(size_t)j * vdd + i] = frand();
        Kr[j] = K + (size_t)j * kd; Vr[j] = V + (size_t)j * vdd;
    }
    if (sinks) for (int h = 0; h < sh.nh; h++) sinks[h] = 2.0f * frand();
    size_t out = (size_t)n * sh.nh * sh.vd;
    float *a = xmalloc(out * sizeof(float), "heads"), *b = xmalloc(out * sizeof(float), "keys");
    memset(a, 0xff, out * sizeof(float)); memset(b, 0xfe, out * sizeof(float));
    float scale = 1.0f / sqrtf((float)sh.hd);
    double t0 = now_s();
    for (int r = 0; r < reps; r++)
        attend_heads(qkv, rows, n, p0, 0, 0, Kr, Vr, nk, sinks, scale, sh.nh, sh.kvh, sh.hd, sh.vd, a);
    double t1 = now_s();
    for (int r = 0; r < reps; r++)
        attend_keys(qkv, rows, n, p0, Kr, Vr, nk, sinks, scale, sh.nh, sh.kvh, sh.hd, sh.vd, b);
    double t2 = now_s();
    if (ms_heads) *ms_heads = (t1 - t0) / reps * 1e3;
    if (ms_keys) *ms_keys = (t2 - t1) / reps * 1e3;
    size_t bad = 0, first = 0;
    for (size_t i = 0; i < out; i++)
        if (memcmp(a + i, b + i, sizeof(float)) && !bad++) first = i;
    if (bad) {
        uint32_t ua, ub;
        memcpy(&ua, a + first, 4); memcpy(&ub, b + first, 4);
        printf("FAIL nh %d kvh %d hd %d vd %d sink %d, %d rows at %d: %zu of %zu outputs differ, the first "
               "(row %zu, head %zu, column %zu) %08x against %08x\n", sh.nh, sh.kvh, sh.hd, sh.vd, sh.sink, n, p0,
               bad, out, first / ((size_t)sh.nh * sh.vd), first / sh.vd % sh.nh, first % sh.vd, ua, ub);
    }
    free(qkv); free(K); free(V); free(sinks); free(Kr); free(Vr); free(a); free(b);
    return bad != 0;
}

int main(int argc, char **argv) {
    const Shape pro = { 128, 8, 192, 128, 0 }, tiny = { 4, 2, 48, 32, 1 }, odd = { 24, 2, 32, 24, 1 },
                wide = { 48, 2, 32, 16, 0 };
    if (argc >= 4 && !strcmp(argv[1], "bench")) {
        int nk = atoi(argv[2]), n = atoi(argv[3]), reps = argc > 4 ? atoi(argv[4]) : 5;
        double h, k;
        one(pro, n, nk - n, 1, NULL, NULL);   /* warm: pages, scratch */
        int bad = one(pro, n, nk - n, reps, &h, &k);
        printf("%d keys, %d rows, %d threads: attend_heads %.2f ms, attend_keys %.2f ms (%.1fx)%s\n", nk, n,
               omp_get_max_threads(), h, k, h / k, bad ? ", OUTPUTS DIFFER" : ", the same bytes");
        return bad;
    }
    int fails = 0, cases = 0;
    const int pro_at[] = { 0, 1, 63, 64, 65, 1000, 4097 };
    for (int n = 1; n <= 4; n++)
        for (size_t i = 0; i < sizeof pro_at / sizeof *pro_at; i++, cases++) fails += one(pro, n, pro_at[i], 1, NULL, NULL);
    const int small_at[] = { 0, 2, 7, 64, 129, 1023, 3000 };
    for (int n = 1; n <= 4; n++)
        for (size_t i = 0; i < sizeof small_at / sizeof *small_at; i++, cases += 3) {
            fails += one(tiny, n, small_at[i], 1, NULL, NULL);
            fails += one(odd, n, small_at[i], 1, NULL, NULL);
            fails += one(wide, n, small_at[i], 1, NULL, NULL);
        }
    printf("%d cases, %d differ\n", cases, fails);
    puts(fails ? "FAIL" : "ok");
    return fails != 0;
}
#endif
