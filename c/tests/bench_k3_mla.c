/* A decode row's MLA core on the CPU at Kimi K3.1 Pro's widths (kimi_k3.c, mla_forward's loop over
 * the picked rows): 96 heads, qk_nope 128 + rope 64, latent 512, v 128, 2,048 picked rows of a
 * longer cache. Times each phase of one layer and the whole of it, and checks that k3_mla_row_core
 * (the loop as mla_forward runs it) gives the bytes of the reference spelled out below.
 *   make tests/bench_k3_mla && tests/bench_k3_mla [cache positions] */
#define main coli_k3_main_unused
#include "../kimi_k3.c"
#undef main

static uint32_t rng = 20261111u;
static float rnd(void){ rng = rng * 1664525u + 1013904223u; return (float)(rng >> 9) / 4194304.0f - 1.0f; }

int main(int argc, char **argv){
    const int N = argc > 1 ? atoi(argv[1]) : 16384, H = 96, nope = 128, qr = 64, kvl = 512, vh = 128, K = 2048;
    const int qh = nope + qr;
    Cfg c; memset(&c, 0, sizeof c);
    c.qk_nope = nope; c.qk_rope = qr; c.qk_head = qh; c.kv_lora = kvl; c.v_head = vh; c.n_heads = H; c.index_topk = K;
    c.attn_scale = 1.0f / sqrtf((float)qh);
    W kvb; memset(&kvb, 0, sizeof kvb);
    kvb.fmt = 1; kvb.O = H * (nope + vh); kvb.I = kvl;
    kvb.q8 = malloc((size_t)kvb.O * kvb.I); kvb.s = malloc((size_t)kvb.O * sizeof(float));
    for(int64_t i = 0; i < (int64_t)kvb.O * kvb.I; i++) kvb.q8[i] = (int8_t)(rnd() * 127);
    for(int i = 0; i < kvb.O; i++) kvb.s[i] = 0.01f + 0.001f * rnd();
    float *Lc = malloc((size_t)N * kvl * 4), *Rc = malloc((size_t)N * qr * 4);
    for(int64_t i = 0; i < (int64_t)N * kvl; i++) Lc[i] = rnd();
    for(int64_t i = 0; i < (int64_t)N * qr; i++) Rc[i] = rnd();
    float *qv = malloc((size_t)H * qh * 4), *gv = malloc((size_t)H * vh * 4);
    for(int i = 0; i < H * qh; i++) qv[i] = rnd();
    for(int i = 0; i < H * vh; i++) gv[i] = rnd();
    int *sel = malloc((size_t)K * sizeof(int)), n = 0;
    for(int t = 0; t < N && n < K; t++) if((int64_t)(K - n) * (N - t) > 0 && (rng = rng * 1664525u + 1013904223u) % (unsigned)(N - t) < (unsigned)(K - n)) sel[n++] = t;
    const int nt = n;
    float *ctx = malloc((size_t)H * vh * 4), *ref = malloc((size_t)H * vh * 4);
    /* the reference: mla_forward's loop for one row, single-threaded */
    for(int h = 0; h < H; h++){
        const float *qp = qv + (int64_t)h * qh; const int rbase = h * (nope + vh);
        float qabs[512]; memset(qabs, 0, sizeof qabs);
        for(int d = 0; d < nope; d++) w_addrow(&kvb, rbase + d, qp[d], qabs);
        const float *qrp = qp + nope;
        float *sc = malloc((size_t)nt * 4);
        k3_mla_scores(sc, qabs, qrp, Lc, Rc, nt, kvl, qr, c.attn_scale, sel);
        softmax_(sc, nt);
        float clat[4096]; memset(clat, 0, kvl * sizeof(float));
        for(int t = 0; t < nt; t++){ const float *Lt = Lc + (int64_t)sel[t] * kvl; float s2 = sc[t]; for(int i = 0; i < kvl; i++) clat[i] += s2 * Lt[i]; }
        for(int d = 0; d < vh; d++) ref[(int64_t)h * vh + d] = w_rowdot(&kvb, rbase + nope + d, clat) * sigmoidf_(gv[(int64_t)h * vh + d]);
        free(sc);
    }
    /* softmax_ against the loop it was (exp_f a value at a time), rows of every length to 70 and 2,048 */
    {
        int bad = 0;
        for(int n2 = 1; n2 <= 2048; n2 = n2 < 70 ? n2 + 1 : 2048){
            float a[2048], b[2048];
            for(int i = 0; i < n2; i++) a[i] = b[i] = rnd() * 30.0f;
            softmax_(a, n2);
            float mx = b[0]; for(int i = 1; i < n2; i++) if(b[i] > mx) mx = b[i];
            float sm = 0; for(int i = 0; i < n2; i++){ b[i] = exp_f(b[i] - mx); sm += b[i]; } for(int i = 0; i < n2; i++) b[i] /= sm;
            bad += memcmp(a, b, (size_t)n2 * 4) != 0;
            if(n2 == 2048) break;
        }
        printf("softmax_: %d of 71 rows differ from exp_f's loop\n", bad);
        if(bad) return 1;
    }
    const int reps = 20;
    double t0 = now_s();
    for(int r = 0; r < reps; r++) k3_mla_row_core(&c, &kvb, qv, gv, Lc, Rc, sel, nt, ctx);
    const double tl = (now_s() - t0) / reps;
    int diff = 0;
    for(int i = 0; i < H * vh; i++) diff += memcmp(&ctx[i], &ref[i], 4) != 0;
    /* the phases one after the other, each over the team: which one the layer's time is */
    {
        const int ngr = (H + K3_MLA_G - 1) / K3_MLA_G;
        float *qab = malloc((size_t)H * kvl * 4), *scs = malloc((size_t)H * nt * 4), *cl = malloc((size_t)H * kvl * 4);
        double tq = 0, ts = 0, tm = 0, tc = 0, tr = 0;
        for(int r = 0; r < reps; r++){
            double a = now_s();
            #pragma omp parallel for schedule(static)
            for(int h = 0; h < H; h++){
                const float *qp = qv + (int64_t)h * qh; const int rbase = h * (nope + vh);
                float *qabs = qab + (int64_t)h * kvl; memset(qabs, 0, kvl * 4);
                for(int d = 0; d < nope; d++) w_addrow(&kvb, rbase + d, qp[d], qabs);
            }
            double b = now_s(); tq += b - a;
            #pragma omp parallel for schedule(static)
            for(int hg = 0; hg < ngr; hg++){
                const int h0 = hg * K3_MLA_G, nh = H - h0 < K3_MLA_G ? H - h0 : K3_MLA_G;
                const float *qrps[K3_MLA_G];
                for(int g = 0; g < nh; g++) qrps[g] = qv + (int64_t)(h0 + g) * qh + nope;
                k3_mla_scores_g(scs + (int64_t)h0 * nt, qab + (int64_t)h0 * kvl, qrps, nh, Lc, Rc, nt, kvl, qr, c.attn_scale, sel);
            }
            double cc = now_s(); ts += cc - b;
            #pragma omp parallel for schedule(static)
            for(int h = 0; h < H; h++) softmax_(scs + (int64_t)h * nt, nt);
            double d = now_s(); tm += d - cc;
            #pragma omp parallel for schedule(static)
            for(int h = 0; h < H; h++){
                float *clat = cl + (int64_t)h * kvl; memset(clat, 0, kvl * 4); const float *sc = scs + (int64_t)h * nt;
                for(int t = 0; t < nt; t++){ const float *Lt = Lc + (int64_t)sel[t] * kvl; float s2 = sc[t]; for(int i = 0; i < kvl; i++) clat[i] += s2 * Lt[i]; }
            }
            double e = now_s(); tc += e - d;
            #pragma omp parallel for schedule(static)
            for(int h = 0; h < H; h++){
                const int rbase = h * (nope + vh);
                for(int dd = 0; dd < vh; dd++) ctx[(int64_t)h * vh + dd] = w_rowdot(&kvb, rbase + nope + dd, cl + (int64_t)h * kvl) * sigmoidf_(gv[(int64_t)h * vh + dd]);
            }
            tr += now_s() - e;
        }
        printf("phases a layer: q.kvb %.3f ms, scores %.3f, softmax %.3f, latent sums %.3f, v rows %.3f\n",
               tq / reps * 1e3, ts / reps * 1e3, tm / reps * 1e3, tc / reps * 1e3, tr / reps * 1e3);
        free(qab); free(scs); free(cl);
    }
    printf("one layer, %d picked rows of %d, %d threads: %.3f ms (x24 MLA layers %.1f ms a token); %d of %d outputs differ "
           "from the loop's\n", nt, N, omp_get_max_threads(), tl * 1e3, tl * 24e3, diff, H * vh);
    return diff != 0;
}
