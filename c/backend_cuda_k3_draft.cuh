// backend_cuda_k3_draft.cuh -- Kimi K3.1 Pro's draft model on the device (backend_cuda_k3.h's
// k3cu_draft_*; kimi_k3.c's k3_draft.h drives it). Included by backend_cuda_k3.cu.
//
// Inferact/Kimi-K3.1-Pro-DSpark: five Qwen3-style decoder layers of width 7168 (96 query heads
// over 16 key heads of 64, a per-head RMSNorm on q and k, RoPE in halves, SwiGLU 14336) that
// read the target's own hidden rows. A context position's five tapped rows go through fc and
// hidden_norm, and each layer's k_proj and v_proj of that give its key and value there (the
// key normed and rotated at its position): they are computed once, as the target's step ends,
// and stay in a ring of the last positions. A draft is then one pass over a block of 7 rows --
// the last token and six mask rows at the next positions -- attending the ring's rows before the
// block that lie within the window, and the block's own rows causally; norm and the target's
// head give 7 rows of logits, and the Markov bias w2 . w1[previous token] is added left to
// right as each row's token is picked.
//
// None of this is the engine's exact arithmetic and it need not be: a draft is only a guess the
// target then checks. f32 with this file's order of the sums; the matrices are held as int8 rows
// with a scale each, quantized here from the checkpoint's bf16 (|max| / 127, the engine's
// w_load rule), which the measurement on real sessions showed costs no acceptance.

#define KD_L 8          // layers at most
#define KD_R 8          // rows a matmul call takes
#define KD_KT 128       // its tile: columns, and threads (an output each)
#define KD_WMAX 4096    // the widest window

struct KdMat { int8_t *w; float *s; int O, I; };
static struct {
    int on, D, H, G, hd, I, V, R, L, T, B, win, ring;
    float eps;
    KdMat fc, q[KD_L], k[KD_L], v[KD_L], o[KD_L], gate[KD_L], up[KD_L], down[KD_L], w2;
    float *hnorm, *norm, *in_ln[KD_L], *post_ln[KD_L], *qn[KD_L], *kn[KD_L], *mask_e, *fr;
    const int8_t *head_w; const float *head_s;
    float *Kr, *Vr;                      // [L][ring][G hd]
    float *taps, *ctx, *ctxn, *kt;       // a context call's: [R][T D], [R][D], [R][D], [R][G hd]
    float *h, *xn, *qb, *kb, *vb, *att, *g1, *g2, *hh, *lg, *me, *bm, *res;   // a block's
    int *bi;
    uint16_t *stage; size_t stage_cap;   // a matrix's bf16 rows on their way to int8
    cudaStream_t st;
    size_t bytes;
    unsigned long long n_rows, n_blocks, n_picks;
    double t_rows, t_block, t_pick;
} kd;

static int kd_ok(cudaError_t e, const char *what) {   // not `checked`: a failed draft turns drafting off, nothing else
    if (e == cudaSuccess) return 1;
    fprintf(stderr, "[K3-CUDA] the draft model: %s: %s\n", what, cudaGetErrorString(e));
    cudaGetLastError();
    return 0;
}
template <typename T> static int kd_alloc(T **p, size_t n) {
    *p = nullptr;
    if (!kd_ok(cudaMalloc(p, n * sizeof(T)), "memory")) return 0;
    kd.bytes += n * sizeof(T);
    return 1;
}

// a matrix's rows from bf16 to int8 with a scale a row: a block a row
__global__ void __launch_bounds__(256) kd_quant(const uint16_t *__restrict__ src, int I, int8_t *__restrict__ w, float *__restrict__ s) {
    __shared__ float red[256];
    const uint16_t *r = src + (size_t)blockIdx.x * I;
    float m = 0.0f;
    for (int i = threadIdx.x; i < I; i += 256) m = fmaxf(m, fabsf(kv_val(r[i])));
    red[threadIdx.x] = m;
    __syncthreads();
    for (int h = 128; h; h >>= 1) { if (threadIdx.x < h) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + h]); __syncthreads(); }
    float sc = red[0] / 127.0f;
    if (!(sc >= 1e-20f)) sc = 1e-20f;
    if (!threadIdx.x) s[blockIdx.x] = sc;
    const float inv = 1.0f / sc;
    int8_t *o = w + (size_t)blockIdx.x * I;
    for (int i = threadIdx.x; i < I; i += 256) o[i] = (int8_t)fminf(127.0f, fmaxf(-127.0f, rintf(kv_val(r[i]) * inv)));
}
// Y [R][O] = (acc ? Y : 0) + s[o] sum_i X[r][i] W[o][i]: R rows at most KD_R, I a multiple of 8. A
// thread an output; the rows' columns a tile at a time through shared memory, where a thread reads a
// column's rows side by side for each of its weights.
__global__ void __launch_bounds__(KD_KT) kd_mm(const float *__restrict__ X, int ldx, int R, const int8_t *__restrict__ W,
                                               const float *__restrict__ S, int O, int I, float *__restrict__ Y, int ldy, int acc) {
    __shared__ float xs[KD_KT][KD_R];
    const int tid = threadIdx.x, o = blockIdx.x * KD_KT + tid;
    const int8_t *w = W + (size_t)(o < O ? o : 0) * I;
    float a[KD_R];
#pragma unroll
    for (int r = 0; r < KD_R; r++) a[r] = 0.0f;
    for (int k0 = 0; k0 < I; k0 += KD_KT) {
        const int col = k0 + tid;
#pragma unroll
        for (int r = 0; r < KD_R; r++) xs[tid][r] = (r < R && col < I) ? X[(size_t)r * ldx + col] : 0.0f;
        __syncthreads();
        const int n = I - k0 < KD_KT ? I - k0 : KD_KT;
        for (int kk = 0; kk < n; kk += 8) {
            const uint64_t b = *(const uint64_t *)(w + k0 + kk);
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float wv = (float)(int8_t)(b >> (8 * j));
#pragma unroll
                for (int r = 0; r < KD_R; r++) a[r] = fmaf(xs[kk + j][r], wv, a[r]);
            }
        }
        __syncthreads();
    }
    if (o >= O) return;
    const float sc = S[o];
#pragma unroll
    for (int r = 0; r < KD_R; r++)
        if (r < R) { float *y = Y + (size_t)r * ldy + o; *y = acc ? *y + a[r] * sc : a[r] * sc; }
}
// a block a row: y = x / sqrt(mean(x^2) + eps) * w
__global__ void __launch_bounds__(256) kd_rms(const float *x, int ldx, const float *w, float *y, int ldy, int D, float eps) {
    __shared__ float red[256];
    const float *r = x + (size_t)blockIdx.x * ldx;
    float s = 0.0f;
    for (int i = threadIdx.x; i < D; i += 256) s = fmaf(r[i], r[i], s);
    red[threadIdx.x] = s;
    __syncthreads();
    for (int h = 128; h; h >>= 1) { if (threadIdx.x < h) red[threadIdx.x] += red[threadIdx.x + h]; __syncthreads(); }
    const float inv = rsqrtf(red[0] / D + eps);
    float *o = y + (size_t)blockIdx.x * ldy;
    for (int i = threadIdx.x; i < D; i += 256) o[i] = r[i] * inv * w[i];
}
// rows of nh heads of hd (a thread a head): the head's RMSNorm (w [hd]), then RoPE at the row's
// position pos0 + row -- pairs (i, i + hd/2), the angle the position times fr[i], both f32 as
// transformers' Qwen3RotaryEmbedding has them
__global__ void kd_hnr(const float *src, int lds, float *dst, int ldd, int rows, int nh, int hd, const float *w,
                       const float *fr, int pos0, float eps) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * nh) return;
    const int row = idx / nh, h = idx % nh, half = hd / 2;
    const float *s = src + (size_t)row * lds + (size_t)h * hd;
    float *d = dst + (size_t)row * ldd + (size_t)h * hd;
    float ss = 0.0f;
    for (int i = 0; i < hd; i++) ss = fmaf(s[i], s[i], ss);
    const float inv = rsqrtf(ss / hd + eps), pos = (float)(pos0 + row);
    for (int i = 0; i < half; i++) {
        float sn, cs;
        sincosf(pos * fr[i], &sn, &cs);
        const float a = s[i] * inv * w[i], b = s[i + half] * inv * w[i + half];
        d[i] = a * cs - b * sn;
        d[i + half] = b * cs + a * sn;
    }
}
// Query head h of the block's row i (a block of the grid): the scores against the context
// positions first .. a-1 (first = max(lo, a + i - (win - 1)): speculators' mask, the position
// before the block and within the window of the query's own; the ring's slot p mod ring) and the
// block's rows 0 .. i, times scale; softmax; the values' weighted sum. hd 64: a thread a key for
// the scores, a dimension of a half of the keys for the sum.
#define KD_AT 128
__global__ void __launch_bounds__(KD_AT) kd_att(const float *__restrict__ Q, const float *__restrict__ Kb, const float *__restrict__ Vb,
                                                const float *__restrict__ Kr, const float *__restrict__ Vr, int a, int lo,
                                                int win, int ring, int H, int G, int hd, float scale, float *__restrict__ out) {
    __shared__ float q[128], p[KD_WMAX + 16], red[KD_AT];
    const int h = blockIdx.x, i = blockIdx.y, g = h / (H / G), tid = threadIdx.x;
    if (tid < hd) q[tid] = Q[((size_t)i * H + h) * hd + tid];
    __syncthreads();
    int first = a + i - (win - 1);
    if (first < lo) first = lo;
    const int nc = a > first ? a - first : 0, n = nc + i + 1;
    float mx = -INFINITY;
    for (int t = tid; t < n; t += KD_AT) {
        const float *k = t < nc ? Kr + ((size_t)((first + t) % ring) * G + g) * hd : Kb + ((size_t)(t - nc) * G + g) * hd;
        float s = 0.0f;
#pragma unroll 8
        for (int d = 0; d < hd; d++) s = fmaf(q[d], k[d], s);
        s *= scale;
        p[t] = s;
        mx = fmaxf(mx, s);
    }
    red[tid] = mx;
    __syncthreads();
    for (int w = KD_AT / 2; w; w >>= 1) { if (tid < w) red[tid] = fmaxf(red[tid], red[tid + w]); __syncthreads(); }
    mx = red[0];
    __syncthreads();
    float sum = 0.0f;
    for (int t = tid; t < n; t += KD_AT) { const float e = expf(p[t] - mx); p[t] = e; sum += e; }
    red[tid] = sum;
    __syncthreads();
    for (int w = KD_AT / 2; w; w >>= 1) { if (tid < w) red[tid] += red[tid + w]; __syncthreads(); }
    const float inv = 1.0f / red[0];
    __syncthreads();
    const int d = tid % hd, part = tid / hd, np = KD_AT / hd;
    float o = 0.0f;
    for (int t = part; t < n; t += np) {
        const float *v = t < nc ? Vr + ((size_t)((first + t) % ring) * G + g) * hd : Vb + ((size_t)(t - nc) * G + g) * hd;
        o = fmaf(p[t], v[d], o);
    }
    red[tid] = o;
    __syncthreads();
    if (tid < hd) {
        float r = 0.0f;
        for (int s2 = 0; s2 < np; s2++) r += red[tid + s2 * hd];
        out[((size_t)i * H + h) * hd + tid] = r * inv;
    }
}
// g = silu(g) u
__global__ void kd_swiglu(float *g, const float *u, size_t n) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) g[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
}
// a block's rows: the first the caller's, the others the mask's
__global__ void kd_fill(float *h, const float *first, const float *mask, int D, int B) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < (size_t)B * D) h[i] = i < (size_t)D ? first[i] : mask[i % D];
}
// a row of V logits: each piece of 1024's maximum and where (the lowest index of equals); the
// pieces' (one block); each piece's sum of exp(l - max); the sum's inverse: the pick's probability
__global__ void __launch_bounds__(256) kd_max(const float *lg, int V, float *bm, int *bi) {
    __shared__ float rm[256]; __shared__ int ri[256];
    const int base = blockIdx.x * 1024;
    float m = -INFINITY; int at = 0x7fffffff;
    for (int k = threadIdx.x; k < 1024; k += 256) {
        const int v = base + k;
        if (v < V && lg[v] > m) { m = lg[v]; at = v; }
    }
    rm[threadIdx.x] = m; ri[threadIdx.x] = at;
    __syncthreads();
    for (int h = 128; h; h >>= 1) {
        if (threadIdx.x < h) {
            const float b = rm[threadIdx.x + h]; const int j = ri[threadIdx.x + h];
            if (b > rm[threadIdx.x] || (b == rm[threadIdx.x] && j < ri[threadIdx.x])) { rm[threadIdx.x] = b; ri[threadIdx.x] = j; }
        }
        __syncthreads();
    }
    if (!threadIdx.x) { bm[blockIdx.x] = rm[0]; bi[blockIdx.x] = ri[0]; }
}
__global__ void kd_max2(const float *bm, const int *bi, int nb, float *res) {   // res: max, index (its bits)
    float m = -INFINITY; int at = 0x7fffffff;
    for (int b = 0; b < nb; b++) if (bm[b] > m || (bm[b] == m && bi[b] < at)) { m = bm[b]; at = bi[b]; }
    res[0] = m; res[1] = __int_as_float(at);
}
__global__ void __launch_bounds__(256) kd_sumexp(const float *lg, int V, const float *res, float *bm) {
    __shared__ float red[256];
    const int base = blockIdx.x * 1024;
    const float mx = res[0];
    float s = 0.0f;
    for (int k = threadIdx.x; k < 1024; k += 256) { const int v = base + k; if (v < V) s += expf(lg[v] - mx); }
    red[threadIdx.x] = s;
    __syncthreads();
    for (int h = 128; h; h >>= 1) { if (threadIdx.x < h) red[threadIdx.x] += red[threadIdx.x + h]; __syncthreads(); }
    if (!threadIdx.x) bm[blockIdx.x] = red[0];
}
__global__ void kd_sum2(const float *bm, int nb, float *res) {
    float s = 0.0f;
    for (int b = 0; b < nb; b++) s += bm[b];
    res[2] = 1.0f / s;
}

static void kd_mul(const float *X, int ldx, int R, const KdMat &m, float *Y, int ldy, int acc) {
    kd_mm<<<(m.O + KD_KT - 1) / KD_KT, KD_KT, 0, kd.st>>>(X, ldx, R, m.w, m.s, m.O, m.I, Y, ldy, acc);
}
static void kd_free_all() {
    KdMat *ms[] = {&kd.fc, &kd.w2};
    for (KdMat *m : ms) { cudaFree(m->w); cudaFree(m->s); }
    for (int l = 0; l < KD_L; l++) {
        KdMat *ls[] = {&kd.q[l], &kd.k[l], &kd.v[l], &kd.o[l], &kd.gate[l], &kd.up[l], &kd.down[l]};
        for (KdMat *m : ls) { cudaFree(m->w); cudaFree(m->s); }
        cudaFree(kd.in_ln[l]); cudaFree(kd.post_ln[l]); cudaFree(kd.qn[l]); cudaFree(kd.kn[l]);
    }
    float *fs[] = {kd.hnorm, kd.norm, kd.mask_e, kd.fr, kd.Kr, kd.Vr, kd.taps, kd.ctx, kd.ctxn, kd.kt, kd.h, kd.xn, kd.qb,
                   kd.kb, kd.vb, kd.att, kd.g1, kd.g2, kd.hh, kd.lg, kd.me, kd.bm, kd.res};
    for (float *p : fs) cudaFree(p);
    cudaFree(kd.bi); cudaFree(kd.stage);
    if (kd.st) cudaStreamDestroy(kd.st);
    cudaGetLastError();
    memset(&kd, 0, sizeof kd);
}

extern "C" int k3cu_draft_begin(const K3cuDraftCfg *c) {
    if (!ready || failed || !on_device() || !c) return 0;
    if (kd.D) kd_free_all();
    if (c->L < 1 || c->L > KD_L || c->block < 1 || c->block > KD_R || c->hd != 64 || c->H % c->G || c->D % 8 || c->I % 8 ||
        c->R % 8 || (c->T * c->D) % 8 || c->win < 2 || c->win > KD_WMAX || c->ring < c->win + c->block) return 0;
    kd.D = c->D; kd.H = c->H; kd.G = c->G; kd.hd = c->hd; kd.I = c->I; kd.V = c->V; kd.R = c->R; kd.L = c->L; kd.T = c->T;
    kd.B = c->block; kd.win = c->win; kd.ring = c->ring; kd.eps = c->eps;
    const size_t D = kd.D, GH = (size_t)kd.G * kd.hd, QH = (size_t)kd.H * kd.hd, B = kd.B, nb = (kd.V + 1023) / 1024;
    int ok = kd_ok(cudaStreamCreateWithFlags(&kd.st, cudaStreamNonBlocking), "stream") &&
             kd_alloc(&kd.Kr, (size_t)kd.L * kd.ring * GH) && kd_alloc(&kd.Vr, (size_t)kd.L * kd.ring * GH) &&
             kd_alloc(&kd.taps, KD_R * (size_t)kd.T * D) && kd_alloc(&kd.ctx, KD_R * D) && kd_alloc(&kd.ctxn, KD_R * D) &&
             kd_alloc(&kd.kt, KD_R * GH) && kd_alloc(&kd.h, B * D) && kd_alloc(&kd.xn, B * D) && kd_alloc(&kd.qb, B * QH) &&
             kd_alloc(&kd.kb, B * GH) && kd_alloc(&kd.vb, B * GH) && kd_alloc(&kd.att, B * QH) &&
             kd_alloc(&kd.g1, B * (size_t)kd.I) && kd_alloc(&kd.g2, B * (size_t)kd.I) && kd_alloc(&kd.hh, (B + 1) * D) &&
             kd_alloc(&kd.lg, B * (size_t)kd.V) && kd_alloc(&kd.me, (size_t)KD_R * kd.R) && kd_alloc(&kd.bm, nb) &&
             kd_alloc(&kd.bi, nb) && kd_alloc(&kd.res, 4) && kd_alloc(&kd.fr, (size_t)kd.hd / 2);
    if (ok) {
        float fr[64];
        for (int i = 0; i < kd.hd / 2; i++) fr[i] = 1.0f / powf(c->theta, (float)(2 * i) / (float)kd.hd);
        ok = kd_ok(cudaMemcpy(kd.fr, fr, (size_t)(kd.hd / 2) * 4, cudaMemcpyHostToDevice), "rope");
    }
    if (!ok) kd_free_all();
    return ok;
}
extern "C" int k3cu_draft_matrix(int kind, int layer, const uint16_t *w, int O, int I) {
    if (!kd.D || kd.on || !on_device() || !w || layer < 0 || layer >= KD_L || O < 1 || I < 8 || I % 8) return 0;
    KdMat *m = kind == K3CU_DM_FC ? &kd.fc : kind == K3CU_DM_Q ? &kd.q[layer] : kind == K3CU_DM_K ? &kd.k[layer] :
               kind == K3CU_DM_V ? &kd.v[layer] : kind == K3CU_DM_O ? &kd.o[layer] : kind == K3CU_DM_GATE ? &kd.gate[layer] :
               kind == K3CU_DM_UP ? &kd.up[layer] : kind == K3CU_DM_DOWN ? &kd.down[layer] : kind == K3CU_DM_W2 ? &kd.w2 : nullptr;
    if (!m || m->w) return 0;
    const size_t piece = (size_t)64 << 20;   // the bf16 rows go up 64 MB at a time
    if (!kd.stage) { if (!kd_ok(cudaMalloc(&kd.stage, piece), "staging")) return 0; kd.stage_cap = piece; }
    if ((size_t)I * 2 > kd.stage_cap) return 0;
    if (!kd_alloc(&m->w, (size_t)O * I) || !kd_alloc(&m->s, (size_t)O)) return 0;
    m->O = O; m->I = I;
    const int rows = (int)(kd.stage_cap / ((size_t)I * 2));
    for (int r0 = 0; r0 < O; r0 += rows) {
        const int n = O - r0 < rows ? O - r0 : rows;
        if (!kd_ok(cudaMemcpyAsync(kd.stage, w + (size_t)r0 * I, (size_t)n * I * 2, cudaMemcpyHostToDevice, kd.st), "weights")) return 0;
        kd_quant<<<n, 256, 0, kd.st>>>(kd.stage, I, m->w + (size_t)r0 * I, m->s + r0);
        if (!kd_ok(cudaStreamSynchronize(kd.st), "weights")) return 0;
    }
    return 1;
}
extern "C" int k3cu_draft_vector(int kind, int layer, const float *v, int n) {
    if (!kd.D || kd.on || !on_device() || !v || layer < 0 || layer >= KD_L || n < 1) return 0;
    float **p = kind == K3CU_DV_HNORM ? &kd.hnorm : kind == K3CU_DV_NORM ? &kd.norm : kind == K3CU_DV_IN ? &kd.in_ln[layer] :
                kind == K3CU_DV_POST ? &kd.post_ln[layer] : kind == K3CU_DV_QN ? &kd.qn[layer] : kind == K3CU_DV_KN ? &kd.kn[layer] :
                kind == K3CU_DV_MASK ? &kd.mask_e : nullptr;
    if (!p || *p) return 0;
    const int want = kind == K3CU_DV_QN || kind == K3CU_DV_KN ? kd.hd : kd.D;
    if (n != want || !kd_alloc(p, (size_t)n)) return 0;
    return kd_ok(cudaMemcpy(*p, v, (size_t)n * 4, cudaMemcpyHostToDevice), "vectors");
}
extern "C" int k3cu_draft_ready(const void *head_w, const float *head_s) {
    if (!kd.D || !on_device() || !head_w || !head_s) return 0;
    const int D = kd.D, GH = kd.G * kd.hd, QH = kd.H * kd.hd;
    int ok = kd.fc.w && kd.fc.O == D && kd.fc.I == kd.T * D && kd.w2.w && kd.w2.O == kd.V && kd.w2.I == kd.R && kd.hnorm && kd.norm && kd.mask_e;
    for (int l = 0; l < kd.L && ok; l++)
        ok = kd.q[l].w && kd.q[l].O == QH && kd.q[l].I == D && kd.k[l].w && kd.k[l].O == GH && kd.k[l].I == D &&
             kd.v[l].w && kd.v[l].O == GH && kd.v[l].I == D && kd.o[l].w && kd.o[l].O == D && kd.o[l].I == QH &&
             kd.gate[l].w && kd.gate[l].O == kd.I && kd.gate[l].I == D && kd.up[l].w && kd.up[l].O == kd.I && kd.up[l].I == D &&
             kd.down[l].w && kd.down[l].O == D && kd.down[l].I == kd.I && kd.in_ln[l] && kd.post_ln[l] && kd.qn[l] && kd.kn[l];
    if (!ok) { fprintf(stderr, "[K3-CUDA] the draft model: a matrix or a vector is missing or of another shape\n"); return 0; }
    cudaFree(kd.stage); kd.stage = nullptr; kd.stage_cap = 0;
    kd.head_w = (const int8_t *)head_w; kd.head_s = head_s;
    kd.on = 1;
    return 1;
}
extern "C" size_t k3cu_draft_bytes(void) { return kd.bytes; }

// Context rows pos0 .. pos0+n-1: taps [n][T D], a position's tapped rows side by side in the
// layers' order. Their keys and values into the ring's slots.
extern "C" int k3cu_draft_rows(const float *taps, int n, int pos0) {
    if (!kd.on || !on_device() || !taps || n < 1 || pos0 < 0) return 0;
    const double t0 = now_s();
    const int D = kd.D, TD = kd.T * D, GH = kd.G * kd.hd;
    for (int r0 = 0; r0 < n;) {
        const int slot = (pos0 + r0) % kd.ring;
        int R = n - r0 < KD_R ? n - r0 : KD_R;
        if (R > kd.ring - slot) R = kd.ring - slot;   // a call's rows in slots that follow one another
        if (!kd_ok(cudaMemcpyAsync(kd.taps, taps + (size_t)r0 * TD, (size_t)R * TD * 4, cudaMemcpyHostToDevice, kd.st), "context rows")) { kd.on = 0; return 0; }
        kd_mul(kd.taps, TD, R, kd.fc, kd.ctx, D, 0);
        kd_rms<<<R, 256, 0, kd.st>>>(kd.ctx, D, kd.hnorm, kd.ctxn, D, D, kd.eps);
        for (int l = 0; l < kd.L; l++) {
            float *Kl = kd.Kr + ((size_t)l * kd.ring + slot) * GH, *Vl = kd.Vr + ((size_t)l * kd.ring + slot) * GH;
            kd_mul(kd.ctxn, D, R, kd.k[l], kd.kt, GH, 0);
            kd_hnr<<<(R * kd.G + 127) / 128, 128, 0, kd.st>>>(kd.kt, GH, Kl, GH, R, kd.G, kd.hd, kd.kn[l], kd.fr, pos0 + r0, kd.eps);
            kd_mul(kd.ctxn, D, R, kd.v[l], Vl, GH, 0);
        }
        r0 += R;
    }
    if (!kd_ok(cudaGetLastError(), "context kernels") || !kd_ok(cudaStreamSynchronize(kd.st), "context rows")) { kd.on = 0; return 0; }
    kd.n_rows += (unsigned long long)n; kd.t_rows += now_s() - t0;
    return 1;
}
// A draft's pass: the block of rows at positions a .. a+B-1 (first: the embedding of the token at
// a; the others the mask's), over the ring's positions lo .. a-1. The rows' logits stay here for
// k3cu_draft_pick.
extern "C" int k3cu_draft_block(int a, int lo, const float *first) {
    if (!kd.on || !on_device() || !first || a < 0 || lo < 0 || lo > a) return 0;
    const double t0 = now_s();
    const int D = kd.D, B = kd.B, GH = kd.G * kd.hd, QH = kd.H * kd.hd, I = kd.I;
    const float scale = 1.0f / sqrtf((float)kd.hd);
    float *row0 = kd.hh + (size_t)B * D;   // the first row's embedding, on its way
    if (!kd_ok(cudaMemcpyAsync(row0, first, (size_t)D * 4, cudaMemcpyHostToDevice, kd.st), "block")) { kd.on = 0; return 0; }
    kd_fill<<<(unsigned)(((size_t)B * D + 255) / 256), 256, 0, kd.st>>>(kd.h, row0, kd.mask_e, D, B);
    for (int l = 0; l < kd.L; l++) {
        kd_rms<<<B, 256, 0, kd.st>>>(kd.h, D, kd.in_ln[l], kd.xn, D, D, kd.eps);
        kd_mul(kd.xn, D, B, kd.q[l], kd.qb, QH, 0);
        kd_mul(kd.xn, D, B, kd.k[l], kd.kb, GH, 0);
        kd_mul(kd.xn, D, B, kd.v[l], kd.vb, GH, 0);
        kd_hnr<<<(B * kd.H + 127) / 128, 128, 0, kd.st>>>(kd.qb, QH, kd.qb, QH, B, kd.H, kd.hd, kd.qn[l], kd.fr, a, kd.eps);
        kd_hnr<<<(B * kd.G + 127) / 128, 128, 0, kd.st>>>(kd.kb, GH, kd.kb, GH, B, kd.G, kd.hd, kd.kn[l], kd.fr, a, kd.eps);
        kd_att<<<dim3(kd.H, B), KD_AT, 0, kd.st>>>(kd.qb, kd.kb, kd.vb, kd.Kr + (size_t)l * kd.ring * GH, kd.Vr + (size_t)l * kd.ring * GH,
                                                  a, lo, kd.win, kd.ring, kd.H, kd.G, kd.hd, scale, kd.att);
        kd_mul(kd.att, QH, B, kd.o[l], kd.h, D, 1);
        kd_rms<<<B, 256, 0, kd.st>>>(kd.h, D, kd.post_ln[l], kd.xn, D, D, kd.eps);
        kd_mul(kd.xn, D, B, kd.gate[l], kd.g1, I, 0);
        kd_mul(kd.xn, D, B, kd.up[l], kd.g2, I, 0);
        kd_swiglu<<<(unsigned)(((size_t)B * I + 255) / 256), 256, 0, kd.st>>>(kd.g1, kd.g2, (size_t)B * I);
        kd_mul(kd.g1, I, B, kd.down[l], kd.h, D, 1);
    }
    kd_rms<<<B, 256, 0, kd.st>>>(kd.h, D, kd.norm, kd.hh, D, D, kd.eps);
    kd_mm<<<(kd.V + KD_KT - 1) / KD_KT, KD_KT, 0, kd.st>>>(kd.hh, D, B, kd.head_w, kd.head_s, kd.V, D, kd.lg, kd.V, 0);
    if (!kd_ok(cudaGetLastError(), "block kernels") || !kd_ok(cudaStreamSynchronize(kd.st), "block")) { kd.on = 0; return 0; }
    kd.n_blocks++; kd.t_block += now_s() - t0;
    return 1;
}
// Row `slot` of the block's logits plus the Markov bias w2 . me (me [R]: w1's row of the token
// before it); its first maximum and that token's probability.
extern "C" int k3cu_draft_pick(int slot, const float *me, int *id, float *prob) {
    if (!kd.on || !on_device() || !me || !id || !prob || slot < 0 || slot >= kd.B) return 0;
    const double t0 = now_s();
    const int nb = (kd.V + 1023) / 1024;
    float *lg = kd.lg + (size_t)slot * kd.V, res[4];
    if (!kd_ok(cudaMemcpyAsync(kd.me, me, (size_t)kd.R * 4, cudaMemcpyHostToDevice, kd.st), "pick")) { kd.on = 0; return 0; }
    kd_mul(kd.me, kd.R, 1, kd.w2, lg, kd.V, 1);
    kd_max<<<nb, 256, 0, kd.st>>>(lg, kd.V, kd.bm, kd.bi);
    kd_max2<<<1, 1, 0, kd.st>>>(kd.bm, kd.bi, nb, kd.res);
    kd_sumexp<<<nb, 256, 0, kd.st>>>(lg, kd.V, kd.res, kd.bm);
    kd_sum2<<<1, 1, 0, kd.st>>>(kd.bm, nb, kd.res);
    if (!kd_ok(cudaGetLastError(), "pick kernels") ||
        !kd_ok(cudaMemcpyAsync(res, kd.res, 12, cudaMemcpyDeviceToHost, kd.st), "pick") ||
        !kd_ok(cudaStreamSynchronize(kd.st), "pick")) { kd.on = 0; return 0; }
    memcpy(id, &res[1], 4);
    *prob = res[2];
    kd.n_picks++; kd.t_pick += now_s() - t0;
    return 1;
}
extern "C" void k3cu_draft_free(void) {
    if (ready && kd.D && on_device()) kd_free_all();
}
static void kd_report() {
    if (kd.n_blocks || kd.n_rows)
        fprintf(stderr, "[K3-CUDA] the draft model: %llu context rows %.2fs, %llu blocks %.2fs, %llu picks %.2fs (%.2f GB on the device)\n",
                kd.n_rows, kd.t_rows, kd.n_blocks, kd.t_block, kd.n_picks, kd.t_pick, kd.bytes / 1e9);
}
