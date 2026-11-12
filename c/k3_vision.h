/* k3_vision.h -- Kimi K3.1's vision tower: MoonViT 3D and the patchmergerv2 projector, images only.
 *
 * Mirrors MoonViT3dPretrainedModel and PatchMergerMLPV2 of Moonshot's modeling_kimi_k3.py (shipped
 * with the Kimi-K3 release; K3.1's vision_config is the same) for one image (t = 1):
 *
 *   pixels     the gateway sends the processor's resized and padded picture (tools/kimi_image.py:
 *              media_utils' navit_resize_image and image_to_np), RGB uint8, H x W x 3 with H and
 *              W multiples of 28; normalized here as media_utils.normalize does it
 *   patches    14 x 14, row-major over the picture, each (channel, row, column): Conv2d(3, 1024,
 *              14, stride 14) without a bias is a Linear over those 588 values
 *   positions  pos_emb.weight [64][64][1024], resized to the grid bilinearly (align_corners
 *              False, F.interpolate) unless the grid is 64 x 64, added
 *   blocks     x += wo(attn(rope(wqkv(rms(x))))), x += fc1(gelu_tanh(fc0(rms(x)))); 12 heads of
 *              128 (qkv_hidden_size 1536, wider than the 1024 of the stream), every patch of the
 *              picture attends every other; 2-D RoPE on interleaved pairs: pair 2i turns by the
 *              column times 10000^(-4i/128), pair 2i+1 by the row (Rope2DPosEmbRepeated)
 *   final      RMSNorm
 *   merge      the 2 x 2 neighbours side by side, 4096 wide, row-major over the merged grid
 *              (tpool_patch_merger; the temporal mean of one frame is the frame)
 *   projector  Linear 4096 -> 4096, exact (erf) GELU, Linear 4096 -> 7168, RMSNorm with
 *              projector_ln_eps: one row of the language model's input a merged patch
 *
 * The norms of the tower are nn.RMSNorm(hidden) with no eps: torch takes it from the type it
 * computes in. Moonshot's code run in bf16 (torch 2.15) agrees with f32 and FLT_EPSILON within
 * bf16's noise (rows 4.3% apart) and not with bf16's own 2^-7 (55% apart, cosine 0.09), and so
 * does vLLM, which builds the same norms: FLT_EPSILON here (K3_VISION_EPS to try another).
 *
 * Everything is f32 over the checkpoint's bf16 weights (widened exactly, one matrix at a time):
 * the matrices through matmul_f32.h's matmul, the attention as two of them a head (scores, then
 * the values against the probabilities), softmax and GELU through exp_f8 where the build has it.
 * Weights 0.89 GB in RAM, kept as bf16. */
#ifndef K3_VISION_H
#define K3_VISION_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct { float *n0, *n1; uint16_t *qkv, *wo, *fc0, *fc1; } K3VBlock;
typedef struct {
    int depth, d, heads, hd, qd, inter, out, ph, pw, patch, pin, pinp, merge;
    float eps, peps;
    uint16_t *patch_w, *pos, *p0, *p2;
    float *fnorm, *pnorm;
    K3VBlock *b;
    size_t bytes;
    int max_side;          /* the RoPE grid's side (Rope2DPosEmbRepeated's 512) */
} K3Vision;

static double k3v_now(void) {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}

static uint16_t *k3v_bf16(shards *S, const char *name, int64_t numel, size_t *bytes) {
    st_tensor *t = st_find(S, name);
    if (!t || t->dtype != 0 || t->numel != numel || t->nbytes != numel * 2) {
        fprintf(stderr, "[K3-VISION] %s: %s -- the tower stays off\n", name,
                t ? "not BF16 of the configured shape" : "not in the checkpoint");
        return NULL;
    }
    uint16_t *p = malloc((size_t)numel * 2);
    if (!p) { fprintf(stderr, "[K3-VISION] out of memory for %s\n", name); return NULL; }
    st_read_raw_cap(S, name, p, numel * 2, 1);
    *bytes += (size_t)numel * 2;
    return p;
}

static float *k3v_vec(shards *S, const char *name, int n, size_t *bytes) {
    uint16_t *h = k3v_bf16(S, name, n, bytes);
    if (!h) return NULL;
    float *f = malloc((size_t)n * sizeof(float));
    if (f) for (int i = 0; i < n; i++) f[i] = bf16_to_f32(h[i]);
    free(h);
    return f;
}

static int k3v_num(jval *o, const char *k, int *out) {
    jval *v = json_get(o, k);
    if (!v || v->t != J_NUM) return 0;
    *out = (int)v->num;
    return 1;
}

static int k3v_str_is(jval *o, const char *k, const char *want) {
    jval *v = json_get(o, k);
    return v && v->t == J_STR && !strcmp(v->str, want);
}

static int k3v_false(jval *o, const char *k) {   /* absent, false or 0 */
    jval *v = json_get(o, k);
    return !v || (v->t == J_BOOL && !v->boolean) || (v->t == J_NUM && v->num == 0);
}

/* The tower of the checkpoint in `snap` (its config.json), or NULL with the reason logged. *pad
 * gets media_placeholder_token_id, the id a picture's rows replace. */
static K3Vision *k3v_load(shards *S, const char *snap, int *pad) {
    *pad = -1;
    if (!st_has(S, "vision_tower.patch_embed.proj.weight")) {
        fprintf(stderr, "[K3-VISION] off: the checkpoint has no vision_tower.* (text only)\n");
        return NULL;
    }
    char path[4096];
    snprintf(path, sizeof path, "%s/config.json", snap);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[K3-VISION] off: %s unreadable\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *text = n > 0 && n < (64L << 20) ? malloc((size_t)n + 1) : NULL;
    if (!text || fread(text, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(text); return NULL; }
    fclose(f);
    text[n] = 0;
    char *arena = NULL;
    jval *root = json_parse(text, &arena);
    jval *vc = root ? json_get(root, "vision_config") : NULL;
    K3Vision *v = calloc(1, sizeof *v);
    int ok = v && vc && vc->t == J_OBJ, mk = 0, pid = -1;
    if (ok) {
        jval *mks = json_get(vc, "merge_kernel_size");
        mk = mks && mks->t == J_ARR && mks->len == 2 && mks->kids[0]->num == 2 && mks->kids[1]->num == 2;
        jval *pe = json_get(vc, "projector_ln_eps");
        v->peps = pe && pe->t == J_NUM ? (float)pe->num : 1e-5f;
        ok = k3v_num(vc, "vt_num_hidden_layers", &v->depth) && k3v_num(vc, "vt_hidden_size", &v->d) &&
             k3v_num(vc, "vt_num_attention_heads", &v->heads) && k3v_num(vc, "vt_intermediate_size", &v->inter) &&
             k3v_num(vc, "qkv_hidden_size", &v->qd) && k3v_num(vc, "text_hidden_size", &v->out) &&
             k3v_num(vc, "patch_size", &v->patch) && k3v_num(vc, "init_pos_emb_height", &v->ph) &&
             k3v_num(vc, "init_pos_emb_width", &v->pw) && k3v_num(root, "media_placeholder_token_id", &pid) &&
             mk && k3v_str_is(vc, "norm_type", "rmsnorm") && k3v_str_is(vc, "merge_type", "sd2_tpool") &&
             k3v_str_is(vc, "mm_projector_type", "patchmergerv2") && k3v_str_is(vc, "mlp_type", "mlp2") &&
             k3v_str_is(vc, "pos_emb_type", "divided_fixed") &&
             k3v_str_is(vc, "pos_emb_interpolation_mode", "bilinear") &&
             k3v_str_is(vc, "activation_func", "gelu_pytorch_tanh") &&
             k3v_str_is(vc, "projector_hidden_act", "gelu") &&
             k3v_false(vc, "attn_bias") && k3v_false(vc, "linear_bias") && k3v_false(vc, "patch_embed_proj_bias");
        if (ok) {
            v->hd = v->heads > 0 ? v->qd / v->heads : 0;
            ok = v->depth >= 1 && v->depth <= 64 && v->d >= 32 && v->d <= 8192 && v->d % 32 == 0 &&
                 v->heads >= 1 && v->qd % v->heads == 0 && v->hd % 4 == 0 && v->hd % 32 == 0 &&
                 v->inter % 32 == 0 && v->inter <= 65536 && v->qd % 32 == 0 && v->out >= 32 &&
                 v->patch >= 1 && v->patch <= 32 && v->ph >= 1 && v->ph <= 1024 && v->pw >= 1 && v->pw <= 1024;
        }
    }
    json_free(root);
    free(arena);
    free(text);
    if (!ok) {
        fprintf(stderr, "[K3-VISION] off: config.json's vision_config is not the MoonViT 3D + patchmergerv2 "
                        "tower this engine computes\n");
        free(v);
        return NULL;
    }
    v->merge = 2;
    v->max_side = 512;
    v->pin = 3 * v->patch * v->patch;
    v->pinp = (v->pin + 31) / 32 * 32;
    v->eps = FLT_EPSILON;
    const char *e = getenv("K3_VISION_EPS");
    if (e && *e) v->eps = (float)atof(e);
    double t0 = k3v_now();
    char nm[256];
    int bad = 0;
    v->patch_w = k3v_bf16(S, "vision_tower.patch_embed.proj.weight", (int64_t)v->d * v->pin, &v->bytes);
    v->pos = k3v_bf16(S, "vision_tower.patch_embed.pos_emb.weight", (int64_t)v->ph * v->pw * v->d, &v->bytes);
    bad |= !v->patch_w || !v->pos;
    v->b = calloc((size_t)v->depth, sizeof(K3VBlock));
    if (!v->b) bad = 1;
    for (int i = 0; i < v->depth && !bad; i++) {
        K3VBlock *b = &v->b[i];
#define K3VN(s_) (snprintf(nm, sizeof nm, "vision_tower.encoder.blocks.%d." s_, i), nm)
        b->n0 = k3v_vec(S, K3VN("norm0.weight"), v->d, &v->bytes);
        b->n1 = k3v_vec(S, K3VN("norm1.weight"), v->d, &v->bytes);
        b->qkv = k3v_bf16(S, K3VN("wqkv.weight"), (int64_t)3 * v->qd * v->d, &v->bytes);
        b->wo = k3v_bf16(S, K3VN("wo.weight"), (int64_t)v->d * v->qd, &v->bytes);
        b->fc0 = k3v_bf16(S, K3VN("mlp.fc0.weight"), (int64_t)v->inter * v->d, &v->bytes);
        b->fc1 = k3v_bf16(S, K3VN("mlp.fc1.weight"), (int64_t)v->d * v->inter, &v->bytes);
#undef K3VN
        bad |= !b->n0 || !b->n1 || !b->qkv || !b->wo || !b->fc0 || !b->fc1;
    }
    if (!bad) {
        const int m4 = v->d * 4;
        v->fnorm = k3v_vec(S, "vision_tower.encoder.final_layernorm.weight", v->d, &v->bytes);
        v->p0 = k3v_bf16(S, "mm_projector.proj.0.weight", (int64_t)m4 * m4, &v->bytes);
        v->p2 = k3v_bf16(S, "mm_projector.proj.2.weight", (int64_t)v->out * m4, &v->bytes);
        v->pnorm = k3v_vec(S, "mm_projector.post_norm.weight", v->out, &v->bytes);
        bad |= !v->fnorm || !v->p0 || !v->p2 || !v->pnorm;
    }
    if (bad) {
        fprintf(stderr, "[K3-VISION] off: the tower's weights did not load\n");
        return NULL;   /* what loaded stays: a refused tower is a start-up path, once */
    }
    *pad = pid;
    fprintf(stderr, "[K3-VISION] the tower: %d blocks of %d (%d heads of %d), projector to %d, %.2f GB of "
            "bf16 in %.1fs; a picture's rows replace id %d\n", v->depth, v->d, v->heads, v->hd, v->out,
            v->bytes / 1e9, k3v_now() - t0, pid);
    return v;
}

/* w [O][I] bf16 to f32, the exact values */
static void k3v_widen(float *dst, const uint16_t *src, int64_t n) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i++) dst[i] = bf16_to_f32(src[i]);
}

/* y [S][O] = x [S][I] . W^T over a bf16 W, widened into wbuf */
static void k3v_linear(float *y, const float *x, int S, const uint16_t *w, int O, int I, float *wbuf) {
    k3v_widen(wbuf, w, (int64_t)O * I);
    matmul(y, x, wbuf, S, I, O);
}

static void k3v_rms(float *y, const float *x, const float *w, int S, int D, float eps) {
    #pragma omp parallel for schedule(static)
    for (int s = 0; s < S; s++) {
        const float *r = x + (int64_t)s * D;
        double q = 0;
        for (int i = 0; i < D; i++) q += (double)r[i] * r[i];
        const float inv = 1.0f / sqrtf((float)(q / D) + eps);
        float *o = y + (int64_t)s * D;
        for (int i = 0; i < D; i++) o[i] = r[i] * inv * w[i];
    }
}

/* torch's gelu(approximate='tanh'): 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))), the tanh
 * as 1 - 2 / (e^2u + 1) */
static void k3v_gelu_tanh(float *x, int64_t n) {
    const float kb = 0.7978845608028654f, kk = 0.044715f;
    #pragma omp parallel for schedule(static)
    for (int64_t c = 0; c < (n + 4095) / 4096; c++) {
        const int64_t a = c * 4096, z = a + 4096 < n ? a + 4096 : n;
        int64_t i = a;
#ifdef EXP_F8
        for (; i + 8 <= z; i += 8) {
            __m256 v = _mm256_loadu_ps(x + i);
            __m256 u = _mm256_mul_ps(_mm256_set1_ps(kb),
                                     _mm256_add_ps(v, _mm256_mul_ps(_mm256_set1_ps(kk),
                                                                    _mm256_mul_ps(_mm256_mul_ps(v, v), v))));
            __m256 e = exp_f8(_mm256_add_ps(u, u));
            __m256 t = _mm256_sub_ps(_mm256_set1_ps(1.0f),
                                     _mm256_div_ps(_mm256_set1_ps(2.0f), _mm256_add_ps(e, _mm256_set1_ps(1.0f))));
            _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5f), v),
                                                  _mm256_add_ps(_mm256_set1_ps(1.0f), t)));
        }
#endif
        for (; i < z; i++) {
            const float v = x[i], u = kb * (v + kk * (v * v * v));
            const float t = 1.0f - 2.0f / (exp_f(u + u) + 1.0f);
            x[i] = 0.5f * v * (1.0f + t);
        }
    }
}

/* softmax of a row of n scores times `scale`, into p (n values; p may be s) */
static void k3v_softmax(float *p, const float *s, int n, float scale) {
    float mx = -INFINITY;
    for (int j = 0; j < n; j++) { const float v = s[j] * scale; if (v > mx) mx = v; }
    float sum = 0.0f;
    int j = 0;
#ifdef EXP_F8
    __m256 acc = _mm256_setzero_ps();
    for (; j + 8 <= n; j += 8) {
        __m256 e = exp_f8(_mm256_sub_ps(_mm256_mul_ps(_mm256_loadu_ps(s + j), _mm256_set1_ps(scale)),
                                        _mm256_set1_ps(mx)));
        _mm256_storeu_ps(p + j, e);
        acc = _mm256_add_ps(acc, e);
    }
    float l[8];
    _mm256_storeu_ps(l, acc);
    sum = ((l[0] + l[4]) + (l[2] + l[6])) + ((l[1] + l[5]) + (l[3] + l[7]));
#endif
    for (; j < n; j++) { p[j] = exp_f(s[j] * scale - mx); sum += p[j]; }
    const float inv = 1.0f / sum;
    for (j = 0; j < n; j++) p[j] *= inv;
}

/* The grid's rows of a [ph][pw][d] bf16 table, resized bilinearly to gh x gw (F.interpolate,
 * align_corners False: source = (dst + 0.5) in/out - 0.5, clamped at 0), added to x */
static void k3v_add_pos(const K3Vision *v, float *x, int gh, int gw) {
    const int D = v->d;
    const float sh = (float)v->ph / gh, sw = (float)v->pw / gw;
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < gh; y++) {
        float fy = sh * (y + 0.5f) - 0.5f;
        if (fy < 0) fy = 0;
        int y0 = (int)fy, y1 = y0 + (y0 < v->ph - 1);
        float ly1 = fy - y0, ly0 = 1.0f - ly1;
        if (gh == v->ph) { y0 = y1 = y; ly0 = 1; ly1 = 0; }
        for (int xx = 0; xx < gw; xx++) {
            float fx = sw * (xx + 0.5f) - 0.5f;
            if (fx < 0) fx = 0;
            int x0 = (int)fx, x1 = x0 + (x0 < v->pw - 1);
            float lx1 = fx - x0, lx0 = 1.0f - lx1;
            if (gw == v->pw) { x0 = x1 = xx; lx0 = 1; lx1 = 0; }
            const uint16_t *a = v->pos + ((int64_t)y0 * v->pw + x0) * D, *b = v->pos + ((int64_t)y0 * v->pw + x1) * D;
            const uint16_t *c = v->pos + ((int64_t)y1 * v->pw + x0) * D, *d = v->pos + ((int64_t)y1 * v->pw + x1) * D;
            float *o = x + ((int64_t)y * gw + xx) * D;
            for (int i = 0; i < D; i++)
                o[i] += ly0 * (lx0 * bf16_to_f32(a[i]) + lx1 * bf16_to_f32(b[i])) +
                        ly1 * (lx0 * bf16_to_f32(c[i]) + lx1 * bf16_to_f32(d[i]));
        }
    }
}

static void k3v_dump(const char *dir, const char *name, const float *x, int64_t n) {
    if (!dir) return;
    char p[4096];
    snprintf(p, sizeof p, "%s/%s", dir, name);
    FILE *f = fopen(p, "wb");
    if (f) { fwrite(x, sizeof(float), (size_t)n, f); fclose(f); }
}

/* The picture's rows: rgb [gh*patch][gw*patch][3] uint8 (gh, gw even, at most max_side) ->
 * [(gh/2)*(gw/2)][out] f32, malloc'd; NULL when it cannot. */
static float *k3v_run(const K3Vision *v, const uint8_t *rgb, int gh, int gw, int *n_rows) {
    *n_rows = 0;
    if (!v || gh < 2 || gw < 2 || gh % 2 || gw % 2 || gh > v->max_side || gw > v->max_side) return NULL;
    const double t0 = k3v_now();
    const char *dump = getenv("K3_VISION_DUMP");
    const int N = gh * gw, D = v->d, P = v->patch, W = gw * P, H = v->heads, hd = v->hd, qd = v->qd;
    const int T = (gh / 2) * (gw / 2), M4 = 4 * D, R3 = 3 * qd;
    const int Np = (N + 31) / 32 * 32;
    int64_t wmax = (int64_t)M4 * M4;
    if ((int64_t)v->out * M4 > wmax) wmax = (int64_t)v->out * M4;
    if ((int64_t)R3 * D > wmax) wmax = (int64_t)R3 * D;
    if ((int64_t)v->inter * D > wmax) wmax = (int64_t)v->inter * D;
    if ((int64_t)v->pinp * D > wmax) wmax = (int64_t)v->pinp * D;
    int B = (int)((128ll << 20) / ((int64_t)Np * 4));   /* query rows a pass: 128 MB of scores */
    if (B > N) B = N;
    B = B < 16 ? 16 : B / 16 * 16;
    float *x = malloc((size_t)N * D * 4), *xn = malloc((size_t)N * (D > v->pinp ? D : v->pinp) * 4);
    float *qkv = malloc((size_t)N * R3 * 4), *ctx = malloc((size_t)N * qd * 4);
    float *h = malloc((size_t)N * (v->inter > qd ? v->inter : qd) * 4), *wbuf = malloc((size_t)wmax * 4);
    float *Qh = malloc((size_t)N * hd * 4), *Kh = calloc((size_t)Np * hd, 4), *Vt = calloc((size_t)hd * Np, 4);
    float *sc = malloc((size_t)B * Np * 4), *ot = malloc((size_t)hd * B * 4);
    float *rope = malloc((size_t)(gh + gw) * (hd / 4) * 2 * 4);
    float *rows = NULL;
    if (!x || !xn || !qkv || !ctx || !h || !wbuf || !Qh || !Kh || !Vt || !sc || !ot || !rope) {
        fprintf(stderr, "[K3-VISION] out of memory for a %dx%d grid\n", gh, gw);
        goto done;
    }
    /* patches: (channel, row, column) of each, normalized as media_utils.normalize: x/255 to
     * f32, minus the mean 0.5, times 1/std 2 (both exact here); zero columns up to pinp */
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        const int py = n / gw, px = n % gw;
        float *o = xn + (int64_t)n * v->pinp;
        for (int c = 0; c < 3; c++)
            for (int i = 0; i < P; i++)
                for (int j = 0; j < P; j++) {
                    const uint8_t u = rgb[((int64_t)(py * P + i) * W + px * P + j) * 3 + c];
                    const float f = (float)((double)u / 255.0);
                    o[(c * P + i) * P + j] = (f - 0.5f) * 2.0f;
                }
        for (int k = v->pin; k < v->pinp; k++) o[k] = 0.0f;
    }
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < D; o++)
        for (int k = 0; k < v->pinp; k++)
            wbuf[(int64_t)o * v->pinp + k] = k < v->pin ? bf16_to_f32(v->patch_w[(int64_t)o * v->pin + k]) : 0.0f;
    matmul(x, xn, wbuf, N, v->pinp, D);
    k3v_add_pos(v, x, gh, gw);
    k3v_dump(dump, "pe.f32", x, (int64_t)N * D);
    /* RoPE: cos and sin of position p times freq i, columns then rows: rope[(p) * nf + i] pairs */
    const int nf = hd / 4;
    float *rc = rope, *rs = rope + (gh + gw) * nf;   /* [gw columns then gh rows][nf] */
    for (int i = 0; i < nf; i++) {
        const float fr = 1.0f / powf(10000.0f, (float)(4 * i) / (float)hd);
        for (int p = 0; p < gw + gh; p++) {
            const float pos = (float)(p < gw ? p : p - gw), ang = pos * fr;
            rc[p * nf + i] = (float)cos((double)ang);
            rs[p * nf + i] = (float)sin((double)ang);
        }
    }
    const float scale = 1.0f / sqrtf((float)hd);
    double t_lin = 0, t_att = 0;
    for (int li = 0; li < v->depth; li++) {
        const K3VBlock *b = &v->b[li];
        double ta = k3v_now();
        k3v_rms(xn, x, b->n0, N, D, v->eps);
        k3v_linear(qkv, xn, N, b->qkv, R3, D, wbuf);
        /* RoPE on q and k: pair j of a head, j = 2i the column's angle i, 2i+1 the row's */
        #pragma omp parallel for schedule(static)
        for (int n = 0; n < N; n++) {
            const int py = n / gw, px = n % gw;
            for (int part = 0; part < 2; part++)
                for (int hh = 0; hh < H; hh++) {
                    float *q = qkv + (int64_t)n * R3 + part * qd + hh * hd;
                    for (int j = 0; j < hd / 2; j++) {
                        const int i = j >> 1, p = (j & 1) ? gw + py : px;
                        const float c = rc[p * nf + i], s = rs[p * nf + i];
                        const float a = q[2 * j], bb = q[2 * j + 1];
                        q[2 * j] = a * c - bb * s;
                        q[2 * j + 1] = a * s + bb * c;
                    }
                }
        }
        t_lin += k3v_now() - ta;
        ta = k3v_now();
        for (int hh = 0; hh < H; hh++) {
            #pragma omp parallel for schedule(static)
            for (int n = 0; n < N; n++) {
                const float *r = qkv + (int64_t)n * R3 + hh * hd;
                memcpy(Qh + (int64_t)n * hd, r, (size_t)hd * 4);
                memcpy(Kh + (int64_t)n * hd, r + qd, (size_t)hd * 4);
                for (int e = 0; e < hd; e++) Vt[(int64_t)e * Np + n] = r[2 * qd + e];
            }
            for (int q0 = 0; q0 < N; q0 += B) {
                const int nb = N - q0 < B ? N - q0 : B;
                /* [nb][Np]: Kh's rows past N are zeros, and so are the probabilities there */
                matmul(sc, Qh + (int64_t)q0 * hd, Kh, nb, hd, Np);
                #pragma omp parallel for schedule(static)
                for (int r = 0; r < nb; r++) {
                    float *row = sc + (int64_t)r * Np;
                    k3v_softmax(row, row, N, scale);
                    for (int k = N; k < Np; k++) row[k] = 0.0f;
                }
                matmul(ot, Vt, sc, hd, Np, nb);                    /* [hd][nb] = V^T . P^T */
                #pragma omp parallel for schedule(static)
                for (int r = 0; r < nb; r++)
                    for (int e = 0; e < hd; e++) ctx[(int64_t)(q0 + r) * qd + hh * hd + e] = ot[(int64_t)e * nb + r];
            }
        }
        t_att += k3v_now() - ta;
        ta = k3v_now();
        k3v_linear(h, ctx, N, b->wo, D, qd, wbuf);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)N * D; i++) x[i] += h[i];
        k3v_rms(xn, x, b->n1, N, D, v->eps);
        k3v_linear(h, xn, N, b->fc0, v->inter, D, wbuf);
        k3v_gelu_tanh(h, (int64_t)N * v->inter);
        k3v_linear(xn, h, N, b->fc1, D, v->inter, wbuf);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)N * D; i++) x[i] += xn[i];
        t_lin += k3v_now() - ta;
        if (li == 0) k3v_dump(dump, "b0.f32", x, (int64_t)N * D);
    }
    k3v_rms(x, x, v->fnorm, N, D, v->eps);
    k3v_dump(dump, "enc.f32", x, (int64_t)N * D);
    /* merge: merged row (i, j) is patches (2i, 2j), (2i, 2j+1), (2i+1, 2j), (2i+1, 2j+1) side by side */
    float *mg = h;   /* N * inter >= T * 4D */
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {
        const int i = t / (gw / 2), j = t % (gw / 2);
        for (int k = 0; k < 4; k++) {
            const int n = (2 * i + (k >> 1)) * gw + 2 * j + (k & 1);
            memcpy(mg + (int64_t)t * M4 + k * D, x + (int64_t)n * D, (size_t)D * 4);
        }
    }
    float *p1 = qkv;   /* N * 3qd >= T * 4D */
    k3v_linear(p1, mg, T, v->p0, M4, M4, wbuf);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)T * M4; i++) p1[i] = 0.5f * p1[i] * (1.0f + erff(p1[i] * 0.70710678118654752f));
    rows = malloc((size_t)T * v->out * 4);
    float *p2 = rows ? malloc((size_t)T * v->out * 4) : NULL;
    if (!rows || !p2) { free(rows); rows = NULL; free(p2); goto done; }
    k3v_linear(p2, p1, T, v->p2, v->out, M4, wbuf);
    k3v_rms(rows, p2, v->pnorm, T, v->out, v->peps);
    free(p2);
    k3v_dump(dump, "rows.f32", rows, (int64_t)T * v->out);
    *n_rows = T;
    fprintf(stderr, "[K3-VISION] a picture of %dx%d patches -> %d rows in %.1fs (matrices %.1f, attention %.1f)\n",
            gh, gw, T, k3v_now() - t0, t_lin, t_att);
done:
    free(x); free(xn); free(qkv); free(ctx); free(h); free(wbuf); free(Qh); free(Kh); free(Vt);
    free(sc); free(ot); free(rope);
    return rows;
}

#endif /* K3_VISION_H */
