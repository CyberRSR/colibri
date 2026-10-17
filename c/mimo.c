/* mimo.c -- Xiaomi MiMo-V2.6 (Flash 309B/15B, Pro 1.02T/42B) on a machine that
 * cannot hold it.
 *
 * The released checkpoint is read as it is, no conversion step:
 *
 *   experts   MXFP4 g32 (e2m1 nibble pairs, low nibble = even column, one e8m0
 *             byte per 32 columns), the six tensors of an expert back to back in
 *             the shard, so an expert is ONE read. 256 of them per layer on
 *             Flash, 8 routed per token, cached in RAM by a per-layer LRU.
 *   dense     qkv_proj and the layer-0 MLP in FP8 e4m3 with a F32 scale per
 *             128x128 block; o_proj, router, norms, sinks, embed, lm_head BF16.
 *
 * What makes MiMo MiMo, each mirroring modeling_mimo_v2.py:
 *   hybrid attention   hybrid_layer_pattern: 1 = sliding window (128 keys,
 *                      itself included), 0 = full attention; the two kinds have
 *                      their own KV head count and their own RoPE theta
 *   partial RoPE       on the first int(head_dim * 0.334) dims of each head,
 *                      rotate-half style; head_dim 192 for q/k, 128 for v
 *   value scale        v *= attention_value_scale before attention
 *   attention sink     SWA layers carry one logit per head that joins the
 *                      softmax denominator and nothing else
 *   router             sigmoid scores, e_score_correction_bias picks but does
 *                      not weigh, top-k renormalised; no shared expert
 *
 * One storage detail the modeling code does not describe: the fused qkv_proj is
 * PRE-SHARDED in num_key_value_heads chunks [Q_0|K_0|V_0|Q_1|K_1|V_1|...] and
 * its FP8 scale grid is tiled per chunk (a Flash full-attention layer has 108
 * scale rows, not the 106 a contiguous grid would have). The layout is vLLM's
 * (vllm/model_executor/models/mimo_v2.py, _shard_fp8_qkv_proj); the loader
 * undoes it and the tiny oracle stores it the same way so a wrong reading fails.
 *
 * Held to Xiaomi's own modeling code through tools/make_mimo_ref.py (vendor
 * files pinned by SHA-256) on the tiny fixture of tools/make_mimo_tiny.py.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <errno.h>
#include <time.h>
#include <stdarg.h>
#include <unistd.h>
#include <sys/stat.h>
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/resource.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#include "cli_args.h"
#include "compat.h"
#include "json.h"
#include "st.h"
#include "quant.h"
#include "omp_tune.h"
#include "kv_prefix.h"
#include "pin_pool.h"
#include "tok.h"
#include "serve_codec.h"
#include "serve_poll.h"
#include "stop_ids.h"
#ifdef COLI_MIMO_CUDA
#include "backend_cuda_mimo.h"
static int g_mimo_cuda;
static int g_mimo_cuda_release, g_mimo_cuda_dropped, g_mimo_cuda_reloaded;
static size_t g_mimo_cuda_host_bytes, g_mimo_cuda_reload_bytes;
static unsigned long long g_mimo_cuda_fault, g_mimo_cuda_calls;
#endif
#include <pthread.h>       /* the expert readers; CUDA's and Vulkan's reload lock */
#include <stdatomic.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include "vk_tier.h"       /* the shared Vulkan routed-expert tier (COLI_VULKAN=1; stubs without VK=1) */
#ifdef COLI_VULKAN
/* VK=1 and COLI_VULKAN=1: the routed experts go to the shared expert tier
 * (vk_tier.c) as the release's MXFP4, and the dense matrices of the trunk and of
 * the vision tower to the device where coli_vk_dense() puts them, in whatever
 * form MIMO_DENSE_BITS gave them (FP8, BF16, int8 or f32). The router, and every
 * matrix whose upload fails, stays on the CPU path below. The backend's dense
 * matmuls share one command buffer: main thread only. */
#include "backend_vulkan.h"
#include <pthread.h>
static int g_vk_ready;
/* COLI_VK_DENSE_HOST: the dense matrices on the device only (mimo_dho_start); the
 * per-matrix path then runs on the device whatever coli_vk_dense() says */
static int g_mimo_dho;
#endif

#define MIMO_MAX_LAYERS 128
#define MIMO_MAX_TOPK 16
#define MIMO_MAX_CHUNKS 16

/* ------------------------------------------------------------------ utils ---- */

static double now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static double rss_gb(void) {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
#ifdef __APPLE__
    return usage.ru_maxrss / 1e9;
#else
    return usage.ru_maxrss / 1e6;
#endif
}

static void *xmalloc(size_t bytes, const char *what) {
    void *p = malloc(bytes ? bytes : 1);
    if (!p) { fprintf(stderr, "[mimo] out of memory allocating %s (%zu bytes)\n", what, bytes); exit(1); }
    return p;
}

static void *xcalloc(size_t n, size_t size, const char *what) {
    void *p = calloc(n ? n : 1, size ? size : 1);
    if (!p) { fprintf(stderr, "[mimo] out of memory allocating %s\n", what); exit(1); }
    return p;
}

static int env_int(const char *name, int fallback) {
    const char *v = getenv(name);
    return (v && *v) ? atoi(v) : fallback;
}

static inline float silu(float x) { return x / (1.0f + expf(-x)); }
static inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

/* ---------------------------------------------------------------- config ---- */

/* kind 0 = full attention, 1 = sliding window */
typedef struct {
    int vocab, hidden, n_layers, dense_inter, moe_inter, n_experts, topk;
    int heads[2], kv_heads[2], head_dim[2], v_dim[2], rope_dim[2];
    float theta[2];
    int sink[2];
    int swa[MIMO_MAX_LAYERS], moe[MIMO_MAX_LAYERS];
    int window, chunks, norm_topk;
    float v_scale, eps, route_scale;
    int max_positions;
    int image_token_id, vision_start_id, vision_end_id;
} Cfg;

static double jnum(jval *o, const char *key, double fallback) {
    jval *v = o ? json_get(o, key) : NULL;
    return (v && v->t == J_NUM) ? v->num : fallback;
}

static int jbool(jval *o, const char *key, int fallback) {
    jval *v = o ? json_get(o, key) : NULL;
    if (!v) return fallback;
    if (v->t == J_BOOL) return v->boolean != 0;
    if (v->t == J_NUM) return v->num != 0;
    return fallback;
}

static char *read_text(const char *path, long *size_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long size = ftell(f); fseek(f, 0, SEEK_SET);
    if (size < 0 || size > (64l << 20)) { fclose(f); return NULL; }
    char *text = xmalloc((size_t)size + 1, path);
    if (fread(text, 1, (size_t)size, f) != (size_t)size) { fclose(f); free(text); return NULL; }
    text[size] = 0; fclose(f);
    if (size_out) *size_out = size;
    return text;
}

static void die_cfg(const char *what) {
    fprintf(stderr, "[mimo] config.json: %s -- refusing\n", what);
    exit(1);
}

static void load_cfg(Cfg *c, const char *dir) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/config.json", dir);
    char *text = read_text(path, NULL);
    if (!text) { fprintf(stderr, "[mimo] cannot read %s\n", path); exit(1); }
    char *arena = NULL;
    jval *root = json_parse(text, &arena);
    if (!root) die_cfg("not JSON");
    jval *mt = json_get(root, "model_type");
    if (!mt || mt->t != J_STR || strcmp(mt->str, "mimo_v2"))
        die_cfg("model_type is not mimo_v2");
    memset(c, 0, sizeof(*c));
    c->vocab = (int)jnum(root, "vocab_size", 0);
    c->hidden = (int)jnum(root, "hidden_size", 0);
    c->n_layers = (int)jnum(root, "num_hidden_layers", 0);
    c->dense_inter = (int)jnum(root, "intermediate_size", 0);
    c->moe_inter = (int)jnum(root, "moe_intermediate_size", 0);
    c->n_experts = (int)jnum(root, "n_routed_experts", 0);
    c->topk = (int)jnum(root, "num_experts_per_tok", 0);
    c->heads[0] = (int)jnum(root, "num_attention_heads", 0);
    c->kv_heads[0] = (int)jnum(root, "num_key_value_heads", 0);
    c->head_dim[0] = (int)jnum(root, "head_dim", 0);
    c->v_dim[0] = (int)jnum(root, "v_head_dim", c->head_dim[0]);
    c->heads[1] = (int)jnum(root, "swa_num_attention_heads", c->heads[0]);
    c->kv_heads[1] = (int)jnum(root, "swa_num_key_value_heads", c->kv_heads[0]);
    c->head_dim[1] = (int)jnum(root, "swa_head_dim", c->head_dim[0]);
    c->v_dim[1] = (int)jnum(root, "swa_v_head_dim", c->v_dim[0]);
    double prf = jnum(root, "partial_rotary_factor", 1.0);
    for (int k = 0; k < 2; k++) c->rope_dim[k] = (int)(c->head_dim[k] * prf);
    c->theta[0] = (float)jnum(root, "rope_theta", 10000.0);
    c->theta[1] = (float)jnum(root, "swa_rope_theta", c->theta[0]);
    c->sink[0] = jbool(root, "add_full_attention_sink_bias", 0);
    c->sink[1] = jbool(root, "add_swa_attention_sink_bias", 0);
    c->window = (int)jnum(root, "sliding_window", 0);
    c->v_scale = (float)jnum(root, "attention_value_scale", 1.0);
    c->eps = (float)jnum(root, "layernorm_epsilon", 1e-6);
    c->route_scale = (float)jnum(root, "routed_scaling_factor", 1.0);
    c->norm_topk = jbool(root, "norm_topk_prob", 1);
    c->max_positions = (int)jnum(root, "max_position_embeddings", 32768);
    c->image_token_id = (int)jnum(root, "image_token_id", -1);
    c->vision_start_id = (int)jnum(root, "vision_start_token_id", -1);
    c->vision_end_id = (int)jnum(root, "vision_end_token_id", -1);
    c->chunks = c->kv_heads[0];
    /* MIMO_LAYERS=N: only the first N layers, then the final norm and head. A
     * checking aid (tools/mimo_real_check.py compares it with Xiaomi's code on
     * the real weights), never an answer anyone should read. */
    int keep = env_int("MIMO_LAYERS", 0);

    jval *pat = json_get(root, "hybrid_layer_pattern");
    jval *freq = json_get(root, "moe_layer_freq");
    if (c->n_layers < 1 || c->n_layers > MIMO_MAX_LAYERS) die_cfg("num_hidden_layers out of range");
    if (!pat || pat->t != J_ARR || pat->len != c->n_layers) die_cfg("hybrid_layer_pattern must list every layer");
    if (!freq || freq->t != J_ARR || freq->len != c->n_layers) die_cfg("moe_layer_freq must list every layer");
    for (int i = 0; i < c->n_layers; i++) {
        c->swa[i] = (int)pat->kids[i]->num == 1;
        c->moe[i] = (int)freq->kids[i]->num != 0;
    }
    if (keep > 0 && keep < c->n_layers) {
        fprintf(stderr, "[mimo] MIMO_LAYERS=%d: a truncated model for checking, not for answers\n", keep);
        c->n_layers = keep;
    }
    /* Everything below sizes a buffer or indexes one: refuse a config that
     * would make any of it overflow or divide by zero, before reading a byte. */
    if (c->vocab < 2 || c->vocab > (1 << 22)) die_cfg("vocab_size out of range");
    if (c->hidden < 32 || c->hidden > 65536 || c->hidden % 32) die_cfg("hidden_size must be a positive multiple of 32");
    if (c->moe_inter < 32 || c->moe_inter % 32 || c->moe_inter > 65536) die_cfg("moe_intermediate_size must be a positive multiple of 32");
    if (c->n_experts < 1 || c->n_experts > 4096) die_cfg("n_routed_experts out of range");
    if (c->topk < 1 || c->topk > MIMO_MAX_TOPK || c->topk > c->n_experts) die_cfg("num_experts_per_tok out of range");
    for (int k = 0; k < 2; k++) {
        if (c->heads[k] < 1 || c->kv_heads[k] < 1 || c->heads[k] % c->kv_heads[k]) die_cfg("attention heads must be a multiple of kv heads");
        if (c->heads[k] > 1024 || c->kv_heads[k] > 1024) die_cfg("too many attention heads");
        if (c->head_dim[k] < 2 || c->head_dim[k] > 1024 || c->v_dim[k] < 1 || c->v_dim[k] > 1024) die_cfg("head dims out of range");
        if (c->rope_dim[k] < 0 || c->rope_dim[k] > c->head_dim[k] || c->rope_dim[k] % 2) die_cfg("rotary dims must be even and fit the head");
    }
    if (c->chunks < 1 || c->chunks > MIMO_MAX_CHUNKS) die_cfg("num_key_value_heads (the qkv pre-shard count) out of range");
    for (int k = 0; k < 2; k++)
        if (c->heads[k] % c->chunks || c->kv_heads[k] % c->chunks) die_cfg("head counts do not divide into the qkv pre-shard chunks");
    int any_swa = 0, any_moe = 0;
    for (int i = 0; i < c->n_layers; i++) { any_swa |= c->swa[i]; any_moe |= c->moe[i]; }
    if (any_swa && (c->window < 1 || c->window > (1 << 20))) die_cfg("sliding_window out of range");
    for (int i = 0; i < c->n_layers; i++)
        if (!c->moe[i] && (c->dense_inter < 1 || c->dense_inter > (1 << 20))) die_cfg("intermediate_size out of range");
    if (!any_moe && keep <= 0) die_cfg("no MoE layer");
    if (!(c->eps > 0) || !(c->eps < 1)) die_cfg("layernorm_epsilon out of range");
    if (c->max_positions < 1) c->max_positions = 32768;
    json_free(root); free(arena); free(text);
}

/* ----------------------------------------------------------- dense weights ---- */

/* A dense matrix [O, I] in one of four forms. FP8 keeps the checkpoint bytes and
 * expands the 128x128 block grid to one scale per (row, column block), which is
 * what lets a row move (the qkv de-interleave) without touching its bytes. */
enum { DW_F32 = 0, DW_BF16 = 1, DW_FP8 = 2, DW_I8 = 3, DW_BF16P = 4 };
typedef struct {
    int fmt, O, I, nblk;
    void *w;           /* f32 / bf16 / e4m3 / int8 / BF16P: the sign/mantissa bytes */
    float *s;          /* FP8: [O][nblk]; I8: [O] */
    /* BF16P (mimo_bf16_pack): the 4-bit exponent codes [O][I/2], each row's lowest
     * exponent, where its exceptions start, the exceptions; one allocation from w */
    uint8_t *pk_codes, *pk_lo, *pk_exc;
    uint32_t *pk_off;
    size_t pk_bytes;
    void *vk;          /* COLI_VULKAN: the device copy, uploaded at start-up */
    int vk_off;        /* its upload failed: this matrix stays on the CPU */
    /* COLI_VULKAN: how to read it back from disk (dw_reload), and 1 while the device
     * holds it alone (COLI_VK_DENSE_HOST, the host copy dropped) */
    char *vk_name;
    int vk_kind, vk_bits, vk_li, vk_gone;
#ifdef COLI_MIMO_CUDA
    MimoCudaTensor *cuda; /* resident native dense matrix; host stays canonical */
#endif
} DW;

/* 0 (default): the checkpoint's own FP8 and BF16 bytes, exact.
 * 8: int8 per row, about 30% less RAM for the dense part, not exact.
 * 32: f32, exact and four times the RAM: the oracle's configuration. */
static int g_dense_bits;

/* The router: matmul() (matmul_f32.h) on the released bf16 bytes. Each weight is
 * widened as st_read_f32 widened it and every lane adds in matmul()'s order, so the
 * scores are the bits a f32 copy gave, from half the bytes. */
static inline float dot_bf16_lanes(const float *x, const uint16_t *w, int I) {
    float a[MATMUL_F32_LANES] = {0};
    int i = 0;
    for (; i + MATMUL_F32_LANES <= I; i += MATMUL_F32_LANES)
        for (int j = 0; j < MATMUL_F32_LANES; j++) a[j] = MATMUL_F32_MADD(a[j], x[i + j], bf16_to_f32(w[i + j]));
    for (int j = 0; i + j < I; j++) a[j] = MATMUL_F32_MADD(a[j], x[i + j], bf16_to_f32(w[i + j]));
    for (int h = MATMUL_F32_LANES / 2; h > 0; h >>= 1)
        for (int j = 0; j < h; j++) a[j] += a[j + h];
    return a[0];
}
static void matmul_router(float *y, const float *x, const uint16_t *W, int S, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint16_t *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) y[(int64_t)s * O + o] = dot_bf16_lanes(x + (int64_t)s * I, w, I);
    }
}

static void matmul_bf16(float *y, const float *x, const uint16_t *W, int S, int I, int O) {
#ifdef __AVX2__
    if (S >= 4 && I % 8 == 0) {
        /* A prompt block: 4 rows x 2 outputs at a time, eight independent chains
         * instead of one, each (row, output) summed in the one-row loop's order
         * below (the same lanes, the same hsum256), so the bits do not move. The
         * rows go in tiles of 64, whose activations stay in cache while every
         * output reads them (a 1024-row block's would not). */
        const float *x_all = x;
        float *y_all = y;
        int S_all = S;
        for (int t0 = 0; t0 < S_all; t0 += 64) {
        S = S_all - t0 < 64 ? S_all - t0 : 64;
        x = x_all + (int64_t)t0 * I;
        y = y_all + (int64_t)t0 * O;
        #pragma omp parallel for schedule(static)
        for (int o2 = 0; o2 < (O + 1) / 2; o2++) {
            int o = o2 * 2, two = o + 1 < O;
            const uint16_t *wa = W + (int64_t)o * I, *wb = two ? wa + I : wa;
            int s = 0;
            for (; s + 4 <= S; s += 4) {
                const float *x0 = x + (int64_t)s * I, *x1 = x0 + I, *x2 = x1 + I, *x3 = x2 + I;
                __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0, b2 = a0, b3 = a0;
                for (int i = 0; i < I; i += 8) {
                    __m256 va = bf16_decode8(wa + i), vb = bf16_decode8(wb + i);
                    __m256 v0 = _mm256_loadu_ps(x0 + i), v1 = _mm256_loadu_ps(x1 + i);
                    __m256 v2 = _mm256_loadu_ps(x2 + i), v3 = _mm256_loadu_ps(x3 + i);
                    a0 = _mm256_fmadd_ps(v0, va, a0); b0 = _mm256_fmadd_ps(v0, vb, b0);
                    a1 = _mm256_fmadd_ps(v1, va, a1); b1 = _mm256_fmadd_ps(v1, vb, b1);
                    a2 = _mm256_fmadd_ps(v2, va, a2); b2 = _mm256_fmadd_ps(v2, vb, b2);
                    a3 = _mm256_fmadd_ps(v3, va, a3); b3 = _mm256_fmadd_ps(v3, vb, b3);
                }
                y[(int64_t)(s + 0) * O + o] = hsum256(a0); y[(int64_t)(s + 1) * O + o] = hsum256(a1);
                y[(int64_t)(s + 2) * O + o] = hsum256(a2); y[(int64_t)(s + 3) * O + o] = hsum256(a3);
                if (two) {
                    y[(int64_t)(s + 0) * O + o + 1] = hsum256(b0); y[(int64_t)(s + 1) * O + o + 1] = hsum256(b1);
                    y[(int64_t)(s + 2) * O + o + 1] = hsum256(b2); y[(int64_t)(s + 3) * O + o + 1] = hsum256(b3);
                }
            }
            for (; s < S; s++)
                for (int k = 0; k < 1 + two; k++) {
                    const uint16_t *w = k ? wb : wa;
                    const float *xs = x + (int64_t)s * I;
                    __m256 acc = _mm256_setzero_ps();
                    for (int i = 0; i < I; i += 8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(xs + i), bf16_decode8(w + i), acc);
                    y[(int64_t)s * O + o + k] = hsum256(acc);
                }
        }
        }
        return;
    }
#endif
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint16_t *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            float a = 0; int i = 0;
#ifdef __AVX2__
            __m256 acc = _mm256_setzero_ps();
            for (; i + 8 <= I; i += 8)
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(xs + i), bf16_decode8(w + i), acc);
            a = hsum256(acc);
#endif
            for (; i < I; i++) a += xs[i] * bf16_to_f32(w[i]);
            y[(int64_t)s * O + o] = a;
        }
    }
}

/* ---- BF16 packed without loss (DW_BF16P, MIMO_BF16_PACK) ----------------------------
 * The o_proj and the head the CPU computes, packed as the CUDA side packs them
 * (backend_cuda_mimo.cu, fmt 5): per row the window of 15 consecutive exponents that
 * holds most of its weights (from pk_lo), per weight a 4-bit code (exponent - lo, or
 * 15: the next byte of the row's exception list) and the sign/mantissa byte. A quarter
 * fewer bytes to read a token. Each group of 8 weights is rebuilt into the bf16 words
 * matmul_bf16 loads and summed in its order, so the results are its bytes. */
typedef struct { const uint8_t *sm, *code, *exc; uint64_t base; } PkRow;
static inline PkRow pk_row(const DW *d, int o) {
    PkRow r = { (const uint8_t *)d->w + (size_t)o * d->I, d->pk_codes + (size_t)o * (d->I / 2),
                d->pk_exc + d->pk_off[o], (uint64_t)d->pk_lo[o] * 0x0101010101010101ull };
    return r;
}
/* Weights i..i+7 (i a multiple of 8): the low and the high bytes of their bf16 words.
 * Byte b of the group's 4 code bytes holds weight b's code low and weight b+4's high. */
static inline void pk_group(PkRow *r, int i, uint64_t *lo, uint64_t *hi) {
    uint32_t q; memcpy(&q, r->code + i / 2, 4);
    uint64_t e = ((uint64_t)(q & 0x0F0F0F0Fu) | (uint64_t)((q >> 4) & 0x0F0F0F0Fu) << 32) + r->base;
    uint32_t esc = q & (q >> 1) & (q >> 2) & (q >> 3) & 0x11111111u;
    if (__builtin_expect(esc != 0, 0))
        for (int k = 0; k < 8; k++)
            if (esc >> (k < 4 ? 8 * k : 8 * (k - 4) + 4) & 1)
                e = (e & ~(0xFFull << (8 * k))) | (uint64_t)*r->exc++ << (8 * k);
    uint64_t sm; memcpy(&sm, r->sm + i, 8);
    *hi = (sm & 0x8080808080808080ull) | ((e >> 1) & 0x7F7F7F7F7F7F7F7Full);
    *lo = (sm & 0x7F7F7F7F7F7F7F7Full) | ((e << 7) & 0x8080808080808080ull);
}
/* A whole row back to its bf16 words. */
static void pk_unpack_row(const DW *d, int o, uint16_t *out) {
    PkRow r = pk_row(d, o);
    for (int i = 0; i < d->I; i += 8) {
        uint64_t lo, hi;
        pk_group(&r, i, &lo, &hi);
        for (int k = 0; k < 8; k++)
            out[i + k] = (uint16_t)(((lo >> (8 * k)) & 0xFF) | ((hi >> (8 * k)) & 0xFF) << 8);
    }
}
#ifdef __AVX2__
static inline __m256 pk_decode8(PkRow *r, int i) {   /* = bf16_decode8 of the 8 words */
    uint64_t lo, hi;
    pk_group(r, i, &lo, &hi);
    __m128i w = _mm_unpacklo_epi8(_mm_cvtsi64_si128((long long)lo), _mm_cvtsi64_si128((long long)hi));
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(w), 16));
}
#endif
static void matmul_bf16p(float *y, const float *x, const DW *d, int S) {
    int I = d->I, O = d->O;
    if (S < 1) return;
#ifdef __AVX2__
    if (S >= 4) {   /* matmul_bf16's prompt block on the two outputs' rows unpacked once a tile */
        const float *x_all = x;
        float *y_all = y;
        int S_all = S;
        for (int t0 = 0; t0 < S_all; t0 += 64) {
        S = S_all - t0 < 64 ? S_all - t0 : 64;
        x = x_all + (int64_t)t0 * I;
        y = y_all + (int64_t)t0 * O;
        #pragma omp parallel
        {
        uint16_t *ua = xmalloc((size_t)I * 2 * sizeof(uint16_t), "unpacked rows"), *ub = ua + I;
        #pragma omp for schedule(static)
        for (int o2 = 0; o2 < (O + 1) / 2; o2++) {
            int o = o2 * 2, two = o + 1 < O;
            pk_unpack_row(d, o, ua);
            if (two) pk_unpack_row(d, o + 1, ub);
            const uint16_t *wa = ua, *wb = two ? ub : ua;
            int s = 0;
            for (; s + 4 <= S; s += 4) {
                const float *x0 = x + (int64_t)s * I, *x1 = x0 + I, *x2 = x1 + I, *x3 = x2 + I;
                __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0, b2 = a0, b3 = a0;
                for (int i = 0; i < I; i += 8) {
                    __m256 va = bf16_decode8(wa + i), vb = bf16_decode8(wb + i);
                    __m256 v0 = _mm256_loadu_ps(x0 + i), v1 = _mm256_loadu_ps(x1 + i);
                    __m256 v2 = _mm256_loadu_ps(x2 + i), v3 = _mm256_loadu_ps(x3 + i);
                    a0 = _mm256_fmadd_ps(v0, va, a0); b0 = _mm256_fmadd_ps(v0, vb, b0);
                    a1 = _mm256_fmadd_ps(v1, va, a1); b1 = _mm256_fmadd_ps(v1, vb, b1);
                    a2 = _mm256_fmadd_ps(v2, va, a2); b2 = _mm256_fmadd_ps(v2, vb, b2);
                    a3 = _mm256_fmadd_ps(v3, va, a3); b3 = _mm256_fmadd_ps(v3, vb, b3);
                }
                y[(int64_t)(s + 0) * O + o] = hsum256(a0); y[(int64_t)(s + 1) * O + o] = hsum256(a1);
                y[(int64_t)(s + 2) * O + o] = hsum256(a2); y[(int64_t)(s + 3) * O + o] = hsum256(a3);
                if (two) {
                    y[(int64_t)(s + 0) * O + o + 1] = hsum256(b0); y[(int64_t)(s + 1) * O + o + 1] = hsum256(b1);
                    y[(int64_t)(s + 2) * O + o + 1] = hsum256(b2); y[(int64_t)(s + 3) * O + o + 1] = hsum256(b3);
                }
            }
            for (; s < S; s++)
                for (int k = 0; k < 1 + two; k++) {
                    const uint16_t *w = k ? wb : wa;
                    const float *xs = x + (int64_t)s * I;
                    __m256 acc = _mm256_setzero_ps();
                    for (int i = 0; i < I; i += 8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(xs + i), bf16_decode8(w + i), acc);
                    y[(int64_t)s * O + o + k] = hsum256(acc);
                }
        }
        free(ua);
        }
        }
        return;
    }
    /* a decode step's rows (verified drafts among them): each group decoded once for all
     * of them, every row summed in the one-row order */
    const float *x1 = x + (S > 1 ? I : 0), *x2 = x + (S > 2 ? 2 * (int64_t)I : 0);
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        PkRow r = pk_row(d, o);
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0;
        for (int i = 0; i < I; i += 8) {
            __m256 w = pk_decode8(&r, i);
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), w, a0);
            if (S > 1) a1 = _mm256_fmadd_ps(_mm256_loadu_ps(x1 + i), w, a1);
            if (S > 2) a2 = _mm256_fmadd_ps(_mm256_loadu_ps(x2 + i), w, a2);
        }
        y[o] = hsum256(a0);
        if (S > 1) y[(int64_t)O + o] = hsum256(a1);
        if (S > 2) y[2 * (int64_t)O + o] = hsum256(a2);
    }
#else
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        for (int s = 0; s < S; s++) {
            PkRow r = pk_row(d, o);
            const float *xs = x + (int64_t)s * I;
            float a = 0;
            for (int i = 0; i < I; i += 8) {
                uint64_t lo, hi;
                pk_group(&r, i, &lo, &hi);
                for (int k = 0; k < 8; k++)
                    a += xs[i + k] * bf16_to_f32((uint16_t)(((lo >> (8 * k)) & 0xFF) | ((hi >> (8 * k)) & 0xFF) << 8));
            }
            y[(int64_t)s * O + o] = a;
        }
    }
#endif
}

/* Pack a bf16 matrix in place (rows a multiple of 8 long); 1 if it was. */
static int dw_pack_bf16(DW *d, size_t *from, size_t *to) {
    if (d->fmt != DW_BF16 || !d->w || d->I % 8 || d->O < 1) return 0;
    int O = d->O, I = d->I;
    const uint16_t *w = (const uint16_t *)d->w;
    uint8_t *lo = xmalloc((size_t)O, "pack lo");
    uint32_t *off = xmalloc((size_t)O * sizeof(uint32_t), "pack offsets");
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint16_t *row = w + (size_t)o * I;
        unsigned h[256] = {0};
        for (int i = 0; i < I; i++) h[(row[i] >> 7) & 0xff]++;
        unsigned sum = 0, best;
        int bl = 0;
        for (int e = 0; e < 15; e++) sum += h[e];
        best = sum;
        for (int l = 1; l <= 240; l++) {   /* lo + 15 stays a byte: the decoder adds bytewise */
            sum += h[l + 14] - h[l - 1];
            if (sum > best) { best = sum; bl = l; }
        }
        lo[o] = (uint8_t)bl; off[o] = (uint32_t)I - best;
    }
    uint64_t nexc = 0;
    for (int o = 0; o < O; o++) { uint32_t n = off[o]; off[o] = (uint32_t)nexc; nexc += n; }
    if (nexc > UINT32_MAX) { free(lo); free(off); return 0; }
    size_t e = (size_t)O * I, a_codes = e, a_lo = e + e / 2, a_off = (a_lo + (size_t)O + 15) & ~(size_t)15,
           a_exc = a_off + (size_t)O * sizeof(uint32_t), total = a_exc + (size_t)nexc;
    uint8_t *buf = xmalloc(total, "packed bf16");
    memcpy(buf + a_lo, lo, (size_t)O);
    memcpy(buf + a_off, off, (size_t)O * sizeof(uint32_t));
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint16_t *row = w + (size_t)o * I;
        uint8_t *sm = buf + (size_t)o * I, *c = buf + a_codes + (size_t)o * (I / 2), *x = buf + a_exc + off[o];
        unsigned l = lo[o];
        for (int g = 0; g < I; g += 8) {
            uint8_t code[8];
            for (int k = 0; k < 8; k++) {
                unsigned v = row[g + k], ex = (v >> 7) & 0xff;
                sm[g + k] = (uint8_t)(((v >> 8) & 0x80) | (v & 0x7f));
                if (ex >= l && ex <= l + 14) code[k] = (uint8_t)(ex - l);
                else { code[k] = 15; *x++ = (uint8_t)ex; }
            }
            for (int b = 0; b < 4; b++) c[g / 2 + b] = (uint8_t)(code[b] | code[b + 4] << 4);
        }
    }
    free(lo); free(off); free(d->w);
    d->w = buf; d->pk_codes = buf + a_codes; d->pk_lo = buf + a_lo;
    d->pk_off = (uint32_t *)(buf + a_off); d->pk_exc = buf + a_exc; d->pk_bytes = total;
    d->fmt = DW_BF16P;
    *from += e * 2; *to += total;
    return 1;
}

/* FP8 with a scale per (row, 128-column block): float inside a block, double
 * across blocks, the convention of quant.h's matmul_fp8. */
static void matmul_fp8_rows(float *y, const float *x, const uint8_t *q8, const float *rs,
                            int nblk, int S, int I, int O) {
#ifdef __AVX2__
    if (S > 1) {
        /* rows four at a time (a decode step's verified drafts, a prompt's): each group of
         * 8 weights decoded once for them, every row summed in the one-row order below */
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            const uint8_t *w = q8 + (int64_t)o * I;
            const float *scl = rs + (int64_t)o * nblk;
            for (int s0 = 0; s0 < S; s0 += 4) {
                int R = S - s0 < 4 ? S - s0 : 4;
                const float *x0 = x + (int64_t)s0 * I, *x1 = x0 + (R > 1 ? I : 0),
                            *x2 = x0 + (R > 2 ? 2 * (int64_t)I : 0), *x3 = x0 + (R > 3 ? 3 * (int64_t)I : 0);
                double a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                for (int b = 0; b < nblk; b++) {
                    int base = b * 128, end = base + 128 < I ? base + 128 : I, i = base;
                    __m256 v0 = _mm256_setzero_ps(), v1 = v0, v2 = v0, v3 = v0;
                    for (; i + 8 <= end; i += 8) {
                        __m256 wd = e4m3_decode8(w + i);
                        v0 = _mm256_fmadd_ps(_mm256_loadu_ps(x0 + i), wd, v0);
                        if (R > 1) v1 = _mm256_fmadd_ps(_mm256_loadu_ps(x1 + i), wd, v1);
                        if (R > 2) v2 = _mm256_fmadd_ps(_mm256_loadu_ps(x2 + i), wd, v2);
                        if (R > 3) v3 = _mm256_fmadd_ps(_mm256_loadu_ps(x3 + i), wd, v3);
                    }
                    float c0 = hsum256(v0), c1 = hsum256(v1), c2 = hsum256(v2), c3 = hsum256(v3);
                    for (; i < end; i++) {
                        float wi = e4m3_decode(w[i]);
                        c0 += wi * x0[i]; c1 += wi * x1[i]; c2 += wi * x2[i]; c3 += wi * x3[i];
                    }
                    a0 += (double)c0 * scl[b]; a1 += (double)c1 * scl[b];
                    a2 += (double)c2 * scl[b]; a3 += (double)c3 * scl[b];
                }
                y[(int64_t)s0 * O + o] = (float)a0;
                if (R > 1) y[(int64_t)(s0 + 1) * O + o] = (float)a1;
                if (R > 2) y[(int64_t)(s0 + 2) * O + o] = (float)a2;
                if (R > 3) y[(int64_t)(s0 + 3) * O + o] = (float)a3;
            }
        }
        return;
    }
#endif
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint8_t *w = q8 + (int64_t)o * I;
        const float *scl = rs + (int64_t)o * nblk;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            double a = 0;
            for (int b = 0; b < nblk; b++) {
                int base = b * 128, end = base + 128 < I ? base + 128 : I, i = base;
                float acc = 0;
#ifdef __AVX2__
                __m256 v = _mm256_setzero_ps();
                for (; i + 8 <= end; i += 8)
                    v = _mm256_fmadd_ps(_mm256_loadu_ps(xs + i), e4m3_decode8(w + i), v);
                acc = hsum256(v);
#endif
                for (; i < end; i++) acc += e4m3_decode(w[i]) * xs[i];
                a += (double)acc * scl[b];
            }
            y[(int64_t)s * O + o] = (float)a;
        }
    }
}

#ifdef COLI_VULKAN
/* Each form is one of the shader's formats as it is stored, so nothing is
 * converted on the way up:
 *   DW_F32   fmt 10, f32 [O, I], no scales
 *   DW_BF16  fmt 11, bf16 [O, I] (the low half of a word is the even column)
 *   DW_FP8   fmt 12, e4m3 [O, I] with the per-(row, 128-column block) scales
 *            rs[O][nblk] that dw_load_fp8 expanded: a group size of 128
 *   DW_I8    fmt 1, int8 [O, I] and one f32 scale per row after the sum
 * Every one is f32 weights times f32 activations, as on the CPU; what differs
 * is the order of the sums (the CPU's FP8 kernel adds its blocks in double). */
static int vk_main_thread(void) {
#ifdef _OPENMP
    return !omp_in_parallel();
#else
    return 1;
#endif
}

static int dw_vk_fmt(const DW *d, int *gs, const float **sc) {
    *gs = 0; *sc = NULL;
    switch (d->fmt) {
    case DW_F32:  return 10;
    case DW_BF16: return 11;
    case DW_FP8:  *gs = 128; *sc = d->s; return 12;
    case DW_I8:   *sc = d->s; return 1;
    }
    return -1;
}

static int g_mimo_vk_dev;   /* the device dw_upload sends a new matrix to: 1 for the chain's layers on
                            * COLI_VK_DEV2's (mimo_chain.h) */
static int dw_upload(DW *d) {
    int gs; const float *sc;
    int fmt = dw_vk_fmt(d, &gs, &sc);
    if (d->vk || d->vk_off || !d->w) return d->vk != NULL;
    if (fmt < 0 || !(g_mimo_vk_dev ? coli_vk_tensor_ensure2((ColiVkTensor **)&d->vk, d->w, sc, fmt, d->I, d->O, gs)
                                   : coli_vk_tensor_ensure((ColiVkTensor **)&d->vk, d->w, sc, fmt, d->I, d->O, gs))) d->vk_off = 1;
    return d->vk != NULL;
}

static int dw_matmul_vk(float *y, const float *x, int S, const DW *d) {
    if (!g_vk_ready || !(coli_vk_dense() || g_mimo_dho) || d->vk_off || !vk_main_thread()) return 0;
    DW *dev = (DW *)d;     /* the device copy is a cache inside a read-only matrix */
    if (!dw_upload(dev) || coli_vk_tensor_dev((ColiVkTensor *)dev->vk)) return 0;   /* the second device's: only its chain */
    int gs; const float *sc;
    int fmt = dw_vk_fmt(d, &gs, &sc);
    return coli_vk_matmul((ColiVkTensor **)&dev->vk, y, x, d->w, sc, fmt, S, d->I, d->O, gs);
}
#endif
#if defined(COLI_VULKAN) || defined(COLI_MIMO_CUDA)
static void dw_reload(DW *d);   /* below, beside the loaders it re-runs */
static size_t dw_bytes(const DW *d) {
    if (!d->w) return 0;
    size_t e = (size_t)d->O * d->I;
    switch (d->fmt) {
    case DW_F32:  return e * 4;
    case DW_BF16: return e * 2;
    case DW_FP8:  return e + (size_t)d->O * d->nblk * 4;
    case DW_I8:   return e + (size_t)d->O * 4;
    case DW_BF16P: return d->pk_bytes;
    }
    return 0;
}
#endif

static void dw_matmul(float *y, const float *x, int S, const DW *d) {
#ifdef COLI_MIMO_CUDA
    /* Vision may call dw_matmul inside an OpenMP region. The CUDA scratch is
     * main-thread only; those callers keep the ordinary CPU path. */
    int cuda_thread = 1;
#ifdef _OPENMP
    cuda_thread = !omp_in_parallel();
#endif
    if (g_mimo_cuda && cuda_thread && d->cuda) {
        if (g_mimo_cuda_fault && ++g_mimo_cuda_calls == g_mimo_cuda_fault) {
            fprintf(stderr, "[CUDA] mimo fault injection: returning to CPU at matmul %llu\n", g_mimo_cuda_calls);
            g_mimo_cuda = 0;
        } else if (mimo_cuda_matmul(d->cuda, y, x, S)) return;
    }
#endif
#ifdef COLI_VULKAN
    if (dw_matmul_vk(y, x, S, d)) return;
#endif
#if defined(COLI_VULKAN) || defined(COLI_MIMO_CUDA)
    if (d->vk_gone) dw_reload((DW *)d);   /* the CPU needs a matrix the device held alone */
#endif
    switch (d->fmt) {
    case DW_F32:  matmul(y, x, (const float *)d->w, S, d->I, d->O); break;
    case DW_BF16: matmul_bf16(y, x, (const uint16_t *)d->w, S, d->I, d->O); break;
    case DW_BF16P: matmul_bf16p(y, x, d, S); break;
    case DW_FP8:  matmul_fp8_rows(y, x, (const uint8_t *)d->w, d->s, d->nblk, S, d->I, d->O); break;
    case DW_I8:   matmul_q(y, x, (const int8_t *)d->w, d->s, S, d->I, d->O); break;
    }
}

static st_tensor *need(shards *S, const char *name, int dtype, int O, int I) {
    st_tensor *t = st_find(S, name);
    if (!t) st_die_missing(S, name);
    int esz = st_dtype_esz(dtype);
    int64_t numel = I > 0 ? (int64_t)O * I : O;
    int rank_ok = I > 0 ? (t->rank == 2 && t->shape[0] == O && t->shape[1] == I)
                        : (t->rank == 1 && t->shape[0] == O);
    if (t->dtype != dtype || !rank_ok || t->nbytes != numel * esz) {
        fprintf(stderr, "[mimo] %s: %s %lld bytes, expected %s [%d%s%d] -- refusing (untrusted container)\n",
                name, st_dtype_name(t->dtype), (long long)t->nbytes, st_dtype_name(dtype), O,
                I > 0 ? ", " : "", I > 0 ? I : 0);
        exit(1);
    }
    return t;
}

/* f32 copy of a BF16 or F32 vector */
static float *load_vec(shards *S, const char *name, int n) {
    st_tensor *t = st_find(S, name);
    if (!t) st_die_missing(S, name);
    int dtype = t->dtype == 2 ? 2 : 0;
    need(S, name, dtype, n, 0);
    float *out = xmalloc((size_t)n * sizeof(float), name);
    st_read_f32(S, name, out, 1);
    return out;
}

/* Rows of an f32 matrix [O, I] into the configured dense form. Takes ownership. */
static void dw_from_f32(DW *d, float *w, int O, int I) {
    d->O = O; d->I = I;
    if (g_dense_bits == 8) {
        d->fmt = DW_I8;
        d->w = xmalloc((size_t)O * I, "int8 weights");
        d->s = xmalloc((size_t)O * sizeof(float), "int8 scales");
        quantize_rows(w, (int8_t *)d->w, d->s, O, I, 8);
        free(w);
    } else {
        d->fmt = DW_F32; d->w = w; d->s = NULL;
    }
}

#if defined(COLI_VULKAN) || defined(COLI_MIMO_CUDA)
enum { DWK_NONE = 0, DWK_BF16, DWK_FP8, DWK_QKV };
/* What dw_reload needs to read the matrix back exactly as it was loaded: its name,
 * its loader, the MIMO_DENSE_BITS it was loaded under (the tower forces its own). */
static void dw_note(DW *d, const char *name, int kind) {
    free(d->vk_name);
    d->vk_name = strdup(name);
    if (!d->vk_name) { fprintf(stderr, "OOM weight name\n"); exit(1); }
    d->vk_kind = kind; d->vk_bits = g_dense_bits; d->vk_li = -1;
}
#define DW_NOTE(d, name, kind) dw_note(d, name, kind)
#else
#define DW_NOTE(d, name, kind) ((void)0)
#endif

static void dw_load_bf16(DW *d, shards *S, const char *name, int O, int I) {
    st_tensor *t = need(S, name, 0, O, I);
    DW_NOTE(d, name, DWK_BF16);
    if (g_dense_bits == 0) {
        d->fmt = DW_BF16; d->O = O; d->I = I; d->s = NULL;
        d->w = xmalloc((size_t)t->nbytes, name);
        st_read_raw_cap(S, name, d->w, t->nbytes, 1);
        return;
    }
    float *w = xmalloc((size_t)O * I * sizeof(float), name);
    st_read_f32(S, name, w, 1);
    dw_from_f32(d, w, O, I);
}

/* An FP8 [O, I] with its block grid; `row_block[r]` names the scale row of
 * stored row r (NULL: r / 128). `order[r]` names which stored row becomes row r
 * of the result (NULL: identity). */
static void dw_load_fp8(DW *d, shards *S, const char *name, int O, int I,
                        const int *row_block, int scale_rows, const int *order) {
    char sname[600];
    snprintf(sname, sizeof(sname), "%s_scale_inv", name);
    int nblk = (I + 127) / 128;
    if (scale_rows <= 0) scale_rows = (O + 127) / 128;
    st_tensor *t = need(S, name, 4, O, I);
    need(S, sname, 2, scale_rows, nblk);
    DW_NOTE(d, name, DWK_FP8);
    uint8_t *raw = xmalloc((size_t)t->nbytes, name);
    st_read_raw_cap(S, name, raw, t->nbytes, 1);
    float *grid = xmalloc((size_t)scale_rows * nblk * sizeof(float), sname);
    st_read_f32(S, sname, grid, 1);
    for (int64_t i = 0; i < (int64_t)scale_rows * nblk; i++)
        if (!isfinite(grid[i])) { fprintf(stderr, "[mimo] %s: non-finite scale -- refusing\n", sname); exit(1); }
    d->O = O; d->I = I; d->nblk = nblk;
    if (g_dense_bits == 0) {
        d->fmt = DW_FP8;
        uint8_t *w = xmalloc((size_t)O * I, name);
        float *rs = xmalloc((size_t)O * nblk * sizeof(float), sname);
        for (int r = 0; r < O; r++) {
            int src = order ? order[r] : r;
            int sb = row_block ? row_block[src] : src / 128;
            memcpy(w + (int64_t)r * I, raw + (int64_t)src * I, (size_t)I);
            memcpy(rs + (int64_t)r * nblk, grid + (int64_t)sb * nblk, (size_t)nblk * sizeof(float));
        }
        d->w = w; d->s = rs;
        free(raw); free(grid);
        return;
    }
    float *w = xmalloc((size_t)O * I * sizeof(float), name);
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < O; r++) {
        int src = order ? order[r] : r;
        int sb = row_block ? row_block[src] : src / 128;
        const uint8_t *q = raw + (int64_t)src * I;
        float *dst = w + (int64_t)r * I;
        for (int i = 0; i < I; i++) dst[i] = e4m3_decode(q[i]) * grid[(int64_t)sb * nblk + i / 128];
    }
    free(raw); free(grid);
    dw_from_f32(d, w, O, I);
}

/* The fused qkv of layer `li`, read in its pre-sharded order and handed back as
 * [Q | K | V] rows. Two scale grids are legal (vLLM accepts the same two): one
 * tiled per chunk, ceil(rows_per_chunk / 128) rows each, or one continuous grid;
 * they coincide when a chunk is a whole number of blocks. */
static void load_qkv_named(DW *d, shards *S, const Cfg *c, int k, const char *name) {
    int T = c->chunks, hd = c->head_dim[k], vd = c->v_dim[k];
    int q_rows = (c->heads[k] / T) * hd, k_rows = (c->kv_heads[k] / T) * hd,
        v_rows = (c->kv_heads[k] / T) * vd;
    int rpc = q_rows + k_rows + v_rows, O = T * rpc;
    int per_chunk = (rpc + 127) / 128, continuous = (O + 127) / 128;
    char sname[600];
    snprintf(sname, sizeof(sname), "%s_scale_inv", name);
    st_tensor *st = st_find(S, sname);
    if (!st) st_die_missing(S, sname);
    int scale_rows = st->rank == 2 ? (int)st->shape[0] : -1;
    int *row_block = xmalloc((size_t)O * sizeof(int), "qkv scale rows");
    for (int r = 0; r < O; r++) {
        if (scale_rows == T * per_chunk) row_block[r] = (r / rpc) * per_chunk + (r % rpc) / 128;
        else if (scale_rows == continuous) row_block[r] = r / 128;
        else {
            fprintf(stderr, "[mimo] %s: %d scale rows fit neither the per-chunk grid (%d) nor a "
                            "continuous one (%d) -- refusing\n", sname, scale_rows, T * per_chunk, continuous);
            exit(1);
        }
    }
    int *order = xmalloc((size_t)O * sizeof(int), "qkv order");
    int w = 0;
    for (int ch = 0; ch < T; ch++) for (int r = 0; r < q_rows; r++) order[w++] = ch * rpc + r;
    for (int ch = 0; ch < T; ch++) for (int r = 0; r < k_rows; r++) order[w++] = ch * rpc + q_rows + r;
    for (int ch = 0; ch < T; ch++) for (int r = 0; r < v_rows; r++) order[w++] = ch * rpc + q_rows + k_rows + r;
    dw_load_fp8(d, S, name, O, c->hidden, row_block, scale_rows, order);
    free(row_block); free(order);
}
static void load_qkv(DW *d, shards *S, const Cfg *c, int li) {
    char name[512];
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.qkv_proj.weight", li);
    load_qkv_named(d, S, c, c->swa[li], name);
#if defined(COLI_VULKAN) || defined(COLI_MIMO_CUDA)
    d->vk_kind = DWK_QKV; d->vk_li = li;   /* read back through this function: the de-interleave */
#endif
}

#if defined(COLI_VULKAN) || defined(COLI_MIMO_CUDA)
/* A matrix whose host copy was dropped by CUDA or Vulkan and that the CPU needs
 * after all: read it back from the checkpoint with the loader and the MIMO_DENSE_BITS
 * it was first read with, the same bytes, and keep it from there on. */
static size_t dw_bytes(const DW *d);
static pthread_mutex_t g_dw_dho_mx = PTHREAD_MUTEX_INITIALIZER;
static shards *g_dw_dho_S;
static const Cfg *g_dw_dho_c;
static void dw_reload(DW *d) {
    pthread_mutex_lock(&g_dw_dho_mx);
    if (d->vk_gone) {
        DW t = {0};
        int saved = g_dense_bits;
        g_dense_bits = d->vk_bits;
        if (!g_dw_dho_S || !d->vk_name) { fprintf(stderr, "[VK] mimo: a dense matrix the device held alone cannot be read back\n"); exit(1); }
        switch (d->vk_kind) {
        case DWK_BF16: dw_load_bf16(&t, g_dw_dho_S, d->vk_name, d->O, d->I); break;
        case DWK_FP8:  dw_load_fp8(&t, g_dw_dho_S, d->vk_name, d->O, d->I, NULL, 0, NULL); break;
        case DWK_QKV:  load_qkv(&t, g_dw_dho_S, g_dw_dho_c, d->vk_li); break;
        default: fprintf(stderr, "[VK] mimo: %s has no loader to read it back\n", d->vk_name); exit(1);
        }
        g_dense_bits = saved;
        if (t.fmt != d->fmt || t.O != d->O || t.I != d->I) { fprintf(stderr, "[VK] mimo: %s came back in another form\n", d->vk_name); exit(1); }
        d->w = t.w; d->s = t.s; d->nblk = t.nblk;
        free(t.vk_name);
        d->vk_gone = 0;
#ifdef COLI_MIMO_CUDA
        if (d->cuda) { g_mimo_cuda_reloaded++; g_mimo_cuda_reload_bytes += dw_bytes(d); }
#endif
#ifdef COLI_VULKAN
        if (g_vk_ready) coli_vk_dense_host_reloaded(dw_bytes(d));
#endif
    }
    pthread_mutex_unlock(&g_dw_dho_mx);
}
#endif

/* ------------------------------------------------------------------ model ---- */

/* An expert slot of a layer's cache. The main thread owns its identity (eid, the
 * layer's by_expert, pin, used, pref); a reader thread only fills the bytes of a
 * LOADING slot and then marks it READY (the expert streaming below). */
enum { SLOT_EMPTY = 0, SLOT_LOADING, SLOT_READY };
typedef struct {
    int eid; uint64_t used; uint8_t *base, *buf;
    _Atomic int state;
    int pin;           /* uses still to come in the MoE step being computed */
    int pref;          /* read ahead by the prefetcher and not asked for yet */
    _Atomic int pending;   /* pieces of its read not done yet (the readers count down) */
    int qdrive, qn;        /* a prefetch's pieces: qn jobs from qpos in drive qdrive's queue */
    unsigned qpos;
} Slot;
typedef struct { Slot *s; int n, cap, max, nbuf; int *by_expert; uint8_t *arena; } LCache;
/* MIMO_CACHE_GLOBAL (default 1; 0 in a VK=1 build, whose expert tier steers the
 * per-layer choice): the MoE layers share one pool of cap x layers buffers. A layer
 * that needs a slot in a decode step when the pool is full takes the least recently
 * used buffer of any layer (not pinned, not loading, not read ahead and still
 * waiting), and may hold up to twice cap, so the layers whose routing spreads wider
 * hold more: on Pro a decode step reads about 10% fewer experts on demand. A prompt
 * block's step reuses the layer's own least recently used buffer once the layer
 * holds cap of them: a new prompt's many experts would otherwise push out what the
 * next layers hold. 0: each layer its own cap, least recently used within it. */
#ifdef COLI_VULKAN
#define MIMO_CACHE_GLOBAL_DEFAULT 0
#else
#define MIMO_CACHE_GLOBAL_DEFAULT 1
#endif
static int g_cache_global, g_slots_total, g_slots_used;
typedef struct { int fd[6], fi[6]; int64_t off[6]; int contig; } ERef;

typedef struct {
    float *ln1, *ln2, *sink;
    DW qkv, o;
    DW gate, up, down;           /* dense MLP (layer 0) */
    uint16_t *router;            /* [E, H] bf16, as released (matmul_router) */
    float *bias;                 /* [E] */
    /* KV: full layers [ctx][kvh][hd]; SWA layers a ring of `window` rows */
    float *K, *V;
    int *ring_pos;
    int rows;
    int kv_lo;                   /* a window's first position held: 0, or where the MTP block began */
} Layer;

/* One conversation's caches for a multiplexed serve (KV_SLOTS>1, serve_mux below):
 * every layer's K and V (a full layer's rows, a sliding window's ring and its
 * positions), where it stands, and the record of the tokens it holds. The Model's
 * layers hold the conversation a prefill runs on (mimo_seq_swap trades it for a parked
 * one); a multiplexed decode step parks them all and reads each row's from its MimoRow. */
typedef struct {
    float *K[MIMO_MAX_LAYERS], *V[MIMO_MAX_LAYERS];
    int *ring_pos[MIMO_MAX_LAYERS];
    int pos;
    kv_prefix kvp;
} MimoSeq;
typedef struct { MimoSeq *seq; int pos; } MimoRow;
static int g_mimo_mux_slots = 1;   /* KV_SLOTS: the conversations a serve decodes at once */

typedef struct {
    Cfg c;
    shards S;
    Layer L[MIMO_MAX_LAYERS];
    uint16_t *embed;             /* bf16 [V, H], rows widened on use */
    DW head;
    float *norm;
    ERef *eref;
    int64_t part[6], e_bytes;    /* down, down scale, gate, gate scale, up, up scale */
    LCache cache[MIMO_MAX_LAYERS];
    int ctx, pos;
    kv_prefix kvp;
    uint64_t clock, hits, miss, bytes_read, forwards;
    double t_disk, t_expert, t_attn;
    int idot, direct;
    uint8_t **ehit;              /* [layer][expert] routed this turn, for HITS */
    /* The prefetcher (MIMO_PREFETCH): the next MoE layer's router applied early to the
     * residual, its likely experts read while this layer computes. pf_h is the residual
     * a prediction is waiting for (taken right after the current layer's own reads). */
    int pf_on, pf_rows, pf_extra, pf_limit;
    int bulk;          /* the MoE step being computed has more rows than pf_rows (a prompt block) */
    int pf_ka, pf_kb;            /* experts predicted per row after the attention, after the MLP */
    uint64_t waves, wave_experts;   /* moe_wave calls and the experts they computed */
    const float *pf_h; int pf_n, pf_li;
    uint8_t *pf_mark;            /* [layer][expert] predicted for the layer's next step */
    uint8_t *pf_made;            /* [layer] a prediction stands */
    uint64_t pf_issued, pf_used, pf_wasted, pf_pred, pf_pred_hit;
    double t_pred, t_qkv, t_oproj;   /* t_attn's matmuls: qkv, and o_proj */
    /* MIMO_CACHE_DECAY > 0: a slot to reuse is the one whose expert has the lowest
     * decayed use count (each of its layer's steps multiplies it by the decay), the
     * least recently used among equals; 0 keeps plain LRU */
    float decay, *efreq, *dpow;      /* efreq [layer][expert]; dpow[k] = decay^k */
    uint32_t *etime, lstep[MIMO_MAX_LAYERS];
#ifdef COLI_VULKAN
    void *vkchain;               /* the dense chain's device state (mimo_chain.h), NULL until it runs */
    void *vkchain2;              /* its layers on COLI_VK_DEV2's device, after the primary's (mimo_chain.h) */
#endif
    /* A multiplexed decode step (forward_rows): row t is the token at mux_rows[t].pos
     * of mux_rows[t].seq; NULL in every other forward. */
    const MimoRow *mux_rows;
} Model;

/* ---- MTP drafts (MIMO_MTP) -----------------------------------------------------------
 * The checkpoint's multi-token-prediction layers (model.mtp.layers.k) draft the tokens
 * after the next one. Block k's row at position p takes the trunk's final hidden state
 * at p, normed by the trunk's final norm (what the serving stacks hand it), and the
 * embedding of the token at p + k + 1, through eh_proj ([enorm(embedding) |
 * hnorm(hidden)]), one sliding-window attention block with a dense MLP of its own
 * (m->L[n_layers + k], its own K/V ring), its final norm and the shared lm_head: it
 * predicts the token at p + k + 2. Every block reads the trunk's hidden state, none the
 * block before it, as SGLang's multi-layer MTP runs MiMo-V2's. Once the token at P is
 * chosen, block 0's row at P - 1 drafts the token at P + 1, block 1's row there (with
 * that draft for its token) the one at P + 2, and so on; a decode step runs the token
 * and the drafts the blocks are sure of (MIMO_MTP_MIN_P) as rows of one block. Every
 * kernel computes a row's bytes whatever rows are beside it, so each row's logits are
 * those a plain step would compute after the tokens before it, and so is the token they
 * give (sample_draft: the argmax, or a draw that keeps the draft at the probability the
 * sampler gives it). The drafts stand up to the first one that is not that token; the
 * token drawn instead ends the step, and the rows after it are dropped -- the position
 * goes back and the next step overwrites their K/V. A block's rows whose token was a
 * draft run again once the token is known, so its window holds the rows it would have
 * computed one token at a time. */
#define MIMO_MTP_MAX 3                /* the MTP layers Pro's checkpoint carries */
/* Tests (tests/mimo_tiny_harness.py, the by-hand path): MIMO_MTP_ORACLE, the reference's
 * ids from position 0, stands in for the blocks' drafts, so the steps keep them;
 * MIMO_MTP_ORACLE_MISS=m makes every m-th draft one the trunk does not choose. The blocks
 * run as always. */
static int *g_mtp_oracle, g_mtp_oracle_n, g_mtp_oracle_miss, g_mtp_oracle_drafts;
typedef struct {
    DW eh;                        /* eh_proj [H, 2H] */
    float *enorm, *hnorm, *fnorm;
    int next;                     /* its first position whose row still waits for its token */
    uint64_t tried, kept;         /* its drafts verified, and those that stood */
} MtpBlock;
typedef struct {
    int on, n, li;                /* n blocks: m->L[li + k], li = n_layers, sliding-window layers */
    float min_p;                  /* MIMO_MTP_MIN_P: the drafts verified multiply to at least this */
    FILE *trace;                  /* MIMO_MTP_TRACE */
    MtpBlock b[MIMO_MTP_MAX];
    float *keep;                  /* the trunk's final hidden rows of the last keep_rows positions */
    int keep_rows;
    int tok[8];                   /* the tokens of the last positions, tok[p & 7] */
    uint64_t steps;               /* verified steps */
    double t;
} Mtp;
static Mtp g_mtp;

static void mtp_keep(Model *m, const float *hc, int n, int p0) {
    int H = m->c.hidden;
    for (int t = n > g_mtp.keep_rows ? n - g_mtp.keep_rows : 0; t < n; t++)
        memcpy(g_mtp.keep + (size_t)((p0 + t) % g_mtp.keep_rows) * H, hc + (size_t)t * H, (size_t)H * sizeof(float));
}

static void expert_table_init(Model *m) {
    Cfg *c = &m->c;
    int H = c->hidden, MI = c->moe_inter;
    m->part[0] = (int64_t)H * (MI / 2);  m->part[1] = (int64_t)H * (MI / 32);
    m->part[2] = (int64_t)MI * (H / 2);  m->part[3] = (int64_t)MI * (H / 32);
    m->part[4] = m->part[2];             m->part[5] = m->part[3];
    m->e_bytes = 0;
    for (int k = 0; k < 6; k++) m->e_bytes += m->part[k];
    m->eref = xcalloc((size_t)c->n_layers * c->n_experts, sizeof(ERef), "expert table");
    static const char *mat[3] = {"down_proj", "gate_proj", "up_proj"};
    int missing = 0, split = 0;
    for (int li = 0; li < c->n_layers; li++) {
        if (!c->moe[li]) continue;
        for (int e = 0; e < c->n_experts; e++) {
            ERef *er = &m->eref[(int64_t)li * c->n_experts + e];
            for (int k = 0; k < 6; k++) {
                char nm[512];
                snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.%d.%s.%s", li, e,
                         mat[k / 2], (k & 1) ? "weight_scale" : "weight");
                st_tensor *t = st_find(&m->S, nm);
                if (!t) { missing++; er->fd[k] = -1; er->fi[k] = -1; continue; }
                if (t->dtype != 3 || t->nbytes != m->part[k]) {
                    fprintf(stderr, "[mimo] %s: %s %lld bytes, expected U8 %lld -- refusing "
                                    "(untrusted container)\n", nm, st_dtype_name(t->dtype),
                            (long long)t->nbytes, (long long)m->part[k]);
                    exit(1);
                }
                er->fd[k] = t->fd; er->off[k] = t->off;
                er->fi[k] = st_fidx(&m->S, t->fd);   /* the shard, for the readers' own handles */
            }
            /* The release stores an expert's six tensors back to back in this
             * order (down, down scale, gate, gate scale, up, up scale): one read. */
            er->contig = er->fd[0] >= 0;
            for (int k = 0; k < 5 && er->contig; k++)
                if (er->fd[k] != er->fd[k + 1] || er->off[k] + m->part[k] != er->off[k + 1]) er->contig = 0;
            split += !er->contig;
        }
    }
    if (missing) fprintf(stderr, "[mimo] WARNING: %d expert tensors missing (incomplete download?) -- touching one aborts\n", missing);
    if (split) fprintf(stderr, "[mimo] %d experts are not contiguous on disk: six reads each\n", split);
}

static void cache_init(Model *m, int cap) {
    Cfg *c = &m->c;
    if (cap < c->topk) cap = c->topk;
    if (cap > c->n_experts) cap = c->n_experts;
    for (int li = 0; li < c->n_layers; li++) {
        if (!c->moe[li]) continue;
        LCache *lc = &m->cache[li];
        lc->cap = cap; lc->n = 0;
        lc->max = g_cache_global ? (2 * cap < c->n_experts ? 2 * cap : c->n_experts) : cap;
        lc->s = xcalloc((size_t)lc->max, sizeof(Slot), "expert slots");
        lc->by_expert = xmalloc((size_t)c->n_experts * sizeof(int), "expert index");
        for (int e = 0; e < c->n_experts; e++) lc->by_expert[e] = -1;
        for (int k = 0; k < lc->max; k++) lc->s[k].eid = -1;
        g_slots_total += cap;
    }
}

/* ---- expert streaming ------------------------------------------------------------
 * Every expert read runs on a pool of reader threads, so the CPU computes the experts
 * already in RAM while the missing ones arrive, and each arrives to be computed at
 * once (moe_cpu). Two queues: a demand (the router of the step being computed chose
 * the expert) is always taken before a prefetch (mimo_prefetch: the next layer's
 * router applied early predicts it), and a demand for an expert whose prefetch still
 * waits moves it to the front. On Windows each reader opens its own unbuffered
 * handle per shard: reads through one synchronous handle are serialized by its file
 * object's lock.
 *
 * None of this changes a number: an expert is computed only once its slot is READY,
 * from the same bytes, and every row adds its experts in its own routing order. What
 * the streaming changes is when the bytes arrive and which experts stay cached. */
#define MIMO_TRACE_RANK 32
static FILE *g_rtrace;             /* MIMO_ROUTE_TRACE (mimo_trace_step, below) */
static void mimo_trace_close(void);

#define MIMO_IOQ 4096
#define MIMO_IO_DRIVES 8
typedef struct { Slot *s; int li, eid, part; } IoJob;   /* part: a piece of the read, -1 the whole */
static struct {
    pthread_t th[64];
    int nth, started, quit;
    pthread_mutex_t mx;
    pthread_cond_t job, done;
    int ndrive, drive_of[ST_MAX_SHARDS];   /* the shards' devices: one queue pair each */
    IoJob q[MIMO_IO_DRIVES][2][MIMO_IOQ];   /* [drive][0: demands, 1: prefetches], rings of head..tail */
    unsigned head[MIMO_IO_DRIVES][2], tail[MIMO_IO_DRIVES][2];
    unsigned long long finished;           /* reads completed, for a wait on any of them */
    Model *m;
} g_io;                                     /* zeroed: its locks are made by io_start */

/* A reader's handle on shard fi: its own on Windows (opened on first use), the shared
 * one elsewhere, where pread on one descriptor runs concurrently. */
static int io_fd(Model *m, int *own, int fi, int direct) {
    if (fi < 0) return -1;
#ifdef _WIN32
    int *h = &own[(direct ? 0 : ST_MAX_SHARDS) + fi];
    if (*h == -2) *h = direct ? compat_open_direct(m->S.paths[fi]) : open(m->S.paths[fi], COMPAT_O_RDONLY);
    return *h;
#else
    (void)own;
    return direct ? m->S.dfds[fi] : m->S.fds[fi];
#endif
}

/* An expert's read, optionally cut into pieces of MIMO_IO_PIECE_MB so that a demand
 * waits behind at most a piece of each reader's current read and one expert arrives
 * through several readers at once. Unset, an expert is one read: with two readers
 * a drive, 4 MB pieces kept too few bytes in flight for the NVMe drives measured and
 * doubled the stalls. The pieces tile [start, start + len) of the shard, into the
 * slot from base; buf is base + pad. Direct reads are aligned to 4 KiB: an expert
 * whose aligned range runs past its shard's end is read buffered. An expert not
 * stored in one run (six tensors apart) is one piece. */
static int64_t g_io_piece = (int64_t)1 << 40;
typedef struct { int64_t start, len, pad; int direct, whole; } ExpertSpan;
static int expert_span(const Model *m, int li, int eid, ExpertSpan *sp) {
    const ERef *er = &m->eref[(int64_t)li * m->c.n_experts + eid];
    memset(sp, 0, sizeof *sp);
    if (!er->contig) { sp->whole = 1; return 1; }
    int64_t a0 = er->off[0] & ~4095LL, pad = er->off[0] - a0;
    int64_t dlen = (pad + m->e_bytes + 4095) & ~4095LL, size = m->S.sizes[er->fi[0]];
    if (m->direct && a0 + dlen <= size) { sp->start = a0; sp->len = dlen; sp->pad = pad; sp->direct = 1; }
    else { sp->start = er->off[0]; sp->len = m->e_bytes; sp->pad = 0; }
    return (int)((sp->len + g_io_piece - 1) / g_io_piece);
}

static void expert_read(Model *m, int li, int eid, Slot *s, int *own, int part) {
    ERef *er = &m->eref[(int64_t)li * m->c.n_experts + eid];
    if (er->fd[0] < 0) { fprintf(stderr, "[mimo] expert L%d E%d missing on disk\n", li, eid); exit(1); }
    ExpertSpan sp;
    expert_span(m, li, eid, &sp);
    if (part >= 0 && !sp.whole) {
        int64_t at = (int64_t)part * g_io_piece, n = sp.len - at < g_io_piece ? sp.len - at : g_io_piece;
        int fd = sp.direct ? io_fd(m, own, er->fi[0], 1) : -1;
        if (fd < 0) fd = io_fd(m, own, er->fi[0], 0);   /* no unbuffered handle: the same bytes buffered */
        if (fd < 0) fd = er->fd[0];
        st_pread_full(fd, s->base + at, n, sp.start + at, "pread expert piece");
        return;
    }
    if (er->contig) {
        int dfd = m->direct ? io_fd(m, own, er->fi[0], 1) : -1;
        int bfd = io_fd(m, own, er->fi[0], 0);
        if (bfd < 0) bfd = er->fd[0];
        if (dfd >= 0) {
            int64_t a0 = er->off[0] & ~4095LL, pad = er->off[0] - a0, want = pad + m->e_bytes;
            int64_t dlen = (want + 4095) & ~4095LL, size = m->S.sizes[er->fi[0]];
            if (a0 + dlen > size) dlen = (size - a0) & ~4095LL;
            if (dlen > 0) st_pread_full(dfd, s->base, dlen, a0, "pread expert direct");
            if (dlen < want) st_pread_full(bfd, s->base + dlen, want - dlen, a0 + dlen, "pread expert tail");
            s->buf = s->base + pad;
        } else {
            st_pread_full(bfd, s->base, m->e_bytes, er->off[0], "pread expert");
            s->buf = s->base;
        }
    } else {
        uint8_t *dst = s->base;
        for (int k = 0; k < 6; k++) {
            if (er->fd[k] < 0) { fprintf(stderr, "[mimo] expert L%d E%d tensor %d missing\n", li, eid, k); exit(1); }
            int fd = io_fd(m, own, er->fi[k], 0);
            st_pread_full(fd >= 0 ? fd : er->fd[k], dst, m->part[k], er->off[k], "pread expert");
            dst += m->part[k];
        }
        s->buf = s->base;
    }
}

/* A reader serves one drive: its demands first, then its prefetches. A drive has a
 * few readers (MIMO_IO_PER_DRIVE), enough for its bandwidth with 20 MB reads, so a
 * demand never queues behind more than that many prefetches on its own drive. */
typedef struct { Model *m; int drive; } IoArg;
static void *io_main(void *arg) {
    IoArg a = *(IoArg *)arg;
    free(arg);
    Model *m = a.m;
    int d = a.drive, nh = 2 * ST_MAX_SHARDS;
    int *own = xmalloc((size_t)nh * sizeof(int), "reader handles");
    for (int i = 0; i < nh; i++) own[i] = -2;
    pthread_mutex_lock(&g_io.mx);
    for (;;) {
        while (!g_io.quit && g_io.head[d][0] == g_io.tail[d][0] && g_io.head[d][1] == g_io.tail[d][1])
            pthread_cond_wait(&g_io.job, &g_io.mx);
        if (g_io.quit) break;
        int p = g_io.head[d][0] != g_io.tail[d][0] ? 0 : 1;
        IoJob j = g_io.q[d][p][g_io.head[d][p]++ % MIMO_IOQ];
        if (!j.s) continue;                 /* a prefetch a demand moved to the front */
        pthread_mutex_unlock(&g_io.mx);
        expert_read(m, j.li, j.eid, j.s, own, j.part);
        pthread_mutex_lock(&g_io.mx);
        if (atomic_fetch_sub_explicit(&j.s->pending, 1, memory_order_acq_rel) == 1) {
            atomic_store_explicit(&j.s->state, SLOT_READY, memory_order_release);
            g_io.finished++;
            pthread_cond_broadcast(&g_io.done);
        }
    }
    pthread_mutex_unlock(&g_io.mx);
    for (int i = 0; i < nh; i++) if (own[i] >= 0) close(own[i]);
    free(own);
    return NULL;
}

static void io_stop(void) {
    if (!g_io.started) return;
    pthread_mutex_lock(&g_io.mx);
    g_io.quit = 1;
    pthread_cond_broadcast(&g_io.job);
    pthread_mutex_unlock(&g_io.mx);
    for (int i = 0; i < g_io.nth; i++) pthread_join(g_io.th[i], NULL);
    g_io.started = 0; g_io.nth = 0;
}

/* The drive of an expert: the device of the shard holding its first tensor. */
static int io_drive(Model *m, int li, int eid) {
    int fi = m->eref[(int64_t)li * m->c.n_experts + eid].fi[0];
    return fi >= 0 ? g_io.drive_of[fi] : 0;
}

/* MIMO_IO_PER_DRIVE readers (default 2) on each device the shards are on -- a RAID
 * volume is one device, two drive letters are two -- or MIMO_IO_THREADS in all on one
 * queue; MIMO_IO_THREADS=0 reads on the caller, one expert after the other. */
static void io_start(Model *m) {
    if (g_io.started) return;
    pthread_mutex_init(&g_io.mx, NULL);
    pthread_cond_init(&g_io.job, NULL);
    pthread_cond_init(&g_io.done, NULL);
    int per = env_int("MIMO_IO_PER_DRIVE", 2), total = env_int("MIMO_IO_THREADS", -1);
    const char *pm = getenv("MIMO_IO_PIECE_MB");
    if (pm && *pm && atof(pm) > 0) g_io_piece = ((int64_t)(atof(pm) * 1048576.0) + 4095) & ~4095LL;
    long long dev[MIMO_IO_DRIVES];
    g_io.ndrive = 0;
    for (int i = 0; i < m->S.nfd && i < ST_MAX_SHARDS; i++) {
        struct stat sb;
        long long id = stat(m->S.paths[i], &sb) == 0 ? (long long)sb.st_dev : -1;
        int k = 0;
        while (k < g_io.ndrive && dev[k] != id) k++;
        if (k == g_io.ndrive) {
            if (g_io.ndrive < MIMO_IO_DRIVES && total < 0) dev[g_io.ndrive++] = id;
            else k = 0;
        }
        g_io.drive_of[i] = k;
    }
    if (!g_io.ndrive) g_io.ndrive = 1;
    int n = total >= 0 ? total : per * g_io.ndrive;
    if (n < 0) n = 0;
    if (n > 64) n = 64;
    g_io.m = m; g_io.quit = 0;
    for (int i = 0; i < n; i++) {
        IoArg *a = xmalloc(sizeof *a, "reader");
        a->m = m; a->drive = i % g_io.ndrive;
        if (pthread_create(&g_io.th[i], NULL, io_main, a)) {
            free(a);
            fprintf(stderr, "[mimo] reader thread %d did not start; %d readers\n", i, i);
            break;
        }
        g_io.nth++;
    }
    if (g_io.nth && g_io.nth < g_io.ndrive) {   /* too few to give each drive one: one shared queue */
        for (int i = 0; i < ST_MAX_SHARDS; i++) g_io.drive_of[i] = 0;
        g_io.ndrive = 1;
    }
    g_io.started = 1;
    atexit(io_stop);
    fprintf(stderr, "[mimo] expert readers: %d on %d device(s)\n", g_io.nth, g_io.ndrive);
}

static void io_submit(Model *m, Slot *s, int li, int eid, int prefetch) {
    if (!g_io.started) io_start(m);
    ExpertSpan sp;
    int parts = expert_span(m, li, eid, &sp);
    s->buf = s->base + sp.pad;
    if (!g_io.nth) {                        /* no reader thread: read it now */
        int nh = 2 * ST_MAX_SHARDS;
        static int *own;
        if (!own) { own = xmalloc((size_t)nh * sizeof(int), "reader handles"); for (int i = 0; i < nh; i++) own[i] = -2; }
        double t0 = now_s();
        expert_read(m, li, eid, s, own, -1);
        if (!prefetch) m->t_disk += now_s() - t0;
        atomic_store_explicit(&s->state, SLOT_READY, memory_order_release);
        return;
    }
    int d = io_drive(m, li, eid), p = prefetch ? 1 : 0;
    atomic_store_explicit(&s->pending, parts, memory_order_relaxed);
    pthread_mutex_lock(&g_io.mx);
    if (g_io.tail[d][p] - g_io.head[d][p] + (unsigned)parts > MIMO_IOQ) { fprintf(stderr, "[mimo] reader queue overflow\n"); exit(1); }
    s->qdrive = d; s->qpos = g_io.tail[d][p]; s->qn = p ? parts : 0;
    for (int k = 0; k < parts; k++)
        g_io.q[d][p][g_io.tail[d][p]++ % MIMO_IOQ] = (IoJob){s, li, eid, sp.whole ? -1 : k};
    pthread_cond_broadcast(&g_io.job);
    pthread_mutex_unlock(&g_io.mx);
}

/* A slot whose prefetch is still queued: its read moves to the demand queue. */
static void io_promote(Slot *s) {
    if (!g_io.nth) return;
    pthread_mutex_lock(&g_io.mx);
    int d = s->qdrive, moved = 0;
    for (int k = 0; k < s->qn; k++) {
        unsigned pos = s->qpos + (unsigned)k;
        if (pos - g_io.head[d][1] >= g_io.tail[d][1] - g_io.head[d][1]) continue;   /* taken already */
        IoJob *j = &g_io.q[d][1][pos % MIMO_IOQ];
        if (j->s != s) continue;
        g_io.q[d][0][g_io.tail[d][0]++ % MIMO_IOQ] = *j;
        j->s = NULL;
        moved = 1;
    }
    s->qn = 0;
    if (moved) pthread_cond_broadcast(&g_io.job);
    pthread_mutex_unlock(&g_io.mx);
}

static unsigned long long io_seen(void) {
    if (!g_io.nth) return 0;
    pthread_mutex_lock(&g_io.mx);
    unsigned long long f = g_io.finished;
    pthread_mutex_unlock(&g_io.mx);
    return f;
}

/* Blocks until a read completes after `seen` (io_seen before looking at the slots). */
static void io_wait_any(Model *m, unsigned long long seen) {
    if (!g_io.nth) return;
    double t0 = now_s();
    pthread_mutex_lock(&g_io.mx);
    while (g_io.finished == seen) pthread_cond_wait(&g_io.done, &g_io.mx);
    pthread_mutex_unlock(&g_io.mx);
    m->t_disk += now_s() - t0;
}

static inline int slot_ready(Slot *s) {
    return atomic_load_explicit(&s->state, memory_order_acquire) == SLOT_READY;
}

static void io_wait_slot(Model *m, Slot *s) {
    while (!slot_ready(s)) {
        unsigned long long seen = io_seen();
        if (slot_ready(s)) break;
        io_wait_any(m, seen);
    }
}

#define MIMO_DPOW 4096
/* The decayed use count of expert e of layer li now (MIMO_CACHE_DECAY). */
static inline float expert_freq(const Model *m, int li, int e) {
    size_t k = (size_t)li * m->c.n_experts + e;
    uint32_t age = m->lstep[li] - m->etime[k];
    return age < MIMO_DPOW ? m->efreq[k] * m->dpow[age] : 0.0f;
}
/* A step of layer li used experts uni[0..nu): their counts decay to now and grow by one. */
static void expert_freq_step(Model *m, int li, const int *uni, int nu) {
    if (!m->efreq) return;
    m->lstep[li]++;
    for (int j = 0; j < nu; j++) {
        size_t k = (size_t)li * m->c.n_experts + uni[j];
        m->efreq[k] = expert_freq(m, li, uni[j]) + 1.0f;
        m->etime[k] = m->lstep[li];
    }
}

/* MIMO_LARGE_PAGES=1 (Windows): a layer's slots in one arena of 2 MiB pages, made at
 * the layer's first read. 144 GB of experts in 4 KiB pages need 288 MB of page tables,
 * far more than the caches hold, so every page the kernel streams through costs a
 * table walk from RAM; 2 MiB pages need 512 times fewer entries. The pages are locked
 * (the account's "Lock pages in memory" right, SeLockMemoryPrivilege, enabled for the
 * process). An arena that cannot be made leaves the layer on ordinary pages. */
static int g_large_pages = -1;    /* -1 not tried yet, 0 off, 1 on */
static size_t g_large_page;
#ifdef _WIN32
static int large_pages_enable(void) {
    HANDLE tok; TOKEN_PRIVILEGES tp;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok)) return 0;
    int ok = LookupPrivilegeValueA(NULL, "SeLockMemoryPrivilege", &tp.Privileges[0].Luid);
    if (ok) {
        tp.PrivilegeCount = 1; tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
        ok = AdjustTokenPrivileges(tok, FALSE, &tp, 0, NULL, NULL) && GetLastError() == ERROR_SUCCESS;
    }
    CloseHandle(tok);
    g_large_page = GetLargePageMinimum();
    return ok && g_large_page > 0;
}
#endif
static size_t slot_stride(const Model *m) { return ((size_t)m->e_bytes + 8192 + 4095) & ~(size_t)4095; }
static void slot_alloc(Model *m, int li, Slot *s, int idx) {
#ifdef _WIN32
    if (g_cache_global) g_large_pages = 0;
    if (g_large_pages < 0) {
        g_large_pages = env_int("MIMO_LARGE_PAGES", 0) ? large_pages_enable() : 0;
        if (env_int("MIMO_LARGE_PAGES", 0) && !g_large_pages)
            fprintf(stderr, "[mimo] MIMO_LARGE_PAGES: the lock-pages right is not available; ordinary pages\n");
    }
    LCache *lc = &m->cache[li];
    if (g_large_pages > 0 && !lc->arena) {
        size_t bytes = slot_stride(m) * (size_t)lc->cap;
        bytes = (bytes + g_large_page - 1) / g_large_page * g_large_page;
        lc->arena = VirtualAlloc(NULL, bytes, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
        if (!lc->arena) {
            fprintf(stderr, "[mimo] MIMO_LARGE_PAGES: no %.2f GB of 2 MiB pages for layer %d (error %lu); "
                            "ordinary pages from here on\n", bytes / 1e9, li, GetLastError());
            g_large_pages = 0;
        }
    }
    if (lc->arena) { s->base = lc->arena + slot_stride(m) * (size_t)idx; return; }
#else
    (void)idx;
#endif
    (void)li;
    if (posix_memalign((void **)&s->base, 4096, (size_t)m->e_bytes + 8192)) {
        fprintf(stderr, "[mimo] out of memory for an expert slot\n"); exit(1);
    }
}

/* MIMO_CACHE_GLOBAL: an entry of layer li for a new expert, with a buffer -- a fresh
 * one while the pool has room, else the oldest of any layer's (its expert leaves that
 * layer). -1 when nothing can give one. */
static int slot_entry_global(Model *m, int li) {
    LCache *lc = &m->cache[li];
    int idx = -1;
    for (int k = 0; k < lc->n && idx < 0; k++)
        if (lc->s[k].eid < 0 && !lc->s[k].pin &&
            atomic_load_explicit(&lc->s[k].state, memory_order_acquire) != SLOT_LOADING) idx = k;
    if (idx < 0 && lc->n < lc->max) idx = lc->n++;
    if (idx >= 0 && lc->s[idx].base) return idx;
    if (m->bulk && lc->nbuf >= lc->cap) return -1;   /* a prompt block: the layer's own LRU */
    if (idx >= 0 && g_slots_used < g_slots_total) {
        if (posix_memalign((void **)&lc->s[idx].base, 4096, (size_t)m->e_bytes + 8192)) {
            fprintf(stderr, "[mimo] out of memory for an expert slot\n"); exit(1);
        }
        g_slots_used++; lc->nbuf++;
        return idx;
    }
    if (idx < 0) return -1;                /* the layer holds its most: its own LRU (slot_take) */
    Slot *v = NULL; int vl = -1, vk = -1;
    uint64_t oldest = UINT64_MAX;
    for (int l2 = 0; l2 < m->c.n_layers; l2++) {
        LCache *c2 = &m->cache[l2];
        if (!c2->s) continue;
        for (int k = 0; k < c2->n; k++) {
            Slot *x = &c2->s[k];
            if (!x->base || x->eid < 0 || x->pin || x->pref ||
                atomic_load_explicit(&x->state, memory_order_acquire) == SLOT_LOADING) continue;
            if (x->used < oldest) { oldest = x->used; v = x; vl = l2; vk = k; }
        }
    }
    if (!v) return -1;
    m->cache[vl].by_expert[v->eid] = -1;
    v->eid = -1; v->used = 0;
    if (vl == li) return vk;               /* this layer's own: reused in place */
    lc->s[idx].base = v->base; v->base = NULL; v->buf = NULL;
    m->cache[vl].nbuf--; lc->nbuf++;
    return idx;
}

/* A slot for expert e of layer li: the one that holds it (resident or on its way), or
 * the least recently used (MIMO_CACHE_DECAY: the least used) that is neither pinned
 * nor loading, its read submitted -- before the others, one whose expert the Vulkan
 * tier holds (vkt_ram_first). A prefetch spares the slots read ahead and not asked
 * for yet. MIMO_CACHE_GLOBAL: a buffer from the pool first (slot_entry_global).
 * NULL when no slot is free. */
static Slot *slot_take(Model *m, int li, int e, int prefetch) {
    LCache *lc = &m->cache[li];
    int si = lc->by_expert[e];
    if (si >= 0) return &lc->s[si];
    int best = -1;
    if (g_cache_global && (best = slot_entry_global(m, li)) >= 0) { /* an entry with a buffer */ }
    else if (lc->n < lc->cap && !g_cache_global) best = lc->n++;
    else {
        uint64_t oldest = UINT64_MAX, dev_oldest = UINT64_MAX;
        float least = INFINITY;
        int dev = -1;
        for (int k = 0; k < lc->n; k++) {
            Slot *s = &lc->s[k];
            if (s->pin || (prefetch && s->pref) || atomic_load_explicit(&s->state, memory_order_acquire) == SLOT_LOADING) continue;
            if (g_cache_global && !s->base) continue;   /* an entry whose buffer went to another layer */
            if (s->eid >= 0 && vkt_ram_first(li, s->eid)) {
                if (s->used < dev_oldest) { dev_oldest = s->used; dev = k; }
            } else if (m->efreq && s->eid >= 0) {
                float f = expert_freq(m, li, s->eid);
                if (f < least || (f == least && s->used < oldest)) { least = f; oldest = s->used; best = k; }
            } else if (s->used < oldest) { oldest = s->used; best = k; least = -INFINITY; }
        }
        if (dev >= 0) { best = dev; vkt_ram_gave(); }
    }
    if (best < 0) return NULL;
    Slot *s = &lc->s[best];
    if (s->eid >= 0) {
        lc->by_expert[s->eid] = -1;
        if (s->pref) m->pf_wasted++;
    }
    if (!s->base) slot_alloc(m, li, s, best);
    s->eid = e; lc->by_expert[e] = best;
    s->pin = 0; s->pref = prefetch; s->used = 0;
    atomic_store_explicit(&s->state, SLOT_LOADING, memory_order_release);
    m->bytes_read += (uint64_t)m->e_bytes;
    if (prefetch) m->pf_issued++; else m->miss++;
    io_submit(m, s, li, e, prefetch);
    return s;
}

/* The step's experts uni[0..nu) of layer li: those the cache holds (resident or on
 * their way) are pinned, the prefetcher's leftovers of this layer become the first to
 * go, and the others take free slots while there are any, their reads submitted.
 * sl[j] stays NULL for an expert that found no slot yet. Returns the step's stamp. */
static uint64_t slots_demand(Model *m, int li, const int *uni, int nu, Slot **sl) {
    LCache *lc = &m->cache[li];
    uint64_t stamp = ++m->clock;
    expert_freq_step(m, li, uni, nu);
    for (int j = 0; j < nu; j++) {
        int si = lc->by_expert[uni[j]];
        sl[j] = NULL;
        if (si < 0) continue;
        Slot *s = sl[j] = &lc->s[si];
        s->pin++; s->used = stamp; m->hits++;
        if (s->pref) { s->pref = 0; m->pf_used++; }
        if (atomic_load_explicit(&s->state, memory_order_acquire) == SLOT_LOADING) io_promote(s);
    }
    for (int k = 0; k < lc->n; k++)
        if (lc->s[k].pref) { lc->s[k].pref = 0; lc->s[k].used = 0; m->pf_wasted++; }
    for (int j = 0; j < nu; j++) {
        if (sl[j]) continue;
        Slot *s = slot_take(m, li, uni[j], 0);
        if (!s) break;
        s->pin++; s->used = stamp;
        sl[j] = s;
    }
    return stamp;
}

/* Make the experts `want[0..n)` of layer li resident, n <= cap, and return their
 * slots in `out` (the Vulkan tier's loads): every read submitted at once, then waited. */
static void experts_ensure(Model *m, int li, const int *want, int n, Slot **out) {
    uint64_t stamp = slots_demand(m, li, want, n, out);
    for (int j = 0; j < n; j++)
        while (!out[j]) {
            unsigned long long seen = io_seen();
            out[j] = slot_take(m, li, want[j], 0);
            if (out[j]) { out[j]->pin++; out[j]->used = stamp; }
            else io_wait_any(m, seen);
        }
    for (int j = 0; j < n; j++) io_wait_slot(m, out[j]);
    for (int j = 0; j < n; j++) out[j]->pin--;
}

static void kv_alloc(Model *m, int ctx) {
    Cfg *c = &m->c;
    m->ctx = ctx;
    /* With n MTP drafts a step, a window's ring keeps n - 1 more positions than the window:
     * the rows a step drops have overwritten the slots of the positions a ring earlier, of
     * which the next step would still read n - 1 (with one draft, none) -- the trunk's, and
     * the rows a block runs again. */
    int extra = g_mtp.n > 1 ? g_mtp.n - 1 : 0;
    for (int li = 0; li < c->n_layers + g_mtp.n; li++) {   /* the MTP blocks' rings too */
        Layer *l = &m->L[li];
        int k = c->swa[li];
        l->rows = k ? (c->window + extra < ctx ? c->window + extra : ctx) : ctx;
        free(l->K); free(l->V); free(l->ring_pos);
        l->K = xmalloc((size_t)l->rows * c->kv_heads[k] * c->head_dim[k] * sizeof(float), "K cache");
        l->V = xmalloc((size_t)l->rows * c->kv_heads[k] * c->v_dim[k] * sizeof(float), "V cache");
        l->ring_pos = xmalloc((size_t)l->rows * sizeof(int), "ring positions");
        for (int r = 0; r < l->rows; r++) l->ring_pos[r] = -1;
    }
    kv_prefix_alloc(&m->kvp, ctx);
}

static void model_reset(Model *m) {
    Cfg *c = &m->c;
    for (int li = 0; li < c->n_layers + g_mtp.n; li++)
        for (int r = 0; r < m->L[li].rows; r++) m->L[li].ring_pos[r] = -1;
    m->pos = 0;
    kv_prefix_clear(&m->kvp);
}

/* MIMO_MTP (default 2; 0 in a VK=1 build, whose dense chain keeps the trunk's hidden
 * states on the device): the drafts a decode step makes, one from each of the
 * checkpoint's first MIMO_MTP MTP layers (up to MIMO_MTP_MAX) that are in the layout this
 * reads (Pro's: fused qkv, a dense MLP of the trunk's width), sliding-window blocks at
 * m->L[n_layers + k]. MIMO_MTP_MIN_P (default 0.6): the step verifies them while the
 * probabilities the blocks give them multiply to at least this, 0 every one. On the RTX
 * 4090 profile two drafts at 0.6 were the fastest setting measured, on the benchmark and
 * on eight different prompts, greedy and sampled (docs/mimo.md). */
#ifdef COLI_VULKAN
#define MIMO_MTP_DEFAULT 0
#else
#define MIMO_MTP_DEFAULT 2
#endif
#define MIMO_MTP_MIN_P_DEFAULT 0.6f
static int mtp_tensor(shards *S, const char *name, int O, int I) {
    st_tensor *t = st_find(S, name);
    return t && (I > 0 ? t->rank == 2 && t->shape[0] == O && t->shape[1] == I : t->rank == 1 && t->shape[0] == O);
}
static int mtp_layer_ok(Model *m, int k) {   /* model.mtp.layers.k, in that layout */
    Cfg *c = &m->c;
    int H = c->hidden, F = c->dense_inter;
    int qkv_o = c->heads[1] * c->head_dim[1] + c->kv_heads[1] * (c->head_dim[1] + c->v_dim[1]);
    char nm[512];
#define MTP_HAS(t, O, I) (snprintf(nm, sizeof nm, "model.mtp.layers.%d.%s", k, t), mtp_tensor(&m->S, nm, O, I))
#define MTP_FOUND(t) (snprintf(nm, sizeof nm, "model.mtp.layers.%d.%s", k, t), st_find(&m->S, nm) != NULL)
    int ok = MTP_HAS("eh_proj.weight", H, 2 * H) &&
             MTP_HAS("self_attn.qkv_proj.weight", qkv_o, H) && MTP_FOUND("self_attn.qkv_proj.weight_scale_inv") &&
             MTP_HAS("self_attn.o_proj.weight", H, c->heads[1] * c->v_dim[1]) &&
             MTP_HAS("self_attn.attention_sink_bias", c->heads[1], 0) &&
             MTP_HAS("mlp.gate_proj.weight", F, H) && MTP_HAS("mlp.up_proj.weight", F, H) &&
             MTP_HAS("mlp.down_proj.weight", H, F) && MTP_FOUND("mlp.gate_proj.weight_scale_inv") &&
             MTP_FOUND("mlp.up_proj.weight_scale_inv") && MTP_FOUND("mlp.down_proj.weight_scale_inv") &&
             MTP_HAS("enorm.weight", H, 0) && MTP_HAS("hnorm.weight", H, 0) && MTP_HAS("final_layernorm.weight", H, 0) &&
             MTP_HAS("input_layernorm.weight", H, 0) && MTP_HAS("pre_mlp_layernorm.weight", H, 0);
#undef MTP_HAS
#undef MTP_FOUND
    return ok;
}
static void mtp_load(Model *m) {
    Cfg *c = &m->c;
    int want = env_int("MIMO_MTP", MIMO_MTP_DEFAULT), explicit = getenv("MIMO_MTP") != NULL;
    if (want <= 0) return;
    int H = c->hidden, n = 0;
    const char *why = NULL;
    while (n < want && n < MIMO_MTP_MAX && mtp_layer_ok(m, n)) n++;
    if (!st_find(&m->S, "model.mtp.layers.0.eh_proj.weight")) why = "the checkpoint has no MTP layer";
    else if (!n) why = "its MTP layer is not in the layout this engine reads";
    else if (c->n_layers + n > MIMO_MAX_LAYERS) why = "no room for its layers";
    else if (g_mimo_mux_slots > 1) why = "KV_SLOTS > 1 decodes several conversations a step";
    else if (!c->window || !c->sink[1]) why = "the checkpoint has no sliding window with sinks";
    if (why) { if (explicit) fprintf(stderr, "[mimo] MIMO_MTP off: %s\n", why); return; }
    if (n < want && explicit)
        fprintf(stderr, "[mimo] MIMO_MTP=%d: %d draft%s a step, the MTP layers in the layout this engine reads\n",
                want, n, n == 1 ? "" : "s");
    g_mtp.li = c->n_layers;
    g_mtp.keep_rows = c->window + 2;
    for (int k = 0; k < n; k++) {
        int li = g_mtp.li + k;
        c->swa[li] = 1; c->moe[li] = 0;
        Layer *l = &m->L[li];
        MtpBlock *b = &g_mtp.b[k];
        char nm[512];
#define MTP_NAME(t) (snprintf(nm, sizeof nm, "model.mtp.layers.%d.%s", k, t), nm)
        l->ln1 = load_vec(&m->S, MTP_NAME("input_layernorm.weight"), H);
        l->ln2 = load_vec(&m->S, MTP_NAME("pre_mlp_layernorm.weight"), H);
        load_qkv_named(&l->qkv, &m->S, c, 1, MTP_NAME("self_attn.qkv_proj.weight"));
        dw_load_bf16(&l->o, &m->S, MTP_NAME("self_attn.o_proj.weight"), H, c->heads[1] * c->v_dim[1]);
        l->sink = load_vec(&m->S, MTP_NAME("self_attn.attention_sink_bias"), c->heads[1]);
        dw_load_fp8(&l->gate, &m->S, MTP_NAME("mlp.gate_proj.weight"), c->dense_inter, H, NULL, 0, NULL);
        dw_load_fp8(&l->up, &m->S, MTP_NAME("mlp.up_proj.weight"), c->dense_inter, H, NULL, 0, NULL);
        dw_load_fp8(&l->down, &m->S, MTP_NAME("mlp.down_proj.weight"), H, c->dense_inter, NULL, 0, NULL);
        dw_load_bf16(&b->eh, &m->S, MTP_NAME("eh_proj.weight"), H, 2 * H);
        b->enorm = load_vec(&m->S, MTP_NAME("enorm.weight"), H);
        b->hnorm = load_vec(&m->S, MTP_NAME("hnorm.weight"), H);
        b->fnorm = load_vec(&m->S, MTP_NAME("final_layernorm.weight"), H);
#undef MTP_NAME
    }
    g_mtp.n = n; g_mtp.on = 1;
    g_mtp.keep = xmalloc((size_t)g_mtp.keep_rows * H * sizeof(float), "mtp hidden rows");
    const char *mp = getenv("MIMO_MTP_MIN_P"), *tr = getenv("MIMO_MTP_TRACE");
    g_mtp.min_p = mp && *mp ? (float)atof(mp) : MIMO_MTP_MIN_P_DEFAULT;
    if (!(g_mtp.min_p >= 0.0f && g_mtp.min_p <= 1.0f)) { fprintf(stderr, "[mimo] MIMO_MTP_MIN_P must be between 0 and 1\n"); exit(2); }
    if (tr && *tr && !(g_mtp.trace = fopen(tr, "a"))) { fprintf(stderr, "[mimo] cannot write %s\n", tr); exit(1); }
    if (n == 1) fprintf(stderr, "[mimo] MTP: decode steps verify a draft of the checkpoint's first MTP layer");
    else fprintf(stderr, "[mimo] MTP: decode steps verify up to %d drafts, from the checkpoint's first %d MTP layers", n, n);
    if (g_mtp.min_p > 0 && n == 1) fprintf(stderr, " when the layer gives it %g or more", g_mtp.min_p);
    else if (g_mtp.min_p > 0) fprintf(stderr, ", while the probabilities the layers give them multiply to %g or more", g_mtp.min_p);
    fprintf(stderr, " (MIMO_MTP=0: off)\n");
}

static void model_load(Model *m, const char *dir, int cap) {
    Cfg *c = &m->c;
    load_cfg(c, dir);
    st_init_multi(&m->S, dir, getenv("MIMO_DIRS"));
    g_dense_bits = env_int("MIMO_DENSE_BITS", 0);
    if (g_dense_bits != 0 && g_dense_bits != 8 && g_dense_bits != 32) {
        fprintf(stderr, "[mimo] MIMO_DENSE_BITS must be 0 (native FP8/BF16), 8 or 32\n"); exit(1);
    }
    m->idot = env_int("MIMO_IDOT", 0);
    m->direct = env_int("MIMO_DIRECT", 1);
    int H = c->hidden;
    need(&m->S, "model.embed_tokens.weight", 0, c->vocab, H);
    m->embed = xmalloc((size_t)c->vocab * H * sizeof(uint16_t), "embeddings");
    st_read_raw_cap(&m->S, "model.embed_tokens.weight", m->embed, (int64_t)c->vocab * H * 2, 1);
    m->norm = load_vec(&m->S, "model.norm.weight", H);
    dw_load_bf16(&m->head, &m->S, "lm_head.weight", c->vocab, H);
    for (int li = 0; li < c->n_layers; li++) {
        Layer *l = &m->L[li];
        int k = c->swa[li];
        char nm[512];
        snprintf(nm, sizeof(nm), "model.layers.%d.input_layernorm.weight", li);
        l->ln1 = load_vec(&m->S, nm, H);
        snprintf(nm, sizeof(nm), "model.layers.%d.post_attention_layernorm.weight", li);
        l->ln2 = load_vec(&m->S, nm, H);
        load_qkv(&l->qkv, &m->S, c, li);
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.o_proj.weight", li);
        dw_load_bf16(&l->o, &m->S, nm, H, c->heads[k] * c->v_dim[k]);
        if (c->sink[k]) {
            snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.attention_sink_bias", li);
            l->sink = load_vec(&m->S, nm, c->heads[k]);
        }
        if (!c->moe[li]) {
            snprintf(nm, sizeof(nm), "model.layers.%d.mlp.gate_proj.weight", li);
            dw_load_fp8(&l->gate, &m->S, nm, c->dense_inter, H, NULL, 0, NULL);
            snprintf(nm, sizeof(nm), "model.layers.%d.mlp.up_proj.weight", li);
            dw_load_fp8(&l->up, &m->S, nm, c->dense_inter, H, NULL, 0, NULL);
            snprintf(nm, sizeof(nm), "model.layers.%d.mlp.down_proj.weight", li);
            dw_load_fp8(&l->down, &m->S, nm, H, c->dense_inter, NULL, 0, NULL);
        } else {
            snprintf(nm, sizeof(nm), "model.layers.%d.mlp.gate.weight", li);
            st_tensor *rt = need(&m->S, nm, 0, c->n_experts, H);
            l->router = xmalloc((size_t)rt->nbytes, nm);
            st_read_raw_cap(&m->S, nm, l->router, rt->nbytes, 1);
            snprintf(nm, sizeof(nm), "model.layers.%d.mlp.gate.e_score_correction_bias", li);
            l->bias = load_vec(&m->S, nm, c->n_experts);
        }
    }
    mtp_load(m);
    expert_table_init(m);
    g_cache_global = env_int("MIMO_CACHE_GLOBAL", MIMO_CACHE_GLOBAL_DEFAULT) != 0;
    if (g_cache_global && env_int("MIMO_LARGE_PAGES", 0))
        fprintf(stderr, "[mimo] MIMO_CACHE_GLOBAL moves buffers between layers: ordinary pages\n");
    cache_init(m, cap);
    m->pf_on = env_int("MIMO_PREFETCH", 1);
    /* MIMO_PREFETCH=1 predicts top-k after the attention, 2 also after the MLP;
     * MIMO_PREFETCH_ATTN / MIMO_PREFETCH_MLP set either point's size (0: not there) */
    m->pf_ka = env_int("MIMO_PREFETCH_ATTN", m->pf_on ? c->topk : 0);
    m->pf_kb = env_int("MIMO_PREFETCH_MLP", m->pf_on >= 2 ? c->topk : 0);
    if (!m->pf_on) m->pf_ka = m->pf_kb = 0;
    if (m->pf_ka < 0) m->pf_ka = 0;
    if (m->pf_kb < 0) m->pf_kb = 0;
    m->pf_on = m->pf_ka > 0 || m->pf_kb > 0;
    m->pf_rows = env_int("MIMO_PREFETCH_ROWS", 4);
    m->pf_extra = env_int("MIMO_PREFETCH_EXTRA", 0);
    m->pf_limit = env_int("MIMO_PREFETCH_LIMIT", 64);
    if (m->pf_extra < 0) m->pf_extra = 0;
    const char *dv = getenv("MIMO_CACHE_DECAY");
    m->decay = dv && *dv ? (float)atof(dv) : 0.0f;
    if (g_cache_global) fprintf(stderr, "[mimo] expert cache: one pool of %d slots for every layer (MIMO_CACHE_GLOBAL)\n", g_slots_total);
    if (m->decay > 0 && m->decay < 1) {
        m->efreq = xcalloc((size_t)c->n_layers * c->n_experts, sizeof(float), "expert counts");
        m->etime = xcalloc((size_t)c->n_layers * c->n_experts, sizeof(uint32_t), "expert count times");
        m->dpow = xmalloc(MIMO_DPOW * sizeof(float), "decay powers");
        m->dpow[0] = 1.0f;
        for (int k = 1; k < MIMO_DPOW; k++) m->dpow[k] = m->dpow[k - 1] * m->decay;
    }
    m->pf_mark = xcalloc((size_t)c->n_layers * c->n_experts, 1, "prefetch marks");
    m->pf_made = xcalloc((size_t)c->n_layers, 1, "prefetch made");
    io_start(m);
    const char *rt = getenv("MIMO_ROUTE_TRACE");
    if (rt && *rt) {
        g_rtrace = fopen(rt, "wb");
        if (!g_rtrace) { fprintf(stderr, "[mimo] cannot write %s\n", rt); exit(1); }
        int32_t head[5] = {0x3154524d /* "MRT1" */, c->n_layers, c->n_experts, c->topk, MIMO_TRACE_RANK};
        fwrite(head, sizeof head, 1, g_rtrace);
        atexit(mimo_trace_close);
    }
    /* coli --ctx arrives as CTX (the family's context_env); MIMO_CTX wins over it */
    const char *ctx_text = getenv("MIMO_CTX");
    if (!ctx_text || !*ctx_text) ctx_text = getenv("CTX");
    int ctx = (ctx_text && *ctx_text) ? atoi(ctx_text) : 8192;
    if (ctx > c->max_positions) ctx = c->max_positions;
    if (ctx < 16) ctx = 16;
    kv_alloc(m, ctx);
    m->ehit = xmalloc((size_t)c->n_layers * sizeof(uint8_t *), "hits");
    for (int li = 0; li < c->n_layers; li++) m->ehit[li] = xcalloc((size_t)c->n_experts, 1, "hits row");
}

/* ------------------------------------------------------------ the forward ---- */

static void rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float r = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) out[i] = w[i] * (x[i] * r);
}

/* rotate-half RoPE on the first `rd` dims of a head, as the reference builds it:
 * inv_freq and the angle in float32, cos/sin of the float angle. The angles depend
 * on the position only, so a row's cosines and sines are made once (rope_table) for
 * all of its heads, the values one head at a time would get. */
#define MIMO_MAX_ROPE 1024
static void rope_table(float *cs, float *sn, int rd, float theta, int pos) {
    int half = rd / 2;
    for (int i = 0; i < half; i++) {
        float inv = 1.0f / powf(theta, (float)(2 * i) / (float)rd);
        float ang = inv * (float)pos;
        cs[i] = cosf(ang); sn[i] = sinf(ang);
    }
}
static void rope_apply(float *v, int rd, const float *cs, const float *sn) {
    int half = rd / 2;
    for (int i = 0; i < half; i++) {
        float a = v[i], b = v[i + half];
        v[i] = a * cs[i] - b * sn[i];
        v[i + half] = b * cs[i] + a * sn[i];
    }
}
/* a row of qkv at position pos: RoPE on every q and k head, values scaled */
static void rope_row(const Cfg *c, int k, float *q, int pos) {
    int nh = c->heads[k], kvh = c->kv_heads[k], hd = c->head_dim[k], rd = c->rope_dim[k];
    float cs[MIMO_MAX_ROPE / 2], sn[MIMO_MAX_ROPE / 2];
    rope_table(cs, sn, rd, c->theta[k], pos);
    float *kk = q + nh * hd, *vv = kk + kvh * hd;
    for (int h = 0; h < nh; h++) rope_apply(q + h * hd, rd, cs, sn);
    for (int h = 0; h < kvh; h++) rope_apply(kk + h * hd, rd, cs, sn);
    for (int i = 0; i < kvh * c->v_dim[k]; i++) vv[i] *= c->v_scale;
}

#if defined(__AVX2__)
/* A product the compiler may not fuse into the add that follows: the scores' sum
 * below is the one gcc makes of `s += q[i] * k[i]` -- products rounded, then
 * added one after the other (an in-order reduction, never an fma) -- and only an
 * opaque product keeps -ffp-contract from changing that. */
static inline __m256 mul_apart(__m256 a, __m256 b) {
    __m256 r = _mm256_mul_ps(a, b);
    __asm__("" : "+x"(r));
    return r;
}
/* The scores of one row's G heads that share key/value head kh, eight heads a
 * vector: qT [hd][lanes] holds the heads' queries by dimension (lanes = G rounded up
 * to 8, the spare lanes zero), sc [lanes][..] receives score j - first of head g.
 * Each head's sum is the scalar loop's: products rounded, added in dimension order
 * from zero, times the scale -- the same bits, eight heads and two keys at a time. */
static void scores_by_head(const float *qT, int lanes, int hd, const float *const *Kr, int lo, int first,
                           int last, size_t koff, float scale, float *sc, int stride) {
    const __m256 vs = _mm256_set1_ps(scale);
    for (int g0 = 0; g0 < lanes; g0 += 8) {
        int j = first;
        for (; j + 1 <= last; j += 2) {
            const float *k0 = Kr[j - lo] + koff, *k1 = Kr[j + 1 - lo] + koff;
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            for (int i = 0; i < hd; i++) {
                __m256 q = _mm256_loadu_ps(qT + (size_t)i * lanes + g0);
                a0 = _mm256_add_ps(a0, mul_apart(q, _mm256_set1_ps(k0[i])));
                a1 = _mm256_add_ps(a1, mul_apart(q, _mm256_set1_ps(k1[i])));
            }
            float t0[8], t1[8];
            _mm256_storeu_ps(t0, _mm256_mul_ps(a0, vs));
            _mm256_storeu_ps(t1, _mm256_mul_ps(a1, vs));
            for (int g = 0; g < 8; g++) {
                sc[(size_t)(g0 + g) * stride + (j - first)] = t0[g];
                sc[(size_t)(g0 + g) * stride + (j + 1 - first)] = t1[g];
            }
        }
        for (; j <= last; j++) {
            const float *k0 = Kr[j - lo] + koff;
            __m256 a0 = _mm256_setzero_ps();
            for (int i = 0; i < hd; i++)
                a0 = _mm256_add_ps(a0, mul_apart(_mm256_loadu_ps(qT + (size_t)i * lanes + g0), _mm256_set1_ps(k0[i])));
            float t0[8];
            _mm256_storeu_ps(t0, _mm256_mul_ps(a0, vs));
            for (int g = 0; g < 8; g++) sc[(size_t)(g0 + g) * stride + (j - first)] = t0[g];
        }
    }
}
#endif

/* Attention of `n` rows at positions p0.. of layer li. `xn` is the normed input
 * [n, H], `out` receives o_proj's output [n, H]. */
/* A multiplexed decode step's attention (KV_SLOTS, m->mux_rows): row t is the next
 * token of its own conversation, at mux_rows[t].pos, over that conversation's cache --
 * a full layer's rows, or a sliding window's ring. Each row's numbers are attention()'s
 * for a block of that one token; the projections run once over the n rows. */
static void attention_rows(Model *m, int li, const float *xn, int n, float *out) {
    Cfg *c = &m->c;
    Layer *l = &m->L[li];
    const MimoRow *mr = m->mux_rows;
    int k = c->swa[li];
    int nh = c->heads[k], kvh = c->kv_heads[k], hd = c->head_dim[k], vd = c->v_dim[k];
    int qd = nh * hd, kd = kvh * hd, vdd = kvh * vd, rows = qd + kd + vdd;
    double t0 = now_s();
    float *qkv = xmalloc((size_t)n * rows * sizeof(float), "qkv");
    double tq = now_s();
    dw_matmul(qkv, xn, n, &l->qkv);
    m->t_qkv += now_s() - tq;
    for (int t = 0; t < n; t++) rope_row(c, k, qkv + (size_t)t * rows, mr[t].pos);
    int window = k ? c->window : 0, most = 1;
    const float **Kc = xmalloc((size_t)n * sizeof *Kc, "row keys"), **Vc = xmalloc((size_t)n * sizeof *Vc, "row values");
    int *lo = xmalloc((size_t)n * sizeof(int), "row first keys");
    for (int t = 0; t < n; t++) {
        MimoSeq *q = mr[t].seq;
        int p = mr[t].pos;
        const float *src = qkv + (size_t)t * rows + qd;
        if (k) {   /* the window's keys before p from the ring, then this token's */
            int l0 = p - (window - 1) > 0 ? p - (window - 1) : 0;
            float *kc = xmalloc((size_t)(p + 1 - l0) * kd * sizeof(float), "window keys");
            float *vc = xmalloc((size_t)(p + 1 - l0) * vdd * sizeof(float), "window values");
            for (int pp = l0; pp < p; pp++) {
                int slot = pp % l->rows;
                if (q->ring_pos[li][slot] != pp) { fprintf(stderr, "[mimo] window ring lost position %d\n", pp); exit(1); }
                memcpy(kc + (size_t)(pp - l0) * kd, q->K[li] + (size_t)slot * kd, (size_t)kd * sizeof(float));
                memcpy(vc + (size_t)(pp - l0) * vdd, q->V[li] + (size_t)slot * vdd, (size_t)vdd * sizeof(float));
            }
            memcpy(kc + (size_t)(p - l0) * kd, src, (size_t)kd * sizeof(float));
            memcpy(vc + (size_t)(p - l0) * vdd, src + kd, (size_t)vdd * sizeof(float));
            Kc[t] = kc; Vc[t] = vc; lo[t] = l0;
        } else {   /* the full cache, this token written in first */
            memcpy(q->K[li] + (size_t)p * kd, src, (size_t)kd * sizeof(float));
            memcpy(q->V[li] + (size_t)p * vdd, src + kd, (size_t)vdd * sizeof(float));
            Kc[t] = q->K[li]; Vc[t] = q->V[li]; lo[t] = 0;
        }
        if (p + 1 - lo[t] > most) most = p + 1 - lo[t];
    }
    float *ctxv = xmalloc((size_t)n * nh * vd * sizeof(float), "attention output");
    float scale = 1.0f / sqrtf((float)hd);
    int group = nh / kvh;
    #pragma omp parallel
    {
        float *score = xmalloc((size_t)most * sizeof(float), "scores");
        #pragma omp for collapse(2) schedule(static)
        for (int t = 0; t < n; t++) {
            for (int h = 0; h < nh; h++) {
                int p = mr[t].pos, kh = h / group, first = lo[t];
                const float *q = qkv + (size_t)t * rows + (size_t)h * hd;
                float best = -INFINITY;
                for (int j = first; j <= p; j++) {
                    const float *kr = Kc[t] + (size_t)(j - lo[t]) * kd + (size_t)kh * hd;
                    float s = 0;
                    for (int i = 0; i < hd; i++) s += q[i] * kr[i];
                    s *= scale;
                    score[j - first] = s;
                    if (s > best) best = s;
                }
                float sink = l->sink ? l->sink[h] : -INFINITY;
                if (sink > best) best = sink;
                float den = l->sink ? expf(sink - best) : 0.0f;
                for (int j = first; j <= p; j++) { score[j - first] = expf(score[j - first] - best); den += score[j - first]; }
                float *o = ctxv + ((size_t)t * nh + h) * vd;
                for (int i = 0; i < vd; i++) o[i] = 0;
                for (int j = first; j <= p; j++) {
                    const float *vr = Vc[t] + (size_t)(j - lo[t]) * vdd + (size_t)kh * vd;
                    float w = score[j - first] / den;
                    for (int i = 0; i < vd; i++) o[i] += w * vr[i];
                }
            }
        }
        free(score);
    }
    if (k)   /* the ring keeps each conversation's last `window` positions */
        for (int t = 0; t < n; t++) {
            MimoSeq *q = mr[t].seq;
            int p = mr[t].pos, slot = p % l->rows;
            const float *src = qkv + (size_t)t * rows + qd;
            memcpy(q->K[li] + (size_t)slot * kd, src, (size_t)kd * sizeof(float));
            memcpy(q->V[li] + (size_t)slot * vdd, src + kd, (size_t)vdd * sizeof(float));
            q->ring_pos[li][slot] = p;
            free((void *)Kc[t]); free((void *)Vc[t]);
        }
    double to = now_s();
    dw_matmul(out, ctxv, n, &l->o);
    m->t_oproj += now_s() - to;
    free(ctxv); free(qkv); free(Kc); free(Vc); free(lo);
    m->t_attn += now_s() - t0;
}

static void attention(Model *m, int li, const float *xn, int n, int p0, float *out) {
    if (m->mux_rows) { attention_rows(m, li, xn, n, out); return; }
    Cfg *c = &m->c;
    Layer *l = &m->L[li];
    int k = c->swa[li];
    int nh = c->heads[k], kvh = c->kv_heads[k], hd = c->head_dim[k], vd = c->v_dim[k];
    int qd = nh * hd, kd = kvh * hd, vdd = kvh * vd, rows = qd + kd + vdd;
    double t0 = now_s();
    float *qkv = xmalloc((size_t)n * rows * sizeof(float), "qkv");
    double tq = now_s();
    dw_matmul(qkv, xn, n, &l->qkv);
    m->t_qkv += now_s() - tq;
    for (int t = 0; t < n; t++) rope_row(c, k, qkv + (size_t)t * rows, p0 + t);
    /* Keys this block can see: [lo, p0 + n). For a sliding window the ring still
     * holds the ones before p0; for full attention the cache holds everything,
     * and the block is written in first. */
    int window = k ? c->window : 0;
    int lo = k ? (p0 - (window - 1) > 0 ? p0 - (window - 1) : 0) : 0;
    if (k && lo < l->kv_lo) lo = l->kv_lo;
    int nkeys = p0 + n - lo;
    /* key and value rows lo.. read where they are: a window's ring for the positions
     * before the block and the block's own rows of qkv (the ring is rewritten after
     * the attention), or the full cache, the block written in first */
    const float **Kr = xmalloc((size_t)nkeys * sizeof *Kr, "key rows"), **Vr = xmalloc((size_t)nkeys * sizeof *Vr, "value rows");
    if (k) {
        for (int p = lo; p < p0; p++) {
            int slot = p % l->rows;
            if (l->ring_pos[slot] != p) { fprintf(stderr, "[mimo] window ring lost position %d\n", p); exit(1); }
            Kr[p - lo] = l->K + (size_t)slot * kd;
            Vr[p - lo] = l->V + (size_t)slot * vdd;
        }
        for (int t = 0; t < n; t++) {
            const float *src = qkv + (size_t)t * rows + qd;
            Kr[p0 + t - lo] = src;
            Vr[p0 + t - lo] = src + kd;
        }
    } else {
        for (int t = 0; t < n; t++) {
            const float *src = qkv + (size_t)t * rows + qd;
            memcpy(l->K + (size_t)(p0 + t) * kd, src, (size_t)kd * sizeof(float));
            memcpy(l->V + (size_t)(p0 + t) * vdd, src + kd, (size_t)vdd * sizeof(float));
        }
        for (int j = 0; j < nkeys; j++) { Kr[j] = l->K + (size_t)j * kd; Vr[j] = l->V + (size_t)j * vdd; }
    }
    float *ctxv = xmalloc((size_t)n * nh * vd * sizeof(float), "attention output");
    float scale = 1.0f / sqrtf((float)hd);
    int group = nh / kvh;
    int vec = 0;
#if defined(__AVX2__)
    vec = hd % 16 == 0;
    if (vec) {
        /* The heads of a key/value head at once (scores_by_head); the softmax and the
         * values per head as below. The sums gcc vectorizes in blocks of 16 are the ones
         * this path reproduces, hence hd % 16 (MiMo's 192, the tiny fixture's 48). */
        int lanes = (group + 7) / 8 * 8;
        #pragma omp parallel
        {
            float *qT = xmalloc((size_t)hd * lanes * sizeof(float), "queries by head");
            float *sc = xmalloc((size_t)lanes * nkeys * sizeof(float), "scores by head");
            #pragma omp for collapse(2) schedule(static)
            for (int t = 0; t < n; t++) {
                for (int kh = 0; kh < kvh; kh++) {
                    int p = p0 + t;
                    int first = window ? (p - (window - 1) > lo ? p - (window - 1) : lo) : 0;
                    for (int g = 0; g < lanes; g++)
                        for (int i = 0; i < hd; i++)
                            qT[(size_t)i * lanes + g] = g < group ? qkv[(size_t)t * rows + (size_t)(kh * group + g) * hd + i] : 0.0f;
                    scores_by_head(qT, lanes, hd, Kr, lo, first, p, (size_t)kh * hd, scale, sc, nkeys);
                    for (int g = 0; g < group; g++) {
                        int h = kh * group + g;
                        float *score = sc + (size_t)g * nkeys;
                        float best = -INFINITY;
                        for (int j = first; j <= p; j++) if (score[j - first] > best) best = score[j - first];
                        float sink = l->sink ? l->sink[h] : -INFINITY;
                        if (sink > best) best = sink;
                        float den = l->sink ? expf(sink - best) : 0.0f;
                        for (int j = first; j <= p; j++) { score[j - first] = expf(score[j - first] - best); den += score[j - first]; }
                        float *o = ctxv + ((size_t)t * nh + h) * vd;
                        for (int i = 0; i < vd; i++) o[i] = 0;
                        for (int j = first; j <= p; j++) {
                            const float *vr = Vr[j - lo] + (size_t)kh * vd;
                            float w = score[j - first] / den;
                            for (int i = 0; i < vd; i++) o[i] += w * vr[i];
                        }
                    }
                }
            }
            free(qT); free(sc);
        }
    }
#endif
    if (!vec) {
    #pragma omp parallel
    {
        float stack_score[1024];
        float *score = nkeys <= 1024 ? stack_score : xmalloc((size_t)nkeys * sizeof(float), "scores");
        #pragma omp for collapse(2) schedule(static)
        for (int t = 0; t < n; t++) {
            for (int h = 0; h < nh; h++) {
                int p = p0 + t, kh = h / group;
                int first = window ? (p - (window - 1) > lo ? p - (window - 1) : lo) : 0;
                const float *q = qkv + (size_t)t * rows + (size_t)h * hd;
                float best = -INFINITY;
                for (int j = first; j <= p; j++) {
                    const float *kr = Kr[j - lo] + (size_t)kh * hd;
                    float s = 0;
                    for (int i = 0; i < hd; i++) s += q[i] * kr[i];
                    s *= scale;
                    score[j - first] = s;
                    if (s > best) best = s;
                }
                float sink = l->sink ? l->sink[h] : -INFINITY;
                if (sink > best) best = sink;
                float den = l->sink ? expf(sink - best) : 0.0f;
                for (int j = first; j <= p; j++) { score[j - first] = expf(score[j - first] - best); den += score[j - first]; }
                float *o = ctxv + ((size_t)t * nh + h) * vd;
                for (int i = 0; i < vd; i++) o[i] = 0;
                for (int j = first; j <= p; j++) {
                    const float *vr = Vr[j - lo] + (size_t)kh * vd;
                    float w = score[j - first] / den;
                    for (int i = 0; i < vd; i++) o[i] += w * vr[i];
                }
            }
        }
        if (score != stack_score) free(score);
    }
    }
    if (k) {
        /* the ring keeps the last `window` positions of the block */
        int from = n > l->rows ? n - l->rows : 0;
        for (int t = from; t < n; t++) {
            int p = p0 + t, slot = p % l->rows;
            const float *src = qkv + (size_t)t * rows + qd;
            memcpy(l->K + (size_t)slot * kd, src, (size_t)kd * sizeof(float));
            memcpy(l->V + (size_t)slot * vdd, src + kd, (size_t)vdd * sizeof(float));
            l->ring_pos[slot] = p;
        }
    }
    free(Kr); free(Vr);
    double to = now_s();
    dw_matmul(out, ctxv, n, &l->o);
    m->t_oproj += now_s() - to;
    free(ctxv); free(qkv);
    m->t_attn += now_s() - t0;
}

static void dense_mlp(Model *m, Layer *l, const float *xn, int n, float *out) {
    int F = l->gate.O;
    float *g = xmalloc((size_t)n * F * sizeof(float), "mlp gate");
    float *u = xmalloc((size_t)n * F * sizeof(float), "mlp up");
    dw_matmul(g, xn, n, &l->gate);
    dw_matmul(u, xn, n, &l->up);
    /* over the workers: each element alone, the same expf, so the split moves no bit */
    #pragma omp parallel for schedule(dynamic, 512)
    for (int64_t i = 0; i < (int64_t)n * F; i++) g[i] = silu(g[i]) * u[i];
    dw_matmul(out, g, n, &l->down);
    free(g); free(u);
    (void)m;
}

/* An expert's bytes as the shared Vulkan tier reads them (buf: down, its
 * scales, gate, its scales, up, its scales; one e8m0 byte per 32 columns). */
static VktExpertSrc mimo_vk_src(const Model *m, const uint8_t *buf) {
    const uint8_t *dq = buf, *ds = dq + m->part[0], *gq = ds + m->part[1], *gs = gq + m->part[2],
                  *uq = gs + m->part[3], *us = uq + m->part[4];
    return (VktExpertSrc){gq, uq, dq, gs, us, ds};
}
/* The tier's streaming (a big prompt chunk's cold experts on the device): the experts'
 * bytes through the layer cache as moe_cpu gets them (experts_ensure: a group at once,
 * read in parallel, up to the cache's capacity), valid until the next ensure. */
static int mimo_vk_load_batch(void *ctx, int li, const int *e, int n, VktExpertSrc *srcs, void **h) {
    Model *m = ctx;
    int cap = m->cache[li].cap;
    if (n > cap) n = cap;
    if (n < 1 || n > 64) return 0;
    Slot *sl[64];
    experts_ensure(m, li, e, n, sl);
    for (int i = 0; i < n; i++) { srcs[i] = mimo_vk_src(m, sl[i]->buf); h[i] = sl[i]; }
    return n;
}
static int mimo_vk_load(void *ctx, int li, int e, VktExpertSrc *src, void **h) {
    return mimo_vk_load_batch(ctx, li, &e, 1, src, h) == 1;
}
static void mimo_vk_release(void *ctx, void *h) { (void)ctx; (void)h; }

static void mimo_prefetch_pending(Model *m);   /* below: the next layer's reads, behind this step's */

/* A worker's scratch for one task's kernel output, kept between tasks. */
static _Thread_local float *t_scratch;
static _Thread_local size_t t_scratch_n;
static float *task_scratch(size_t n) {
    if (n > t_scratch_n) {
        free(t_scratch);
        t_scratch = xmalloc(n * sizeof(float), "expert scratch");
        t_scratch_n = n;
    }
    return t_scratch;
}

/* One wave of moe_cpu: the experts wave[0..nw) (indices into the step's union),
 * resident in sl[], over the pairs [first[j], first[j+1]) of each. A first-pass
 * task computes a block of gate rows and the same block of up rows and applies
 * silu(gate) * up to them at once, in every worker; the second pass is down. */
static void moe_wave(Model *m, const int *wave, int nw, Slot *const *sl, const int *first,
                     const float *xg, float *g, float *y, int base) {
    Cfg *c = &m->c;
    int H = c->hidden, MI = c->moe_inter;
    void (*mm)(float *, const float *, const uint8_t *, const uint8_t *, int, int, int)
        = m->idot ? matmul_mxfp4_i8 : matmul_mxfp4;
    /* rows per task: a wave of one or two arrivals is cut finer, so every worker has
     * a few tasks (a row's numbers do not depend on the cut) */
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    int BLOCK_ROWS = 256;
    while (BLOCK_ROWS > 64 && nw * ((MI + BLOCK_ROWS - 1) / BLOCK_ROWS) < 4 * threads) BLOCK_ROWS /= 2;
    int up_blocks = (MI + BLOCK_ROWS - 1) / BLOCK_ROWS, down_blocks = (H + BLOCK_ROWS - 1) / BLOCK_ROWS;
    int rb_mi = MI / 2, gb_mi = MI / 32, rb_h = H / 2, gb_h = H / 32;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int task = 0; task < nw * up_blocks; task++) {
        int j = wave[task / up_blocks], blk = task % up_blocks;
        int r = first[j + 1] - first[j];
        if (!r) continue;
        int o0 = blk * BLOCK_ROWS, rows = MI - o0 < BLOCK_ROWS ? MI - o0 : BLOCK_ROWS;
        const uint8_t *gq = sl[j]->buf + m->part[0] + m->part[1], *gs = gq + m->part[2];
        const uint8_t *uq = gs + m->part[3], *us = uq + m->part[4];
        /* rows [o0, o0+rows) of this expert's gate and up for its r inputs, written
         * [r, rows] by the kernel, then silu(gate) * up into g */
        float *sg = task_scratch((size_t)2 * r * rows), *su = sg + (size_t)r * rows;
        const float *xr = xg + (size_t)first[j] * H;
        mm(sg, xr, gq + (size_t)o0 * rb_h, gs + (size_t)o0 * gb_h, r, H, rows);
        mm(su, xr, uq + (size_t)o0 * rb_h, us + (size_t)o0 * gb_h, r, H, rows);
        float *dst = g + (size_t)first[j] * MI + o0;
        for (int i = 0; i < r; i++)
            for (int k = 0; k < rows; k++)
                dst[(size_t)i * MI + k] = silu(sg[(size_t)i * rows + k]) * su[(size_t)i * rows + k];
    }
    #pragma omp parallel for schedule(dynamic, 1)
    for (int task = 0; task < nw * down_blocks; task++) {
        int j = wave[task / down_blocks], blk = task % down_blocks;
        int r = first[j + 1] - first[j];
        if (!r) continue;
        int o0 = blk * BLOCK_ROWS, rows = H - o0 < BLOCK_ROWS ? H - o0 : BLOCK_ROWS;
        const uint8_t *dq = sl[j]->buf, *ds = dq + m->part[0];
        float *scratch = task_scratch((size_t)r * rows);
        mm(scratch, g + (size_t)first[j] * MI, dq + (size_t)o0 * rb_mi, ds + (size_t)o0 * gb_mi, r, MI, rows);
        float *dst = y + (size_t)(base + first[j]) * H;
        for (int i = 0; i < r; i++)
            memcpy(dst + (size_t)i * H + o0, scratch + (size_t)i * rows, (size_t)rows * sizeof(float));
    }
}

/* The CPU's experts of one MoE step: every (row, choice) pair i of sel -- or,
 * with a mask, those whose mask[i] is mval -- runs its expert, and its output
 * lands in y at row pair[i], numbered on from `total`; returns the new count.
 *
 * The experts are streamed: those the cache holds are computed at once while the
 * others are read, then each wave takes whatever has arrived. A step whose union
 * is larger than the cache gets slots as earlier experts finish. A wave is two
 * parallel passes (moe_wave) -- gate and up of every expert, then down of every
 * expert -- split in row blocks, instead of three OpenMP regions per expert (a
 * thousand a token on Flash, and every barrier costs a scheduling round on a busy
 * machine). The kernel inside a task is the same matmul_mxfp4 on the same rows,
 * so the numbers do not move: they are those of a step computed all at once.
 *
 * Every (row, choice) pair has its own slot in y, numbered from `total` in the
 * union's order, so the weighted sum can wait for the last wave and run row by
 * row in that row's own routing order. With `note`, every expert computed here
 * is offered to the Vulkan tier while its bytes are in RAM (vkt_note; nothing
 * without the tier). */
static int moe_cpu(Model *m, int li, const float *xn, int n, const int *sel,
                   const uint8_t *mask, int mval, int *pair, float *y, int total, int note) {
    Cfg *c = &m->c;
    int H = c->hidden, E = c->n_experts, K = c->topk, MI = c->moe_inter;
#define MIMO_WANT(i) (!mask || (mask[i] != 0) == mval)
    /* the union of chosen experts, in first-use order */
    int *uni = xmalloc((size_t)E * sizeof(int), "expert union");
    int *seen = xcalloc((size_t)E, sizeof(int), "expert seen");
    int nu = 0;
    for (int i = 0; i < n * K; i++) if (MIMO_WANT(i) && !seen[sel[i]]) { seen[sel[i]] = 1; uni[nu++] = sel[i]; }
    free(seen);
    int *row = xmalloc((size_t)n * K * sizeof(int), "expert rows index");
    int *first = xmalloc((size_t)(nu + 1) * sizeof(int), "expert row offsets");
    int base = total;                /* y rows before this call are someone else's */
    for (int j = 0; j < nu; j++) {
        first[j] = total - base;
        for (int t = 0; t < n; t++)
            for (int q = 0; q < K; q++)
                if (sel[(size_t)t * K + q] == uni[j] && MIMO_WANT((size_t)t * K + q)) {
                    row[total - base] = t;
                    pair[(size_t)t * K + q] = total;
                    total++;
                }
    }
    first[nu] = total - base;
#undef MIMO_WANT
    Slot **sl = xmalloc((size_t)(nu ? nu : 1) * sizeof(Slot *), "expert slots of a step");
    m->bulk = n > m->pf_rows;
    uint64_t stamp = slots_demand(m, li, uni, nu, sl);
    if (m->pf_h) mimo_prefetch_pending(m);
    double t0 = now_s(), waited = m->t_disk;
    float *xg = xmalloc((size_t)(total - base ? total - base : 1) * H * sizeof(float), "expert inputs");
    float *g = xmalloc((size_t)(total - base ? total - base : 1) * MI * sizeof(float), "expert gate");
    for (int i = base; i < total; i++) memcpy(xg + (size_t)(i - base) * H, xn + (size_t)row[i - base] * H, (size_t)H * sizeof(float));
    int *wave = xmalloc((size_t)(nu ? nu : 1) * sizeof(int), "expert wave");
    uint8_t *done = xcalloc((size_t)(nu ? nu : 1), 1, "experts done");
    for (int left = nu; left > 0;) {
        unsigned long long seen_io = io_seen();
        int nw = 0;
        for (int j = 0; j < nu; j++) if (!done[j] && sl[j] && slot_ready(sl[j])) wave[nw++] = j;
        if (!nw) {
            io_wait_any(m, seen_io);
        } else {
            moe_wave(m, wave, nw, sl, first, xg, g, y, base);
            m->waves++; m->wave_experts += (uint64_t)nw;
            for (int w = 0; w < nw; w++) {
                int j = wave[w];
                done[j] = 1; sl[j]->pin--; left--;
                /* its bytes are still in RAM: the tier may promote it */
                if (note && vkt_ready()) { VktExpertSrc vs = mimo_vk_src(m, sl[j]->buf); vkt_note(li, uni[j], &vs); }
            }
        }
        for (int j = 0; j < nu; j++) {   /* a step larger than the cache: slots as experts finish */
            if (done[j] || sl[j]) continue;
            Slot *s = slot_take(m, li, uni[j], 0);
            if (!s) break;
            s->pin++; s->used = stamp;
            sl[j] = s;
        }
    }
    m->t_expert += now_s() - t0 - (m->t_disk - waited);
    free(done); free(wave); free(sl);
    free(first); free(xg); free(g); free(row); free(uni);
    return total;
}

/* The prefetcher (MIMO_PREFETCH=1): layer li's router applied to the residual h of
 * the layer before it -- after that layer's attention, normed by li's own
 * post-attention norm -- predicts li's experts while the layer before computes its
 * own, and the ones the cache does not hold are read ahead. The residual changes
 * little from one layer to the next, so most of the prediction comes true; a wrong
 * one costs a read and a slot that is the first to be reused (slots_demand), never
 * a number: the routing that counts is li's own, on its own input.
 * MIMO_PREFETCH_EXTRA adds that many experts beyond top-k to each row's prediction;
 * MIMO_PREFETCH_ROWS bounds the rows a prediction is made for (a decode step, the
 * conversations of a multiplexed one; not a prefill block); MIMO_PREFETCH_LIMIT the
 * prefetches waiting at once. */
static void mimo_prefetch(Model *m, int li, const float *h, int n, int k) {
    Cfg *c = &m->c;
    if (k < 1 || !g_io.nth || li >= c->n_layers || !c->moe[li] || n < 1 || n > m->pf_rows) return;
    double t0 = now_s();
    int H = c->hidden, E = c->n_experts, K = k + m->pf_extra;
    if (K > E) K = E;
    Layer *l = &m->L[li];
    float *xn = xmalloc((size_t)n * H * sizeof(float), "prefetch input");
    float *lg = xmalloc((size_t)n * E * sizeof(float), "prefetch logits");
    for (int t = 0; t < n; t++) rmsnorm(xn + (size_t)t * H, h + (size_t)t * H, l->ln2, H, c->eps);
    matmul_router(lg, xn, l->router, n, H, E);
    uint8_t *mark = m->pf_mark + (size_t)li * E;
    int *order = xmalloc((size_t)n * K * sizeof(int), "prefetch order"), no = 0;
    int *votes = xcalloc((size_t)E, sizeof(int), "prefetch votes");
    for (int t = 0; t < n; t++) {
        float *v = lg + (size_t)t * E;
        for (int e = 0; e < E; e++) v[e] = sigmoid(v[e]) + l->bias[e];
        for (int j = 0; j < K; j++) {
            int best = -1; float bv = -INFINITY;
            for (int e = 0; e < E; e++) if (v[e] > bv) { bv = v[e]; best = e; }
            if (best < 0) break;
            v[best] = -INFINITY;
            if (!mark[best]) { mark[best] = 1; order[no++] = best; }
            votes[best]++;
        }
    }
    /* the experts most rows chose first (a block of rows; one row keeps its rank order) */
    for (int a = 1; a < no; a++) {
        int e = order[a], b = a;
        while (b > 0 && votes[order[b - 1]] < votes[e]) { order[b] = order[b - 1]; b--; }
        order[b] = e;
    }
    free(votes);
    m->pf_made[li] = 1;
    unsigned waiting = 0;
    if (g_io.nth) {
        pthread_mutex_lock(&g_io.mx);
        for (int d = 0; d < g_io.ndrive; d++) waiting += g_io.tail[d][1] - g_io.head[d][1];
        pthread_mutex_unlock(&g_io.mx);
    }
    /* never more than half the layer's cache in one prediction: a prompt block's
     * union can be most of the layer, and its residents are the next step's hits */
    int room = m->cache[li].cap / 2;
    for (int k = 0; k < no && room > 0; k++) {
        if (m->cache[li].by_expert[order[k]] >= 0) continue;
        if (waiting >= (unsigned)m->pf_limit) break;
        if (!slot_take(m, li, order[k], 1)) break;
        waiting++; room--;
    }
    free(order); free(xn); free(lg);
    m->t_pred += now_s() - t0;
}

static void mimo_prefetch_pending(Model *m) {
    if (!m->pf_h) return;
    const float *h = m->pf_h;
    m->pf_h = NULL;
    mimo_prefetch(m, m->pf_li, h, m->pf_n, m->pf_ka);
}

/* A step's routing against the prediction that stood for it, then the prediction is spent. */
static void mimo_prefetch_score(Model *m, int li, const int *sel, int n) {
    if (!m->pf_made || !m->pf_made[li]) return;
    int E = m->c.n_experts, K = m->c.topk;
    uint8_t *mark = m->pf_mark + (size_t)li * E;
    for (int i = 0; i < n * K; i++) { m->pf_pred++; m->pf_pred_hit += mark[sel[i]] != 0; }
    memset(mark, 0, (size_t)E);
    m->pf_made[li] = 0;
}

/* MIMO_ROUTE_TRACE=<file>: every MoE step's routing, for offline study of the expert
 * cache and of the prefetcher's predictions (tools/mimo_route_sim.py). A record is
 * int32 layer, rows, position, then rows * topk int16 expert ids; a one-row step adds
 * int16 have[2] and the experts layer li's router ranks first (MIMO_TRACE_RANK of
 * them, best first) from the residual at the two points the prefetcher can use: after
 * the previous layer's attention (0) and after its MLP (1). Timing is not recorded
 * and nothing computed changes. */
static int16_t g_rt_rank[2][MIMO_MAX_LAYERS][MIMO_TRACE_RANK];
static int16_t g_rt_have[2][MIMO_MAX_LAYERS];

static void mimo_trace_rank(Model *m, int li, const float *h, int at) {
    Cfg *c = &m->c;
    if (!g_rtrace || li >= c->n_layers || !c->moe[li]) return;
    int H = c->hidden, E = c->n_experts, R = MIMO_TRACE_RANK < E ? MIMO_TRACE_RANK : E;
    Layer *l = &m->L[li];
    float *xn = xmalloc((size_t)H * sizeof(float), "trace input");
    float *v = xmalloc((size_t)E * sizeof(float), "trace logits");
    rmsnorm(xn, h, l->ln2, H, c->eps);
    matmul_router(v, xn, l->router, 1, H, E);
    for (int e = 0; e < E; e++) v[e] = sigmoid(v[e]) + l->bias[e];
    for (int j = 0; j < MIMO_TRACE_RANK; j++) {
        int best = -1; float bv = -INFINITY;
        for (int e = 0; e < E && j < R; e++) if (v[e] > bv) { bv = v[e]; best = e; }
        g_rt_rank[at][li][j] = (int16_t)best;
        if (best >= 0) v[best] = -INFINITY;
    }
    g_rt_have[at][li] = 1;
    free(xn); free(v);
}

static void mimo_trace_close(void) { if (g_rtrace) { fclose(g_rtrace); g_rtrace = NULL; } }

static void mimo_trace_step(Model *m, int li, const int *sel, int n) {
    if (!g_rtrace) return;
    int K = m->c.topk;
    int32_t head[3] = {li, n, m->mux_rows ? -1 : m->pos};
    fwrite(head, sizeof head, 1, g_rtrace);
    for (int i = 0; i < n * K; i++) { int16_t e = (int16_t)sel[i]; fwrite(&e, 2, 1, g_rtrace); }
    if (n == 1) {
        int16_t have[2] = {g_rt_have[0][li], g_rt_have[1][li]};
        fwrite(have, sizeof have, 1, g_rtrace);
        fwrite(g_rt_rank[0][li], 2, MIMO_TRACE_RANK, g_rtrace);
        fwrite(g_rt_rank[1][li], 2, MIMO_TRACE_RANK, g_rtrace);
    }
    g_rt_have[0][li] = g_rt_have[1][li] = 0;
}

/* Route every row, then run each chosen expert once over the rows that chose it.
 * `out` [n, H] receives the weighted sum.
 *
 * With the Vulkan tier on, vkt_issue first hands the pairs whose expert is
 * resident on the device to one asynchronous batch; moe_cpu reads and computes
 * the others meanwhile (an expert the device took entirely is never read from
 * disk), and after the join every pair joins the sum in its row's routing order,
 * the device's and the CPU's alike: the order never depends on which experts were
 * resident. The experts the CPU computed are offered to the tier after the join
 * (those still in the RAM cache), not while the batch runs: a promotion that
 * displaces a resident then frees it at once, and its upload starts on a pool
 * with room instead of waiting on the uploader thread for the join's free. */
#define MIMO_VK_ROWS 64              /* rows whose pairs one device batch carries at most */
static void moe(Model *m, int li, const float *xn, int n, float *out) {
    Cfg *c = &m->c;
    Layer *l = &m->L[li];
    int H = c->hidden, E = c->n_experts, K = c->topk;
    int *sel = xmalloc((size_t)n * K * sizeof(int), "routing");
    float *wt = xmalloc((size_t)n * K * sizeof(float), "routing weights");
    float *logits = xmalloc((size_t)n * E * sizeof(float), "router logits");
    matmul_router(logits, xn, l->router, n, H, E);
    for (int t = 0; t < n; t++) {
        float *score = logits + (size_t)t * E;
        for (int e = 0; e < E; e++) score[e] = sigmoid(score[e]);
        int *s = sel + (size_t)t * K;
        float *w = wt + (size_t)t * K;
        for (int j = 0; j < K; j++) {
            int best = -1; float bv = -INFINITY;
            for (int e = 0; e < E; e++) {
                int taken = 0;
                for (int q = 0; q < j; q++) if (s[q] == e) { taken = 1; break; }
                if (taken) continue;
                float v = score[e] + l->bias[e];
                if (v > bv) { bv = v; best = e; }
            }
            s[j] = best; w[j] = score[best];
        }
        float sum = 0;
        for (int j = 0; j < K; j++) sum += w[j];
        for (int j = 0; j < K; j++) w[j] = (c->norm_topk && K > 1 ? w[j] / (sum + 1e-20f) : w[j]) * c->route_scale;
        for (int j = 0; j < K; j++) m->ehit[li][s[j]] = 1;
    }
    free(logits);
    mimo_prefetch_score(m, li, sel, n);
    mimo_trace_step(m, li, sel, n);
    memset(out, 0, (size_t)n * H * sizeof(float));
    int *pair = xmalloc((size_t)n * K * sizeof(int), "expert pair of a choice");
    float *y = xmalloc((size_t)n * K * H * sizeof(float), "expert out");
    uint8_t *taken = NULL;
    const float **dev = NULL;
    int ndev = 0;
    if (vkt_ready()) {
        taken = xmalloc((size_t)n * K, "device pairs");
        dev = xmalloc((size_t)n * K * sizeof(*dev), "device rows");
        ndev = vkt_issue(li, xn, n, K, sel, taken);
    }
    int total = moe_cpu(m, li, xn, n, sel, ndev ? taken : NULL, 0, pair, y, 0, !ndev);
    if (ndev && !vkt_join(dev)) {    /* the batch failed (the tier stops): those pairs here */
        moe_cpu(m, li, xn, n, sel, taken, 1, pair, y, total, 0);
        ndev = 0;
    } else if (ndev) {               /* the CPU's experts still in RAM, now that nothing is in flight */
        uint8_t *noted = xcalloc((size_t)E, 1, "noted experts");
        int *by = m->cache[li].by_expert;
        for (int i = 0; i < n * K; i++) {
            int e = sel[i];
            if (taken[i] || noted[e] || by[e] < 0) continue;
            noted[e] = 1;
            VktExpertSrc vs = mimo_vk_src(m, m->cache[li].s[by[e]].buf);
            vkt_note(li, e, &vs);
        }
        free(noted);
    }
    /* The weighted sum, row by row, each row's experts in the order ITS router
     * chose them. A row's sum used to follow the block's first-use order, which
     * depends on the rows that happen to share the block: the same position
     * came out with different low bits in a 64-row prefill block and in a
     * one-row resume, so a prompt resumed from a prefix was not the prompt
     * computed cold. Now a row is a function of its own inputs and nothing else,
     * whatever the block boundaries. One row (a decode step) adds in the same
     * order as before: its first-use order IS its routing order. */
    double t1 = now_s();
    for (int t = 0; t < n; t++) {
        float *o = out + (size_t)t * H;
        for (int q = 0; q < K; q++) {
            size_t i = (size_t)t * K + q;
            const float *yy = ndev && taken[i] ? dev[i] : y + (size_t)pair[i] * H;
            float w = wt[i];
            for (int d = 0; d < H; d++) o[d] += w * yy[d];
        }
    }
    m->t_expert += now_s() - t1;
    free(pair); free(y); free(taken); free(dev); free(sel); free(wt);
}

static void embed_rows(Model *m, const int *ids, int n, float *h) {
    int H = m->c.hidden;
    for (int t = 0; t < n; t++) {
        int id = ids[t];
        if (id < 0 || id >= m->c.vocab) { fprintf(stderr, "[mimo] token id %d out of range\n", id); exit(1); }
        for (int i = 0; i < H; i++) h[(size_t)t * H + i] = bf16_to_f32(m->embed[(size_t)id * H + i]);
    }
}

/* Image rows to splice in: `rows[r]` replaces the embedding of the r-th
 * image-pad position of the WHOLE prompt (the first at `first_pad`). */
typedef struct { const float *rows; int n_rows; } ImageRows;

/* Layers [l0, L) on the CPU for the nc rows hc at position pc (in place; xn and tmp hold
 * nc rows each). trace: MIMO_TRACE's file, the residual after every block. */
static void layers_cpu(Model *m, float *hc, int nc, int pc, int l0, float *xn, float *tmp, FILE *trace) {
    Cfg *c = &m->c;
    int H = c->hidden;
    for (int li = l0; li < c->n_layers && nc > 0; li++) {
        Layer *l = &m->L[li];
        for (int t = 0; t < nc; t++) rmsnorm(xn + (size_t)t * H, hc + (size_t)t * H, l->ln1, H, c->eps);
        attention(m, li, xn, nc, pc, tmp);
        for (size_t i = 0; i < (size_t)nc * H; i++) hc[i] += tmp[i];
        if (trace) fwrite(hc, sizeof(float), (size_t)nc * H, trace);
        for (int t = 0; t < nc; t++) rmsnorm(xn + (size_t)t * H, hc + (size_t)t * H, l->ln2, H, c->eps);
        /* the next layer's experts, predicted from this residual, are read while
         * this layer's MLP runs (after this layer's own reads: moe_cpu) */
        if (m->pf_on) { m->pf_h = hc; m->pf_n = nc; m->pf_li = li + 1; }
        if (g_rtrace && nc == 1) mimo_trace_rank(m, li + 1, hc, 0);
        if (c->moe[li]) moe(m, li, xn, nc, tmp);
        else { mimo_prefetch_pending(m); dense_mlp(m, l, xn, nc, tmp); }
        m->pf_h = NULL;
        for (size_t i = 0; i < (size_t)nc * H; i++) hc[i] += tmp[i];
        if (trace) fwrite(hc, sizeof(float), (size_t)nc * H, trace);
        if (g_rtrace && nc == 1) mimo_trace_rank(m, li + 1, hc, 1);
        /* MIMO_PREFETCH=2 or MIMO_PREFETCH_MLP: again from the layer's output, the
         * experts the first prediction missed */
        if (m->pf_kb) mimo_prefetch(m, li + 1, hc, nc, m->pf_kb);
    }
}
/* The final norm and lm_head on the CPU: every row of hc into lc (all_rows), or its last. */
static void head_cpu(Model *m, const float *hc, int nc, float *lc, int all_rows, float *xn) {
    Cfg *c = &m->c;
    int H = c->hidden, from = all_rows ? 0 : nc - 1, rows = nc - from;
    for (int t = from; t < nc; t++) rmsnorm(xn + (size_t)(t - from) * H, hc + (size_t)t * H, m->norm, H, c->eps);
    dw_matmul(lc, xn, rows, &m->head);
}

#ifdef COLI_VULKAN
#include "mimo_chain.h"   /* COLI_VK_CHAIN: every layer's dense chain on the device */
#endif

/* One block of n tokens at m->pos. logits: NULL, or [n, V] (all rows) when
 * all_rows, else [V] for the last row. */
static void forward(Model *m, const int *ids, int n, float *logits, int all_rows,
                    const ImageRows *img, int *img_used) {
    Cfg *c = &m->c;
    int H = c->hidden;
    if (m->pos + n > m->ctx) { fprintf(stderr, "[mimo] context full (%d + %d > %d)\n", m->pos, n, m->ctx); exit(1); }
    float *h = xmalloc((size_t)n * H * sizeof(float), "residual");
    float *xn = xmalloc((size_t)n * H * sizeof(float), "normed");
    float *tmp = xmalloc((size_t)n * H * sizeof(float), "block out");
    embed_rows(m, ids, n, h);
    if (img && img->rows && c->image_token_id >= 0)
        for (int t = 0; t < n; t++)
            if (ids[t] == c->image_token_id && *img_used < img->n_rows) {
                memcpy(h + (size_t)t * H, img->rows + (size_t)(*img_used) * H, (size_t)H * sizeof(float));
                (*img_used)++;
            }
    /* MIMO_TRACE=<file>: the residual after the embedding and after every
     * attention and every MLP of the first block, f32 [2L+1][n][H] -- what
     * tools/make_mimo_ref.py --trace compares against the vendor's hooks. */
    static FILE *trace;
    static int traced;
    if (!traced && getenv("MIMO_TRACE")) { trace = fopen(getenv("MIMO_TRACE"), "wb"); }
    if (trace) fwrite(h, sizeof(float), (size_t)n * H, trace);
    /* The rows the CPU computes: all of them, or with the dense chain on (COLI_VK_CHAIN,
     * mimo_chain.h) the ones it did not (a lost device), from where the host's caches end. */
    int done = 0;
#ifdef COLI_VULKAN
    if (g_vk_chain && !trace) {
        done = mc_forward(m, h, n, m->pos, logits, all_rows, xn, tmp);
        if (done < n) mc_cpu_step(m, m->pos + done);
    }
#endif
    float *hc = h + (size_t)done * H, *lc = logits && all_rows ? logits + (size_t)done * c->vocab : logits;
    int nc = n - done, pc = m->pos + done;
    layers_cpu(m, hc, nc, pc, 0, xn, tmp, trace);
    if (g_mtp.on && nc > 0) mtp_keep(m, hc, nc, pc);
    if (trace) { fclose(trace); trace = NULL; }
    traced = 1;
    if (lc && nc > 0) head_cpu(m, hc, nc, lc, all_rows, xn);
    kv_prefix_record(&m->kvp, ids, m->pos, n);
    m->pos += n;
    m->forwards++;
    free(h); free(xn); free(tmp);
}

/* sum exp(x[i] - top) over x[0..n), the softmax's denominator: over the workers in
 * chunks of a fixed size, each chunk's sum in double in order, then the chunks' in order,
 * so the bits do not depend on the thread count or on which worker took a chunk. (One
 * region: on MinGW's libgomp a fork or a barrier costs tens of microseconds, so the
 * argmax before it stays a single pass.) */
static double exp_sum(const float *x, int n, float top) {
    enum { CH = 512 };
    int nc = (n + CH - 1) / CH;
    double *part = xmalloc((size_t)nc * sizeof(double), "softmax chunks");
    #pragma omp parallel for schedule(dynamic, 1)
    for (int c = 0; c < nc; c++) {
        int lo = c * CH, hi = lo + CH < n ? lo + CH : n;
        double a = 0;
        for (int i = lo; i < hi; i++) a += expf(x[i] - top);
        part[c] = a;
    }
    double s = 0;
    for (int c = 0; c < nc; c++) s += part[c];
    free(part);
    return s;
}

/* MTP block k over rows at positions p0.., row t from the trunk's hidden state at p0 + t
 * and tok[t], the token at p0 + t + k + 1; with `p`, the last row's draft (else -1) and in
 * *p the probability the block gives it. The trunk's attention counters stay as they
 * were: the blocks' time is g_mtp.t. */
static int mtp_run(Model *m, int k, const int *tok, int p0, int n, float *p) {
    Cfg *c = &m->c;
    int H = c->hidden;
    MtpBlock *b = &g_mtp.b[k];
    double t0 = now_s(), ta = m->t_attn, tq = m->t_qkv, to = m->t_oproj;
    float *cat = xmalloc((size_t)n * 2 * H * sizeof(float), "mtp input");
    float *h = xmalloc((size_t)n * H * sizeof(float), "mtp residual");
    float *xn = xmalloc((size_t)n * H * sizeof(float), "mtp normed");
    float *tmp = xmalloc((size_t)n * H * sizeof(float), "mtp block out");
    embed_rows(m, tok, n, tmp);
    for (int t = 0; t < n; t++) {
        const float *hid = g_mtp.keep + (size_t)((p0 + t) % g_mtp.keep_rows) * H;
        float *e = cat + (size_t)t * 2 * H;
        rmsnorm(e, tmp + (size_t)t * H, b->enorm, H, c->eps);
        rmsnorm(xn + (size_t)t * H, hid, m->norm, H, c->eps);
        rmsnorm(e + H, xn + (size_t)t * H, b->hnorm, H, c->eps);
    }
    dw_matmul(h, cat, n, &b->eh);
    int li = g_mtp.li + k;
    Layer *l = &m->L[li];
    for (int t = 0; t < n; t++) rmsnorm(xn + (size_t)t * H, h + (size_t)t * H, l->ln1, H, c->eps);
    attention(m, li, xn, n, p0, tmp);
    for (size_t i = 0; i < (size_t)n * H; i++) h[i] += tmp[i];
    for (int t = 0; t < n; t++) rmsnorm(xn + (size_t)t * H, h + (size_t)t * H, l->ln2, H, c->eps);
    dense_mlp(m, l, xn, n, tmp);
    for (size_t i = 0; i < (size_t)n * H; i++) h[i] += tmp[i];
    int d = -1;
    if (p) {
        float *lg = xmalloc((size_t)c->vocab * sizeof(float), "mtp logits");
        rmsnorm(xn, h + (size_t)(n - 1) * H, b->fnorm, H, c->eps);
        dw_matmul(lg, xn, 1, &m->head);
        d = 0;
        for (int i = 1; i < c->vocab; i++) if (lg[i] > lg[d]) d = i;
        *p = (float)(1.0 / exp_sum(lg, c->vocab, lg[d]));
        free(lg);
    }
    free(cat); free(h); free(xn); free(tmp);
    m->t_attn = ta; m->t_qkv = tq; m->t_oproj = to;
    g_mtp.t += now_s() - t0;
    return d;
}
/* After a prompt whose positions a..b-1 were computed (ids from position 0): each block's
 * window filled from the last of them -- block k's rows up to b - 2 - k, the last whose
 * token is in the prompt -- so the first drafts see what they would later. */
static void mtp_warm(Model *m, const int *ids, int a, int b) {
    int lo = b - 1 - m->c.window;
    if (lo < a) lo = a;
    for (int p = b > 8 ? b - 8 : 0; p < b; p++) g_mtp.tok[p & 7] = ids[p];
    for (int k = 0; k < g_mtp.n; k++) {
        Layer *l = &m->L[g_mtp.li + k];
        for (int r = 0; r < l->rows; r++) l->ring_pos[r] = -1;
        l->kv_lo = lo;
        int end = b - 1 - k;
        if (end > lo) mtp_run(m, k, ids + lo + k + 1, lo, end - lo, NULL);
        g_mtp.b[k].next = end > lo ? end : lo;
    }
}
/* A dropped draft: the trunk's state goes back to `pos` (its K/V rows beyond are rewritten
 * before they are read again). */
static void mtp_back(Model *m, int pos) {
    m->pos = pos;
    if (m->kvp.len > pos) m->kvp.len = pos;
}
/* ---- several conversations at once (KV_SLOTS>1, serve_mux) -------------------
 * Each conversation owns a MimoSeq. The Model's layers hold the conversation a
 * prefill runs on (mimo_seq_swap trades it for a parked one); a decode step parks
 * them all and runs one forward over a row of each (forward_rows). */
static void mimo_seq_swap(Model *m, MimoSeq *q) {
    for (int li = 0; li < m->c.n_layers; li++) {
        Layer *l = &m->L[li];
        float *K = l->K, *V = l->V; int *rp = l->ring_pos;
        l->K = q->K[li]; l->V = q->V[li]; l->ring_pos = q->ring_pos[li];
        q->K[li] = K; q->V[li] = V; q->ring_pos[li] = rp;
    }
    int pos = m->pos; kv_prefix p = m->kvp;
    m->pos = q->pos; m->kvp = q->kvp;
    q->pos = pos; q->kvp = p;
}
/* A conversation's caches of its own, the shape kv_alloc gives the Model's. */
static void mimo_seq_alloc(Model *m, MimoSeq *q) {
    Cfg *c = &m->c;
    memset(q, 0, sizeof *q);
    for (int li = 0; li < c->n_layers; li++) {
        Layer *l = &m->L[li];
        int k = c->swa[li];
        q->K[li] = xmalloc((size_t)l->rows * c->kv_heads[k] * c->head_dim[k] * sizeof(float), "K cache");
        q->V[li] = xmalloc((size_t)l->rows * c->kv_heads[k] * c->v_dim[k] * sizeof(float), "V cache");
        q->ring_pos[li] = xmalloc((size_t)l->rows * sizeof(int), "ring positions");
        for (int r = 0; r < l->rows; r++) q->ring_pos[li][r] = -1;
    }
    kv_prefix_alloc(&q->kvp, m->ctx);
}

/* One decode step of several conversations: row t is the token ids[t] at rows[t].pos
 * of the conversation rows[t].seq, every conversation parked. The matrices, the routed
 * experts and lm_head run once over the n rows; the attention reads and writes each
 * row's own caches (attention_rows). The CPU kernels give a row the same bits whatever
 * n is, so each conversation gets the logits it would alone: [n, V] into logits. */
static void forward_rows(Model *m, const MimoRow *rows, const int *ids, int n, float *logits) {
    Cfg *c = &m->c;
    int H = c->hidden;
    float *h = xmalloc((size_t)n * H * sizeof(float), "residual");
    float *xn = xmalloc((size_t)n * H * sizeof(float), "normed");
    float *tmp = xmalloc((size_t)n * H * sizeof(float), "block out");
    embed_rows(m, ids, n, h);
    m->mux_rows = rows;
    layers_cpu(m, h, n, 0, 0, xn, tmp, NULL);
    m->mux_rows = NULL;
    head_cpu(m, h, n, logits, 1, xn);
    for (int t = 0; t < n; t++) {
        kv_prefix_record(&rows[t].seq->kvp, ids + t, rows[t].pos, 1);
        rows[t].seq->pos = rows[t].pos + 1;
    }
    m->forwards++;
    free(h); free(xn); free(tmp);
}

/* The read-out of a prefill, for the logprobs channel (SUBMIT logprobs=k): one
 * ECHO frame per prompt position this prefill feeds, in position order, before
 * any DATA. The logits at position p predict the token at p + 1, so the frame
 * of a position carries the log-probability of the token that is actually
 * there, over the whole vocabulary, and not only when it is among the top k:
 * that is what scores an option the model would never have written.
 *
 * The first position fed has no predictor among the rows computed here. On a
 * cold prompt it is position 0, which has nothing to condition on, and its
 * frame carries " nan 0". Resumed from a photo, `first` is the photo's own last
 * logits, which predict exactly that token. There is no third case: a prompt
 * read out never resumes from a live prefix, because the positions before it
 * would have no frame at all (see the serve loop).
 *
 * This is its own argument and not a mode of something else: it asks for every
 * row's logits and changes nothing about how the rows are computed. */
typedef struct {
    const char *id;
    int k;
    Tok *tok;
    const float *first;
} Echo;

static void echo_frame(const Echo *e, int pos, int token, const float *lo, int vocab) {
    char tail[1024], piece[512];
    coli_logprob_tail(tail, sizeof(tail), lo, vocab, token, e->k);
    int n = tok_decode(e->tok, &token, 1, piece, (int)sizeof(piece));
    if (n < 0) n = 0;
    printf("ECHO %s %d %d%s\n", e->id, n, pos, tail);
    if (n > 0) fwrite(piece, 1, (size_t)n, stdout);
    fputc('\n', stdout);
    fflush(stdout);
}

/* A prompt in blocks of MIMO_CHUNK rows (default 512): the last block's last
 * row (or every row, for the teacher-forcing dump) comes back in `logits`.
 * With `echo`, every row of every block goes through the head and is read out
 * as it comes; only one block of logits is ever held, not the prompt's. A block
 * computes each expert once for all of its rows that chose it, so a long prompt
 * in blocks of 64 read and streamed most of a layer's experts for every 64 rows:
 * 2260 tokens took 498 s in blocks of 64 and 213 s in blocks of 512 (the logits
 * are the same bytes whatever the blocks). */
static void prefill(Model *m, const int *ids, int n, float *logits, float *all_logits,
                    const ImageRows *img, const Echo *echo) {
    int chunk = env_int("MIMO_CHUNK", 512);
#ifdef COLI_VULKAN
    /* the dense chain with a chunk from the budget (vkc_chunk_auto): blocks of its chunk,
     * so a long prompt's MoE steps are big ones the tier can stream (not with a read-out,
     * which holds a block's logits) */
    if (!getenv("MIMO_CHUNK") && g_vk_chain && vkc_chunk_auto() && !echo) {
        MimoChain *ch = mc_setup(m);
        if (ch && !ch->failed) chunk = mc_chunk_rows(ch, m);
    }
#endif
    if (chunk < 1) chunk = 1;
    int used = 0, V = m->c.vocab;
    float *rows = NULL;
    if (echo && !all_logits) {
        rows = xmalloc((size_t)(n < chunk ? n : chunk) * V * sizeof(float), "read-out logits");
        echo_frame(echo, m->pos, ids[0], echo->first, V);
    }
    for (int b = 0; b < n; b += chunk) {
        int nb = n - b < chunk ? n - b : chunk;
        if (all_logits) forward(m, ids + b, nb, all_logits + (size_t)b * V, 1, img, &used);
        else if (rows) {
            int p0 = m->pos;
            forward(m, ids + b, nb, rows, 1, img, &used);
            for (int t = 0; t < nb; t++) {
                if (b + t + 1 < n) echo_frame(echo, p0 + t + 1, ids[b + t + 1], rows + (size_t)t * V, V);
                else if (logits) memcpy(logits, rows + (size_t)t * V, (size_t)V * sizeof(float));
            }
        }
        else forward(m, ids + b, nb, b + nb == n ? logits : NULL, 0, img, &used);
    }
    free(rows);
    if (all_logits && logits) memcpy(logits, all_logits + (size_t)(n - 1) * V, (size_t)V * sizeof(float));
}

/* ----------------------------------------------------------------- vision ---- */

/* The tower, when the container carries one (visual.*): vision_run turns the
 * gateway's patches into the rows that replace the image-pad embeddings. */
typedef struct Vision Vision;
static Vision *g_vision;
#include "mimo_vision.h"

#ifdef COLI_VULKAN
/* The dense matrices go up first, at start-up, the trunk before the tower, so
 * the experts take only what they leave; one that does not fit stays on the
 * CPU. Returns how many went up. The dense chain alone (the per-matrix path off)
 * takes the trunk and leaves the tower on the CPU (vision 0). */
static void mc_place(Model *m, int *dropped);   /* below: a fit's layers, layer by layer */
static int vk_dense_upload(Model *m, int vision) {
    int n = 0;
    if (g_mc_fit.L && !g_mc_placed) mc_place(m, NULL);   /* the chain's fit: its layers, each whole or none */
    for (int li = 0; li < m->c.n_layers; li++) {
        Layer *l = &m->L[li];
        n += dw_upload(&l->qkv) + dw_upload(&l->o);
        if (!m->c.moe[li]) n += dw_upload(&l->gate) + dw_upload(&l->up) + dw_upload(&l->down);
    }
    n += dw_upload(&m->head);
    if (g_vision && vision) {
        Vision *v = g_vision;
        n += dw_upload(&v->embed) + dw_upload(&v->fc1) + dw_upload(&v->fc2);
        for (int i = 0; i < v->depth; i++)
            n += dw_upload(&v->b[i].qkv) + dw_upload(&v->b[i].proj) + dw_upload(&v->b[i].gate) +
                 dw_upload(&v->b[i].up) + dw_upload(&v->b[i].down);
    }
    return n;
}

static size_t vk_dense_bytes(const Model *m, int vision) {
    size_t b = dw_bytes(&m->head);
    for (int li = 0; li < m->c.n_layers; li++) {
        const Layer *l = &m->L[li];
        b += dw_bytes(&l->qkv) + dw_bytes(&l->o);
        if (!m->c.moe[li]) b += dw_bytes(&l->gate) + dw_bytes(&l->up) + dw_bytes(&l->down);
    }
    if (g_vision && vision) {
        const Vision *v = g_vision;
        b += dw_bytes(&v->embed) + dw_bytes(&v->fc1) + dw_bytes(&v->fc2);
        for (int i = 0; i < v->depth; i++)
            b += dw_bytes(&v->b[i].qkv) + dw_bytes(&v->b[i].proj) + dw_bytes(&v->b[i].gate) +
                 dw_bytes(&v->b[i].up) + dw_bytes(&v->b[i].down);
    }
    return b;
}

/* ---- a partial chain (vk_chain.h, vkc_fit): the first n layers on the device --------
 * mc_fit_start decides n at start-up, before any upload and before the tier sizes its
 * budget. A layer's bytes: its matrices in their dense form (MIMO_DENSE_BITS: the fused
 * qkv and o_proj, the dense layer's MLP too), its K/V mirror at the size the chain
 * allocates (a sliding layer's ring; a full layer's context, or COLI_VK_KV_DEVICE_ROWS
 * when that asks for fewer: the split covers the rest) and its norms and sink logits.
 * Fixed: the final norm and the scratch of one block of vkc_fit_rows(MIMO_CHUNK) rows
 * (the residual's handoff reuses the normed rows' read-back). Tail: the head, and the
 * vision tower when the per-matrix path takes it (COLI_VK_DENSE). The CPU's layers, the
 * head and the tower of a partial chain stay on the CPU with their host copies (vk_off:
 * the per-matrix path leaves them there). */
static void mc_fit_start(Model *m) {
    Cfg *c = &m->c; int L = c->n_layers, H = c->hidden;
    for (int k = 0; k < 2; k++) if (c->head_dim[k] > 256 || c->v_dim[k] > 256) return;   /* the chain declines */
    size_t *per = calloc((size_t)L, sizeof *per), *mat = calloc((size_t)L, sizeof *mat);
    int ok = per && mat;
    long forced = vkc_kv_env("COLI_VK_KV_DEVICE_ROWS", 0);
#define MC_FITW(d) do { const DW *d_ = (d); int g_; const float *s_; int f_ = d_->w ? dw_vk_fmt(d_, &g_, &s_) : -1; \
        if (f_ < 0) ok = 0; else { per[i] += vkc_fit_tensor(f_, d_->I, d_->O, g_); mat[i] += coli_vk_tensor_payload(f_, d_->I, d_->O, g_); } } while (0)
    for (int i = 0; i < L && ok; i++) {
        const Layer *l = &m->L[i];
        MC_FITW(&l->qkv); MC_FITW(&l->o);
        if (!c->moe[i]) { MC_FITW(&l->gate); MC_FITW(&l->up); MC_FITW(&l->down); }
        McGeo g = mc_geo(m, i);
        size_t rows = !g.swa && forced > 0 && forced < g.rows ? (size_t)forced : (size_t)g.rows;
        per[i] += vkc_fit_buf(rows * g.kd * sizeof(float)) + vkc_fit_buf(rows * g.vdd * sizeof(float)) +
                  ((size_t)2 * H + (l->sink ? (size_t)c->heads[c->swa[i]] : 0)) * sizeof(float);
    }
#undef MC_FITW
    int hg; const float *hs;
    int hf = m->head.w ? dw_vk_fmt(&m->head, &hg, &hs) : -1;
    if (!ok || hf < 0) { free(per); free(mat); return; }   /* a matrix the device does not take: the chain declines as before */
    size_t tail = vkc_fit_tensor(hf, m->head.I, m->head.O, hg);
    if (g_vision && coli_vk_dense()) {   /* the tower the per-matrix path puts on the device */
        Vision *v = g_vision;
        const DW *t[3] = {&v->embed, &v->fc1, &v->fc2};
        for (int k = 0; k < 3; k++) if (t[k]->w && dw_vk_fmt(t[k], &hg, &hs) >= 0) tail += vkc_fit_tensor(dw_vk_fmt(t[k], &hg, &hs), t[k]->I, t[k]->O, hg);
        for (int b = 0; b < v->depth; b++) {
            const DW *d[5] = {&v->b[b].qkv, &v->b[b].proj, &v->b[b].gate, &v->b[b].up, &v->b[b].down};
            for (int k = 0; k < 5; k++) if (d[k]->w && dw_vk_fmt(d[k], &hg, &hs) >= 0) tail += vkc_fit_tensor(dw_vk_fmt(d[k], &hg, &hs), d[k]->I, d[k]->O, hg);
        }
    }
    MimoChain g; memset(&g, 0, sizeof g);
    g.o_kvd = calloc((size_t)L, sizeof(size_t));
    const char *mc = getenv("MIMO_CHUNK");
    int R = vkc_fit_rows(mc && *mc && atoi(mc) > 0 ? atoi(mc) : 64);
    g_mc_count = 0; g_mc_fitcount = 1;
    mc_scratch(&g, m, R, 1);
    size_t fixed = (size_t)g_mc_count + vkc_fit_buf((size_t)H * sizeof(float));
    g_mc_count = -1; g_mc_fitcount = 0;
    free(g.o_kvd);
    vkc_fit("mimo", L, per, mat, fixed, tail, &g_mc_fit);
    /* the layers the primary leaves, on COLI_VK_DEV2's device: a fit of its own from layer
     * n0 with that device's free memory, its pipelines up now (the head goes with it when it
     * ends at the last layer with room; the tower stays on the CPU) */
    int n0 = g_mc_fit.n;
    if (vkc_fit_partial(&g_mc_fit) && n0 > 0 && n0 < L && mc_dev2_wanted() && getenv("COLI_VK_DEV2") &&
        coli_vk_dev2_open_env()) {
        vkc_device(1);
        size_t tail2 = vkc_fit_tensor(hf, m->head.I, m->head.O, hg);
        int n2 = vkc_fit("mimo dev2", L - n0, per + n0, mat + n0, fixed, tail2, &g_mc_fit2);
        if (n2 > 0 && !vkc_init()) {
            fprintf(stderr, "[VK] mimo chain: the second device's pipelines did not come up; its layers stay on the CPU\n");
            n2 = 0;
        }
        g_mc_fit2_on = n2 > 0;
        if (!g_mc_fit2_on) g_mc_fit2.tail = 0;
        vkc_device(0);
    }
    free(per); free(mat);
}
/* Layers from..L-1, the head and the tower refused to the device: what of them is there
 * is freed, and the per-matrix path leaves them on the CPU. */
static void mc_refuse(DW *d) {
    if (d->vk && !d->vk_gone) { coli_vk_tensor_free((ColiVkTensor *)d->vk); d->vk = NULL; }
    if (!d->vk_gone) d->vk_off = 1;
}
static void mc_refuse_from(Model *m, int from) {
    if (vkc_ready()) vkc_finish();
    for (int li = from; li < m->c.n_layers; li++) {
        Layer *l = &m->L[li];
        mc_refuse(&l->qkv); mc_refuse(&l->o); mc_refuse(&l->gate); mc_refuse(&l->up); mc_refuse(&l->down);
    }
    if (!(m->head.vk && coli_vk_tensor_dev((ColiVkTensor *)m->head.vk)))   /* the second device's chain keeps it */
        mc_refuse(&m->head);
    if (g_vision) {
        Vision *v = g_vision;
        mc_refuse(&v->embed); mc_refuse(&v->fc1); mc_refuse(&v->fc2);
        for (int b = 0; b < v->depth; b++) {
            mc_refuse(&v->b[b].qkv); mc_refuse(&v->b[b].proj); mc_refuse(&v->b[b].gate); mc_refuse(&v->b[b].up); mc_refuse(&v->b[b].down);
        }
    }
}
static void dw_drop(DW *d, int *n);   /* below */
/* The fit's layers up, layer by layer: each whole or not at all (a layer that fails is
 * freed, it and the rest refused, vkc_fit_shrink); dropped: with COLI_VK_DENSE_HOST, each
 * layer's host copies given back once all of it is on the device (counted there). */
static void mc_place(Model *m, int *dropped) {
    int n0 = g_mc_fit.n;
    for (int i = 0; i < g_mc_fit.n; i++) {
        Layer *l = &m->L[i];
        DW *d[5] = {&l->qkv, &l->o, &l->gate, &l->up, &l->down};
        int nd = m->c.moe[i] ? 2 : 5, ok = 1;
        for (int k = 0; k < nd && ok; k++) ok = dw_upload(d[k]);
        if (!ok) {
            mc_refuse_from(m, i);
            vkc_fit_shrink("mimo", &g_mc_fit, i, "a dense matrix did not go up");
            break;
        }
        if (dropped) for (int k = 0; k < nd; k++) dw_drop(d[k], dropped);
        vkc_fit_mark(&g_mc_fit, i);
    }
    int end = g_mc_fit.n;
    if (g_mc_fit2_on && g_mc_fit.n < n0) {
        /* the primary placed fewer layers than its fit: the second device's would not follow
         * them, so they stay on the CPU too */
        fprintf(stderr, "[VK] mimo chain: the primary device stopped before layer %d; layers %d..%d stay on the CPU, "
                        "not on the second device\n", n0, n0, n0 + g_mc_fit2.n - 1);
        g_mc_fit2_on = 0;
        vkc_device(1); vkc_shutdown(); vkc_device(0);
    } else if (g_mc_fit2_on) {   /* the second device's layers (their host copies kept), and the head with the last one */
        vkc_device(1); g_mimo_vk_dev = 1;
        for (int i = n0; i < n0 + g_mc_fit2.n; i++) {
            Layer *l = &m->L[i];
            DW *d[5] = {&l->qkv, &l->o, &l->gate, &l->up, &l->down};
            int nd = m->c.moe[i] ? 2 : 5, ok = 1;
            for (int k = 0; k < nd && ok; k++) ok = dw_upload(d[k]);
            if (!ok) {
                g_mimo_vk_dev = 0;
                mc_refuse_from(m, i);
                vkc_fit_shrink("mimo dev2", &g_mc_fit2, i - n0, "a dense matrix did not go up");
                g_mimo_vk_dev = 1;
                break;
            }
            vkc_fit_mark(&g_mc_fit2, i - n0);
        }
        end = n0 + g_mc_fit2.n;
        if (end == m->c.n_layers && g_mc_fit2.tail && !dw_upload(&m->head)) g_mc_fit2.tail = 0;
        g_mimo_vk_dev = 0;
        vkc_fit_placed("mimo dev2", &g_mc_fit2);
        if (!g_mc_fit2.n) { g_mc_fit2_on = 0; vkc_shutdown(); }
        vkc_device(0);
    }
    if (vkc_fit_partial(&g_mc_fit)) mc_refuse_from(m, end);   /* the CPU's layers, the head (unless the second device's), the tower */
    vkc_fit_placed("mimo", &g_mc_fit);
    g_mc_placed = 1;
}

/* ---- the dense matrices on the device only (COLI_VK_DENSE_HOST) ----------------
 * With the trunk on the device (the chain, or COLI_VK_DENSE), every trunk matrix goes
 * up now, before the expert tier sizes its budget, and its host copy is dropped; the
 * tower too when the per-matrix path is on (COLI_VK_DENSE), else it stays on the CPU
 * with its copy. The per-matrix path then answers on the device whatever
 * coli_vk_dense() says (the CPU holds no copy), and the CPU reads a matrix back from
 * disk only when it needs it after all (dw_reload: a lost device). Kept on the host:
 * the embedding (its rows are gathered on the CPU), the routers (the host computes
 * them), norms and vectors, the patch embedding. */
static void dw_drop(DW *d, int *n) {
    if (!d->w || d->vk_off || d->vk_gone || !d->vk_name || !dw_upload(d)) return;
    size_t b = dw_bytes(d);
    free(d->w); free(d->s);
    d->w = NULL; d->s = NULL; d->vk_gone = 1;
    coli_vk_dense_host_dropped(b);
    (*n)++;
}
static void mimo_dho_start(Model *m) {
    if (!g_vk_ready) return;
    int vision = coli_vk_dense();
    /* a partial chain's fit: the first n layers' bytes only (the head and the tower with
     * the tail); n = 0 leaves the dense part on the CPU */
    size_t bytes = vk_dense_bytes(m, vision);
    int fit = g_mc_fit.L > 0, tail = !fit || (g_mc_fit.n == g_mc_fit.L && g_mc_fit.tail);
    if (fit) {
        bytes = 0;
        for (int li = 0; li < g_mc_fit.n; li++) {
            const Layer *l = &m->L[li];
            bytes += dw_bytes(&l->qkv) + dw_bytes(&l->o);
            if (!m->c.moe[li]) bytes += dw_bytes(&l->gate) + dw_bytes(&l->up) + dw_bytes(&l->down);
        }
        if (tail) bytes += vk_dense_bytes(m, vision) - vk_dense_bytes(m, 0) + dw_bytes(&m->head);
    }
    if (!coli_vk_dense_host_decide("mimo", (coli_vk_dense() || g_vk_chain) && !(fit && !g_mc_fit.n), bytes)) return;
    g_mimo_dho = 1;
    g_dw_dho_S = &m->S; g_dw_dho_c = &m->c;
    int n = 0;
    if (fit) {
        coli_vk_dense_host_layers(g_mc_fit.n, g_mc_fit.L);
        mc_place(m, &n);   /* each layer whole, then its host copies */
        tail = g_mc_fit.n == g_mc_fit.L && g_mc_fit.tail;
    } else
        for (int li = 0; li < m->c.n_layers; li++) {
            Layer *l = &m->L[li];
            dw_drop(&l->qkv, &n); dw_drop(&l->o, &n);
            if (!m->c.moe[li]) { dw_drop(&l->gate, &n); dw_drop(&l->up, &n); dw_drop(&l->down, &n); }
        }
    if (tail) dw_drop(&m->head, &n);
    if (g_vision && tail) {
        Vision *v = g_vision;
        DW *t[3] = {&v->embed, &v->fc1, &v->fc2};
        for (int k = 0; k < 3; k++) { if (vision) dw_drop(t[k], &n); else t[k]->vk_off = 1; }
        for (int i = 0; i < v->depth; i++) {
            DW *b[5] = {&v->b[i].qkv, &v->b[i].proj, &v->b[i].gate, &v->b[i].up, &v->b[i].down};
            for (int k = 0; k < 5; k++) { if (vision) dw_drop(b[k], &n); else b[k]->vk_off = 1; }
        }
    }
    coli_vk_dense_host_placed("mimo", vision || !g_vision
        ? "the embedding (its rows are gathered on the CPU), the routers, norms, the patch embedding"
        : "the embedding (its rows are gathered on the CPU), the routers, norms, the vision tower (on the CPU: COLI_VK_DENSE=1 puts it on the device)");
}

/* Is the expert in this layer's RAM cache now (the tier's balance asks)? */
static int vk_in_ram(void *ctx, int li, int e) {
    const Model *m = ctx;
    return m->cache[li].by_expert && m->cache[li].by_expert[e] >= 0;
}

/* The routed experts on the shared tier (vk_tier.c): the release's MXFP4 with its
 * e8m0 group scales, SwiGLU. This engine keeps no expert history, so the tier
 * starts empty and fills as experts pass by, evicting the coldest once its budget
 * is full. MIMO_VK_EXPERTS, the switch of the per-engine tier before this one, is
 * read as follows: 0 keeps the experts on the CPU (no tier), N > 0 sizes the
 * budget at N experts (COLI_VK_TIER_GB, which wins when set); unset, the tier's own
 * budget applies. */
static void vk_tier_start(Model *m) {
    Cfg *c = &m->c;
    VktFmt f = {VKT_SRC_MXFP4_E8M0, 32};
    const char *mx = getenv("MIMO_VK_EXPERTS"), *gb = getenv("COLI_VK_TIER_GB");
    if (mx && *mx && atoi(mx) > 0 && !(gb && *gb)) {
        size_t eb = vkt_expert_bytes(c->hidden, c->moe_inter, f, f);
        char buf[64];
        snprintf(buf, sizeof(buf), "%.12g", ((double)atoi(mx) * eb + eb / 2) / 1073741824.0);
        setenv("COLI_VK_TIER_GB", buf, 1);
        fprintf(stderr, "[VK] mimo: MIMO_VK_EXPERTS=%s read as a tier budget of %s experts (COLI_VK_TIER_GB=%s)\n",
                mx, mx, buf);
    }
    int cap = 0, nmoe = 0;
    for (int li = 0; li < c->n_layers; li++) if (c->moe[li]) { nmoe++; if (m->cache[li].cap > cap) cap = m->cache[li].cap; }
    VktConfig vc = {.engine = "mimo", .layers = c->n_layers, .experts = c->n_experts,
                    .hidden = c->hidden, .inter = c->moe_inter, .topk = c->topk,
                    .gate_up = f, .down = f, .act = VKT_ACT_SWIGLU,
                    .max_rows = MIMO_VK_ROWS * c->topk,
                    .ram_reserve = (size_t)(m->e_bytes + 8192) * (size_t)cap * (size_t)nmoe,
                    .dense_bytes = g_mc_fit.L && vkc_fit_partial(&g_mc_fit) ? mc_kv_bytes_n(m, g_mc_fit.n) :   /* a partial chain: placed already, its caches to come */
                                   (coli_vk_dense_device_only() ? 0 :   /* placed already (mimo_dho_start) */
                                    coli_vk_dense() ? vk_dense_bytes(m, 1) : g_vk_chain ? vk_dense_bytes(m, 0) : 0) +
                                   (g_vk_chain ? mc_kv_bytes(m) : 0),   /* the chain's KV caches */
                    .in_ram = vk_in_ram, .ram_ctx = m,
                    .load = mimo_vk_load, .release = mimo_vk_release, .load_ctx = m, .load_batch = mimo_vk_load_batch};
    atexit(coli_vk_shutdown);   /* before vkt_init, which makes the expert batch's pipelines and can still refuse (no room): the device goes at exit either way, after the tier's teardown */
    if (vkt_init(&vc, NULL)) atexit(vkt_shutdown);
}
#endif

/* --------------------------------------------------------------- sampling ---- */

static int argmax(const float *x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[best]) best = i;
    return best;
}

typedef struct { float p; int id; } SampleProb;
static int sample_desc(const void *a, const void *b) {
    float pa = ((const SampleProb *)a)->p, pb = ((const SampleProb *)b)->p;
    return (pb > pa) - (pa > pb);
}

static int sample(const float *logits, int vocab, float temperature, float top_p) {
    if (temperature <= 0.0f) return argmax(logits, vocab);
    SampleProb *rank = xmalloc((size_t)vocab * sizeof(SampleProb), "sampling");
    float top = logits[0];
    for (int i = 1; i < vocab; i++) if (logits[i] > top) top = logits[i];
    double total = 0.0;
    for (int i = 0; i < vocab; i++) {
        float p = expf((logits[i] - top) / temperature);
        total += p; rank[i].p = p; rank[i].id = i;
    }
    qsort(rank, (size_t)vocab, sizeof(SampleProb), sample_desc);
    double cut = (top_p > 0.0f && top_p < 1.0f) ? top_p * total : total, kept = 0.0;
    int n = 0;
    while (n < vocab && kept < cut) kept += rank[n++].p;
    double draw = ((double)rand() / RAND_MAX) * kept, acc = 0.0;
    int pick = rank[0].id;
    for (int i = 0; i < n; i++) { acc += rank[i].p; if (acc >= draw) { pick = rank[i].id; break; } }
    free(rank);
    return pick;
}

/* A draft d against the distribution sample() draws from (speculative sampling with a
 * draft that is one token): d stands with the probability sample() gives it, its share
 * of the nucleus; otherwise the token is drawn from the nucleus without d. Either way it
 * has sample()'s distribution. Greedy: the argmax, kept when it is d. */
static int sample_draft(const float *logits, int vocab, float temperature, float top_p, int d, int *kept) {
    if (temperature <= 0.0f) { int y = argmax(logits, vocab); *kept = y == d; return y; }
    SampleProb *rank = xmalloc((size_t)vocab * sizeof(SampleProb), "sampling");
    float top = logits[0];
    for (int i = 1; i < vocab; i++) if (logits[i] > top) top = logits[i];
    double total = 0.0;
    for (int i = 0; i < vocab; i++) {
        float p = expf((logits[i] - top) / temperature);
        total += p; rank[i].p = p; rank[i].id = i;
    }
    qsort(rank, (size_t)vocab, sizeof(SampleProb), sample_desc);
    double cut = (top_p > 0.0f && top_p < 1.0f) ? top_p * total : total, nucleus = 0.0, pd = 0.0;
    int n = 0;
    while (n < vocab && nucleus < cut) nucleus += rank[n++].p;
    for (int i = 0; i < n; i++) if (rank[i].id == d) { pd = rank[i].p; break; }
    int pick = d;
    *kept = ((double)rand() / RAND_MAX) * nucleus < pd;
    if (!*kept) {
        double draw = ((double)rand() / RAND_MAX) * (nucleus - pd), acc = 0.0;
        for (int i = 0; i < n; i++) {
            if (rank[i].id == d) continue;
            pick = rank[i].id;
            acc += rank[i].p;
            if (acc >= draw) break;
        }
    }
    free(rank);
    return pick;
}

/* ---------------------------------------------------------------- photos ---- */

/* SUBMIT pin=1: a photo of the state at the end of this prompt, so that the
 * prompts that begin with it (the options of a closed question) resume from
 * there instead of computing it again. pin_pool.h keeps several, nested, and
 * restores the deepest one that is a strict prefix of the new prompt.
 *
 * WHAT A LAYER CARRIES BETWEEN TOKENS, and so what the photo must hold:
 *
 *   full attention   K/V rows indexed by position. Nothing after the photo
 *                    writes below it, so they stay where they are, and the
 *                    photo holds only the token ids; kv_prefix_holds() is what
 *                    says they are still the photo's rows (an unrelated prompt
 *                    in between rewrites them from position 0).
 *   sliding window   a RING of `rows` slots, position p in slot p % rows, and
 *                    the position each slot holds. Every option overwrites the
 *                    ring slots of the photo's last positions -- its own token at
 *                    P lands where P - rows was -- so after one option the ring
 *                    no longer describes the photo. The ring, keys, values and
 *                    positions, is copied into the photo and copied back.
 *   the position     m->pos, which is the photo's length.
 *
 * Nothing else carries over: the sinks are weights, the router has no state,
 * and the expert cache changes the time, never the numbers. On Flash the rings
 * of the 39 windowed layers are 128 x 8 x (192 + 128) floats each, about 50 MB
 * a photo, so COLI_PIN_SLOTS (default 4) is what bounds the memory. */
typedef struct {
    int n_layers;
    int rows[MIMO_MAX_LAYERS];               /* 0: a full-attention layer */
    size_t kd[MIMO_MAX_LAYERS], vd[MIMO_MAX_LAYERS];
    float *K[MIMO_MAX_LAYERS], *V[MIMO_MAX_LAYERS];
    int *ring_pos[MIMO_MAX_LAYERS];
} PinState;

static ColiPinPool g_pins;

static void pin_state_free(void *v) {
    PinState *st = (PinState *)v;
    if (!st) return;
    for (int li = 0; li < st->n_layers; li++) { free(st->K[li]); free(st->V[li]); free(st->ring_pos[li]); }
    free(st);
}

/* Copies the rings into `st` (allocated on first use). NULL when memory runs
 * out: a photo is an optimisation and never the reason a request fails. */
static PinState *pin_state_save(Model *m, PinState *st) {
    Cfg *c = &m->c;
    if (st && st->n_layers != c->n_layers) { pin_state_free(st); st = NULL; }
    if (!st) {
        st = calloc(1, sizeof(*st));
        if (!st) return NULL;
        st->n_layers = c->n_layers;
        for (int li = 0; li < c->n_layers; li++) {
            if (!c->swa[li]) continue;
            Layer *l = &m->L[li];
            st->rows[li] = l->rows;
            st->kd[li] = (size_t)l->rows * c->kv_heads[1] * c->head_dim[1];
            st->vd[li] = (size_t)l->rows * c->kv_heads[1] * c->v_dim[1];
            st->K[li] = malloc(st->kd[li] * sizeof(float));
            st->V[li] = malloc(st->vd[li] * sizeof(float));
            st->ring_pos[li] = malloc((size_t)l->rows * sizeof(int));
            if (!st->K[li] || !st->V[li] || !st->ring_pos[li]) { pin_state_free(st); return NULL; }
        }
    }
    for (int li = 0; li < c->n_layers; li++) {
        if (!st->rows[li]) continue;
        Layer *l = &m->L[li];
        memcpy(st->K[li], l->K, st->kd[li] * sizeof(float));
        memcpy(st->V[li], l->V, st->vd[li] * sizeof(float));
        memcpy(st->ring_pos[li], l->ring_pos, (size_t)l->rows * sizeof(int));
    }
    return st;
}

static int pin_state_load(Model *m, const PinState *st) {
    Cfg *c = &m->c;
    if (!st || st->n_layers != c->n_layers) return 0;
    for (int li = 0; li < c->n_layers; li++)
        if (st->rows[li] != (c->swa[li] ? m->L[li].rows : 0)) return 0;
    for (int li = 0; li < c->n_layers; li++) {
        if (!st->rows[li]) continue;
        Layer *l = &m->L[li];
        memcpy(l->K, st->K[li], st->kd[li] * sizeof(float));
        memcpy(l->V, st->V[li], st->vd[li] * sizeof(float));
        memcpy(l->ring_pos, st->ring_pos[li], (size_t)l->rows * sizeof(int));
    }
#ifdef COLI_VULKAN
    mc_rings_rewritten(m);   /* the dense chain's copies of the rings go up again */
#endif
    return 1;
}

/* Only a request that asks for it takes a photo, and it goes into the pool,
 * where no other request writes: an ordinary prompt can make a photo stale
 * (it rewrites the rows the photo stands on), never replace it. */
static void pin_save(Model *m, const int *ids, int n, const float *logits) {
    coli_pin_pool_init(&g_pins, m->c.vocab);
    ColiPin *k = coli_pin_store(&g_pins, ids, n, logits);
    if (!k) return;
    PinState *st = pin_state_save(m, (PinState *)k->state);
    if (!st) { k->state = NULL; k->len = 0; return; }   /* without its rings it would lie */
    k->state = st;
    fprintf(stderr, "[PIN] photo of %d tokens\n", n);
    fflush(stderr);
}

/* The deepest photo that is a strict prefix of this prompt and whose rows the
 * state still holds, put back: the rings, the position and the record. Returns
 * its length and its logits (the predictor of the first fresh token), or 0.
 * A photo whose rows are gone cannot help any more: it is released, memory and
 * all, and the next deepest is tried instead of giving up. */
static int pin_restore(Model *m, const int *ids, int n, const float **logit) {
    *logit = NULL;
    int s = coli_pin_best(&g_pins, ids, n);
    while (s >= 0) {
        ColiPin *k = &g_pins.slot[s];
        if (k->logit && kv_prefix_holds(&m->kvp, k->ids, k->len) &&
            pin_state_load(m, (const PinState *)k->state)) {
            m->pos = k->len;
            kv_prefix_clear(&m->kvp);
            kv_prefix_record(&m->kvp, k->ids, 0, k->len);
            *logit = k->logit;
            coli_pin_touch(&g_pins, s);
            return k->len;
        }
        fprintf(stderr, "[PIN] released the photo of %d tokens: the state no longer holds it\n", k->len);
        k->len = 0;
        pin_state_free(k->state);
        k->state = NULL;
        s = coli_pin_best(&g_pins, ids, n);
    }
    return 0;
}

/* ------------------------------------------------------------------ serve ---- */

/* The 7th numeric field (the byte count of a payload extension) is accepted, but
 * only as 0: the gateway puts it ahead of the logprobs= key on chat requests, and
 * this engine takes no grammar, so any other value is still a bad header. */
static const ColiServeWireProfile mimo_wire = {
    .max_header_bytes = 511,
    .max_payload_bytes = 1u << 26,
    .max_extension_bytes = 0,
    .max_tokens = 1 << 20,
    .require_exact_lf = 1,
    .require_finite_sampling = 1,
    .allow_extension_bytes = 1,
};

static void serve_line(const char *format, ...) {
    va_list args;
    va_start(args, format);
    vfprintf(stdout, format, args);
    va_end(args);
    fflush(stdout);
}

/* the dashboard's grid: rows are the MoE layers, columns the routed experts */
static void serve_emap(Model *m) {
    Cfg *c = &m->c;
    int rows = 0;
    for (int li = 0; li < c->n_layers; li++) rows += c->moe[li];
    int cols = c->n_experts;
    char *hex = xmalloc((size_t)rows * cols * 2 + 1, "emap");
    int w = 0;
    for (int li = 0; li < c->n_layers; li++) {
        if (!c->moe[li]) continue;
        for (int e = 0; e < cols; e++) {
            int byte = (vkt_resident(li, e) ? 2 : m->cache[li].by_expert[e] >= 0) << 6;   /* 2 = on the Vulkan device */
            hex[w++] = "0123456789abcdef"[byte >> 4];
            hex[w++] = "0123456789abcdef"[byte & 15];
        }
    }
    hex[w] = 0;
    serve_line("EMAP %d %d %s\n", rows, cols, hex);
    free(hex);
}

static void serve_hits(Model *m) {
    Cfg *c = &m->c;
    int rows = 0;
    for (int li = 0; li < c->n_layers; li++) rows += c->moe[li];
    int cols = c->n_experts, nb = (rows * cols + 7) / 8, bit = 0;
    uint8_t *bitmap = xcalloc((size_t)nb, 1, "hits bitmap");
    for (int li = 0; li < c->n_layers; li++) {
        if (!c->moe[li]) continue;
        for (int e = 0; e < cols; e++, bit++)
            if (m->ehit[li][e]) { bitmap[bit >> 3] |= (uint8_t)(1 << (bit & 7)); m->ehit[li][e] = 0; }
    }
    char *hex = xmalloc((size_t)nb * 2 + 1, "hits hex");
    for (int b = 0; b < nb; b++) {
        hex[2 * b] = "0123456789abcdef"[bitmap[b] >> 4];
        hex[2 * b + 1] = "0123456789abcdef"[bitmap[b] & 15];
    }
    hex[2 * nb] = 0;
    serve_line("HITS %d %d %s\n", rows, cols, hex);
    free(hex); free(bitmap);
}

/* max_tokens is a ceiling; generation needs at least one free position. A
 * read-only request (max_tokens=0, legal only with logprobs) generates nothing,
 * so its prompt may fill the context. */
static int serve_budget(int prompt, int requested, int context, int logprobs) {
    if (prompt < 1 || prompt > context) return -1;
    int budget = requested > 0 ? requested : (logprobs > 0 ? 0 : 256);
    int room = context - prompt;
    if (budget > 0 && room == 0) return -1;
    return budget < room ? budget : room;
}

/* One verified step from `token` at P = m->pos, with at most `most` drafts (those whose
 * tokens could still go out; it only falls at a turn's last steps, so a block skipped
 * once is not run again before mtp_warm): each block's rows from its `next` (the tokens
 * after P are the drafts of the blocks before it), so block k drafts the token at
 * P + k + 1; the token and the drafts the blocks are sure enough of through the trunk as
 * rows into `rows`; then each of those against the row before it (sample_draft), up to
 * the first one that does not stand. Writes the step's tokens to out[] and returns how
 * many: the drafts kept, computed already (*done of them; when all stood, `logits`
 * predicts the token after them), then, when one did not, the token drawn instead,
 * still to run (the rows after it dropped). No room for a draft: the token's row alone,
 * nothing returned. */
static int mtp_step(Model *m, int token, int most, float *logits, float *rows,
                    float temperature, float top_p, int *out, int *done) {
    int V = m->c.vocab, P = m->pos, N = g_mtp.n, d[MIMO_MTP_MAX], tok[MIMO_MTP_MAX + 1];
    float conf[MIMO_MTP_MAX];
    if (N > most) N = most;
    if (N > m->ctx - P - 1) N = m->ctx - P - 1;
    *done = 0;
    if (N < 1) { prefill(m, &token, 1, logits, NULL, NULL, NULL); return 0; }
    g_mtp.tok[P & 7] = token;
    for (int k = 0; k < N; k++) {
        int p0 = g_mtp.b[k].next, n = P - p0;   /* at most N + 1 rows: what the last step kept */
        if (n > MIMO_MTP_MAX + 1) { fprintf(stderr, "[mimo] MTP block %d is %d rows behind\n", k, n); exit(1); }
        for (int t = 0; t < n; t++) {
            int j = p0 + t + k + 1;   /* row t's token: chosen, or a draft of this step */
            tok[t] = j <= P ? g_mtp.tok[j & 7] : d[j - P - 1];
        }
        d[k] = mtp_run(m, k, tok, p0, n, &conf[k]);
        int j = P + k + 1;
        if (g_mtp_oracle && j < g_mtp_oracle_n)   /* tests: the reference's token, or one beside it */
            d[k] = g_mtp_oracle_miss > 0 && ++g_mtp_oracle_drafts % g_mtp_oracle_miss == 0
                 ? (g_mtp_oracle[j] + 1) % V : g_mtp_oracle[j];
    }
    /* the drafts verified: the first K, while the probabilities the blocks give them
     * multiply to at least MIMO_MTP_MIN_P (a less likely one's row would mostly be dropped,
     * its experts computed for nothing) */
    int K = 0;
    for (float q = 1.0f; K < N && (q *= conf[K]) >= g_mtp.min_p; K++) {}
    int seq[MIMO_MTP_MAX + 1] = {token};
    memcpy(seq + 1, d, (size_t)K * sizeof(int));
    prefill(m, seq, K + 1, NULL, rows, NULL, NULL);
    g_mtp.steps++;
    int a = 0, kept = 0, y = -1;
    while (a < K) {
        y = sample_draft(rows + (size_t)a * V, V, temperature, top_p, d[a], &kept);
        g_mtp.b[a].tried++;
        if (!kept) break;
        g_mtp.b[a].kept++;
        g_mtp.tok[(P + a + 1) & 7] = out[a] = d[a];
        a++;
    }
    if (g_mtp.trace) {   /* MIMO_MTP_TRACE: the position, drafts made, verified and kept, their probabilities */
        fprintf(g_mtp.trace, "%d %d %d %d", P, N, K, a);
        for (int k = 0; k < N; k++) fprintf(g_mtp.trace, " %.5f", conf[k]);
        fprintf(g_mtp.trace, "\n");
        fflush(g_mtp.trace);
    }
    /* the rows a block ran with a token that is not the one chosen run again */
    for (int k = 0; k < N; k++) g_mtp.b[k].next = P + a - k < P ? P + a - k : P;
    memcpy(logits, rows + (size_t)a * V, (size_t)V * sizeof(float));
    *done = a;
    if (a == K) return K;
    mtp_back(m, P + a + 1);
    out[a] = y;
    return a + 1;
}

/* When a request was accepted, and the counters then: DONE and PROF report its share. */
typedef struct {
    double started, disk0, expert0, attn0, pred0, qkv0, oproj0;
    double prefilled, pdisk, pexpert, pattn;   /* the counters when the prompt was in */
    uint64_t pmiss, ppfi, waves0, wexp0;
    uint64_t hits0, miss0, fw0, pfi0, pfu0, pfw0, pp0, pph0, br0;
    uint64_t mtps0, mtpd0[MIMO_MTP_MAX], mtpk0[MIMO_MTP_MAX];   /* MTP steps, each block's drafts tried and kept */
    double mtpt0;                                                /* and the drafting time */
    int n_prompt, budget, reuse;
} MimoReq;

/* The streaming's share of a request, on stderr: demand reads (the stalls PROF
 * counts as disk time), the prefetcher's reads and how many were asked for, and how
 * often a step's experts were the ones predicted for it. */
static void mimo_stream_report(const Model *m, const MimoReq *rq) {
    uint64_t pi = m->pf_issued - rq->pfi0, pu = m->pf_used - rq->pfu0, pw = m->pf_wasted - rq->pfw0,
             pp = m->pf_pred - rq->pp0, ph = m->pf_pred_hit - rq->pph0;
    fprintf(stderr, "[mimo] streaming: %llu demand reads, %llu prefetched (%llu used, %llu unused), "
                    "predicted %.1f%% of %llu choices, %.2f GB read, %.2fs stalled, %.2fs predicting; "
                    "attention matmuls: qkv %.2fs, o_proj %.2fs; %.2f experts a wave\n",
            (unsigned long long)(m->miss - rq->miss0), (unsigned long long)pi, (unsigned long long)pu,
            (unsigned long long)pw, pp ? 100.0 * ph / pp : 0.0, (unsigned long long)pp,
            (m->bytes_read - rq->br0) / 1e9, m->t_disk - rq->disk0, m->t_pred - rq->pred0,
            m->t_qkv - rq->qkv0, m->t_oproj - rq->oproj0,
            m->waves > rq->waves0 ? (double)(m->wave_experts - rq->wexp0) / (double)(m->waves - rq->waves0) : 0.0);
    if (g_mtp.steps > rq->mtps0) {
        /* block k's drafts are tried in the steps whose drafts before it stood */
        uint64_t steps = g_mtp.steps - rq->mtps0, kept = 0;
        char per[256] = "";
        for (int k = 0, at = 0; k < g_mtp.n; k++) {
            uint64_t s = g_mtp.b[k].kept - rq->mtpk0[k];
            kept += s;
            if (g_mtp.n > 1)
                at += snprintf(per + at, sizeof per - at, "%s%llu of %llu", k ? ", " : " (", (unsigned long long)s,
                               (unsigned long long)(g_mtp.b[k].tried - rq->mtpd0[k]));
        }
        uint64_t tried = g_mtp.b[0].tried - rq->mtpd0[0];   /* a step may verify none (MIMO_MTP_MIN_P) */
        if (g_mtp.n == 1)
            fprintf(stderr, "[mimo] MTP: %llu drafts, %llu kept (%.1f%%), %.2fs drafting\n", (unsigned long long)tried,
                    (unsigned long long)kept, tried ? 100.0 * (double)kept / (double)tried : 0.0, g_mtp.t - rq->mtpt0);
        else
            fprintf(stderr, "[mimo] MTP: %llu steps of up to %d drafts, %llu kept%s), %.2f tokens a step, %.2fs drafting\n",
                    (unsigned long long)steps, g_mtp.n, (unsigned long long)kept, per,
                    1.0 + (double)kept / (double)steps, g_mtp.t - rq->mtpt0);
    }
    if (rq->prefilled > 0)
        fprintf(stderr, "[mimo] prompt %.2fs (stalled %.2fs, experts %.2fs, attention %.2fs, %llu demand reads); "
                        "generation %.2fs (stalled %.2fs, experts %.2fs, attention %.2fs, %llu demand reads, %llu prefetched)\n",
                rq->prefilled - rq->started, rq->pdisk - rq->disk0, rq->pexpert - rq->expert0, rq->pattn - rq->attn0,
                (unsigned long long)(rq->pmiss - rq->miss0),
                now_s() - rq->prefilled, m->t_disk - rq->pdisk, m->t_expert - rq->pexpert, m->t_attn - rq->pattn,
                (unsigned long long)(m->miss - rq->pmiss), (unsigned long long)(m->pf_issued - rq->ppfi));
    fflush(stderr);
}

/* A request's prompt into the caches the Model's layers hold: its tokens, the budget,
 * the image, the prefix reuse and the photos, ACCEPT, the prefill and its read-out.
 * 1 with the logits after the prompt; 0 when the request ended here, its ERROR written
 * and its command disposed. serve_loop and serve_mux start every request here. */
static int mimo_serve_start(Model *m, Tok *tokenizer, ColiServeCommand *cmd, int *ids, float *logits,
                            float **pending, int *pending_h, int *pending_w, MimoReq *rq) {
    Cfg *c = &m->c;
    double started = now_s();
    double disk0 = m->t_disk, expert0 = m->t_expert, attn0 = m->t_attn;
    uint64_t hits0 = m->hits, miss0 = m->miss, fw0 = m->forwards;
    rq->pred0 = m->t_pred; rq->pfi0 = m->pf_issued; rq->pfu0 = m->pf_used; rq->pfw0 = m->pf_wasted;
    rq->pp0 = m->pf_pred; rq->pph0 = m->pf_pred_hit; rq->br0 = m->bytes_read;
    rq->qkv0 = m->t_qkv; rq->oproj0 = m->t_oproj; rq->waves0 = m->waves; rq->wexp0 = m->wave_experts;
    rq->mtps0 = g_mtp.steps; rq->mtpt0 = g_mtp.t;
    for (int k = 0; k < MIMO_MTP_MAX; k++) { rq->mtpd0[k] = g_mtp.b[k].tried; rq->mtpk0[k] = g_mtp.b[k].kept; }
    int n_prompt = tok_encode(tokenizer, (const char *)cmd->payload, (int)cmd->payload_bytes,
                              ids, m->ctx + 1);
    int budget = serve_budget(n_prompt, cmd->max_tokens, m->ctx, cmd->logprobs);
    if (budget < 0) {
        char message[160];
        snprintf(message, sizeof(message), "CONTEXT_EXCEEDED prompt_tokens=%d requested=%d capacity=%d",
                 n_prompt, cmd->max_tokens, m->ctx);
        coli_serve_write_error(stdout, cmd->id, n_prompt < 1 ? "EMPTY_PROMPT" : message);
        free((*pending)); (*pending) = NULL;
        coli_serve_command_dispose(cmd);
        return 0;
    }
    /* An image: its rows replace the pad ids, which must be exactly as many
     * as the tower produces. Checked before ACCEPT so a mismatch is a clean
     * 400, not a half-answered turn. */
    float *image = NULL;
    int image_rows = 0;
    if ((*pending)) {
        int pads = 0;
        for (int t = 0; t < n_prompt; t++) pads += ids[t] == c->image_token_id;
        int want = ((*pending_h) / 2) * ((*pending_w) / 2);
        if (pads != want) {
            char message[160];
            snprintf(message, sizeof(message), "BAD_IMAGE prompt has %d image pads, the %dx%d grid needs %d",
                     pads, (*pending_h), (*pending_w), want);
            coli_serve_write_error(stdout, cmd->id, message);
            free((*pending)); (*pending) = NULL;
            coli_serve_command_dispose(cmd);
            return 0;
        }
    }
    /* Where this prompt starts. A chat client resends the whole transcript:
     * if this prompt begins with the ids the state was built from, only the
     * tail is fed (COLI_KV_PREFIX=0 turns that off). A photo (SUBMIT pin=1
     * earlier) resumes a prompt that begins with it.
     *
     * With the read-out on (logprobs=k) only a photo will do. The read-out
     * owes a frame to every position it does not inherit a predictor for,
     * and a live prefix comes with none: the frames before it would simply
     * be missing. So such a prompt resumes from the deepest photo, which
     * carries the logits that predict its first fresh token, or is read out
     * from position 0. It resumes at the photo even when the live state
     * shares more: two options share the text before them, and stopping at
     * that would leave the first option token without its predictor.
     *
     * An image refuses both: the pad ids do not describe the picture. */
    int prefix_on = env_int("COLI_KV_PREFIX", 1) != 0;
    int reuse = 0;
    const float *pin_logit = NULL;
    if (!(*pending)) {
        if (cmd->logprobs > 0) reuse = pin_restore(m, ids, n_prompt, &pin_logit);
        else {
            reuse = prefix_on ? kv_prefix_reuse(&m->kvp, ids, n_prompt) : 0;
            if (!reuse) reuse = pin_restore(m, ids, n_prompt, &pin_logit);
        }
    }
    if (getenv("COLI_PREFIX_LOG"))
        fprintf(stderr, "[PREFIX] %s %d of %d prompt tokens%s\n", reuse ? "reusing" : "no reuse,", reuse,
                n_prompt, pin_logit ? " (photo)" : "");
    if (pin_logit) {
        fprintf(stderr, "[PIN] resumed from the photo of %d tokens, %d to prefill\n", reuse, n_prompt - reuse);
        fflush(stderr);
    }
    if (!reuse) model_reset(m);
    coli_serve_write_accept(stdout, cmd->id, n_prompt);
    if ((*pending)) {
        image = vision_run(m, (*pending), (*pending_h), (*pending_w), &image_rows);
        free((*pending)); (*pending) = NULL;
    }
    ImageRows img = { image, image_rows };
    Echo echo = { cmd->id, cmd->logprobs, tokenizer, pin_logit };
    prefill(m, ids + reuse, n_prompt - reuse, logits, NULL, image ? &img : NULL,
            cmd->logprobs > 0 ? &echo : NULL);
    if (image) kv_prefix_taint(&m->kvp);
    if (cmd->pin) {
        /* a photo is described by its ids, and these do not describe the picture */
        if (image) fprintf(stderr, "[PIN] no photo: the prompt carries a picture\n");
        else pin_save(m, ids, n_prompt, logits);
    }
    free(image);
    rq->started = started; rq->disk0 = disk0; rq->expert0 = expert0; rq->attn0 = attn0;
    rq->hits0 = hits0; rq->miss0 = miss0; rq->fw0 = fw0; rq->n_prompt = n_prompt; rq->budget = budget;
    rq->reuse = reuse;
    rq->prefilled = now_s(); rq->pdisk = m->t_disk; rq->pexpert = m->t_expert; rq->pattn = m->t_attn;
    rq->pmiss = m->miss; rq->ppfi = m->pf_issued;
    return 1;
}

/* ---- several conversations at once (KV_SLOTS>1) ---------------------------------
 * The gateway's cache slots, each a conversation with caches of its own (MimoSeq). A
 * SUBMIT on a free slot starts its request at once through mimo_serve_start, on that
 * slot's caches: its prefix reuse and photos work as a lone serve's. Then every step
 * picks the next token of each active request and runs one forward over a row of each
 * (forward_rows): the matrices and the experts are read once for all of them. A
 * request's frames are a lone request's; they interleave by id. As alone, STOP ends a
 * request with DONE and CANCEL with ERROR CANCELLED. */
static MimoSeq *g_mimo_mux_seq;   /* [slots]: the conversations the Model does not hold */
static int g_mimo_mux_cur;        /* the slot the Model holds, -1 when every one is parked */

typedef struct {
    char id[COLI_SERVE_ID_CAP];
    float temperature, top_p;
    int logprobs, active, stop, cancel, limited, emitted;
    float *logits;                /* the logits the next pick reads */
    MimoReq rq;
} MimoMuxReq;

static void mimo_mux_bind(Model *m, int slot) {
    if (g_mimo_mux_cur == slot) return;
    if (g_mimo_mux_cur >= 0) mimo_seq_swap(m, &g_mimo_mux_seq[g_mimo_mux_cur]);
    if (slot >= 0) mimo_seq_swap(m, &g_mimo_mux_seq[slot]);
    g_mimo_mux_cur = slot;
}

/* A request's end, as serve_loop ends one. */
static void mimo_mux_finish(Model *m, MimoMuxReq *r) {
    r->active = 0;
    if (r->cancel) { coli_serve_write_error(stdout, r->id, "CANCELLED"); return; }
    if (r->stop) r->limited = 0;
    double wall = now_s() - r->rq.started;
    uint64_t th = m->hits - r->rq.hits0, tm = m->miss - r->rq.miss0;
    ColiServeDone done = { r->emitted, wall > 0 ? r->emitted / wall : 0.0,
                           (th + tm) ? 100.0 * th / (double)(th + tm) : 0.0, rss_gb(), r->rq.n_prompt, r->limited };
    coli_serve_write_done(stdout, r->id, &done);
    serve_line("PROF %.3f %d %d %.3f %.3f %.3f %.3f %.3f %llu\n", wall, r->rq.n_prompt, r->emitted,
               m->t_disk - r->rq.disk0, 0.0, m->t_expert - r->rq.expert0, m->t_attn - r->rq.attn0, 0.0,
               (unsigned long long)(m->forwards - r->rq.fw0));
    mimo_stream_report(m, &r->rq);
    serve_hits(m);
}

/* The next token of an active request, as serve_loop picks and sends it: 1 with the
 * token when the request goes on, 0 when it ended. */
static int mimo_mux_pick(Model *m, Tok *tokenizer, MimoMuxReq *r, const int *eos, int n_eos, int *tk_out) {
    Cfg *c = &m->c;
    if (r->cancel || r->stop || r->emitted >= r->rq.budget) { mimo_mux_finish(m, r); return 0; }
    int token = sample(r->logits, c->vocab, r->temperature, r->top_p);
    for (int i = 0; i < n_eos; i++) if (token == eos[i]) { r->limited = 0; mimo_mux_finish(m, r); return 0; }
    char piece[512];
    int written = tok_decode(tokenizer, &token, 1, piece, (int)sizeof(piece));
    if (written > 0) {
        if (r->logprobs > 0) {
            char tail[1024];
            coli_logprob_tail(tail, sizeof(tail), r->logits, c->vocab, token, r->logprobs);
            coli_serve_write_data_lp(stdout, r->id, piece, (size_t)written, tail);
        } else coli_serve_write_data(stdout, r->id, piece, (size_t)written);
    }
    r->emitted++;
    if (r->emitted >= r->rq.budget) { mimo_mux_finish(m, r); return 0; }
    *tk_out = token; return 1;
}

static void serve_mux(Model *m, Tok *tokenizer, const char *dir) {
    Cfg *c = &m->c;
    int n = g_mimo_mux_slots, V = c->vocab, input_eof = 0;
    int eos[8];
    int n_eos = coli_load_stop_ids(dir, eos, 8, NULL);
    coli_serve_write_ready_caps(stdout, rss_gb(), g_vision ? "vision=1" : "vision=0");
    serve_emap(m);
    MimoMuxReq *rq = xcalloc((size_t)n, sizeof *rq, "requests");
    MimoRow *rows = xmalloc((size_t)n * sizeof *rows, "rows");
    int *tok = xmalloc((size_t)n * sizeof(int), "tokens"), *who = xmalloc((size_t)n * sizeof(int), "slots");
    float *lo = xmalloc((size_t)n * V * sizeof(float), "step logits");
    int *ids = xmalloc(((size_t)m->ctx + 1) * sizeof(int), "prompt ids");
    for (int i = 0; i < n; i++) rq[i].logits = xmalloc((size_t)V * sizeof(float), "logits");
    float *pending = NULL;
    int pending_h = 0, pending_w = 0;
    unsigned long long steps = 0, nrows = 0;
    fprintf(stderr, "[mimo] serving %d conversations at once (KV_SLOTS)\n", n);
    for (;;) {
        int active = 0; for (int i = 0; i < n; i++) active += rq[i].active;
        /* idle: wait for a command; decoding: take one only when one is there */
        if (!input_eof && (!active || coli_serve_stdin_ready())) {
            ColiServeCommand command;
            ColiServeReadResult result = coli_serve_read_command(stdin, &mimo_wire, &command);
            if (result == COLI_SERVE_READ_EOF || result == COLI_SERVE_READ_BAD_FRAME) input_eof = 1;
            else if (result == COLI_SERVE_READ_NOMEM) { coli_serve_write_error(stdout, command.id, "out of memory"); input_eof = 1; }
            else if (result == COLI_SERVE_READ_BAD_REQUEST) {
                if (command.kind == COLI_SERVE_COMMAND_SUBMIT) coli_serve_write_error(stdout, command.id, "bad submit header");
                coli_serve_command_dispose(&command);
            } else if (command.kind == COLI_SERVE_COMMAND_IMAGE) {
                /* one image waits for the SUBMIT that names it; a second replaces it */
                uint64_t expected = g_vision ? (uint64_t)command.grid_h * command.grid_w * g_vision->patch_in * sizeof(float) : 0;
                if (!g_vision) coli_serve_write_error(stdout, command.id, "this container has no vision tower");
                else if (command.grid_h < 2 || command.grid_w < 2 || command.grid_h % 2 || command.grid_w % 2 ||
                         command.payload_bytes != expected)
                    coli_serve_write_error(stdout, command.id, "BAD_IMAGE payload does not match its grid");
                else {
                    free(pending);
                    pending = (float *)coli_serve_command_take_payload(&command);
                    pending_h = command.grid_h; pending_w = command.grid_w;
                }
                coli_serve_command_dispose(&command);
            } else if (command.kind == COLI_SERVE_COMMAND_CANCEL || command.kind == COLI_SERVE_COMMAND_STOP) {
                int found = 0;
                for (int i = 0; i < n; i++) if (rq[i].active && !strcmp(rq[i].id, command.id)) {
                    found = 1;
                    if (command.kind == COLI_SERVE_COMMAND_STOP) rq[i].stop = 1; else rq[i].cancel = 1;
                }
                if (!found && command.kind == COLI_SERVE_COMMAND_CANCEL) coli_serve_write_error(stdout, command.id, "NOT_FOUND");
                coli_serve_command_dispose(&command);
            } else if (command.kind != COLI_SERVE_COMMAND_SUBMIT) {
                coli_serve_command_dispose(&command);
            } else if (command.slot < 0 || command.slot >= n || rq[command.slot].active) {
                coli_serve_write_error(stdout, command.id, command.slot < 0 || command.slot >= n ? "invalid cache slot" : "SLOT_BUSY");
                coli_serve_command_dispose(&command);
            } else {
                MimoMuxReq *t = &rq[command.slot];
                float *keep = t->logits;
                memset(t, 0, sizeof *t); t->logits = keep;
                snprintf(t->id, sizeof t->id, "%s", command.id);
                t->temperature = command.temperature; t->top_p = command.top_p; t->logprobs = command.logprobs;
                mimo_mux_bind(m, command.slot);
                if (mimo_serve_start(m, tokenizer, &command, ids, t->logits, &pending, &pending_h, &pending_w, &t->rq)) {
                    t->active = 1; t->limited = 1;
                    coli_serve_command_dispose(&command);
                }
            }
        }
        active = 0; for (int i = 0; i < n; i++) active += rq[i].active;
        if (!active) { if (input_eof) break; continue; }
        int S = 0, ended = 0;
        for (int i = 0; i < n; i++) if (rq[i].active) {
            int tk;
            if (!mimo_mux_pick(m, tokenizer, &rq[i], eos, n_eos, &tk)) { ended = 1; continue; }
            rows[S] = (MimoRow){&g_mimo_mux_seq[i], g_mimo_mux_cur == i ? m->pos : g_mimo_mux_seq[i].pos};
            tok[S] = tk; who[S] = i; S++;
        }
        if (S) {
            mimo_mux_bind(m, -1);   /* every conversation parked: the rows read theirs */
            for (int s = 0; s < S; s++) rows[s].pos = g_mimo_mux_seq[who[s]].pos;
            forward_rows(m, rows, tok, S, lo);
            steps++; nrows += (unsigned long long)S;
            for (int s = 0; s < S; s++) memcpy(rq[who[s]].logits, lo + (size_t)s * V, (size_t)V * sizeof(float));
        }
        if (ended) {
            serve_emap(m);
            vkt_report("turn", m->hits, m->miss);
        }
    }
    fprintf(stderr, "[mimo] KV_SLOTS=%d: %llu decode steps, %llu rows (%.2f a step)\n", n, steps, nrows,
            steps ? (double)nrows / (double)steps : 0.0);
    mimo_mux_bind(m, 0);
    for (int i = 0; i < n; i++) free(rq[i].logits);
    free(rq); free(rows); free(tok); free(who); free(lo); free(ids); free(pending);
}

static void serve_loop(Model *m, Tok *tokenizer, const char *dir) {
    if (g_mimo_mux_slots > 1) { serve_mux(m, tokenizer, dir); return; }
    Cfg *c = &m->c;
    coli_serve_stdio_init();
    int eos[8];
    int n_eos = coli_load_stop_ids(dir, eos, 8, NULL);
    /* CAPS: the gateway offers pictures only when the tower actually loaded */
    coli_serve_write_ready_caps(stdout, rss_gb(), g_vision ? "vision=1" : "vision=0");
    serve_emap(m);
    float *logits = xmalloc((size_t)c->vocab * sizeof(float), "logits");
    int *ids = xmalloc(((size_t)m->ctx + 1) * sizeof(int), "prompt ids");
    float *pending = NULL;
    int pending_h = 0, pending_w = 0;
    for (;;) {
        ColiServeCommand command;
        ColiServeReadResult result = coli_serve_read_command(stdin, &mimo_wire, &command);
        if (result == COLI_SERVE_READ_EOF || result == COLI_SERVE_READ_BAD_FRAME) break;
        if (result == COLI_SERVE_READ_NOMEM) { coli_serve_write_error(stdout, command.id, "out of memory"); break; }
        if (result == COLI_SERVE_READ_BAD_REQUEST) {
            if (command.kind == COLI_SERVE_COMMAND_SUBMIT) coli_serve_write_error(stdout, command.id, "bad submit header");
            coli_serve_command_dispose(&command);
            continue;
        }
        if (command.kind == COLI_SERVE_COMMAND_IMAGE) {
            /* One image waits for the SUBMIT that names it; a second replaces it. */
            if (!g_vision) {
                coli_serve_write_error(stdout, command.id, "this container has no vision tower");
                coli_serve_command_dispose(&command);
                continue;
            }
            uint64_t expected = (uint64_t)command.grid_h * command.grid_w * g_vision->patch_in * sizeof(float);
            if (command.grid_h < 2 || command.grid_w < 2 || command.grid_h % 2 || command.grid_w % 2 ||
                command.payload_bytes != expected) {
                coli_serve_write_error(stdout, command.id, "BAD_IMAGE payload does not match its grid");
                coli_serve_command_dispose(&command);
                continue;
            }
            free(pending);
            pending = (float *)coli_serve_command_take_payload(&command);
            pending_h = command.grid_h; pending_w = command.grid_w;
            coli_serve_command_dispose(&command);
            continue;
        }
        if (command.kind != COLI_SERVE_COMMAND_SUBMIT) {
            if (command.kind == COLI_SERVE_COMMAND_CANCEL) coli_serve_write_error(stdout, command.id, "NOT_FOUND");
            coli_serve_command_dispose(&command);
            continue;
        }
        MimoReq rq;
        if (!mimo_serve_start(m, tokenizer, &command, ids, logits, &pending, &pending_h, &pending_w, &rq)) continue;
        double started = rq.started;
        double disk0 = rq.disk0, expert0 = rq.expert0, attn0 = rq.attn0;
        uint64_t hits0 = rq.hits0, miss0 = rq.miss0, fw0 = rq.fw0;
        int n_prompt = rq.n_prompt, budget = rq.budget;
        int emitted = 0, limited = 1, cancelled = 0, done_early = 0;
        char piece[512];
        /* MIMO_MTP: a turn without a read-out verifies drafts each step; the tokens the step
         * chose wait in `queue`, its first `computed` already through the trunk */
        int spec = g_mtp.on && command.logprobs <= 0, warm = 0, queue[MIMO_MTP_MAX + 1], qn = 0, qi = 0, computed = 0;
        float *rows = spec ? xmalloc((size_t)(g_mtp.n + 1) * c->vocab * sizeof(float), "verified logits") : NULL;
        while (emitted < budget && !cancelled && !done_early) {
            int token = qi < qn ? queue[qi] : sample(logits, c->vocab, command.temperature, command.top_p);
            int done = qi < computed;
            if (qi < qn) qi++;
            int stop = 0;
            for (int i = 0; i < n_eos; i++) if (token == eos[i]) stop = 1;
            if (stop) { limited = 0; break; }
            int written = tok_decode(tokenizer, &token, 1, piece, (int)sizeof(piece));
            if (written > 0) {
                if (command.logprobs > 0) {
                    /* the distribution the token was drawn from: sample() only reads it */
                    char tail[1024];
                    coli_logprob_tail(tail, sizeof(tail), logits, c->vocab, token, command.logprobs);
                    coli_serve_write_data_lp(stdout, command.id, piece, (size_t)written, tail);
                } else coli_serve_write_data(stdout, command.id, piece, (size_t)written);
            }
            emitted++;
            while (coli_serve_stdin_ready()) {
                ColiServeCommand control;
                ColiServeReadResult inner = coli_serve_read_command(stdin, &mimo_wire, &control);
                if (inner == COLI_SERVE_READ_EOF) { done_early = 1; break; }
                if (control.kind == COLI_SERVE_COMMAND_CANCEL && !strcmp(control.id, command.id)) cancelled = 1;
                else if (control.kind == COLI_SERVE_COMMAND_STOP && !strcmp(control.id, command.id)) { limited = 0; done_early = 1; }
                else if (control.kind == COLI_SERVE_COMMAND_SUBMIT) coli_serve_write_error(stdout, control.id, "SLOT_BUSY");
                coli_serve_command_dispose(&control);
            }
            if (cancelled || done_early || emitted >= budget) break;
            if (!spec) prefill(m, &token, 1, logits, NULL, NULL, NULL);
            else if (!done) {
                int most = budget - emitted - 1;   /* the drafts whose tokens could still go out */
                if (most > 0 && !warm++) mtp_warm(m, ids, rq.reuse, n_prompt);   /* after the first token is out */
                qn = mtp_step(m, token, most, logits, rows, command.temperature, command.top_p, queue, &computed);
                qi = 0;
            }
        }
        free(rows);
        if (cancelled) {
            coli_serve_write_error(stdout, command.id, "CANCELLED");
            coli_serve_command_dispose(&command);
            continue;
        }
        double wall = now_s() - started;
        uint64_t th = m->hits - hits0, tm = m->miss - miss0;
        ColiServeDone done = { emitted, wall > 0 ? emitted / wall : 0.0,
                               (th + tm) ? 100.0 * th / (double)(th + tm) : 0.0, rss_gb(), n_prompt, limited };
        coli_serve_write_done(stdout, command.id, &done);
        serve_line("PROF %.3f %d %d %.3f %.3f %.3f %.3f %.3f %llu\n", wall, n_prompt, emitted,
                   m->t_disk - disk0, 0.0, m->t_expert - expert0, m->t_attn - attn0, 0.0,
                   (unsigned long long)(m->forwards - fw0));
        mimo_stream_report(m, &rq);
        serve_hits(m);
        serve_emap(m);
        vkt_report("turn", m->hits, m->miss);   /* the Vulkan tier's line, when it runs */
#ifdef COLI_VULKAN
        mc_report(m, "turn");
#endif
        coli_serve_command_dispose(&command);
    }
    free(ids); free(logits); free(pending);
}

/* ------------------------------------------------------------------- main ---- */

/* Once, at the end: how many products the GPU actually took, so a run that
 * asked for Vulkan and quietly stayed on the CPU says so. */
static void vk_report(const Model *m) {
#ifdef COLI_VULKAN
    if (!g_vk_ready) return;
    fprintf(stderr, "[VK] mimo: %llu matmuls on the GPU\n", coli_vk_matmul_calls());
    vkt_report("run", m->hits, m->miss);
    mc_report((Model *)m, "run");
#else
    (void)m;
#endif
}

static int parse_ids(const char *text, int *out, int cap) {
    int n = 0;
    const char *p = text;
    while (*p && n < cap) {
        while (*p == ' ' || *p == ',' || *p == '\t' || *p == '\n') p++;
        if (!*p) break;
        char *end;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        out[n++] = (int)v;
        p = end;
    }
    return n;
}

static int is_dir(const char *path) {
    struct stat sb;
    return stat(path, &sb) == 0 && S_ISDIR(sb.st_mode);
}

#ifdef COLI_MIMO_CUDA
static double mimo_cuda_gb(const char *key, double fallback) {
    const char *v = getenv(key);
    if (!v || !*v) return fallback;
    char *end; double n = strtod(v, &end);
    if (*end || !isfinite(n) || n < 0 || n > 1048576) {
        fprintf(stderr, "[mimo] %s must be a finite nonnegative GiB value\n", key); exit(2);
    }
    return n;
}
/* MIMO_CUDA_BF16_PACK (default 1): a BF16 matrix with rows a multiple of 256 (the head,
 * every o_proj) goes up packed without loss -- 12 bits a weight, the kernel computing the
 * bytes of the plain bf16 one (backend_cuda_mimo.cu, fmt 5) -- so about a third more of
 * them fit. 0: plain bf16. */
static int g_mimo_cuda_pack = 1, g_mimo_cuda_packed;
static void mimo_cuda_place(DW *d, size_t *budget, int *count) {
    if (!d->w || d->O < 1 || d->I < 1) return;
    size_t b = dw_bytes(d), used = b;
    if (g_mimo_cuda_pack && d->fmt == DW_BF16 && d->I % 256 == 0) {
        size_t est = (size_t)d->O * d->I * 3 / 2 + (size_t)d->O * 5;
        if (est > *budget) return;
        d->cuda = mimo_cuda_upload(d->w, NULL, 5, d->I, d->O);
        if (d->cuda) {
            used = mimo_cuda_bytes(d->cuda);
            if (!used) used = est;
            if (used > *budget) used = *budget;   /* its exceptions went past: the budget is spent */
            g_mimo_cuda_packed++;
        }
    }
    if (!d->cuda) {
        if (b > *budget) return;
        d->cuda = mimo_cuda_upload(d->w, d->s, d->fmt, d->I, d->O);
    }
    if (d->cuda) {
        *budget -= used; (*count)++;
        if (g_mimo_cuda_release && d->vk_name && d->vk_kind != DWK_NONE) {
            free(d->w); free(d->s); d->w = NULL; d->s = NULL; d->vk_gone = 1;
            g_mimo_cuda_host_bytes += b; g_mimo_cuda_dropped++;
        }
    }
}
/* MIMO_CUDA_OPROJ_INT8=1: an o_proj goes up as int8 with a scale per (row, 128
 * columns), half its bf16 bytes, so every layer's fits beside the qkv and the head on
 * a 24-GiB card and the CPU computes none. NOT exact: about 0.65% relative error on
 * the projection (measured on Pro's weights; bf16 itself is about 0.1%). The host
 * copy stays bf16, so a CPU that has to take over computes the released bytes. */
static int g_mimo_cuda_oproj_i8, g_mimo_cuda_oproj_i8_n;
static void mimo_cuda_place_i8(DW *d, size_t *budget, int *count) {
    if (!d->w || d->fmt != DW_BF16 || d->O < 1 || d->I < 128 || d->I % 128) { mimo_cuda_place(d, budget, count); return; }
    int nb = d->I / 128;
    size_t qb = (size_t)d->O * d->I, sb = (size_t)d->O * nb * sizeof(float), b = dw_bytes(d);
    if (qb + sb > *budget) return;
    int8_t *q = malloc(qb);
    float *sc = malloc(sb);
    if (!q || !sc) { free(q); free(sc); return; }
    const uint16_t *w = (const uint16_t *)d->w;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < d->O; o++)
        for (int k = 0; k < nb; k++) {
            const uint16_t *src = w + (size_t)o * d->I + (size_t)k * 128;
            float most = 0;
            for (int i = 0; i < 128; i++) { float a = fabsf(bf16_to_f32(src[i])); if (a > most) most = a; }
            float step = most > 0 ? most / 127.0f : 1.0f;
            sc[(size_t)o * nb + k] = step;
            int8_t *dst = q + (size_t)o * d->I + (size_t)k * 128;
            for (int i = 0; i < 128; i++) {
                long v = lrintf(bf16_to_f32(src[i]) / step);
                dst[i] = (int8_t)(v > 127 ? 127 : v < -127 ? -127 : v);
            }
        }
    d->cuda = mimo_cuda_upload(q, sc, 4, d->I, d->O);
    free(q); free(sc);
    if (!d->cuda) return;
    *budget -= qb + sb; (*count)++; g_mimo_cuda_oproj_i8_n++;
    if (g_mimo_cuda_release && d->vk_name && d->vk_kind != DWK_NONE) {
        free(d->w); free(d->s); d->w = NULL; d->s = NULL; d->vk_gone = 1;
        g_mimo_cuda_host_bytes += b; g_mimo_cuda_dropped++;
    }
}
static void mimo_cuda_host_report(void) {
    fprintf(stderr, "[CUDA] mimo host copies: %d released (%.3f GiB), %d restored from checkpoint (%.3f GiB)\n",
            g_mimo_cuda_dropped, g_mimo_cuda_host_bytes / 1073741824.0,
            g_mimo_cuda_reloaded, g_mimo_cuda_reload_bytes / 1073741824.0);
}
static void mimo_cuda_start(Model *m) {
    if (!env_int("MIMO_CUDA", env_int("COLI_CUDA", 0))) return;
    double gb = mimo_cuda_gb("MIMO_CUDA_DENSE_GB", 20);
    double reserve = mimo_cuda_gb("MIMO_CUDA_RESERVE_GB", 2);
    if (!gb) return;
    if (!mimo_cuda_init(env_int("COLI_GPU", 0))) {
        fprintf(stderr, "[CUDA] mimo unavailable; using existing CPU/Vulkan path\n"); return;
    }
    atexit(mimo_cuda_shutdown);
    atexit(mimo_cuda_report);
    atexit(mimo_cuda_host_report);
    g_dw_dho_S = &m->S; g_dw_dho_c = &m->c;
    g_mimo_cuda_release = env_int("MIMO_CUDA_RELEASE_HOST", 0) != 0;
    g_mimo_cuda_pack = env_int("MIMO_CUDA_BF16_PACK", 1) != 0;
    int fault = env_int("MIMO_CUDA_FAULT", 0);
    g_mimo_cuda_fault = fault > 0 ? (unsigned)fault : 0;
    size_t fr = mimo_cuda_free_bytes(), headroom = (size_t)(reserve * 1073741824.0);
    size_t budget = (size_t)(gb * 1073741824.0);
    size_t available = fr > headroom ? fr - headroom : 0;
    if (budget > available) budget = available;
    size_t start = budget; int count = 0;
    /* All QKV first: expensive CPU FP8 decoding, and much smaller than the
     * BF16 output projections. Then the head and layer-0 MLP; use the rest
     * for output projections. A 24-GiB card need not fit complete layers. */
    for (int i = 0; i < m->c.n_layers; i++) mimo_cuda_place(&m->L[i].qkv, &budget, &count);
    mimo_cuda_place(&m->head, &budget, &count);
    for (int i = 0; i < m->c.n_layers; i++) if (!m->c.moe[i]) {
        mimo_cuda_place(&m->L[i].gate, &budget, &count);
        mimo_cuda_place(&m->L[i].up, &budget, &count);
        mimo_cuda_place(&m->L[i].down, &budget, &count);
    }
    g_mimo_cuda_oproj_i8 = env_int("MIMO_CUDA_OPROJ_INT8", 0) != 0;
    /* MIMO_CUDA_OPROJ_SPREAD (default 1): the even layers' o_proj first, then the odd
     * ones, so the layers whose o_proj stays on the CPU (the slow attentions) are spread
     * through the model instead of being its second half. The next layer's experts are
     * read during this layer's MLP and the next attention: a run of fast attentions left
     * the disks behind, a slow one every other layer lets them catch up. 0: in order. */
    int spread = env_int("MIMO_CUDA_OPROJ_SPREAD", 1) != 0;
    for (int pass = 0; pass < (spread ? 2 : 1); pass++)
        for (int i = spread ? pass : 0; i < m->c.n_layers; i += spread ? 2 : 1) {
            if (g_mimo_cuda_oproj_i8) mimo_cuda_place_i8(&m->L[i].o, &budget, &count);
            else mimo_cuda_place(&m->L[i].o, &budget, &count);
        }
    g_mimo_cuda = count > 0;
    fprintf(stderr, "[CUDA] mimo placed %d dense matrices (%d bf16 packed): %.3f / %.3f GiB budget; CPU host copies %s\n",
            count, g_mimo_cuda_packed, (start - budget) / 1073741824.0, start / 1073741824.0, g_mimo_cuda_release ? "released" : "retained");
    if (g_mimo_cuda_oproj_i8)
        fprintf(stderr, "[CUDA] mimo MIMO_CUDA_OPROJ_INT8: %d o_proj as int8 per 128 columns -- NOT exact\n",
                g_mimo_cuda_oproj_i8_n);
}
#endif

/* MIMO_BF16_PACK (default 1): the o_proj and the head the CPU computes (none the GPU
 * holds) packed without loss (DW_BF16P), with MIMO_DENSE_BITS 0 and no Vulkan device.
 * 0: plain bf16. */
static void mimo_bf16_pack(Model *m) {
    if (!env_int("MIMO_BF16_PACK", 1) || g_dense_bits != 0) return;
#ifdef COLI_VULKAN
    if (g_vk_ready) return;   /* the device may take them as bf16 */
#endif
    int n = 0;
    size_t from = 0, to = 0;
    for (int i = 0; i < m->c.n_layers; i++) {
#ifdef COLI_MIMO_CUDA
        if (m->L[i].o.cuda) continue;
#endif
        n += dw_pack_bf16(&m->L[i].o, &from, &to);
    }
#ifdef COLI_MIMO_CUDA
    if (!m->head.cuda)
#endif
    n += dw_pack_bf16(&m->head, &from, &to);
    for (int k = 0; k < g_mtp.n; k++) {
        n += dw_pack_bf16(&m->L[g_mtp.li + k].o, &from, &to);
        n += dw_pack_bf16(&g_mtp.b[k].eh, &from, &to);
    }
    if (n) fprintf(stderr, "[mimo] %d bf16 matrices packed without loss for the CPU: %.2f GB for %.2f GB\n",
                   n, to / 1e9, from / 1e9);
}

int main(int argc, char **argv) {
    coli_omp_tune_threads("mimo");
    /* Two ways in. The gateway: SNAP=<dir> mimo <cap>, SERVE=1. By hand and in
     * the tests: mimo <dir> [--ids "..." | --prompt "..."] [--ngen N] [--cap N]. */
    const char *dir = getenv("SNAP");
    int cap = env_int("MIMO_CAP", 0);
    if (env_int("SERVE", 0)) {   /* KV_SLOTS: the conversations a serve decodes at once */
        const char *ks = getenv("KV_SLOTS");
        if (ks && *ks) {
            char *end = NULL; long v = strtol(ks, &end, 10);
            if (end == ks || *end || v < 1 || v > 16) { fprintf(stderr, "KV_SLOTS must be between 1 and 16\n"); return 2; }
            g_mimo_mux_slots = (int)v;
        }
    }
    const char *ids_text = NULL, *prompt = NULL, *image_path = NULL;
    int ngen = 32, grid_h = 0, grid_w = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--ids") && i + 1 < argc) ids_text = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--ngen") && i + 1 < argc) ngen = coli_arg_int(argv[++i], "--ngen");
        else if (!strcmp(argv[i], "--cap") && i + 1 < argc) cap = coli_arg_int(argv[++i], "--cap");
        else if (!strcmp(argv[i], "--image") && i + 1 < argc) image_path = argv[++i];
        else if (!strcmp(argv[i], "--grid") && i + 2 < argc) {
            grid_h = coli_arg_int(argv[++i], "--grid h"); grid_w = coli_arg_int(argv[++i], "--grid w"); }
        else if (is_dir(argv[i])) dir = argv[i];
        else cap = coli_arg_int(argv[i], "cache/layer");
    }
    if (!dir) {
        fprintf(stderr, "usage: %s <model dir> [--ids \"1 2 3\" | --prompt TEXT] [--ngen N] [--cap N]\n"
                        "       SNAP=<model dir> SERVE=1 %s <cap>\n", argv[0], argv[0]);
        return 2;
    }
    Model *m = xcalloc(1, sizeof(Model), "model");
    double t0 = now_s();
    if (cap <= 0) cap = 64;
    model_load(m, dir, cap);
    Cfg *c = &m->c;
    g_vision = vision_load(m, dir);
#ifdef COLI_MIMO_CUDA
    mimo_cuda_start(m);
#endif
#ifdef COLI_VULKAN
    /* after the weights, so a missing device costs one line and nothing else; the
     * routed-expert tier first, then the dense matrices where coli_vk_dense() puts
     * them (on a device that shares the CPU's RAM they stay on the CPU while the
     * tier is on; COLI_VK_DENSE decides when set) */
    {
#ifdef COLI_MIMO_CUDA
        /* One backend owns dense placement. Do not double-allocate VRAM or
         * free host weights beneath CUDA's CPU recovery path. */
        if (!g_mimo_cuda) {
#endif
        int any_moe = 0;
        for (int i = 0; i < c->n_layers; i++) any_moe |= c->moe[i];
        const char *mx = getenv("MIMO_VK_EXPERTS");
        int tier = vkt_wanted() && any_moe && !(mx && *mx && atoi(mx) <= 0);   /* MIMO_VK_EXPERTS=0: experts on the CPU */
        g_vk_ready = coli_vk_init_env_tier("mimo", tier);
        /* COLI_VK_CHAIN: every layer's dense chain on the device (mimo_chain.h), decided
         * before the tier so the tier's budget leaves the trunk room. Not measured on a
         * MiMo checkpoint: an integrated GPU keeps it opt-in (docs/vulkan.md). */
        if (g_vk_ready) {
            g_vk_chain = coli_vk_chain_decide("mimo", tier, COLI_VK_CHAIN_UNMEASURED);
            if (g_vk_chain && g_mimo_mux_slots > 1) {
                g_vk_chain = 0;
                fprintf(stderr, "[VK] mimo: KV_SLOTS=%d: the dense chain is off (it keeps one conversation's caches on the device); "
                                "the expert tier runs every conversation's experts\n", g_mimo_mux_slots);
            }
            /* the chain's fit before any upload: its first n layers on the device, the
             * others, the head and the tower on the CPU when they do not all fit */
            if (g_vk_chain) mc_fit_start(m);
            if (g_mc_fit.L && !g_mc_fit.n) {   /* its line said the chain stays off: nothing goes up */
                g_vk_chain = 0;
                mc_refuse_from(m, 0);
                vkc_fit_placed("mimo", &g_mc_fit);
                g_mc_placed = 1;
            }
            if (g_vk_chain && !vkc_init()) g_vk_chain = 0;
        }
        mimo_dho_start(m);   /* COLI_VK_DENSE_HOST: the dense matrices on the device only, before the tier sizes its budget */
        if (g_vk_chain && vkc_fit_partial(&g_mc_fit) && !g_mc_placed) mc_place(m, NULL);   /* before the tier, too */
        if (g_vk_ready && tier) vk_tier_start(m);
        if (g_vk_ready && !vkt_ready() && !coli_vk_dense()) coli_vk_dense_decide("mimo", 0, 1);   /* no tier after all */
        if (g_vk_chain) {   /* the chain's teardown before the device's (the tier registered the device's) */
            if (!tier) atexit(coli_vk_shutdown);
            atexit(vkc_shutdown_all);
        }
#ifdef COLI_MIMO_CUDA
        }
#endif
    }
    if (g_vk_ready && (coli_vk_dense() || g_vk_chain) && !(g_mc_fit.L && !g_mc_fit.n)) {   /* COLI_VK_DENSE=0: the trunk stays on the CPU */
        int up = vk_dense_upload(m, coli_vk_dense());
        size_t used = 0, count = 0;
        coli_vk_mem_info(&used, &count);
        fprintf(stderr, "[VK] mimo: %d dense matrices on the GPU (%.2f GB)\n", up, used / 1e9);
    }
#endif
    mimo_bf16_pack(m);
    int swa = 0, shown_cap = 0;
    for (int i = 0; i < c->n_layers; i++) { swa += c->swa[i]; if (c->moe[i] && !shown_cap) shown_cap = m->cache[i].cap; }
    fprintf(stderr, "[mimo] %d layers (%d sliding window %d, %d full), %d experts top-%d, cache %d/layer, "
                    "dense %s, ctx %d%s -- loaded in %.2fs\n",
            c->n_layers, swa, c->window, c->n_layers - swa, c->n_experts, c->topk, shown_cap,
            g_dense_bits == 0 ? "native" : g_dense_bits == 8 ? "int8" : "f32", m->ctx,
            g_vision ? ", vision" : "", now_s() - t0);

    char tok_path[4096];
    snprintf(tok_path, sizeof(tok_path), "%s/tokenizer.json", dir);
    int have_tok = access(tok_path, R_OK) == 0;
    Tok tokenizer;
    if (have_tok) tok_load(&tokenizer, tok_path);
    const char *seed = getenv("SEED");
    srand(seed ? (unsigned)strtoul(seed, NULL, 10) : (unsigned)time(NULL));

    if (env_int("SERVE", 0)) {
        if (!have_tok) { fprintf(stderr, "[mimo] SERVE needs %s\n", tok_path); return 1; }
        if (g_mimo_mux_slots > 1) {   /* slot 0 is the Model's own caches */
            g_mimo_mux_seq = xcalloc((size_t)g_mimo_mux_slots, sizeof *g_mimo_mux_seq, "conversations");
            for (int i = 1; i < g_mimo_mux_slots; i++) mimo_seq_alloc(m, &g_mimo_mux_seq[i]);
            g_mimo_mux_cur = 0;
        }
        serve_loop(m, &tokenizer, dir);
        vk_report(m);
        return 0;
    }
    int *ids = xmalloc(((size_t)m->ctx + 1) * sizeof(int), "ids");
    int n = 0;
    if (ids_text) n = parse_ids(ids_text, ids, m->ctx);
    else if (prompt && have_tok) n = tok_encode(&tokenizer, prompt, (int)strlen(prompt), ids, m->ctx);
    else { fprintf(stderr, "[mimo] give --ids, or --prompt with a tokenizer.json in the model dir\n"); return 2; }
    if (n < 1) { fprintf(stderr, "[mimo] empty prompt\n"); return 2; }
    if (n + ngen > m->ctx) { fprintf(stderr, "[mimo] prompt + ngen exceed the context %d\n", m->ctx); return 2; }
    float *logits = xmalloc((size_t)c->vocab * sizeof(float), "logits");
    const char *dump = getenv("MIMO_LOGITS");
    float *all = dump ? xmalloc((size_t)n * c->vocab * sizeof(float), "all logits") : NULL;
    /* --image patches.f32 --grid H W: the gateway's patches for the image-pad
     * ids already in the prompt */
    float *image = NULL;
    int image_rows = 0;
    if (image_path) {
        if (!g_vision) { fprintf(stderr, "[mimo] --image: this container has no vision tower\n"); return 2; }
        size_t want = (size_t)grid_h * grid_w * g_vision->patch_in;
        float *patches = xmalloc(want * sizeof(float), "patches");
        FILE *f = fopen(image_path, "rb");
        if (!f || fread(patches, sizeof(float), want, f) != want || fgetc(f) != EOF) {
            fprintf(stderr, "[mimo] %s does not hold %dx%d patches\n", image_path, grid_h, grid_w); return 2;
        }
        fclose(f);
        int pads = 0;
        for (int t = 0; t < n; t++) pads += ids[t] == c->image_token_id;
        if (pads != (grid_h / 2) * (grid_w / 2)) {
            fprintf(stderr, "[mimo] the prompt has %d image pads, the grid needs %d\n", pads, (grid_h / 2) * (grid_w / 2));
            return 2;
        }
        image = vision_run(m, patches, grid_h, grid_w, &image_rows);
        free(patches);
    }
    ImageRows img = { image, image_rows };
    double tp = now_s();
    prefill(m, ids, n, logits, all, image ? &img : NULL, NULL);
    double prefill_s = now_s() - tp;
    if (dump) {
        FILE *f = fopen(dump, "wb");
        if (!f || fwrite(all, sizeof(float), (size_t)n * c->vocab, f) != (size_t)n * c->vocab) {
            fprintf(stderr, "[mimo] cannot write %s\n", dump); return 1;
        }
        fclose(f);
    }
    float temperature = getenv("COLI_TEMP") ? (float)atof(getenv("COLI_TEMP")) : 0.0f;
    int eos[8], n_eos = coli_load_stop_ids(dir, eos, 8, NULL);
    double td = now_s();
    int produced = 0;
    char piece[512];
    int spec = g_mtp.on, warm = 0, queue[MIMO_MTP_MAX + 1], qn = 0, qi = 0, computed = 0;   /* MIMO_MTP: as in serve_loop */
    float *rows = spec ? xmalloc((size_t)(g_mtp.n + 1) * c->vocab * sizeof(float), "verified logits") : NULL;
    const char *oracle = getenv("MIMO_MTP_ORACLE");
    if (spec && oracle && *oracle) {
        g_mtp_oracle = xmalloc(((size_t)m->ctx + 1) * sizeof(int), "oracle ids");
        g_mtp_oracle_n = parse_ids(oracle, g_mtp_oracle, m->ctx + 1);
        g_mtp_oracle_miss = env_int("MIMO_MTP_ORACLE_MISS", 0);
    }
    for (int g = 0; g < ngen; g++) {
        int token = qi < qn ? queue[qi] : sample(logits, c->vocab, temperature, 1.0f);
        int done = qi < computed;
        if (qi < qn) qi++;
        int stop = 0;
        for (int i = 0; i < n_eos; i++) stop |= token == eos[i];
        if (have_tok && !ids_text) {
            int w = tok_decode(&tokenizer, &token, 1, piece, (int)sizeof(piece));
            fwrite(piece, 1, (size_t)(w > 0 ? w : 0), stdout);
        } else printf("%d ", token);
        fflush(stdout);
        produced++;
        if (stop && have_tok && !ids_text) break;
        if (g + 1 < ngen && !spec) prefill(m, &token, 1, logits, NULL, NULL, NULL);
        else if (g + 1 < ngen && !done) {
            if (ngen - g - 2 > 0 && !warm++) mtp_warm(m, ids, 0, n);
            qn = mtp_step(m, token, ngen - g - 2, logits, rows, temperature, 1.0f, queue, &computed);
            qi = 0;
        }
    }
    free(rows);
    printf("\n");
    fflush(stdout);
    double decode_s = now_s() - td;
    double total = prefill_s + decode_s;
    fprintf(stderr, "[mimo] prefill %d tokens %.2fs, decode %d tokens %.2fs (%.2f tok/s), experts %llu hits "
                    "%llu misses %.1f MB read, rss %.2f GB\n"
                    "[mimo] time: expert reads %.2fs (%.2f GB/s), expert matmuls %.2fs, attention %.2fs, "
                    "the rest (dense MLP, router, head) %.2fs\n",
            n, prefill_s, produced, decode_s, decode_s > 0 ? produced / decode_s : 0.0,
            (unsigned long long)m->hits, (unsigned long long)m->miss, m->bytes_read / 1e6, rss_gb(),
            m->t_disk, m->t_disk > 0 ? m->bytes_read / 1e9 / m->t_disk : 0.0, m->t_expert, m->t_attn,
            total - m->t_disk - m->t_expert - m->t_attn);
    MimoReq none = {0};
    mimo_stream_report(m, &none);
    if (g_mtp.on && g_mtp.steps) fprintf(stderr, "[mimo] decode %d tokens in %llu verified steps\n", produced,
                                         (unsigned long long)g_mtp.steps);
    vk_report(m);
    return 0;
}
