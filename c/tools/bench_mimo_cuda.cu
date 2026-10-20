/* The prompt kernels of backend_cuda_mimo.cu at MiMo-V2.6-Pro's shapes, on the device alone (no
 * host copies): time per call, TFLOP/s, and each kernel's largest difference from a double
 * reference on sampled outputs. Built with the backend itself:
 *   nvcc -O3 -std=c++17 -arch=sm_89 -ftz=false tools/bench_mimo_cuda.cu -o tools/bench_mimo_cuda.exe
 * usage: bench_mimo_cuda [rows=1024] [expert_rows=520] [keys=25024] */
#include "../backend_cuda_mimo.cu"
#include <cmath>
#include <cstdio>
#include <vector>

/* ---- the previous round's prompt kernels (commit 2147c6b1), for the comparison ---- */
/* FP8 (fmt 2) for a prompt block (S >= MIMO_CUDA_TILED_ROWS, default 64): a tiled product,
 * 64 rows x 64 outputs a block, 4 x 4 a thread out of shared memory. Per 128-column scale
 * block the products are summed in f32 (in two 64-column halves), then times the block's
 * scale into the output in f32. Not the narrow kernels' bits (they sum each block's lanes
 * by a butterfly and the blocks in double): the same quantity, for a prompt block's rows,
 * each row's bytes whatever rows share its block. */
#define TL_B 64
__global__ static void __launch_bounds__(256) fp8_tiled(const uint8_t *__restrict__ w, const float *__restrict__ scales,
                                                        const float *__restrict__ x, float *__restrict__ y, int I, int O, int S) {
    __shared__ float xs[TL_B][TL_B + 4];   /* [column][row] */
    __shared__ float ws[TL_B][TL_B + 4];   /* [column][output] */
    int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    int r0 = blockIdx.y * TL_B, o0 = blockIdx.x * TL_B, nb = I / 128, nt = I / TL_B;
    int lr = tid >> 2, lc = (tid & 3) * 16;   /* this thread's loads: row/output lr, columns lc..lc+15 */
    const float *xr = r0 + lr < S ? x + (size_t)(r0 + lr) * I : nullptr;
    const uint8_t *wr = w + (size_t)(o0 + lr) * I;
    float acc[4][4], part[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = part[i][j] = 0.f;
    /* the next tile's loads go out before this tile's products (a register prefetch) */
    float4 xa[4];
    uint4 wq;
    #pragma unroll
    for (int v = 0; v < 4; v++) xa[v] = xr ? ((const float4 *)(xr + lc))[v] : make_float4(0.f, 0.f, 0.f, 0.f);
    wq = *(const uint4 *)(wr + lc);
    for (int t = 0; t < nt; t++) {
        #pragma unroll
        for (int v = 0; v < 4; v++) {
            xs[lc + 4 * v][lr] = xa[v].x; xs[lc + 4 * v + 1][lr] = xa[v].y; xs[lc + 4 * v + 2][lr] = xa[v].z; xs[lc + 4 * v + 3][lr] = xa[v].w;
        }
        float wf[16];
        fp8x16(wq, wf);
        #pragma unroll
        for (int v = 0; v < 16; v++) ws[lc + v][lr] = wf[v];
        __syncthreads();
        if (t + 1 < nt) {
            int k1 = (t + 1) * TL_B;
            #pragma unroll
            for (int v = 0; v < 4; v++) xa[v] = xr ? ((const float4 *)(xr + k1 + lc))[v] : make_float4(0.f, 0.f, 0.f, 0.f);
            wq = *(const uint4 *)(wr + k1 + lc);
        }
        #pragma unroll 8
        for (int kk = 0; kk < TL_B; kk++) {
            float4 a = *(const float4 *)&xs[kk][ty * 4], c = *(const float4 *)&ws[kk][tx * 4];
            const float av[4] = {a.x, a.y, a.z, a.w}, cv[4] = {c.x, c.y, c.z, c.w};
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++) part[i][j] = fmaf(av[i], cv[j], part[i][j]);
        }
        if (t & 1) {   /* the 128-column scale block is complete */
            int b = t >> 1;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float sc = scales[(size_t)(o0 + tx * 4 + j) * nb + b];
                #pragma unroll
                for (int i = 0; i < 4; i++) { acc[i][j] = fmaf(part[i][j], sc, acc[i][j]); part[i][j] = 0.f; }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int row = r0 + ty * 4 + i;
        if (row >= S) break;
        *(float4 *)(y + (size_t)row * O + o0 + tx * 4) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
    }
}
/* Packed bf16 (fmt 5) for a prompt block: fp8_tiled's tiling, each weight rebuilt from its
 * sign/mantissa byte and its exponent (lo + code, or the next byte of the row's exception
 * list for code 15), summed in f32 straight into the output. A row's four loading threads
 * count their exceptions in the tile and share the counts (a prefix across their lanes), so
 * each finds its own in the row's list. */
__global__ static void __launch_bounds__(256) bf16p_tiled(const uint8_t *__restrict__ sm, const uint8_t *__restrict__ codes,
                                                          const uint8_t *__restrict__ lo, const unsigned *__restrict__ exc_off,
                                                          const uint8_t *__restrict__ exc, const float *__restrict__ x,
                                                          float *__restrict__ y, int I, int O, int S) {
    __shared__ float xs[TL_B][TL_B + 4];
    __shared__ float ws[TL_B][TL_B + 4];
    int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4, lane = tid & 31;
    int r0 = blockIdx.y * TL_B, o0 = blockIdx.x * TL_B, nt = I / TL_B;
    int lr = tid >> 2, lq = tid & 3, lc = lq * 16;
    const float *xr = r0 + lr < S ? x + (size_t)(r0 + lr) * I : nullptr;
    int o = o0 + lr;
    const uint8_t *smr = sm + (size_t)o * I;
    const unsigned *cr = (const unsigned *)(codes + (size_t)o * (I / 2));
    const unsigned base = lo[o];
    const uint8_t *ex = exc + exc_off[o];
    unsigned run = 0;   /* the row's exceptions before this tile */
    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.f;
    /* the next tile's loads go out before this tile's products (a register prefetch) */
    float4 xa[4];
    unsigned q[2];
    uint4 sv;
    #pragma unroll
    for (int v = 0; v < 4; v++) xa[v] = xr ? ((const float4 *)(xr + lc))[v] : make_float4(0.f, 0.f, 0.f, 0.f);
    q[0] = cr[lc / 8]; q[1] = cr[lc / 8 + 1];
    sv = *(const uint4 *)(smr + lc);
    for (int t = 0; t < nt; t++) {
        #pragma unroll
        for (int v = 0; v < 4; v++) {
            xs[lc + 4 * v][lr] = xa[v].x; xs[lc + 4 * v + 1][lr] = xa[v].y; xs[lc + 4 * v + 2][lr] = xa[v].z; xs[lc + 4 * v + 3][lr] = xa[v].w;
        }
        const unsigned sw[4] = {sv.x, sv.y, sv.z, sv.w};
        unsigned esc0 = q[0] & (q[0] >> 1) & (q[0] >> 2) & (q[0] >> 3) & 0x11111111u;
        unsigned esc1 = q[1] & (q[1] >> 1) & (q[1] >> 2) & (q[1] >> 3) & 0x11111111u;
        int n = __popc(esc0) + __popc(esc1), incl = n;
        int up = __shfl_up_sync(0xffffffffu, incl, 1);
        if (lq >= 1) incl += up;
        up = __shfl_up_sync(0xffffffffu, incl, 2);
        if (lq >= 2) incl += up;
        int total = __shfl_sync(0xffffffffu, incl, lane | 3);
        const uint8_t *p = ex + run + (unsigned)(incl - n);
        #pragma unroll
        for (int g = 0; g < 2; g++) {
            #pragma unroll
            for (int k = 0; k < 8; k++) {   /* weight k of the group: its code, then exponent, then bits */
                unsigned code = (q[g] >> (k < 4 ? 8 * k : 8 * (k - 4) + 4)) & 15u;
                unsigned e = code == 15u ? (unsigned)*p++ : base + code;
                int idx = g * 8 + k;
                unsigned b = (sw[idx >> 2] >> (8 * (idx & 3))) & 0xffu;
                ws[lc + idx][lr] = __uint_as_float(((b & 0x80u) << 24) | (e << 23) | ((b & 0x7fu) << 16));
            }
        }
        run += (unsigned)total;
        __syncthreads();
        if (t + 1 < nt) {
            int k1 = (t + 1) * TL_B;
            #pragma unroll
            for (int v = 0; v < 4; v++) xa[v] = xr ? ((const float4 *)(xr + k1 + lc))[v] : make_float4(0.f, 0.f, 0.f, 0.f);
            q[0] = cr[(k1 + lc) / 8]; q[1] = cr[(k1 + lc) / 8 + 1];
            sv = *(const uint4 *)(smr + k1 + lc);
        }
        #pragma unroll 8
        for (int kk = 0; kk < TL_B; kk++) {
            float4 a = *(const float4 *)&xs[kk][ty * 4], c = *(const float4 *)&ws[kk][tx * 4];
            const float av[4] = {a.x, a.y, a.z, a.w}, cv[4] = {c.x, c.y, c.z, c.w};
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] = fmaf(av[i], cv[j], acc[i][j]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int row = r0 + ty * 4 + i;
        if (row >= S) break;
        *(float4 *)(y + (size_t)row * O + o0 + tx * 4) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
    }
}
/* Plain bf16 (fmt 1) for a prompt block: fp8_tiled's tiling, no scales (a router's logits). */
__global__ static void __launch_bounds__(256) bf16_tiled(const uint16_t *__restrict__ w, const float *__restrict__ x,
                                                         float *__restrict__ y, int I, int O, int S) {
    __shared__ float xs[TL_B][TL_B + 4];
    __shared__ float ws[TL_B][TL_B + 4];
    int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    int r0 = blockIdx.y * TL_B, o0 = blockIdx.x * TL_B, nt = I / TL_B;
    int lr = tid >> 2, lc = (tid & 3) * 16;
    const float *xr = r0 + lr < S ? x + (size_t)(r0 + lr) * I : nullptr;
    const uint16_t *wr = w + (size_t)(o0 + lr) * I;
    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.f;
    float4 xa[4];
    uint4 wq[2];
    #pragma unroll
    for (int v = 0; v < 4; v++) xa[v] = xr ? ((const float4 *)(xr + lc))[v] : make_float4(0.f, 0.f, 0.f, 0.f);
    wq[0] = ((const uint4 *)(wr + lc))[0]; wq[1] = ((const uint4 *)(wr + lc))[1];
    for (int t = 0; t < nt; t++) {
        #pragma unroll
        for (int v = 0; v < 4; v++) {
            xs[lc + 4 * v][lr] = xa[v].x; xs[lc + 4 * v + 1][lr] = xa[v].y; xs[lc + 4 * v + 2][lr] = xa[v].z; xs[lc + 4 * v + 3][lr] = xa[v].w;
        }
        const unsigned u[8] = {wq[0].x, wq[0].y, wq[0].z, wq[0].w, wq[1].x, wq[1].y, wq[1].z, wq[1].w};
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            ws[lc + 2 * k][lr] = __uint_as_float(u[k] << 16);
            ws[lc + 2 * k + 1][lr] = __uint_as_float(u[k] & 0xffff0000u);
        }
        __syncthreads();
        if (t + 1 < nt) {
            int k1 = (t + 1) * TL_B;
            #pragma unroll
            for (int v = 0; v < 4; v++) xa[v] = xr ? ((const float4 *)(xr + k1 + lc))[v] : make_float4(0.f, 0.f, 0.f, 0.f);
            wq[0] = ((const uint4 *)(wr + k1 + lc))[0]; wq[1] = ((const uint4 *)(wr + k1 + lc))[1];
        }
        #pragma unroll 8
        for (int kk = 0; kk < TL_B; kk++) {
            float4 a = *(const float4 *)&xs[kk][ty * 4], c = *(const float4 *)&ws[kk][tx * 4];
            const float av[4] = {a.x, a.y, a.z, a.w}, cv[4] = {c.x, c.y, c.z, c.w};
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] = fmaf(av[i], cv[j], acc[i][j]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int row = r0 + ty * 4 + i;
        if (row >= S) break;
        *(float4 *)(y + (size_t)row * O + o0 + tx * 4) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
    }
}
#define MOE_BM 64
#define MOE_BN 64
#define MOE_BK 32
__constant__ float mx4_lut2[16] = {0.f, 1.f, 2.f, 3.f, 4.f, 6.f, 8.f, 12.f, 0.f, -1.f, -2.f, -3.f, -4.f, -6.f, -8.f, -12.f};
__device__ static __forceinline__ float e8m0_half(unsigned s) { return __uint_as_float(s << 23) * 0.5f; }

/* One 64-row x 64-output tile of A [r rows, K] times W^T (W: e2m1 [O][K/2], scales
 * [O][K/32]) for one or two matrices sharing A. A's rows: A + idx[row] * lda (idx NULL:
 * the row itself). 256 threads, 4 x 4 outputs each; acc[i][j] is row ty*4+i, output tx*4+j. */
template<int NW>
__device__ static __forceinline__ void mx4_tile(const float *__restrict__ A, int lda, const int *__restrict__ idx, int r,
                                                int K, const uint8_t *const *q, const uint8_t *const *s,
                                                int r0, int o0, float (*acc)[4][4]) {
    __shared__ float as[MOE_BK][MOE_BM + 4];
    __shared__ float ws[NW][MOE_BK][MOE_BN + 4];
    __shared__ float sc[NW][MOE_BN];
    int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    int lr = tid >> 2, lk = (tid & 3) * 8;          /* the A row and 8 columns this thread loads */
    int row = r0 + lr;
    const float *arow = row < r ? A + (size_t)(idx ? idx[row] : row) * lda : nullptr;
    const int ng = K / 32, rb = K / 2;
    /* the next tile's loads go out before this tile's products (a register prefetch) */
    float4 a0, a1;
    unsigned v[NW], e[NW];
    auto fetch = [&](int k0) {
        a0 = make_float4(0.f, 0.f, 0.f, 0.f); a1 = a0;
        if (arow) { const float4 *src = (const float4 *)(arow + k0 + lk); a0 = src[0]; a1 = src[1]; }
        #pragma unroll
        for (int w = 0; w < NW; w++) {
            v[w] = *(const unsigned *)(q[w] + (size_t)(o0 + lr) * rb + k0 / 2 + (tid & 3) * 4);
            e[w] = tid < MOE_BN ? s[w][(size_t)(o0 + tid) * ng + k0 / 32] : 0u;
        }
    };
    fetch(0);
    for (int k0 = 0; k0 < K; k0 += MOE_BK) {
        as[lk + 0][lr] = a0.x; as[lk + 1][lr] = a0.y; as[lk + 2][lr] = a0.z; as[lk + 3][lr] = a0.w;
        as[lk + 4][lr] = a1.x; as[lk + 5][lr] = a1.y; as[lk + 6][lr] = a1.z; as[lk + 7][lr] = a1.w;
        #pragma unroll
        for (int w = 0; w < NW; w++) {
            /* output o0 + lr, bytes (tid & 3) * 4 .. +3 of the group: columns lk .. lk + 7 */
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                unsigned by = (v[w] >> (8 * b)) & 0xffu;
                ws[w][lk + 2 * b][lr] = mx4_lut2[by & 15u];
                ws[w][lk + 2 * b + 1][lr] = mx4_lut2[by >> 4];
            }
            if (tid < MOE_BN) sc[w][tid] = e8m0_half(e[w]);
        }
        __syncthreads();
        if (k0 + MOE_BK < K) fetch(k0 + MOE_BK);
        float p[NW][4][4];
        #pragma unroll
        for (int w = 0; w < NW; w++)
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++) p[w][i][j] = 0.f;
        #pragma unroll 8
        for (int kk = 0; kk < MOE_BK; kk++) {
            float4 a = *(const float4 *)&as[kk][ty * 4];
            const float av[4] = {a.x, a.y, a.z, a.w};
            #pragma unroll
            for (int w = 0; w < NW; w++) {
                float4 b = *(const float4 *)&ws[w][kk][tx * 4];
                const float bv[4] = {b.x, b.y, b.z, b.w};
                #pragma unroll
                for (int i = 0; i < 4; i++)
                    #pragma unroll
                    for (int j = 0; j < 4; j++) p[w][i][j] = fmaf(av[i], bv[j], p[w][i][j]);
            }
        }
        #pragma unroll
        for (int w = 0; w < NW; w++)
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float g = sc[w][tx * 4 + j];
                #pragma unroll
                for (int i = 0; i < 4; i++) acc[w][i][j] = fmaf(p[w][i][j], g, acc[w][i][j]);
            }
        __syncthreads();
    }
}
/* gate and up of one expert over its r rows (X's rows idx[]): G [r][MI] = silu(gate) * up */
__global__ static void __launch_bounds__(256) moe_gateup(const float *__restrict__ X, const int *__restrict__ idx, int r,
                                                         int H, int MI, const uint8_t *__restrict__ w, size_t gq_off,
                                                         size_t gs_off, size_t uq_off, size_t us_off, float *__restrict__ G) {
    const uint8_t *q[2] = {w + gq_off, w + uq_off}, *s[2] = {w + gs_off, w + us_off};
    float acc[2][4][4];
    #pragma unroll
    for (int a = 0; a < 2; a++)
        #pragma unroll
        for (int i = 0; i < 4; i++)
            #pragma unroll
            for (int j = 0; j < 4; j++) acc[a][i][j] = 0.f;
    int r0 = blockIdx.x * MOE_BM, o0 = blockIdx.y * MOE_BN;
    mx4_tile<2>(X, H, idx, r, H, q, s, r0, o0, acc);
    int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int row = r0 + ty * 4 + i;
        if (row >= r) break;
        float4 o = make_float4(silu_f(acc[0][i][0]) * acc[1][i][0], silu_f(acc[0][i][1]) * acc[1][i][1],
                               silu_f(acc[0][i][2]) * acc[1][i][2], silu_f(acc[0][i][3]) * acc[1][i][3]);
        *(float4 *)(G + (size_t)row * MI + o0 + tx * 4) = o;
    }
}
/* down of one expert: Y [r][H] = G . down^T, the expert's r outputs in its rows' order */
__global__ static void __launch_bounds__(256) moe_down(const float *__restrict__ G, int r, int H, int MI,
                                                       const uint8_t *__restrict__ w, size_t dq_off, size_t ds_off,
                                                       float *__restrict__ Y) {
    const uint8_t *q[1] = {w + dq_off}, *s[1] = {w + ds_off};
    float acc[1][4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) acc[0][i][j] = 0.f;
    int r0 = blockIdx.x * MOE_BM, o0 = blockIdx.y * MOE_BN;
    mx4_tile<1>(G, MI, nullptr, r, MI, q, s, r0, o0, acc);
    int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int row = r0 + ty * 4 + i;
        if (row >= r) break;
        *(float4 *)(Y + (size_t)row * H + o0 + tx * 4) = make_float4(acc[0][i][0], acc[0][i][1], acc[0][i][2], acc[0][i][3]);
    }
}

/* ---- (previous round) a prompt block's attention ------------------------------------------------------
 * The attention of n rows at positions p0.. over the keys and values of positions
 * lo..lo+nkeys-1: for each (row, head) the keys of the sliding window of `window` positions
 * that end at the row (window 0: every key from lo to the row), softmax with the head's sink
 * logit in its sum when there is one, the values weighted by it. The CPU's attention() in
 * f32 with an online softmax instead of a max pass: the same quantity, summed in another
 * order. A block holds BR rows x G heads of one kv head (G = nh / kvh), four threads a
 * (row, head); thread g of a query holds its dimensions 16i + 4g .. 16i + 4g + 3, so a
 * query's four threads read 16 consecutive floats of a key as four float4s, and the dot
 * product is two shuffles across them. Keys and values go through shared memory 32 at a
 * time. When they are more than one upload holds (MIMO_CUDA_ATTN_MB), they go up in chunks
 * and each query's running max and sum (ml) and its output so far (in out) wait in device
 * memory between them. */
template<int HD, int VD>
__global__ static void __launch_bounds__(256) attn_rows(
        const float *__restrict__ q, int qstride, const float *__restrict__ k, const float *__restrict__ v,
        const float *__restrict__ sink, float2 *__restrict__ ml, float *__restrict__ out,
        int n, int nh, int kvh, int G, int BR, int p0, int lo, int window,
        int j0, int nk, float scale, int first_chunk, int last_chunk) {
    constexpr int BC = 32, H4 = HD / 16, V4 = VD / 16, KW = HD / 4, VW = VD / 4;
    extern __shared__ float4 smem[];
    float4 *ks = smem, *vs = smem + BC * KW;
    int kh = blockIdx.y, qv = threadIdx.x >> 2, g = threadIdx.x & 3;
    int rloc = qv / G, t = blockIdx.x * BR + rloc, h = kh * G + (qv - rloc * G);
    bool on = rloc < BR && t < n;
    int p = p0 + t, first = window ? max(p - window + 1, lo) : lo;
    unsigned mask = 0xfu << ((threadIdx.x & 31) & ~3);
    float4 qr[H4], acc[V4];
    float m = __int_as_float(0xff800000), l = 0.0f;     /* -inf */
    float4 *op = (float4 *)(out + ((size_t)t * nh + h) * VD);
    if (on) {
        const float4 *qp = (const float4 *)(q + (size_t)t * qstride + (size_t)h * HD);
        #pragma unroll
        for (int i = 0; i < H4; i++) qr[i] = qp[4 * i + g];
        if (first_chunk) {
            if (sink) { m = sink[h]; l = 1.0f; }
            #pragma unroll
            for (int i = 0; i < V4; i++) acc[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        } else {
            float2 s2 = ml[(size_t)t * nh + h];
            m = s2.x; l = s2.y;
            #pragma unroll
            for (int i = 0; i < V4; i++) acc[i] = op[4 * i + g];
        }
    }
    int t_lo = blockIdx.x * BR, t_hi = min(t_lo + BR, n) - 1;
    int kb = window ? max(p0 + t_lo - window + 1, lo) : lo, ke = p0 + t_hi;
    if (kb < j0) kb = j0;
    if (ke > j0 + nk - 1) ke = j0 + nk - 1;
    for (int jb = kb; jb <= ke; jb += BC) {
        int cnt = min(BC, ke - jb + 1);
        __syncthreads();
        for (int idx = threadIdx.x; idx < cnt * KW; idx += blockDim.x) {
            int kk = idx / KW;
            ks[idx] = ((const float4 *)(k + ((size_t)(jb - j0 + kk) * kvh + kh) * HD))[idx - kk * KW];
        }
        for (int idx = threadIdx.x; idx < cnt * VW; idx += blockDim.x) {
            int kk = idx / VW;
            vs[idx] = ((const float4 *)(v + ((size_t)(jb - j0 + kk) * kvh + kh) * VD))[idx - kk * VW];
        }
        __syncthreads();
        if (!on) continue;
        int a = max(jb, first), b = min(jb + cnt - 1, p);
        for (int j = a; j <= b; j++) {
            const float4 *kr = ks + (j - jb) * KW, *vr = vs + (j - jb) * VW;
            float s = 0.0f;
            #pragma unroll
            for (int i = 0; i < H4; i++) {
                float4 x = kr[4 * i + g];
                s = fmaf(qr[i].x, x.x, s); s = fmaf(qr[i].y, x.y, s);
                s = fmaf(qr[i].z, x.z, s); s = fmaf(qr[i].w, x.w, s);
            }
            s += __shfl_xor_sync(mask, s, 1);
            s += __shfl_xor_sync(mask, s, 2);
            s *= scale;
            if (s > m) {   /* a new max: what is summed so far shrinks by e^(m - s) */
                float c = expf(m - s);
                l = fmaf(l, c, 1.0f);
                #pragma unroll
                for (int i = 0; i < V4; i++) {
                    float4 x = vr[4 * i + g];
                    acc[i].x = fmaf(acc[i].x, c, x.x); acc[i].y = fmaf(acc[i].y, c, x.y);
                    acc[i].z = fmaf(acc[i].z, c, x.z); acc[i].w = fmaf(acc[i].w, c, x.w);
                }
                m = s;
            } else {
                float e = expf(s - m);
                l += e;
                #pragma unroll
                for (int i = 0; i < V4; i++) {
                    float4 x = vr[4 * i + g];
                    acc[i].x = fmaf(e, x.x, acc[i].x); acc[i].y = fmaf(e, x.y, acc[i].y);
                    acc[i].z = fmaf(e, x.z, acc[i].z); acc[i].w = fmaf(e, x.w, acc[i].w);
                }
            }
        }
    }
    if (!on) return;
    if (last_chunk) {
        float r = 1.0f / l;
        #pragma unroll
        for (int i = 0; i < V4; i++) op[4 * i + g] = make_float4(acc[i].x * r, acc[i].y * r, acc[i].z * r, acc[i].w * r);
    } else {
        if (!g) ml[(size_t)t * nh + h] = make_float2(m, l);
        #pragma unroll
        for (int i = 0; i < V4; i++) op[4 * i + g] = acc[i];
    }
}


static uint32_t st = 2026;
static uint32_t rnd() { st = st * 1664525u + 1013904223u; return st >> 8; }
static float frand() { return (float)(rnd() & 0xffffff) / 16777216.0f * 2.0f - 1.0f; }
static float h_fp8(unsigned v) {
    int a = v & 127, e = a >> 3, m = a & 7;
    float f = e ? std::ldexp(1.f + m / 8.f, e - 7) : m / 512.f;
    return v & 128 ? -f : f;
}
static const float h_lut[16] = {0, .5f, 1, 1.5f, 2, 3, 4, 6, 0, -.5f, -1, -1.5f, -2, -3, -4, -6};

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e_)); exit(1); } } while (0)
template<class F> static double timed(F f, int reps) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    f(); CK(cudaStreamSynchronize(stream));
    cudaEventRecord(a, stream);
    for (int i = 0; i < reps; i++) f();
    cudaEventRecord(b, stream); CK(cudaEventSynchronize(b));
    float ms; cudaEventElapsedTime(&ms, a, b);
    return ms / reps / 1e3;
}
template<class T> static T *up(const std::vector<T> &v) {
    T *d; CK(cudaMalloc(&d, v.size() * sizeof(T))); CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return d;
}
template<class T> static std::vector<T> down(const T *d, size_t n) {
    std::vector<T> v(n); CK(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return v;
}
/* the largest |got - want| over sampled outputs, relative to the products' magnitude */
struct Err { double rel = 0; void add(double got, double want, double mag) { double e = std::fabs(got - want) / (mag > 0 ? mag : 1); if (e > rel) rel = e; } };

int main(int argc, char **argv) {
    int S = argc > 1 ? atoi(argv[1]) : 1024, R = argc > 2 ? atoi(argv[2]) : 520, NK = argc > 3 ? atoi(argv[3]) : 25024;
    if (!mimo_cuda_init(0)) return 1;
    const int H = 6144, QKV = 27136, OI = 16384, MI = 2048;
    printf("rows %d, expert rows %d, keys %d\n", S, R, NK);
    /* ---- qkv: FP8 [27136][6144], a scale per (row, 128 columns) */
    {
        std::vector<uint8_t> w((size_t)QKV * H);
        for (auto &b : w) { unsigned v = rnd() & 0xff; if ((v & 127) == 127) v ^= 1; if ((v & 120) > 64) v &= ~8u; b = (uint8_t)v; }
        std::vector<float> sc((size_t)QKV * (H / 128)), x((size_t)S * H);
        for (auto &s : sc) s = 0.001f * (1 + rnd() % 64);
        for (auto &v : x) v = frand();
        uint8_t *dw = up(w); float *ds = up(sc), *dx_ = up(x), *dy_;
        CK(cudaMalloc(&dy_, (size_t)S * QKV * 4));
        dim3 g(QKV / TL_B, (S + TL_B - 1) / TL_B);
        double t = timed([&] { fp8_tiled<<<g, 256, 0, stream>>>(dw, ds, dx_, dy_, H, QKV, S); }, 5);
        auto y = down(dy_, (size_t)S * QKV);
        Err e;
        for (int k = 0; k < 64; k++) {
            int s = rnd() % S, o = rnd() % QKV; double acc = 0, mag = 0;
            for (int i = 0; i < H; i++) { double p = (double)x[(size_t)s * H + i] * h_fp8(w[(size_t)o * H + i]) * sc[(size_t)o * (H / 128) + i / 128]; acc += p; mag += std::fabs(p); }
            e.add(y[(size_t)s * QKV + o], acc, mag);
        }
        printf("qkv fp8_tiled      %8.3f ms  %6.2f TFLOP/s  err %.2g\n", t * 1e3, 2.0 * S * H * QKV / t / 1e12, e.rel);
        {
            MimoCudaTensor tq{}; tq.w = dw; tq.scales = ds; tq.fmt = 2; tq.I = H; tq.O = QKV;
            double t2 = timed([&] { gemm_launch(&tq, dx_, S, dy_, stream); }, 5);
            auto y2 = down(dy_, (size_t)S * QKV);
            Err e2; st = 99;
            for (int k = 0; k < 64; k++) {
                int s = rnd() % S, o = rnd() % QKV; double acc = 0, mag = 0;
                for (int i = 0; i < H; i++) { double p = (double)x[(size_t)s * H + i] * h_fp8(w[(size_t)o * H + i]) * sc[(size_t)o * (H / 128) + i / 128]; acc += p; mag += std::fabs(p); }
                e2.add(y2[(size_t)s * QKV + o], acc, mag);
            }
            printf("qkv gemm_tiled     %8.3f ms  %6.2f TFLOP/s  err %.2g\n", t2 * 1e3, 2.0 * S * H * QKV / t2 / 1e12, e2.rel);
        }
        cudaFree(dw); cudaFree(ds); cudaFree(dx_); cudaFree(dy_);
    }
    /* ---- o_proj: packed bf16 [6144][16384] */
    {
        std::vector<uint16_t> w((size_t)H * OI);
        for (auto &b : w) { float f = 0; for (int k = 0; k < 4; k++) f += frand() * 0.5f; f *= 0.02f; uint32_t u; memcpy(&u, &f, 4); b = (uint16_t)(u >> 16); }
        MimoCudaTensor *t5 = mimo_cuda_upload(w.data(), nullptr, 5, OI, H);
        std::vector<float> x((size_t)S * OI);
        for (auto &v : x) v = frand();
        float *dx_ = up(x), *dy_;
        CK(cudaMalloc(&dy_, (size_t)S * H * 4));
        dim3 g(H / TL_B, (S + TL_B - 1) / TL_B);
        double t = timed([&] { bf16p_tiled<<<g, 256, 0, stream>>>((const uint8_t *)t5->w, t5->codes, t5->lo, t5->exc_off, t5->exc, dx_, dy_, OI, H, S); }, 5);
        auto y = down(dy_, (size_t)S * H);
        Err e;
        for (int k = 0; k < 64; k++) {
            int s = rnd() % S, o = rnd() % H; double acc = 0, mag = 0;
            for (int i = 0; i < OI; i++) { uint32_t u = (uint32_t)w[(size_t)o * OI + i] << 16; float f; memcpy(&f, &u, 4); double p = (double)x[(size_t)s * OI + i] * f; acc += p; mag += std::fabs(p); }
            e.add(y[(size_t)s * H + o], acc, mag);
        }
        printf("o_proj bf16p_tiled %8.3f ms  %6.2f TFLOP/s  err %.2g\n", t * 1e3, 2.0 * S * H * OI / t / 1e12, e.rel);
        {

            double t2 = timed([&] { gemm_launch(t5, dx_, S, dy_, stream); }, 5);
            auto y2 = down(dy_, (size_t)S * H);
            double same = 0;
            for (size_t i = 0; i < y2.size(); i++) same = std::max(same, (double)std::fabs(y2[i] - y[i]) / (std::fabs(y[i]) + 1e-3));
            printf("o_proj gemm_tiled  %8.3f ms  %6.2f TFLOP/s  vs bf16p_tiled %.2g\n", t2 * 1e3, 2.0 * S * H * OI / t2 / 1e12, same);
        }
        cudaFree(dx_); cudaFree(dy_);
    }
    /* ---- an expert: MXFP4 gate/up [2048][6144], down [6144][2048], e8m0 per 32 columns */
    {
        size_t dq = 0, ds = dq + (size_t)H * (MI / 2), gq = ds + (size_t)H * (MI / 32), gs = gq + (size_t)MI * (H / 2);
        size_t uq = gs + (size_t)MI * (H / 32), us = uq + (size_t)MI * (H / 2), eb = us + (size_t)MI * (H / 32);
        std::vector<uint8_t> w(eb);
        for (size_t i = 0; i < eb; i++) w[i] = (uint8_t)rnd();
        for (size_t i = ds; i < gq; i++) w[i] = (uint8_t)(118 + rnd() % 6);
        for (size_t i = gs; i < uq; i++) w[i] = (uint8_t)(118 + rnd() % 6);
        for (size_t i = us; i < eb; i++) w[i] = (uint8_t)(118 + rnd() % 6);
        int n = R * 4;   /* the block's rows; the expert takes every fourth */
        std::vector<float> x((size_t)n * H);
        for (auto &v : x) v = frand();
        std::vector<int> idx(R);
        for (int r = 0; r < R; r++) idx[r] = r * 4 + (int)(rnd() % 4);
        uint8_t *dw = up(w); float *dx_ = up(x), *dg, *dy_; int *di = up(idx);
        CK(cudaMalloc(&dg, (size_t)R * MI * 4)); CK(cudaMalloc(&dy_, (size_t)R * H * 4));
        dim3 gu((R + MOE_BM - 1) / MOE_BM, MI / MOE_BN), dn((R + MOE_BM - 1) / MOE_BM, H / MOE_BN);
        double t1 = timed([&] { moe_gateup<<<gu, 256, 0, stream>>>(dx_, di, R, H, MI, dw, gq, gs, uq, us, dg); }, 10);
        double t2 = timed([&] { moe_down<<<dn, 256, 0, stream>>>(dg, R, H, MI, dw, dq, ds, dy_); }, 10);
        auto g = down(dg, (size_t)R * MI), y = down(dy_, (size_t)R * H);
        auto wq = [&](size_t q, size_t s, int O, int I, int o, int i) {
            uint8_t b = w[q + (size_t)o * (I / 2) + i / 2];
            return (double)h_lut[i & 1 ? b >> 4 : b & 15] * std::ldexp(1.0, (int)w[s + (size_t)o * (I / 32) + i / 32] - 127);
        };
        Err eg, ed;
        for (int k = 0; k < 32; k++) {
            int r = rnd() % R, o = rnd() % MI; double a = 0, b = 0, ma = 0, mb = 0;
            for (int i = 0; i < H; i++) {
                double xv = x[(size_t)idx[r] * H + i], pa = xv * wq(gq, gs, MI, H, o, i), pb = xv * wq(uq, us, MI, H, o, i);
                a += pa; b += pb; ma += std::fabs(pa); mb += std::fabs(pb);
            }
            double want = a / (1 + std::exp(-a)) * b, mag = std::fabs(a / (1 + std::exp(-a))) * mb + ma * std::fabs(b) * 0.5;
            eg.add(g[(size_t)r * MI + o], want, mag);
            int o2 = rnd() % H; double c = 0, mc = 0;
            for (int i = 0; i < MI; i++) { double p = (double)g[(size_t)r * MI + i] * wq(dq, ds, H, MI, o2, i); c += p; mc += std::fabs(p); }
            ed.add(y[(size_t)r * H + o2], c, mc);
        }
        double fl = 2.0 * R * H * MI;
        printf("expert gate+up     %8.3f ms  %6.2f TFLOP/s  err %.2g\n", t1 * 1e3, 2 * fl / t1 / 1e12, eg.rel);
        printf("expert down        %8.3f ms  %6.2f TFLOP/s  err %.2g\n", t2 * 1e3, fl / t2 / 1e12, ed.rel);
        {
            GemmW gp{}; gp.q = dw + gq; gp.s8 = dw + gs; gp.q2 = dw + uq; gp.s82 = dw + us;
            GemmW gd{}; gd.q = dw + dq; gd.s8 = dw + ds;
            float *dg2, *dy2; CK(cudaMalloc(&dg2, (size_t)R * MI * 4)); CK(cudaMalloc(&dy2, (size_t)R * H * 4));
            dim3 ga(MI / (GB_N / 2), (R + GB_M - 1) / GB_M), gb2(H / GB_N, (R + GB_M - 1) / GB_M);
            double t3 = timed([&] { gemm_tiled<GW_PAIR><<<ga, 256, 0, stream>>>(dx_, H, di, R, H, MI, gp, dg2, MI); }, 10);
            double t4 = timed([&] { gemm_tiled<GW_MX4><<<gb2, 256, 0, stream>>>(dg, MI, nullptr, R, MI, H, gd, dy2, H); }, 10);
            auto g2 = down(dg2, (size_t)R * MI), y2 = down(dy2, (size_t)R * H);
            Err eg2, ed2;
            st = 4242;
            for (int k = 0; k < 32; k++) {
                int r = rnd() % R, o = rnd() % MI; double a = 0, b = 0, ma = 0, mb = 0;
                for (int i = 0; i < H; i++) {
                    double xv = x[(size_t)idx[r] * H + i], pa = xv * wq(gq, gs, MI, H, o, i), pb = xv * wq(uq, us, MI, H, o, i);
                    a += pa; b += pb; ma += std::fabs(pa); mb += std::fabs(pb);
                }
                double want = a / (1 + std::exp(-a)) * b, mag = std::fabs(a / (1 + std::exp(-a))) * mb + ma * std::fabs(b) * 0.5;
                eg2.add(g2[(size_t)r * MI + o], want, mag);
                int o2 = rnd() % H; double c = 0, mc = 0;
                for (int i = 0; i < MI; i++) { double p = (double)g[(size_t)r * MI + i] * wq(dq, ds, H, MI, o2, i); c += p; mc += std::fabs(p); }
                ed2.add(y2[(size_t)r * H + o2], c, mc);
            }
            double rms = 0, dmax = 0;
            for (size_t i = 0; i < y2.size(); i++) { rms += (double)y[i] * y[i]; dmax = std::max(dmax, (double)std::fabs(y2[i] - y[i])); }
            rms = std::sqrt(rms / y2.size());
            printf("expert gate+up new %8.3f ms  %6.2f TFLOP/s  err %.2g\n", t3 * 1e3, 2 * fl / t3 / 1e12, eg2.rel);
            printf("expert down new    %8.3f ms  %6.2f TFLOP/s  err %.2g; max |new - old| %.2g of the outputs' rms\n", t4 * 1e3, fl / t4 / 1e12, ed2.rel, dmax / rms);
            cudaFree(dg2); cudaFree(dy2);
        }
        cudaFree(dw); cudaFree(dx_); cudaFree(dg); cudaFree(dy_); cudaFree(di);
    }
    /* ---- attention: 1024 rows at the end of NK keys (a full layer), and a window of 128 */
    for (int full = 1; full >= 0; full--) {
        int n = S, nh = 128, kvh = 8, HD = 192, VD = 128, window = full ? 0 : 128;
        int p0 = NK - n, lo = full ? 0 : (p0 - 127 > 0 ? p0 - 127 : 0), nkeys = p0 + n - lo;
        std::vector<float> q((size_t)n * nh * HD), k((size_t)nkeys * kvh * HD), v((size_t)nkeys * kvh * VD), sink(nh);
        for (auto &a : q) a = frand();
        for (auto &a : k) a = frand();
        for (auto &a : v) a = frand();
        for (auto &a : sink) a = frand();
        float *dq_ = up(q), *dk = up(k), *dv = up(v), *dsink = up(sink), *dout;
        CK(cudaMalloc(&dout, (size_t)n * nh * VD * 4));
        int G = nh / kvh, BR = 64 / G, threads = BR * G * 4;
        size_t shared = (size_t)32 * (HD + VD) * sizeof(float);
        CK(cudaFuncSetAttribute(attn_rows<192, 128>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared));
        size_t fshared = sizeof(float) * (size_t)(64 * (FA_Q + 4) + 64 * (128 + 4));
        CK(cudaFuncSetAttribute(attn_flash<192, 128>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)fshared));
        dim3 grid((n + BR - 1) / BR, kvh), fgrid((n + FA_Q / G - 1) / (FA_Q / G), kvh);
        float scale = 1.0f / sqrtf(192.0f);
        double pairs = full ? (double)n * ((double)p0 + (n + 1) / 2.0) : (double)n * 128;
        double fl = 2.0 * pairs * nh * (HD + VD);
        for (int kern = 0; kern < 2; kern++) {
            double t = timed([&] {
                if (kern == 0)
                    attn_rows<192, 128><<<grid, threads, shared, stream>>>(dq_, nh * HD, dk, dv, full ? nullptr : dsink, nullptr, dout,
                                                                          n, nh, kvh, G, BR, p0, lo, window, lo, nkeys, scale, 1, 1);
                else
                    attn_flash<192, 128><<<fgrid, 256, fshared, stream>>>(dq_, nh * HD, dk, dv, full ? nullptr : dsink, nullptr, dout,
                                                                         n, nh, kvh, G, p0, lo, window, lo, nkeys, scale, 1, 1);
            }, full ? 2 : 10);
            CK(cudaGetLastError());
            auto o = down(dout, (size_t)n * nh * VD);
            Err e;
            st = 77;
            for (int s = 0; s < 8; s++) {
                int t_ = rnd() % n, h = rnd() % nh, kh = h / G, p = p0 + t_, first = window ? std::max(p - window + 1, lo) : lo;
                std::vector<double> sc(p - first + 1);
                double best = full ? -1e300 : sink[h];
                for (int j = first; j <= p; j++) {
                    double a = 0;
                    for (int i = 0; i < HD; i++) a += (double)q[((size_t)t_ * nh + h) * HD + i] * k[((size_t)(j - lo) * kvh + kh) * HD + i];
                    sc[j - first] = a * scale; best = std::max(best, sc[j - first]);
                }
                double den = full ? 0 : std::exp(sink[h] - best);
                for (auto &a : sc) { a = std::exp(a - best); den += a; }
                for (int i = 0; i < VD; i++) {
                    double acc = 0, mag = 0;
                    for (int j = first; j <= p; j++) { double pr = sc[j - first] / den * v[((size_t)(j - lo) * kvh + kh) * VD + i]; acc += pr; mag += std::fabs(pr); }
                    e.add(o[((size_t)t_ * nh + h) * VD + i], acc, mag);
                }
            }
            printf("attention %s %s %8.3f ms  %6.2f TFLOP/s  err %.2g\n", full ? "full" : "swa ", kern ? "flash" : "rows ", t * 1e3, fl / t / 1e12, e.rel);
        }
        cudaFree(dq_); cudaFree(dk); cudaFree(dv); cudaFree(dsink); cudaFree(dout);
    }
    return 0;
}
