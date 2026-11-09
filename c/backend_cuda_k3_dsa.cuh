/* Kimi K3.1's lightning indexer on the GPU: the kernels of k3cu_dsa_mask (backend_cuda_k3.cu).
 * Kept apart so tools/check_k3_dsa_kernels.cpp can compile them as host code, a std::thread a CUDA
 * thread, and check them against kimi_k3.c's k3_dsa_scores and k3_dsa_pick without a card. */
#ifndef BACKEND_CUDA_K3_DSA_CUH
#define BACKEND_CUDA_K3_DSA_CUH
// The scores: S[r][t] = kimi_k3.c's k3_dsa_scores to the bit -- for each head h in order, a =
// fma(q[r][h][d], K[t][d], a) over d from +0, then s = fma(w[r][h], a > 0 ? a : +0, s) from +0; NaN
// -> -inf, -0 -> +0 -- for t below the row's nt. A block: 32 rows by 64 positions, a thread 2 rows
// by 4 positions (64 apart by 16); the keys of its 64 positions transposed in shared memory for
// every head, a head's queries of its 32 rows beside them.
#define DS_R 32
#define DS_T 64
#define DS_HD 128
__global__ void __launch_bounds__(256) k3_dsa_scores(const float *__restrict__ Kc, const float *__restrict__ q,
                                                     const float *__restrict__ w, int rows, int first, int nh, int hd,
                                                     float *__restrict__ S, int ldS) {
    __shared__ float kt[DS_HD][DS_T];
    __shared__ float qs[DS_R][DS_HD];
    const int r0 = blockIdx.y * DS_R, t0 = blockIdx.x * DS_T;
    const int last = first + min(r0 + DS_R, rows) - 1;   // the block's last row's position
    if (t0 > last) return;                                 // past every row's nt
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    for (int e = threadIdx.x; e < DS_T * hd; e += blockDim.x) {
        const int tl = e / hd, d = e - tl * hd, t = t0 + tl;
        kt[d][tl] = t <= last ? Kc[(size_t)t * hd + d] : 0.0f;
    }
    float s[2][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) s[i][j] = 0.0f;
    for (int h = 0; h < nh; h++) {
        __syncthreads();
        for (int e = threadIdx.x; e < DS_R * hd; e += blockDim.x) {
            const int rl = e / hd, d = e - rl * hd, r = r0 + rl;
            qs[rl][d] = r < rows ? q[((size_t)r * nh + h) * hd + d] : 0.0f;
        }
        __syncthreads();
        float a[2][4];
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 4; j++) a[i][j] = 0.0f;
        for (int d = 0; d < hd; d++) {
            const float q0 = qs[2 * ty][d], q1 = qs[2 * ty + 1][d];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const float k = kt[d][tx + 16 * j];
                a[0][j] = __fmaf_rn(q0, k, a[0][j]);
                a[1][j] = __fmaf_rn(q1, k, a[1][j]);
            }
        }
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int r = r0 + 2 * ty + i;
            const float wh = r < rows ? w[(size_t)r * nh + h] : 0.0f;
#pragma unroll
            for (int j = 0; j < 4; j++) s[i][j] = __fmaf_rn(wh, a[i][j] > 0.0f ? a[i][j] : 0.0f, s[i][j]);
        }
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int r = r0 + 2 * ty + i;
        if (r >= rows) continue;
        const int nt = first + r + 1;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int t = t0 + tx + 16 * j;
            if (t >= nt) continue;
            float v = s[i][j];
            if (v != v) v = -INFINITY; else if (v == 0.0f) v = 0.0f;
            S[(size_t)r * ldS + t] = v;
        }
    }
}
// a score as an unsigned key that sorts as the float does (no NaN here; -0 is gone)
__device__ __forceinline__ uint32_t ds_key(float f) {
    const uint32_t b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
// The pick: a block a row. The K-th largest key by a radix select (8 bits a pass, the bins counted
// in shared memory, thread 0 walking them from the top), then the mask: every position above it,
// and of the positions equal to it the lowest ones, as many as K leaves (kimi_k3.c's k3_dsa_pick);
// each thread a run of whole mask words in position order, an exclusive scan of the runs' ties.
__global__ void __launch_bounds__(256) k3_dsa_pick(const float *__restrict__ S, int ldS, int first, int K,
                                                   uint32_t *__restrict__ M, int mw) {
    __shared__ unsigned hist[256];
    __shared__ unsigned ties[256];
    __shared__ uint32_t pre, pmask;
    __shared__ int need;
    const int r = blockIdx.x, nt = first + r + 1;
    const float *Sr = S + (size_t)r * ldS;
    uint32_t *Mr = M + (size_t)r * mw;
    const int nw = (nt + 31) / 32;
    if (nt <= K) {   // every position (the caller's rows all select; kept for safety)
        for (int i = threadIdx.x; i < mw; i += blockDim.x)
            Mr[i] = i < nt / 32 ? 0xffffffffu : (i == nt / 32 && (nt & 31) ? (1u << (nt & 31)) - 1u : 0u);
        return;
    }
    if (threadIdx.x == 0) { pre = 0; pmask = 0; need = K; }
    for (int shift = 24; shift >= 0; shift -= 8) {
        hist[threadIdx.x] = 0;
        __syncthreads();
        const uint32_t p = pre, m = pmask;
        for (int t = threadIdx.x; t < nt; t += blockDim.x) {
            const uint32_t u = ds_key(Sr[t]);
            if ((u & m) == p) atomicAdd(&hist[(u >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned cum = 0;
            for (int b = 255; b >= 0; b--) {
                if (cum + hist[b] >= (unsigned)need) { need -= (int)cum; pre |= (uint32_t)b << shift; pmask |= 255u << shift; break; }
                cum += hist[b];
            }
        }
        __syncthreads();
    }
    const uint32_t v = pre;
    const int per = (nw + (int)blockDim.x - 1) / (int)blockDim.x, w0 = threadIdx.x * per, w1 = min(nw, w0 + per);
    unsigned eq = 0;
    for (int wi = w0; wi < w1; wi++)
        for (int b = 0; b < 32; b++) { const int t = wi * 32 + b; if (t < nt && ds_key(Sr[t]) == v) eq++; }
    ties[threadIdx.x] = eq;
    __syncthreads();
    if (threadIdx.x == 0) { unsigned run = 0; for (int i = 0; i < (int)blockDim.x; i++) { const unsigned c = ties[i]; ties[i] = run; run += c; } }
    __syncthreads();
    unsigned rank = ties[threadIdx.x];
    for (int wi = w0; wi < w1; wi++) {
        uint32_t word = 0;
        for (int b = 0; b < 32; b++) {
            const int t = wi * 32 + b;
            if (t >= nt) break;
            const uint32_t u = ds_key(Sr[t]);
            if (u > v) word |= 1u << b;
            else if (u == v) { if (rank < (unsigned)need) word |= 1u << b; rank++; }
        }
        Mr[wi] = word;
    }
    for (int i = nw + threadIdx.x; i < mw; i += blockDim.x) Mr[i] = 0;
}
#endif
