/* Native FP8/BF16 dense offload for MiMo. No weight/activation requantization.
 * The FP8 scale table is PER ROW, after MiMo's pre-sharded QKV deinterleave;
 * using the generic backend's 128x128 scale grid here would be incorrect.
 * One warp per output row, four warps per block. Every batch size uses the
 * same reduction, preserving prompt-chunk and prefix-reuse consistency. */
#include "backend_cuda_mimo.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <chrono>

struct MimoCudaTensor {
    void *w;
    float *scales;
    int fmt, I, O;
    MimoCudaTensor *next;
};
static cudaStream_t stream;
static int ready, selected = -1, failed;
static MimoCudaTensor *tensors;
static float *dx, *dy, *hx, *hy;
static size_t xcap, ycap, resident, tensor_count;
static unsigned long long calls, rows, fallback;
static double seconds[5];

static int checked(cudaError_t e, const char *where) {
    if (e == cudaSuccess) return 1;
    if (!failed) fprintf(stderr, "[CUDA] mimo %s: %s; retaining CPU weights\n", where, cudaGetErrorString(e));
    failed = 1;
    return 0;
}

extern "C" int mimo_cuda_init(int device) {
    if (ready) return device == selected && !failed;
    failed = 0;
    if (!checked(cudaSetDevice(device), "select") ||
        !checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream")) return 0;
    cudaDeviceProp prop;
    if (!checked(cudaGetDeviceProperties(&prop, device), "properties")) {
        cudaStreamDestroy(stream); stream = nullptr; return 0;
    }
    selected = device; ready = 1;
    fprintf(stderr, "[CUDA] mimo device %d: %s; native dense weights, float32 activations\n", device, prop.name);
    return 1;
}

extern "C" size_t mimo_cuda_free_bytes(void) {
    size_t available = 0, total = 0;
    if (!ready || failed || !checked(cudaMemGetInfo(&available, &total), "memory info")) return 0;
    return available;
}

extern "C" MimoCudaTensor *mimo_cuda_upload(const void *w, const float *scales, int fmt, int I, int O) {
    if (!ready || failed || !w || I < 1 || I > INT_MAX - 127 || O < 1 || O > INT_MAX - 3 || fmt < 0 || fmt > 4 ||
        ((fmt == 2 || fmt == 3 || fmt == 4) && !scales)) return nullptr;
    size_t esz = fmt == 0 ? 4 : fmt == 1 ? 2 : 1;
    if ((size_t)O > SIZE_MAX / (size_t)I / esz) return nullptr;
    size_t wb = (size_t)O * I * esz;
    size_t ns = fmt == 2 || fmt == 4 ? (size_t)O * (((size_t)I + 127) / 128) : fmt == 3 ? (size_t)O : 0;
    if (ns > SIZE_MAX / sizeof(float)) return nullptr;
    size_t sb = ns * sizeof(float);
    MimoCudaTensor *t = (MimoCudaTensor *)calloc(1, sizeof *t);
    if (!t) return nullptr;
    /* OOM is local to this matrix, not a poisoned compute context. */
    if (cudaMalloc(&t->w, wb) != cudaSuccess) { cudaGetLastError(); free(t); return nullptr; }
    if (sb && cudaMalloc(&t->scales, sb) != cudaSuccess) {
        cudaGetLastError(); cudaFree(t->w); free(t); return nullptr;
    }
    if (!checked(cudaMemcpy(t->w, w, wb, cudaMemcpyHostToDevice), "weights") ||
        (sb && !checked(cudaMemcpy(t->scales, scales, sb, cudaMemcpyHostToDevice), "scales"))) {
        cudaFree(t->w); cudaFree(t->scales); free(t); return nullptr;
    }
    t->fmt = fmt; t->I = I; t->O = O; t->next = tensors; tensors = t;
    resident += wb + sb; tensor_count++;
    return t;
}

__device__ static float e4m3(unsigned char v) {
    unsigned int a = v & 127, e = a >> 3, m = a & 7;
    float f = e ? __uint_as_float(((e + 120) << 23) | (m << 20)) : (float)m * (1.0f / 512.0f);
    if (a == 127) f = __uint_as_float(0x7fc00000u);
    return v & 128 ? -f : f;
}
__device__ static float warp_sum(float v) {
    for (int d = 16; d; d >>= 1) v += __shfl_down_sync(0xffffffffu, v, d);
    return v;
}
template<int FMT>
__global__ static void dense(const void *__restrict__ weights, const float *__restrict__ scales,
                             const float *__restrict__ x, float *__restrict__ y, int I, int O) {
    int o = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (o >= O) return;
    const float *xs = x + (size_t)blockIdx.y * I;
    size_t base = (size_t)o * I;
    float sum = 0;
    if (FMT == 2 || FMT == 4) {
        double total = 0;
        int nb = (I + 127) / 128;
        for (int b = 0; b < nb; b++) {
            float part = 0;
            for (int k = lane; k < 128 && b * 128 + k < I; k += 32) {
                int i = b * 128 + k;
                unsigned char q = ((const unsigned char *)weights)[base + i];
                part = fmaf(FMT == 2 ? e4m3(q) : (float)(signed char)q, xs[i], part);
            }
            part = warp_sum(part);
            if (!lane) total += (double)part * scales[(size_t)o * nb + b];
        }
        if (!lane) sum = (float)total;
    } else {
        for (int i = lane; i < I; i += 32) {
            float w;
            if (FMT == 0) w = ((const float *)weights)[base + i];
            else if (FMT == 1) w = __uint_as_float((unsigned int)((const uint16_t *)weights)[base + i] << 16);
            else w = (float)((const int8_t *)weights)[base + i];
            sum = fmaf(w, xs[i], sum);
        }
        sum = warp_sum(sum);
        if (FMT == 3 && !lane) sum *= scales[o];
    }
    if (!lane) y[(size_t)blockIdx.y * O + o] = sum;
}

/* The wide kernels: the trunk's shapes (I a multiple of 512 for FP8, of 256 for
 * BF16), one warp per output row, each lane loading 16 weight bytes at a time.
 * A block of up to TS activation rows shares every weight it decodes, so a
 * prompt chunk streams the matrix once per TS rows instead of once per row.
 * Each row's arithmetic is the same whatever the chunk: the order of every sum
 * depends only on the element index, so S rows at once give the bytes of S
 * calls of one row (the chunk invariance the narrow kernel also keeps).
 *
 * FP8: a lane's 16 products are summed in order, the 8 lanes of a 128-column
 * block are combined by a fixed butterfly, and the block sums are scaled and
 * added in double, block by block, as in the narrow kernel. The e4m3 bytes are
 * widened by the hardware conversion (exact: every e4m3 value is a half). */
#define MIMO_TS 8
__device__ static __forceinline__ void fp8x16(const uint4 q, float *w) {
    const unsigned int v[4] = {q.x, q.y, q.z, q.w};
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        __half2_raw lo = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(v[k] & 0xffffu), __NV_E4M3);
        __half2_raw hi = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(v[k] >> 16), __NV_E4M3);
        float2 a = __half22float2(*(const __half2 *)&lo), b = __half22float2(*(const __half2 *)&hi);
        w[4 * k] = a.x; w[4 * k + 1] = a.y; w[4 * k + 2] = b.x; w[4 * k + 3] = b.y;
    }
}
__device__ static __forceinline__ void i8x16(const uint4 q, float *w) {
    const unsigned int v[4] = {q.x, q.y, q.z, q.w};
    #pragma unroll
    for (int k = 0; k < 4; k++)
        #pragma unroll
        for (int b = 0; b < 4; b++) w[4 * k + b] = (float)(signed char)(v[k] >> (8 * b));
}
/* FP8 (FMT 2) or int8 with the same block scales (FMT 4, the opt-in o_proj form) */
template<int TS, int FMT>
__global__ static void __launch_bounds__(128) fp8_wide(const uint8_t *__restrict__ weights, const float *__restrict__ scales,
                                                       const float *__restrict__ x, float *__restrict__ y,
                                                       int I, int O, int S) {
    int o = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (o >= O) return;
    int s0 = blockIdx.y * TS, ns = S - s0 < TS ? S - s0 : TS, nb = I / 128;
    const uint4 *wr = (const uint4 *)(weights + (size_t)o * I);
    const float *sc = scales + (size_t)o * nb;
    double total[TS];
    #pragma unroll
    for (int s = 0; s < TS; s++) total[s] = 0;
    #pragma unroll 4
    for (int it = 0; it < I / 512; it++) {
        int c = it * 32 + lane;
        float w[16];
        if (FMT == 4) i8x16(__ldcs(wr + c), w);
        else fp8x16(__ldcs(wr + c), w);
        float part[TS];
        #pragma unroll
        for (int s = 0; s < TS; s++) {
            part[s] = 0;
            if (s < ns) {
                const float4 *xs = (const float4 *)(x + (size_t)(s0 + s) * I + (size_t)c * 16);
                float p = 0;
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    float4 v = __ldg(xs + k);
                    p = fmaf(w[4 * k], v.x, p); p = fmaf(w[4 * k + 1], v.y, p);
                    p = fmaf(w[4 * k + 2], v.z, p); p = fmaf(w[4 * k + 3], v.w, p);
                }
                part[s] = p;
            }
        }
        #pragma unroll
        for (int s = 0; s < TS; s++) {
            float p = part[s];
            p += __shfl_xor_sync(0xffffffffu, p, 1);
            p += __shfl_xor_sync(0xffffffffu, p, 2);
            p += __shfl_xor_sync(0xffffffffu, p, 4);
            part[s] = p;
        }
        /* FP64 runs at 1/64 rate on this class of GPU: only the tile's real rows */
        if (!(lane & 7)) {
            double scb = sc[it * 4 + (lane >> 3)];
            #pragma unroll
            for (int s = 0; s < TS; s++) if (s < ns) total[s] += (double)part[s] * scb;
        }
    }
    #pragma unroll
    for (int s = 0; s < TS; s++) {
        if (s >= ns) break;
        double t = total[s];
        t += __shfl_xor_sync(0xffffffffu, t, 8);
        t += __shfl_xor_sync(0xffffffffu, t, 16);
        if (!lane) y[(size_t)(s0 + s) * O + o] = (float)t;
    }
}
/* BF16: a lane's 8 products per load in order, the lanes combined by a fixed
 * butterfly at the end. */
template<int TS>
__global__ static void __launch_bounds__(128) bf16_wide(const uint16_t *__restrict__ weights, const float *__restrict__ x,
                                                        float *__restrict__ y, int I, int O, int S) {
    int o = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (o >= O) return;
    int s0 = blockIdx.y * TS, ns = S - s0 < TS ? S - s0 : TS;
    const uint4 *wr = (const uint4 *)(weights + (size_t)o * I);
    float acc[TS];
    #pragma unroll
    for (int s = 0; s < TS; s++) acc[s] = 0;
    #pragma unroll 4
    for (int it = 0; it < I / 256; it++) {
        int c = it * 32 + lane;
        uint4 q = __ldcs(wr + c);
        const unsigned int v[4] = {q.x, q.y, q.z, q.w};
        float w[8];
        #pragma unroll
        for (int k = 0; k < 4; k++) { w[2 * k] = __uint_as_float(v[k] << 16); w[2 * k + 1] = __uint_as_float(v[k] & 0xffff0000u); }
        #pragma unroll
        for (int s = 0; s < TS; s++) {
            if (s < ns) {
                const float4 *xs = (const float4 *)(x + (size_t)(s0 + s) * I + (size_t)c * 8);
                float4 a = __ldg(xs), b = __ldg(xs + 1);
                float p = acc[s];
                p = fmaf(w[0], a.x, p); p = fmaf(w[1], a.y, p); p = fmaf(w[2], a.z, p); p = fmaf(w[3], a.w, p);
                p = fmaf(w[4], b.x, p); p = fmaf(w[5], b.y, p); p = fmaf(w[6], b.z, p); p = fmaf(w[7], b.w, p);
                acc[s] = p;
            }
        }
    }
    #pragma unroll
    for (int s = 0; s < TS; s++) {
        float p = acc[s];
        for (int d = 16; d; d >>= 1) p += __shfl_xor_sync(0xffffffffu, p, d);
        if (!lane && s < ns) y[(size_t)(s0 + s) * O + o] = p;
    }
}
static int wide(const MimoCudaTensor *t) {
    return ((t->fmt == 2 || t->fmt == 4) && t->I % 512 == 0) || (t->fmt == 1 && t->I % 256 == 0);
}

/* Grow transactionally: a partially successful allocation neither leaks nor
 * destroys the old scratch. Pinned staging avoids pageable async copies. */
static int scratch(float **d, float **h, size_t *cap, size_t n) {
    if (n <= *cap) return 1;
    float *nd = nullptr, *nh = nullptr;
    if (cudaMalloc(&nd, n) != cudaSuccess) { cudaGetLastError(); return 0; }
    if (cudaMallocHost(&nh, n) != cudaSuccess) { cudaGetLastError(); cudaFree(nd); return 0; }
    cudaFree(*d); cudaFreeHost(*h);
    *d = nd; *h = nh; *cap = n;
    return 1;
}
extern "C" int mimo_cuda_matmul(MimoCudaTensor *t, float *y, const float *x, int S) {
    if (!ready || failed || !t || !x || !y || S < 1 || S > 65535) return 0;
    if ((size_t)S > SIZE_MAX / sizeof(float) / (size_t)t->I ||
        (size_t)S > SIZE_MAX / sizeof(float) / (size_t)t->O) return 0;
    auto started = std::chrono::steady_clock::now();
    size_t xb = (size_t)S * t->I * sizeof(float), yb = (size_t)S * t->O * sizeof(float);
    if (!scratch(&dx, &hx, &xcap, xb) || !scratch(&dy, &hy, &ycap, yb)) { fallback++; return 0; }
    memcpy(hx, x, xb);
    if (!checked(cudaMemcpyAsync(dx, hx, xb, cudaMemcpyHostToDevice, stream), "input")) return 0;
    dim3 grid((unsigned)((t->O + 3) / 4), (unsigned)S);
    /* one row (a decode step) gets a one-row tile: fewer registers, more warps in
     * flight; a row's arithmetic is the same in either */
    int ts = S == 1 ? 1 : MIMO_TS;
    dim3 tiles((unsigned)((t->O + 3) / 4), (unsigned)((S + ts - 1) / ts));
    if (wide(t) && t->fmt == 2 && ts == 1)
        fp8_wide<1, 2><<<tiles,128,0,stream>>>((const uint8_t *)t->w,t->scales,dx,dy,t->I,t->O,S);
    else if (wide(t) && t->fmt == 2)
        fp8_wide<MIMO_TS, 2><<<tiles,128,0,stream>>>((const uint8_t *)t->w,t->scales,dx,dy,t->I,t->O,S);
    else if (wide(t) && t->fmt == 4 && ts == 1)
        fp8_wide<1, 4><<<tiles,128,0,stream>>>((const uint8_t *)t->w,t->scales,dx,dy,t->I,t->O,S);
    else if (wide(t) && t->fmt == 4)
        fp8_wide<MIMO_TS, 4><<<tiles,128,0,stream>>>((const uint8_t *)t->w,t->scales,dx,dy,t->I,t->O,S);
    else if (wide(t) && ts == 1)
        bf16_wide<1><<<tiles,128,0,stream>>>((const uint16_t *)t->w,dx,dy,t->I,t->O,S);
    else if (wide(t))
        bf16_wide<MIMO_TS><<<tiles,128,0,stream>>>((const uint16_t *)t->w,dx,dy,t->I,t->O,S);
    else switch (t->fmt) {
    case 0: dense<0><<<grid,128,0,stream>>>(t->w,t->scales,dx,dy,t->I,t->O); break;
    case 1: dense<1><<<grid,128,0,stream>>>(t->w,t->scales,dx,dy,t->I,t->O); break;
    case 2: dense<2><<<grid,128,0,stream>>>(t->w,t->scales,dx,dy,t->I,t->O); break;
    case 3: dense<3><<<grid,128,0,stream>>>(t->w,t->scales,dx,dy,t->I,t->O); break;
    case 4: dense<4><<<grid,128,0,stream>>>(t->w,t->scales,dx,dy,t->I,t->O); break;
    default: return 0;
    }
    if (!checked(cudaGetLastError(), "kernel") ||
        !checked(cudaMemcpyAsync(hy, dy, yb, cudaMemcpyDeviceToHost, stream), "output") ||
        !checked(cudaStreamSynchronize(stream), "synchronize")) {
        cudaStreamSynchronize(stream); fallback++; return 0;
    }
    memcpy(y, hy, yb); calls++; rows += S;
    seconds[t->fmt] += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return 1;
}
extern "C" void mimo_cuda_report(void) {
    if (ready) fprintf(stderr, "[CUDA] mimo dense: %zu matrices, %.3f GiB resident, %llu matmuls, %llu rows, %llu scratch/compute fallbacks%s\n",
                       tensor_count, resident / 1073741824.0, calls, rows, fallback, failed ? ", device failed" : "");
    if (ready) fprintf(stderr, "[CUDA] mimo matmul wall time (copies + compute): f32 %.3fs, bf16 %.3fs, fp8 %.3fs, int8 %.3fs, int8 blocks %.3fs\n",
                       seconds[0], seconds[1], seconds[2], seconds[3], seconds[4]);
}
extern "C" void mimo_cuda_shutdown(void) {
    if (stream) cudaStreamSynchronize(stream);
    while (tensors) {
        MimoCudaTensor *next = tensors->next;
        cudaFree(tensors->w); cudaFree(tensors->scales); free(tensors); tensors = next;
    }
    cudaFree(dx); cudaFree(dy); cudaFreeHost(hx); cudaFreeHost(hy);
    if (stream) cudaStreamDestroy(stream);
    stream = nullptr; dx = dy = hx = hy = nullptr; xcap = ycap = resident = tensor_count = 0;
    calls = rows = fallback = 0; ready = failed = 0; selected = -1;
    memset(seconds, 0, sizeof seconds);
}
