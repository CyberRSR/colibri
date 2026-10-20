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
#include <functional>
#include <thread>
#include <vector>

struct MimoCudaTensor {
    void *w;
    float *scales;
    int fmt, I, O;
    /* fmt 5 (packed bf16): the planes inside w */
    const uint8_t *codes, *lo, *exc;
    const unsigned *exc_off;
    size_t bytes;          /* device bytes, scales included */
    MimoCudaTensor *next;
};
static cudaStream_t stream;
static int ready, selected = -1, failed;
static MimoCudaTensor *tensors;
static float *dx, *dy, *hx, *hy;
static size_t xcap, ycap, resident, tensor_count;
static unsigned long long calls, rows, fallback, packed_count, packed_exc;
static size_t packed_bytes, packed_bf16;
static double seconds[6];

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

/* fmt 5: a bf16 matrix (the input is fmt 1's) packed without loss. Each row keeps the
 * window of 15 consecutive exponents that holds most of its weights (from lo[row]); a
 * weight is its sign/mantissa byte and a 4-bit code, its exponent minus lo, or 15: the
 * exponent is the next byte of the row's exception list. 12 bits a weight instead of 16
 * (Pro's o_proj and head: 0.018% exceptions). bf16x_wide decodes every weight to its bf16
 * and sums as bf16_wide does: the same bytes as fmt 1, from three quarters of the memory.
 * Planes, one allocation: sign/mantissa [O][I], codes [O][I/2] (byte b of a lane's
 * group of 8 holds weight b's code low and weight b+4's high), lo [O], exception
 * offsets [O], exceptions. */
static void pack_count(const uint16_t *w, int I, int r0, int r1, uint8_t *lo, unsigned *cnt) {
    for (int o = r0; o < r1; o++) {
        const uint16_t *row = w + (size_t)o * I;
        unsigned h[256] = {0};
        for (int i = 0; i < I; i++) h[(row[i] >> 7) & 0xff]++;
        unsigned s = 0, best;
        int bl = 0;
        for (int e = 0; e < 15; e++) s += h[e];
        best = s;
        for (int l = 1; l <= 240; l++) {   /* lo + 15 stays a byte: the decoder adds bytewise */
            s += h[l + 14] - h[l - 1];
            if (s > best) { best = s; bl = l; }
        }
        lo[o] = (uint8_t)bl; cnt[o] = (unsigned)I - best;
    }
}
static void pack_fill(const uint16_t *w, int I, int r0, int r1, const uint8_t *lo, const unsigned *off,
                      uint8_t *sm, uint8_t *codes, uint8_t *exc) {
    for (int o = r0; o < r1; o++) {
        const uint16_t *row = w + (size_t)o * I;
        uint8_t *s = sm + (size_t)o * I, *c = codes + (size_t)o * (I / 2), *x = exc + off[o];
        int l = lo[o];
        for (int g = 0; g < I; g += 8) {
            uint8_t code[8];
            for (int k = 0; k < 8; k++) {
                unsigned v = row[g + k], e = (v >> 7) & 0xff;
                s[g + k] = (uint8_t)(((v >> 8) & 0x80) | (v & 0x7f));
                if (e >= (unsigned)l && e <= (unsigned)l + 14) code[k] = (uint8_t)(e - l);
                else { code[k] = 15; *x++ = (uint8_t)e; }
            }
            for (int b = 0; b < 4; b++) c[g / 2 + b] = (uint8_t)(code[b] | code[b + 4] << 4);
        }
    }
}
static void pack_parallel(int O, const std::function<void(int, int)> &f) {
    unsigned hw = std::thread::hardware_concurrency();
    int nt = (int)(hw ? (hw < 16 ? hw : 16) : 1);
    if (nt > O / 64) nt = O / 64 > 0 ? O / 64 : 1;
    std::vector<std::thread> th;
    for (int t = 1; t < nt; t++) th.emplace_back(f, (int)((long long)O * t / nt), (int)((long long)O * (t + 1) / nt));
    f(0, (int)((long long)O / nt));
    for (auto &x : th) x.join();
}
static MimoCudaTensor *upload_packed(const uint16_t *w, int I, int O) {
    if (I % 256) return nullptr;
    std::vector<uint8_t> lo((size_t)O);
    std::vector<unsigned> off((size_t)O);
    pack_parallel(O, [&](int r0, int r1) { pack_count(w, I, r0, r1, lo.data(), off.data()); });
    size_t nexc = 0;
    for (int o = 0; o < O; o++) { unsigned n = off[o]; off[o] = (unsigned)nexc; nexc += n; }
    if (nexc > UINT_MAX) return nullptr;
    size_t e = (size_t)O * I, a_codes = e, a_lo = e + e / 2, a_off = a_lo + (((size_t)O + 15) & ~(size_t)15),
           a_exc = a_off + (size_t)O * sizeof(unsigned), total = a_exc + nexc;
    std::vector<uint8_t> host(total);
    memcpy(host.data() + a_lo, lo.data(), (size_t)O);
    memcpy(host.data() + a_off, off.data(), (size_t)O * sizeof(unsigned));
    pack_parallel(O, [&](int r0, int r1) {
        pack_fill(w, I, r0, r1, lo.data(), off.data(), host.data(), host.data() + a_codes, host.data() + a_exc);
    });
    MimoCudaTensor *t = (MimoCudaTensor *)calloc(1, sizeof *t);
    if (!t) return nullptr;
    if (cudaMalloc(&t->w, total) != cudaSuccess) { cudaGetLastError(); free(t); return nullptr; }
    if (!checked(cudaMemcpy(t->w, host.data(), total, cudaMemcpyHostToDevice), "weights")) {
        cudaFree(t->w); free(t); return nullptr;
    }
    uint8_t *d = (uint8_t *)t->w;
    t->codes = d + a_codes; t->lo = d + a_lo; t->exc_off = (const unsigned *)(d + a_off); t->exc = d + a_exc;
    t->fmt = 5; t->I = I; t->O = O; t->bytes = total; t->next = tensors; tensors = t;
    resident += total; tensor_count++;
    packed_count++; packed_exc += nexc; packed_bytes += total; packed_bf16 += e * 2;
    return t;
}

extern "C" MimoCudaTensor *mimo_cuda_upload(const void *w, const float *scales, int fmt, int I, int O) {
    if (!ready || failed || !w || I < 1 || I > INT_MAX - 127 || O < 1 || O > INT_MAX - 3 || fmt < 0 || fmt > 5 ||
        ((fmt == 2 || fmt == 3 || fmt == 4) && !scales)) return nullptr;
    if (fmt == 5) return upload_packed((const uint16_t *)w, I, O);
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
    t->fmt = fmt; t->I = I; t->O = O; t->bytes = wb + sb; t->next = tensors; tensors = t;
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
/* Packed BF16 (fmt 5): a lane's 8 weights from 8 sign/mantissa bytes and 4 code bytes,
 * rebuilt four at a time inside 32-bit words (exponent = lo + code; the high byte of
 * a bf16 is sign | exponent >> 1, the low (exponent & 1) << 7 | mantissa), then paired
 * as bf16_wide's words. A code 15 takes its exponent from the row's exception list:
 * the warp counts its lanes' exceptions in order (rare: most iterations skip it). */
template<int TS>
__global__ static void __launch_bounds__(128) bf16x_wide(const uint8_t *__restrict__ sm, const uint8_t *__restrict__ codes,
                                                         const uint8_t *__restrict__ lo, const unsigned *__restrict__ exc_off,
                                                         const uint8_t *__restrict__ exc, const float *__restrict__ x,
                                                         float *__restrict__ y, int I, int O, int S) {
    int o = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (o >= O) return;
    int s0 = blockIdx.y * TS, ns = S - s0 < TS ? S - s0 : TS;
    const uint2 *sr = (const uint2 *)(sm + (size_t)o * I);
    const unsigned *cr = (const unsigned *)(codes + (size_t)o * (I / 2));
    const unsigned base = lo[o] * 0x01010101u;
    const uint8_t *ex = exc + exc_off[o];
    unsigned run = 0;
    float acc[TS];
    #pragma unroll
    for (int s = 0; s < TS; s++) acc[s] = 0;
    #pragma unroll 4
    for (int it = 0; it < I / 256; it++) {
        int c = it * 32 + lane;
        uint2 b = __ldcs(sr + c);
        unsigned q = __ldcs(cr + c);
        unsigned e0 = (q & 0x0f0f0f0fu) + base, e1 = ((q >> 4) & 0x0f0f0f0fu) + base;
        unsigned esc = q & (q >> 1) & (q >> 2) & (q >> 3) & 0x11111111u;
        if (__any_sync(0xffffffffu, esc)) {
            int n = __popc(esc), incl = n;
            #pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                int t = __shfl_up_sync(0xffffffffu, incl, d);
                if (lane >= d) incl += t;
            }
            const uint8_t *p = ex + run + (unsigned)(incl - n);
            #pragma unroll
            for (int k = 0; k < 4; k++)
                if (esc >> (8 * k) & 1) e0 = (e0 & ~(0xffu << (8 * k))) | (unsigned)*p++ << (8 * k);
            #pragma unroll
            for (int k = 0; k < 4; k++)
                if (esc >> (8 * k + 4) & 1) e1 = (e1 & ~(0xffu << (8 * k))) | (unsigned)*p++ << (8 * k);
            run += (unsigned)__shfl_sync(0xffffffffu, incl, 31);
        }
        unsigned h0 = (b.x & 0x80808080u) | ((e0 >> 1) & 0x7f7f7f7fu), l0 = (b.x & 0x7f7f7f7fu) | ((e0 << 7) & 0x80808080u);
        unsigned h1 = (b.y & 0x80808080u) | ((e1 >> 1) & 0x7f7f7f7fu), l1 = (b.y & 0x7f7f7f7fu) | ((e1 << 7) & 0x80808080u);
        const unsigned v[4] = {__byte_perm(l0, h0, 0x5140), __byte_perm(l0, h0, 0x7362),
                               __byte_perm(l1, h1, 0x5140), __byte_perm(l1, h1, 0x7362)};
        float w[8];
        #pragma unroll
        for (int k = 0; k < 4; k++) { w[2 * k] = __uint_as_float(v[k] << 16); w[2 * k + 1] = __uint_as_float(v[k] & 0xffff0000u); }
        #pragma unroll
        for (int s = 0; s < TS; s++) {
            if (s < ns) {
                const float4 *xs = (const float4 *)(x + (size_t)(s0 + s) * I + (size_t)c * 8);
                float4 a = __ldg(xs), bb = __ldg(xs + 1);
                float p = acc[s];
                p = fmaf(w[0], a.x, p); p = fmaf(w[1], a.y, p); p = fmaf(w[2], a.z, p); p = fmaf(w[3], a.w, p);
                p = fmaf(w[4], bb.x, p); p = fmaf(w[5], bb.y, p); p = fmaf(w[6], bb.z, p); p = fmaf(w[7], bb.w, p);
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
static int matmul_piece(MimoCudaTensor *t, float *y, const float *x, int S, int tiled_ok) {
    if (S < 1 || S > 65535) return 0;
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
    int tiled = tiled_ok && t->O % TL_B == 0;
    dim3 tgrid((unsigned)(t->O / TL_B), (unsigned)((S + TL_B - 1) / TL_B));
    if (tiled && t->fmt == 2 && t->I % 128 == 0)
        fp8_tiled<<<tgrid, 256, 0, stream>>>((const uint8_t *)t->w, t->scales, dx, dy, t->I, t->O, S);
    else if (tiled && t->fmt == 5 && t->I % TL_B == 0)
        bf16p_tiled<<<tgrid, 256, 0, stream>>>((const uint8_t *)t->w, t->codes, t->lo, t->exc_off, t->exc, dx, dy, t->I, t->O, S);
    else if (tiled && t->fmt == 1 && t->I % TL_B == 0)
        bf16_tiled<<<tgrid, 256, 0, stream>>>((const uint16_t *)t->w, dx, dy, t->I, t->O, S);
    else if (t->fmt == 5 && ts == 1)
        bf16x_wide<1><<<tiles,128,0,stream>>>((const uint8_t *)t->w,t->codes,t->lo,t->exc_off,t->exc,dx,dy,t->I,t->O,S);
    else if (t->fmt == 5)
        bf16x_wide<MIMO_TS><<<tiles,128,0,stream>>>((const uint8_t *)t->w,t->codes,t->lo,t->exc_off,t->exc,dx,dy,t->I,t->O,S);
    else if (wide(t) && t->fmt == 2 && ts == 1)
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
/* A prompt block's rows go through in pieces whose scratch, on the device and pinned on the
 * host, stays under MIMO_CUDA_SCRATCH_MB (192): a 12,000-row block's dense MLP would want 1.6 GB
 * of each, and a WDDM card that runs out pages its memory and crawls. Every piece of a call of
 * MIMO_CUDA_TILED_ROWS (64) rows or more takes the tiled kernels, so a row's bytes do not depend
 * on where its piece begins. */
extern "C" int mimo_cuda_matmul(MimoCudaTensor *t, float *y, const float *x, int S) {
    if (!ready || failed || !t || !x || !y || S < 1) return 0;
    static int tiled_rows = -1;
    static size_t budget;
    if (tiled_rows < 0) {
        const char *e = getenv("MIMO_CUDA_TILED_ROWS"), *b = getenv("MIMO_CUDA_SCRATCH_MB");
        tiled_rows = e && *e ? atoi(e) : 64;
        budget = (size_t)((b && *b ? atof(b) : 192.0) * 1048576.0);
    }
    int tiled = tiled_rows > 0 && S >= tiled_rows;
    size_t per_row = ((size_t)t->I + (size_t)t->O) * sizeof(float);
    int piece = (int)(budget / per_row);
    if (piece < 1) piece = 1;
    if (piece > 65535) piece = 65535;
    for (int s0 = 0; s0 < S; s0 += piece) {
        int ns = S - s0 < piece ? S - s0 : piece;
        if (!matmul_piece(t, y + (size_t)s0 * t->O, x + (size_t)s0 * t->I, ns, tiled)) return 0;
    }
    return 1;
}
extern "C" size_t mimo_cuda_bytes(const MimoCudaTensor *t) { return t ? t->bytes : 0; }

/* A matrix the device does not hold, for a prompt block: the host's packed bf16 (mimo.c's
 * DW_BF16P, whose planes are fmt 5's byte for byte) goes up into one scratch buffer and is
 * used from there, kept while the next call names the same host bytes (the sub-blocks of
 * a prompt block call each layer's o_proj in turn). */
static void *pk_dev;
static size_t pk_cap, pk_bytes_up;
static const void *pk_host;
static unsigned long long pk_uploads;
extern "C" int mimo_cuda_matmul_packed(const void *packed, size_t bytes, int I, int O, float *y, const float *x, int S) {
    if (!ready || failed || !packed || I < 256 || I % 256 || O < 1) return 0;
    size_t e = (size_t)O * I, a_lo = e + e / 2, a_off = a_lo + (((size_t)O + 15) & ~(size_t)15),
           a_exc = a_off + (size_t)O * sizeof(unsigned);
    if (bytes < a_exc) return 0;
    if (pk_host != packed || pk_bytes_up != bytes) {
        if (bytes > pk_cap) {
            void *nd = nullptr;
            if (cudaMalloc(&nd, bytes) != cudaSuccess) { cudaGetLastError(); return 0; }
            cudaFree(pk_dev); pk_dev = nd; pk_cap = bytes;
        }
        pk_host = nullptr;
        if (!checked(cudaMemcpy(pk_dev, packed, bytes, cudaMemcpyHostToDevice), "packed weights")) return 0;
        pk_host = packed; pk_bytes_up = bytes; pk_uploads++;
    }
    MimoCudaTensor t;
    memset(&t, 0, sizeof t);
    uint8_t *d = (uint8_t *)pk_dev;
    t.w = d; t.codes = d + e; t.lo = d + a_lo; t.exc_off = (const unsigned *)(d + a_off); t.exc = d + a_exc;
    t.fmt = 5; t.I = I; t.O = O; t.bytes = bytes;
    return mimo_cuda_matmul(&t, y, x, S);
}
/* The same for a plain bf16 host matrix [O][I] (a router's), in a buffer of its own. */
static void *bf_dev;
static size_t bf_cap;
static const void *bf_host;
extern "C" int mimo_cuda_matmul_bf16_host(const uint16_t *w, int I, int O, float *y, const float *x, int S) {
    if (!ready || failed || !w || I < 1 || O < 1) return 0;
    size_t bytes = (size_t)O * I * sizeof(uint16_t);
    if (bf_host != w) {
        if (bytes > bf_cap) {
            void *nd = nullptr;
            if (cudaMalloc(&nd, bytes) != cudaSuccess) { cudaGetLastError(); return 0; }
            cudaFree(bf_dev); bf_dev = nd; bf_cap = bytes;
        }
        bf_host = nullptr;
        if (!checked(cudaMemcpy(bf_dev, w, bytes, cudaMemcpyHostToDevice), "bf16 weights")) return 0;
        bf_host = w; pk_uploads++;
    }
    MimoCudaTensor t;
    memset(&t, 0, sizeof t);
    t.w = bf_dev; t.fmt = 1; t.I = I; t.O = O; t.bytes = bytes;
    return mimo_cuda_matmul(&t, y, x, S);
}

/* ---- a prompt block's attention ------------------------------------------------------
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
 * and each query's running max, sum and output wait in device memory between them. */
template<int HD, int VD>
__global__ static void __launch_bounds__(256) attn_rows(
        const float *__restrict__ q, int qstride, const float *__restrict__ k, const float *__restrict__ v,
        const float *__restrict__ sink, float *__restrict__ st, float *__restrict__ out,
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
    float *sq = st + ((size_t)t * nh + h) * (VD + 2);   /* chunked: max, sum, output */
    if (on) {
        const float4 *qp = (const float4 *)(q + (size_t)t * qstride + (size_t)h * HD);
        #pragma unroll
        for (int i = 0; i < H4; i++) qr[i] = qp[4 * i + g];
        if (first_chunk) {
            if (sink) { m = sink[h]; l = 1.0f; }
            #pragma unroll
            for (int i = 0; i < V4; i++) acc[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        } else {
            m = sq[0]; l = sq[1];
            #pragma unroll
            for (int i = 0; i < V4; i++) acc[i] = ((const float4 *)(sq + 2))[4 * i + g];
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
        float4 *op = (float4 *)(out + ((size_t)t * nh + h) * VD);
        #pragma unroll
        for (int i = 0; i < V4; i++) op[4 * i + g] = make_float4(acc[i].x * r, acc[i].y * r, acc[i].z * r, acc[i].w * r);
    } else {
        if (!g) { sq[0] = m; sq[1] = l; }
        #pragma unroll
        for (int i = 0; i < V4; i++) ((float4 *)(sq + 2))[4 * i + g] = acc[i];
    }
}

static float *a_q, *a_k, *a_v, *a_out, *a_st, *a_sink;
static size_t a_qcap, a_kcap, a_vcap, a_outcap, a_stcap, a_sinkcap;
static unsigned long long attn_calls, attn_rows_n, attn_fallback;
static double attn_seconds;
static int attn_grow(float **d, size_t *cap, size_t bytes) {
    if (bytes <= *cap) return 1;
    float *nd = nullptr;
    if (cudaMalloc(&nd, bytes) != cudaSuccess) { cudaGetLastError(); return 0; }
    cudaFree(*d); *d = nd; *cap = bytes;
    return 1;
}
template<int HD, int VD>
static int attn_launch(const float *q, int qstride, int n, int nh, int kvh, const float *k, const float *v,
                       int lo, int nkeys, int p0, int window, const float *sink, float scale, float *out) {
    int G = nh / kvh, BR = 64 / G > 0 ? 64 / G : 1, threads = BR * G * 4;
    size_t shared = (size_t)32 * (HD + VD) * sizeof(float);
    static int attr_set;
    if (!attr_set) {
        if (cudaFuncSetAttribute(attn_rows<HD, VD>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared) != cudaSuccess) {
            cudaGetLastError(); return 0;
        }
        attr_set = 1;
    }
    size_t row_kv = (size_t)kvh * (HD + VD) * sizeof(float);
    const char *mb = getenv("MIMO_CUDA_ATTN_MB");
    size_t budget = (size_t)((mb && *mb ? atof(mb) : 256.0) * 1048576.0);
    int chunk = (int)(budget / row_kv);
    if (chunk < 64) chunk = 64;
    if (chunk > nkeys) chunk = nkeys;
    size_t qb = (size_t)n * nh * HD * sizeof(float), ob = (size_t)n * nh * VD * sizeof(float);
    if (!attn_grow(&a_q, &a_qcap, qb) || !attn_grow(&a_out, &a_outcap, ob) ||
        !attn_grow(&a_k, &a_kcap, (size_t)chunk * kvh * HD * sizeof(float)) ||
        !attn_grow(&a_v, &a_vcap, (size_t)chunk * kvh * VD * sizeof(float)) ||
        (sink && !attn_grow(&a_sink, &a_sinkcap, (size_t)nh * sizeof(float))) ||
        (chunk < nkeys && !attn_grow(&a_st, &a_stcap, (size_t)n * nh * (VD + 2) * sizeof(float)))) return 0;
    if (!checked(cudaMemcpy2DAsync(a_q, (size_t)nh * HD * sizeof(float), q, (size_t)qstride * sizeof(float),
                                   (size_t)nh * HD * sizeof(float), (size_t)n, cudaMemcpyHostToDevice, stream), "attention queries") ||
        (sink && !checked(cudaMemcpyAsync(a_sink, sink, (size_t)nh * sizeof(float), cudaMemcpyHostToDevice, stream), "attention sinks")))
        return 0;
    dim3 grid((unsigned)((n + BR - 1) / BR), (unsigned)kvh);
    for (int j0 = lo; j0 < lo + nkeys; j0 += chunk) {
        int nk = lo + nkeys - j0 < chunk ? lo + nkeys - j0 : chunk;
        if (!checked(cudaMemcpyAsync(a_k, k + (size_t)(j0 - lo) * kvh * HD, (size_t)nk * kvh * HD * sizeof(float),
                                     cudaMemcpyHostToDevice, stream), "attention keys") ||
            !checked(cudaMemcpyAsync(a_v, v + (size_t)(j0 - lo) * kvh * VD, (size_t)nk * kvh * VD * sizeof(float),
                                     cudaMemcpyHostToDevice, stream), "attention values")) return 0;
        attn_rows<HD, VD><<<grid, threads, shared, stream>>>(a_q, nh * HD, a_k, a_v, sink ? a_sink : nullptr, a_st, a_out,
                                                              n, nh, kvh, G, BR, p0, lo, window, j0, nk, scale,
                                                              j0 == lo, j0 + nk == lo + nkeys);
        if (!checked(cudaGetLastError(), "attention kernel")) return 0;
    }
    if (!checked(cudaMemcpyAsync(out, a_out, ob, cudaMemcpyDeviceToHost, stream), "attention output") ||
        !checked(cudaStreamSynchronize(stream), "attention")) return 0;
    return 1;
}
extern "C" int mimo_cuda_attention(const float *q, int qstride, int n, int nh, int kvh, int hd, int vd,
                                   const float *k, const float *v, int lo, int nkeys, int p0, int window,
                                   const float *sink, float scale, float *out) {
    if (!ready || failed || !q || !k || !v || !out || n < 1 || nkeys < 1 || kvh < 1 || nh % kvh || nh / kvh > 64 ||
        qstride < nh * hd || p0 < lo || p0 + n != lo + nkeys) return 0;
    auto started = std::chrono::steady_clock::now();
    int ok = hd == 192 && vd == 128 ? attn_launch<192, 128>(q, qstride, n, nh, kvh, k, v, lo, nkeys, p0, window, sink, scale, out)
           : hd == 48 && vd == 32 ? attn_launch<48, 32>(q, qstride, n, nh, kvh, k, v, lo, nkeys, p0, window, sink, scale, out)
           : 0;
    if (!ok) { cudaStreamSynchronize(stream); attn_fallback++; return 0; }
    attn_calls++; attn_rows_n += (unsigned long long)n;
    attn_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return 1;
}
/* ---- a prompt block's routed experts --------------------------------------------------
 * The MXFP4 experts of a prompt block on the device. mimo_cuda_moe_begin sends the block's
 * normed rows up and hands out a pinned host buffer for every (row, choice) pair's output.
 * For each expert, in any order, the host copies its bytes, as the expert cache holds them
 * (down, its scales, gate, its scales, up, its scales), into the pinned buffer
 * mimo_cuda_moe_staging hands out; mimo_cuda_moe_expert sends them up on the copy stream,
 * the compute stream runs gate and up over the rows that chose it, silu(gate) * up and
 * down, and the third stream brings the rows' outputs down into their pairs' places;
 * mimo_cuda_moe_end waits for the last. The host then sums each row's pairs in its routing
 * order, as it does the CPU's (moe). The arithmetic is matmul_mxfp4's grouping: per
 * 32-column group the doubled e2m1 values (exact) times the row, summed, then times the
 * group's scale (halved) into the output -- in f32, summed in another order. MOE_SLOTS
 * experts are in flight: their staging and device buffers. */
#define MOE_SLOTS 6
static int moe_rcap = 1024;   /* the rows of an expert computed at once, its buffers' (MIMO_CUDA_MOE_RCAP for tests) */
#define MOE_BM 64
#define MOE_BN 64
#define MOE_BK 32
__constant__ float mx4_lut2[16] = {0.f, 1.f, 2.f, 3.f, 4.f, 6.f, 8.f, 12.f, 0.f, -1.f, -2.f, -3.f, -4.f, -6.f, -8.f, -12.f};
__device__ static __forceinline__ float e8m0_half(unsigned s) { return __uint_as_float(s << 23) * 0.5f; }
__device__ static __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }

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

static cudaStream_t m_copy, m_down;
static cudaEvent_t m_copied[MOE_SLOTS], m_computed[MOE_SLOTS], m_done[MOE_SLOTS];
static int m_used[MOE_SLOTS], m_cur, m_H, m_MI, m_rmax, m_npairs, m_on;
static size_t m_ebytes, m_metacap, m_pairscap;
static uint8_t *m_stage[MOE_SLOTS], *m_dw[MOE_SLOTS];
static int *m_hrows[MOE_SLOTS], *m_drows[MOE_SLOTS];
static float *m_ye[MOE_SLOTS];
static float *m_x, *m_g, *m_pairs;
static size_t m_xcap, m_gcap, m_yecap;
static unsigned long long moe_blocks, moe_experts, moe_rows_n, moe_fallback;
static double moe_seconds;
static void moe_free(void) {
    for (int b = 0; b < MOE_SLOTS; b++) {
        if (m_copied[b]) cudaEventDestroy(m_copied[b]);
        if (m_computed[b]) cudaEventDestroy(m_computed[b]);
        if (m_done[b]) cudaEventDestroy(m_done[b]);
        cudaFreeHost(m_stage[b]); cudaFree(m_dw[b]); cudaFreeHost(m_hrows[b]); cudaFree(m_drows[b]); cudaFree(m_ye[b]);
        m_copied[b] = m_computed[b] = m_done[b] = nullptr; m_stage[b] = m_dw[b] = nullptr;
        m_hrows[b] = m_drows[b] = nullptr; m_ye[b] = nullptr;
    }
    if (m_copy) cudaStreamDestroy(m_copy);
    if (m_down) cudaStreamDestroy(m_down);
    m_copy = m_down = nullptr; m_ebytes = m_metacap = m_yecap = 0;
    cudaFree(m_x); cudaFree(m_g); cudaFreeHost(m_pairs);
    m_x = m_g = m_pairs = nullptr; m_xcap = m_gcap = m_pairscap = 0;
}
/* Grows every slot's buffer of `per` bytes, or none. */
static int moe_grow_slots(float **d, size_t *cap, size_t per) {
    if (per <= *cap) return 1;
    for (int b = 0; b < MOE_SLOTS; b++) { cudaFree(d[b]); d[b] = nullptr; }
    *cap = 0;
    for (int b = 0; b < MOE_SLOTS; b++)
        if (cudaMalloc(&d[b], per) != cudaSuccess) { cudaGetLastError(); return 0; }
    *cap = per;
    return 1;
}
extern "C" float *mimo_cuda_moe_begin(const float *x, int n, int H, int MI, size_t ebytes, int npairs, int rmax) {
    if (!ready || failed || !x || n < 1 || npairs < 1 || rmax < 1 || rmax > n || H % MOE_BN || MI % MOE_BN ||
        H % MOE_BK || MI % MOE_BK || !ebytes) return nullptr;
    if (ebytes != m_ebytes) {   /* the buffers, made once for the model's expert size */
        moe_free();
        if (cudaStreamCreateWithFlags(&m_copy, cudaStreamNonBlocking) != cudaSuccess ||
            cudaStreamCreateWithFlags(&m_down, cudaStreamNonBlocking) != cudaSuccess) { cudaGetLastError(); moe_free(); return nullptr; }
        for (int b = 0; b < MOE_SLOTS; b++)
            if (cudaEventCreateWithFlags(&m_copied[b], cudaEventDisableTiming) != cudaSuccess ||
                cudaEventCreateWithFlags(&m_computed[b], cudaEventDisableTiming) != cudaSuccess ||
                cudaEventCreateWithFlags(&m_done[b], cudaEventDisableTiming) != cudaSuccess ||
                cudaMallocHost(&m_stage[b], ebytes) != cudaSuccess || cudaMalloc(&m_dw[b], ebytes) != cudaSuccess) {
                cudaGetLastError(); moe_free(); return nullptr;
            }
        m_ebytes = ebytes;
    }
    if ((size_t)rmax > m_metacap) {
        for (int b = 0; b < MOE_SLOTS; b++) {
            cudaFreeHost(m_hrows[b]); cudaFree(m_drows[b]); m_hrows[b] = nullptr; m_drows[b] = nullptr;
        }
        m_metacap = 0;
        for (int b = 0; b < MOE_SLOTS; b++)
            if (cudaMallocHost(&m_hrows[b], (size_t)rmax * sizeof(int)) != cudaSuccess ||
                cudaMalloc(&m_drows[b], (size_t)rmax * sizeof(int)) != cudaSuccess) { cudaGetLastError(); return nullptr; }
        m_metacap = (size_t)rmax;
    }
    size_t xb = (size_t)n * H * sizeof(float), pb = (size_t)npairs * H * sizeof(float);
    { const char *e = getenv("MIMO_CUDA_MOE_RCAP"); moe_rcap = e && *e && atoi(e) > 0 ? atoi(e) : 1024; }
    size_t rcap = (size_t)(rmax < moe_rcap ? rmax : moe_rcap);
    if (!moe_grow_slots(m_ye, &m_yecap, rcap * H * sizeof(float)) ||
        !attn_grow(&m_x, &m_xcap, xb) || !attn_grow(&m_g, &m_gcap, rcap * MI * sizeof(float))) return nullptr;
    if (pb > m_pairscap) {
        cudaFreeHost(m_pairs); m_pairs = nullptr; m_pairscap = 0;
        if (cudaMallocHost(&m_pairs, pb) != cudaSuccess) { cudaGetLastError(); return nullptr; }
        m_pairscap = pb;
    }
    if (!checked(cudaMemcpyAsync(m_x, x, xb, cudaMemcpyHostToDevice, stream), "experts' rows")) return nullptr;
    for (int b = 0; b < MOE_SLOTS; b++) m_used[b] = 0;
    m_cur = 0; m_H = H; m_MI = MI; m_rmax = rmax; m_npairs = npairs; m_on = 1;
    moe_blocks++; moe_rows_n += (unsigned long long)n;
    return m_pairs;
}
/* The next expert's staging buffer, once the copy that read it last is done. */
extern "C" void *mimo_cuda_moe_staging(void) {
    if (!m_on) return nullptr;
    int b = m_cur;
    if (m_used[b] && !checked(cudaEventSynchronize(m_copied[b]), "experts' staging")) return nullptr;
    return m_stage[b];
}
/* The expert in the staging buffer, over rows[0..r) of the block; its outputs go to the
 * pairs pair0 .. pair0 + r - 1 of the buffer begin returned. */
extern "C" int mimo_cuda_moe_expert(const int *rows, int r, int pair0) {
    if (!m_on || r < 1 || r > m_rmax || pair0 < 0 || pair0 + r > m_npairs) return 0;
    auto started = std::chrono::steady_clock::now();
    int b = m_cur, H = m_H, MI = m_MI;
    if (m_used[b] && !checked(cudaEventSynchronize(m_copied[b]), "experts' staging")) return 0;
    memcpy(m_hrows[b], rows, (size_t)r * sizeof(int));
    size_t dq = 0, ds = dq + (size_t)H * (MI / 2), gq = ds + (size_t)H * (MI / 32), gs = gq + (size_t)MI * (H / 2);
    size_t uq = gs + (size_t)MI * (H / 32), us = uq + (size_t)MI * (H / 2);
    /* copy stream: the weights and rows into slot b once its last outputs are down */
    if (!checked(cudaStreamWaitEvent(m_copy, m_done[b], 0), "experts' order") ||
        !checked(cudaMemcpyAsync(m_dw[b], m_stage[b], m_ebytes, cudaMemcpyHostToDevice, m_copy), "expert weights") ||
        !checked(cudaMemcpyAsync(m_drows[b], m_hrows[b], (size_t)r * sizeof(int), cudaMemcpyHostToDevice, m_copy), "expert rows") ||
        !checked(cudaEventRecord(m_copied[b], m_copy), "experts' copies") ||
        !checked(cudaStreamWaitEvent(stream, m_copied[b], 0), "experts' order")) return 0;
    /* an expert with more rows than its buffers hold (moe_rcap) goes in pieces, each piece's
     * outputs down before the next is computed into the same buffer */
    for (int p0 = 0; p0 < r; p0 += moe_rcap) {
        int pr = r - p0 < moe_rcap ? r - p0 : moe_rcap;
        if (p0 && !checked(cudaStreamWaitEvent(stream, m_done[b], 0), "experts' order")) return 0;
        dim3 gu((unsigned)((pr + MOE_BM - 1) / MOE_BM), (unsigned)(MI / MOE_BN)), dn((unsigned)((pr + MOE_BM - 1) / MOE_BM), (unsigned)(H / MOE_BN));
        moe_gateup<<<gu, 256, 0, stream>>>(m_x, m_drows[b] + p0, pr, H, MI, m_dw[b], gq, gs, uq, us, m_g);
        moe_down<<<dn, 256, 0, stream>>>(m_g, pr, H, MI, m_dw[b], dq, ds, m_ye[b]);
        if (!checked(cudaGetLastError(), "expert kernels") || !checked(cudaEventRecord(m_computed[b], stream), "experts' order") ||
            !checked(cudaStreamWaitEvent(m_down, m_computed[b], 0), "experts' order") ||
            !checked(cudaMemcpyAsync(m_pairs + (size_t)(pair0 + p0) * H, m_ye[b], (size_t)pr * H * sizeof(float),
                                     cudaMemcpyDeviceToHost, m_down), "expert outputs") ||
            !checked(cudaEventRecord(m_done[b], m_down), "experts' order")) return 0;
    }
    m_used[b] = 1; m_cur = (b + 1) % MOE_SLOTS;
    moe_experts++;
    moe_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return 1;
}
/* Waits until every expert's outputs are in the pairs buffer. */
extern "C" int mimo_cuda_moe_end(void) {
    if (!m_on) return 0;
    m_on = 0;
    auto started = std::chrono::steady_clock::now();
    int ok = checked(cudaStreamSynchronize(m_copy), "experts' copies") && checked(cudaStreamSynchronize(stream), "experts") &&
             checked(cudaStreamSynchronize(m_down), "experts' outputs");
    moe_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    if (!ok) { cudaDeviceSynchronize(); moe_fallback++; }
    return ok;
}

extern "C" void mimo_cuda_report(void) {
    if (ready) fprintf(stderr, "[CUDA] mimo dense: %zu matrices, %.3f GiB resident, %llu matmuls, %llu rows, %llu scratch/compute fallbacks%s\n",
                       tensor_count, resident / 1073741824.0, calls, rows, fallback, failed ? ", device failed" : "");
    if (ready) fprintf(stderr, "[CUDA] mimo matmul wall time (copies + compute): f32 %.3fs, bf16 %.3fs, fp8 %.3fs, int8 %.3fs, int8 blocks %.3fs, bf16 packed %.3fs\n",
                       seconds[0], seconds[1], seconds[2], seconds[3], seconds[4], seconds[5]);
    if (ready && packed_count)
        fprintf(stderr, "[CUDA] mimo bf16 packed without loss: %llu matrices, %.3f GiB for %.3f GiB of bf16, %llu exceptions\n",
                packed_count, packed_bytes / 1073741824.0, packed_bf16 / 1073741824.0, packed_exc);
    if (ready && (attn_calls || attn_fallback))
        fprintf(stderr, "[CUDA] mimo prompt attention: %llu blocks, %llu rows, %.3fs (copies + compute), %llu left to the CPU\n",
                attn_calls, attn_rows_n, attn_seconds, attn_fallback);
    if (ready && pk_uploads)
        fprintf(stderr, "[CUDA] mimo prompt blocks' host matrices: %llu uploads\n", pk_uploads);
    if (ready && (moe_blocks || moe_fallback))
        fprintf(stderr, "[CUDA] mimo prompt experts: %llu MoE steps, %llu rows, %llu experts, %.3fs queueing and waiting, %llu left to the CPU\n",
                moe_blocks, moe_rows_n, moe_experts, moe_seconds, moe_fallback);
}
extern "C" void mimo_cuda_shutdown(void) {
    if (stream) cudaStreamSynchronize(stream);
    while (tensors) {
        MimoCudaTensor *next = tensors->next;
        cudaFree(tensors->w); cudaFree(tensors->scales); free(tensors); tensors = next;
    }
    cudaFree(dx); cudaFree(dy); cudaFreeHost(hx); cudaFreeHost(hy);
    cudaFree(a_q); cudaFree(a_k); cudaFree(a_v); cudaFree(a_out); cudaFree(a_st); cudaFree(a_sink);
    a_q = a_k = a_v = a_out = a_st = a_sink = nullptr;
    a_qcap = a_kcap = a_vcap = a_outcap = a_stcap = a_sinkcap = 0;
    attn_calls = attn_rows_n = attn_fallback = 0; attn_seconds = 0;
    moe_free();
    moe_blocks = moe_experts = moe_rows_n = moe_fallback = 0; moe_seconds = 0; m_on = 0;
    cudaFree(pk_dev); pk_dev = nullptr; pk_cap = pk_bytes_up = 0; pk_host = nullptr; pk_uploads = 0;
    cudaFree(bf_dev); bf_dev = nullptr; bf_cap = 0; bf_host = nullptr;
    if (stream) cudaStreamDestroy(stream);
    stream = nullptr; dx = dy = hx = hy = nullptr; xcap = ycap = resident = tensor_count = 0;
    calls = rows = fallback = packed_count = packed_exc = 0; packed_bytes = packed_bf16 = 0;
    ready = failed = 0; selected = -1;
    memset(seconds, 0, sizeof seconds);
}
