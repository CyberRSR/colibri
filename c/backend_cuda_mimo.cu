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

/* Pinned host memory for the caller's buffers of a prompt block: copies to and from it go
 * straight between the device and it. */
extern "C" void *mimo_cuda_host_alloc(size_t bytes) {
    void *p = nullptr;
    if (!ready || failed || !bytes) return nullptr;
    if (cudaMallocHost(&p, bytes) != cudaSuccess) { cudaGetLastError(); return nullptr; }
    return p;
}
extern "C" void mimo_cuda_host_free(void *p) { if (p) cudaFreeHost(p); }
static int pinned(const void *p) {
    cudaPointerAttributes a;
    if (cudaPointerGetAttributes(&a, p) != cudaSuccess) { cudaGetLastError(); return 0; }
    return a.type == cudaMemoryTypeHost;
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
__device__ static __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }
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
/* ---- a prompt block's matrices, 128 x 128 ------------------------------------------------
 * C [S][O] = A [S][K] . W^T for every weight format a prompt block meets: W's tile rebuilt in
 * f32 into shared memory 16 columns at a time (FP8, bf16, packed bf16, MXFP4: every weight's
 * exact value), 128 rows x 128 outputs a block of 256 threads, 8 x 8 a thread, the next 16
 * columns' loads in flight while these are used (two buffers, one barrier a step). The scaled
 * formats sum each scale group's products apart -- FP8's 128 columns, MXFP4's 32 -- and add the
 * sum times the group's scale to the output, as the CPU and the decode kernels group them; bf16
 * is one f32 FMA chain an output. Every sum runs over the columns in order, whatever rows share
 * the tile: a row's bytes do not depend on its block. (One chain for the scaled formats too,
 * the scales folded into the weights, was ten times less accurate -- 2.5e-7 of the products'
 * magnitude against 2.6e-8 -- and a long prompt's massive-activation channels felt it: the
 * argmax left the CPU's at 7 of 658 positions. Blocks of 128 x 256, 8 x 16 a thread, read shared
 * memory a third less a product but hold one block an SM: no faster, 39.7 against 40.1 TFLOP/s
 * for the qkv.) A's rows may be gathered (idx: an expert's rows of the block). GW_PAIR computes
 * an expert's gate and up at once (64 outputs of each a block) and stores silu(gate) * up. */
#define GB_M 128
#define GB_N 128
#define GB_K 16
enum { GW_FP8, GW_BF16, GW_BF16P, GW_MX4, GW_PAIR };
struct GemmW {
    const uint8_t *q, *q2;        /* fp8 / bf16 / packed sign-mantissa / e2m1 bytes; packed codes or the pair's up */
    const float *scales;          /* fp8: [O][K/128] */
    const uint8_t *s8, *s82;      /* e2m1: e8m0 [O][K/32], the pair's up */
    const uint8_t *lo, *exc;      /* packed bf16 */
    const unsigned *exc_off;
};
__device__ static __forceinline__ float e8m0_f(unsigned s) { return s ? __uint_as_float(s << 23) : __uint_as_float(0x00400000u); }
template<int WF>
__global__ static void __launch_bounds__(256, WF == GW_BF16 || WF == GW_BF16P ? 2 : 1)
gemm_tiled(const float *__restrict__ A, int lda, const int *__restrict__ idx, int S, int K, int O, GemmW w, float *__restrict__ C, int ldc) {
    constexpr bool scaled = WF == GW_FP8 || WF == GW_MX4 || WF == GW_PAIR;
    constexpr int GS = WF == GW_FP8 ? 128 / GB_K : 32 / GB_K;   /* steps a scale group */
    __shared__ __align__(16) float as[2][GB_K][GB_M + 4];   /* [column][row] */
    __shared__ __align__(16) float ws[2][GB_K][GB_N + 4];   /* [column][output] */
    __shared__ float lut[16];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int r0 = blockIdx.y * GB_M, o0 = blockIdx.x * (WF == GW_PAIR ? GB_N / 2 : GB_N);
    if ((WF == GW_MX4 || WF == GW_PAIR) && tid < 16) {   /* e2m1: sign, 2 exponent bits, 1 mantissa bit */
        int e = (tid >> 1) & 3, m = tid & 1;
        float v = (float)(e ? (2 + m) << (e - 1) : m) * 0.5f;
        lut[tid] = tid & 8 && v != 0.f ? -v : v;
    }
    /* A: rows lr and lr + 64 of the tile, columns kq..kq+3 of the step */
    const int lr = tid >> 2, kq = (tid & 3) * 4;
    const float *a0 = r0 + lr < S ? A + (size_t)(idx ? idx[r0 + lr] : r0 + lr) * lda + kq : nullptr;
    const float *a1 = r0 + lr + 64 < S ? A + (size_t)(idx ? idx[r0 + lr + 64] : r0 + lr + 64) * lda + kq : nullptr;
    /* W: output ol of the tile, columns kh..kh+7 of the step */
    const int ol = tid >> 1, kh = (tid & 1) * 8;
    int orow = WF == GW_PAIR ? o0 + (ol & 63) : o0 + ol;
    const bool wok = orow < O;
    if (!wok) orow = 0;
    const uint8_t *wq = WF == GW_PAIR && ol >= 64 ? w.q2 : w.q;
    unsigned run = 0, base = 0;
    const uint8_t *ex = nullptr;
    if (WF == GW_BF16P) { base = w.lo[orow]; ex = w.exc + w.exc_off[orow]; }
    /* the scales of this thread's outputs: column j of acc is output oc[j] (a pair's up for j >= 4) */
    int oc[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int o = WF == GW_PAIR ? o0 + tx * 4 + (j & 3) : o0 + (j < 4 ? tx * 4 + j : 64 + tx * 4 + j - 4);
        oc[j] = o < O ? o : 0;
    }
    float4 ra0, ra1;
    uint4 rq;             /* the step's raw weights */
    auto fetch = [&](int k) {
        ra0 = a0 ? *(const float4 *)(a0 + k) : make_float4(0.f, 0.f, 0.f, 0.f);
        ra1 = a1 ? *(const float4 *)(a1 + k) : make_float4(0.f, 0.f, 0.f, 0.f);
        if (WF == GW_FP8) {
            uint2 v = wok ? *(const uint2 *)(w.q + (size_t)orow * K + k + kh) : make_uint2(0u, 0u);
            rq.x = v.x; rq.y = v.y;
        } else if (WF == GW_BF16) {
            rq = wok ? *(const uint4 *)((const uint16_t *)w.q + (size_t)orow * K + k + kh) : make_uint4(0u, 0u, 0u, 0u);
        } else if (WF == GW_BF16P) {
            uint2 v = wok ? *(const uint2 *)(w.q + (size_t)orow * K + k + kh) : make_uint2(0u, 0u);
            rq.x = v.x; rq.y = v.y;
            rq.z = wok ? *(const unsigned *)(w.q2 + (size_t)orow * (K / 2) + (k + kh) / 2) : 0u;
        } else {
            rq.x = wok ? *(const unsigned *)(wq + (size_t)orow * (K / 2) + (k + kh) / 2) : 0u;
        }
    };
    auto stash = [&](int b) {
        float (*xs)[GB_M + 4] = as[b];
        xs[kq + 0][lr] = ra0.x; xs[kq + 1][lr] = ra0.y; xs[kq + 2][lr] = ra0.z; xs[kq + 3][lr] = ra0.w;
        xs[kq + 0][lr + 64] = ra1.x; xs[kq + 1][lr + 64] = ra1.y; xs[kq + 2][lr + 64] = ra1.z; xs[kq + 3][lr + 64] = ra1.w;
        float (*wt)[GB_N + 4] = ws[b];
        if (WF == GW_FP8) {
            const unsigned v[2] = {rq.x, rq.y};
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                __half2_raw l = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(v[h] & 0xffffu), __NV_E4M3);
                __half2_raw u = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(v[h] >> 16), __NV_E4M3);
                float2 fa = __half22float2(*(const __half2 *)&l), fb = __half22float2(*(const __half2 *)&u);
                wt[kh + 4 * h + 0][ol] = fa.x; wt[kh + 4 * h + 1][ol] = fa.y;
                wt[kh + 4 * h + 2][ol] = fb.x; wt[kh + 4 * h + 3][ol] = fb.y;
            }
        } else if (WF == GW_BF16) {
            const unsigned u[4] = {rq.x, rq.y, rq.z, rq.w};
            #pragma unroll
            for (int m = 0; m < 4; m++) {
                wt[kh + 2 * m][ol] = __uint_as_float(u[m] << 16);
                wt[kh + 2 * m + 1][ol] = __uint_as_float(u[m] & 0xffff0000u);
            }
        } else if (WF == GW_BF16P) {
            /* the row's two loading threads (adjacent lanes) count their exceptions in column order */
            const unsigned q = rq.z, esc = q & (q >> 1) & (q >> 2) & (q >> 3) & 0x11111111u;
            const int n = __popc(esc);
            const int before = __shfl_up_sync(0xffffffffu, n, 1);
            const int pair = __shfl_xor_sync(0xffffffffu, n, 1);
            const uint8_t *p = ex + run + (kh ? (unsigned)before : 0u);
            run += (unsigned)(n + pair);
            const unsigned sw[2] = {rq.x, rq.y};
            #pragma unroll
            for (int m = 0; m < 8; m++) {   /* weight m: code nibble 2m (m < 4) or 2(m - 4) + 1 */
                unsigned code = (q >> (m < 4 ? 8 * m : 8 * (m - 4) + 4)) & 15u;
                unsigned e = code == 15u ? (unsigned)*p++ : base + code;
                unsigned b = (sw[m >> 2] >> (8 * (m & 3))) & 0xffu;
                wt[kh + m][ol] = __uint_as_float(((b & 0x80u) << 24) | (e << 23) | ((b & 0x7fu) << 16));
            }
        } else {
            #pragma unroll
            for (int m = 0; m < 8; m++) wt[kh + m][ol] = lut[(rq.x >> (4 * m)) & 15u];
        }
    };
    float acc[8][8], part[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = part[i][j] = 0.f;
    if (WF == GW_MX4 || WF == GW_PAIR) __syncthreads();   /* the table */
    fetch(0);
    stash(0);
    __syncthreads();
    const int nk = K / GB_K;
    for (int t = 0; t < nk; t++) {
        const int b = t & 1;
        if (t + 1 < nk) fetch((t + 1) * GB_K);
        #pragma unroll
        for (int kk = 0; kk < GB_K; kk++) {
            const float4 x0 = *(const float4 *)&as[b][kk][ty * 4], x1 = *(const float4 *)&as[b][kk][64 + ty * 4];
            const float4 y0 = *(const float4 *)&ws[b][kk][tx * 4], y1 = *(const float4 *)&ws[b][kk][64 + tx * 4];
            const float av[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
            const float bv[8] = {y0.x, y0.y, y0.z, y0.w, y1.x, y1.y, y1.z, y1.w};
            #pragma unroll
            for (int i = 0; i < 8; i++)
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    if (scaled) part[i][j] = fmaf(av[i], bv[j], part[i][j]);
                    else acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
                }
        }
        if (scaled && t % GS == GS - 1) {   /* the scale group is complete: its sum times its scale */
            const int g = t / GS;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                float sc;
                if (WF == GW_FP8) sc = w.scales[(size_t)oc[j] * (K / 128) + g];
                else sc = e8m0_f((WF == GW_PAIR && j >= 4 ? w.s82 : w.s8)[(size_t)oc[j] * (K / 32) + g]);
                #pragma unroll
                for (int i = 0; i < 8; i++) { acc[i][j] = fmaf(part[i][j], sc, acc[i][j]); part[i][j] = 0.f; }
            }
        }
        if (t + 1 < nk) stash(b ^ 1);
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int row = r0 + (i < 4 ? ty * 4 + i : 64 + ty * 4 + i - 4);
        if (row >= S) continue;
        float *cr = C + (size_t)row * ldc;
        if (WF == GW_PAIR) {
            const int o = o0 + tx * 4;
            if (o < O)
                *(float4 *)(cr + o) = make_float4(silu_f(acc[i][0]) * acc[i][4], silu_f(acc[i][1]) * acc[i][5],
                                                  silu_f(acc[i][2]) * acc[i][6], silu_f(acc[i][3]) * acc[i][7]);
        } else {
            if (o0 + tx * 4 < O) *(float4 *)(cr + o0 + tx * 4) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
            if (o0 + 64 + tx * 4 < O) *(float4 *)(cr + o0 + 64 + tx * 4) = make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
        }
    }
}
/* A resident matrix gemm_tiled takes: FP8 (fmt 2, K a multiple of 128), packed bf16 (5) or bf16
 * (1), K a multiple of 16 and O of 4. */
static int gemm_ok(const MimoCudaTensor *t) {
    return t->O % 4 == 0 && t->I % GB_K == 0 && ((t->fmt == 2 && t->I % 128 == 0) || t->fmt == 5 || t->fmt == 1);
}
static void gemm_launch(const MimoCudaTensor *t, const float *x, int S, float *y, cudaStream_t s) {
    GemmW w{};
    w.q = (const uint8_t *)t->w; w.scales = t->scales;
    w.q2 = t->codes; w.lo = t->lo; w.exc = t->exc; w.exc_off = t->exc_off;
    dim3 g((unsigned)((t->O + GB_N - 1) / GB_N), (unsigned)((S + GB_M - 1) / GB_M));
    if (t->fmt == 2) gemm_tiled<GW_FP8><<<g, 256, 0, s>>>(x, t->I, nullptr, S, t->I, t->O, w, y, t->O);
    else if (t->fmt == 5) gemm_tiled<GW_BF16P><<<g, 256, 0, s>>>(x, t->I, nullptr, S, t->I, t->O, w, y, t->O);
    else gemm_tiled<GW_BF16><<<g, 256, 0, s>>>(x, t->I, nullptr, S, t->I, t->O, w, y, t->O);
}
static int wide(const MimoCudaTensor *t) {
    return ((t->fmt == 2 || t->fmt == 4) && t->I % 512 == 0) || (t->fmt == 1 && t->I % 256 == 0);
}

/* ---- the prompt phases' device memory ------------------------------------------------------
 * A prompt block's attention (mimo_cuda_attn_block: its rows, qkv, attention output, keys and
 * values), its MoE steps (their rows, the router's logits, the experts' weights, gate * up and
 * outputs) and its other matrix calls never run at once: one allocation holds the buffers of
 * the one that runs (ar_use), where each had its own -- 1.6 GB beside the dense weights for a
 * 25,000-row block, a gigabyte now. A phase's buffers keep their contents from one call to the
 * next of the same phase; another phase taking the memory drops what the first kept (a full
 * layer's keys, a MoE step's rows). Growing it waits for the work queued on the old one. */
enum { AR_NONE, AR_ATTN, AR_MOE, AR_MAT };
static uint8_t *ar_mem;
static size_t ar_cap;
static int ar_owner;
static long long kv_tag;   /* what the attention's keys and values hold (attn_run): the tag's positions [kv_lo, kv_hi) */
static int kv_lo, kv_hi;
static int m_xn, m_xh;     /* the rows mimo_cuda_moe_rows sent */
static int c_pending;
extern "C" int mimo_cuda_attn_wait(void);
static size_t ar_align(size_t b) { return (b + 255) & ~(size_t)255; }
static uint8_t *ar_use(int phase, size_t bytes) {
    if (ar_owner != phase) {
        if (c_pending) mimo_cuda_attn_wait();
        if (ar_owner == AR_ATTN) kv_tag = 0;
        if (ar_owner == AR_MOE) m_xn = 0;
        ar_owner = phase;
    }
    if (bytes > ar_cap) {
        cudaDeviceSynchronize();
        cudaFree(ar_mem); ar_mem = nullptr; ar_cap = 0;
        kv_tag = 0;
        if (cudaMalloc(&ar_mem, bytes) != cudaSuccess) { cudaGetLastError(); ar_owner = AR_NONE; return nullptr; }
        ar_cap = bytes;
    }
    return ar_mem;
}
/* Grow transactionally: a partially successful allocation neither leaks nor
 * destroys the old scratch. Pinned staging avoids pageable async copies. */
static size_t hxcap, hycap;   /* the pinned scratch's own sizes (it also serves arena pieces) */
static int scratch(float **d, float **h, size_t *cap, size_t n) {
    if (n <= *cap && n <= (h == &hx ? hxcap : hycap)) return 1;
    float *nd = nullptr, *nh = nullptr;
    if (cudaMalloc(&nd, n) != cudaSuccess) { cudaGetLastError(); return 0; }
    if (cudaMallocHost(&nh, n) != cudaSuccess) { cudaGetLastError(); cudaFree(nd); return 0; }
    cudaFree(*d); cudaFreeHost(*h);
    *d = nd; *h = nh; *cap = n;
    if (h == &hx) hxcap = n; else hycap = n;
    return 1;
}
static int scratch_host(float **h, size_t *cap, size_t n) {
    if (n <= *cap) return 1;
    float *nh = nullptr;
    if (cudaMallocHost(&nh, n) != cudaSuccess) { cudaGetLastError(); return 0; }
    cudaFreeHost(*h); *h = nh; *cap = n;
    return 1;
}
static int matmul_piece(MimoCudaTensor *t, float *y, const float *x, int S, int tiled_ok) {
    if (S < 1 || S > 65535) return 0;
    if ((size_t)S > SIZE_MAX / sizeof(float) / (size_t)t->I ||
        (size_t)S > SIZE_MAX / sizeof(float) / (size_t)t->O) return 0;
    auto started = std::chrono::steady_clock::now();
    size_t xb = (size_t)S * t->I * sizeof(float), yb = (size_t)S * t->O * sizeof(float);
    /* a prompt block's pinned buffer goes straight, any other through the pinned scratch: a
     * decode step's rows are a few KB, and asking the driver costs more than copying them */
    const int px = tiled_ok && pinned(x), py = tiled_ok && pinned(y);
    float *dx = ::dx, *dy = ::dy;
    if (tiled_ok) {   /* a prompt block's: the device side from the arena */
        uint8_t *a = ar_use(AR_MAT, ar_align(xb) + yb);
        if (!a || (!px && !scratch_host(&hx, &hxcap, xb)) || (!py && !scratch_host(&hy, &hycap, yb))) { fallback++; return 0; }
        dx = (float *)a; dy = (float *)(a + ar_align(xb));
    } else if (!scratch(&::dx, &hx, &xcap, xb) || !scratch(&::dy, &hy, &ycap, yb)) { fallback++; return 0; }
    else { dx = ::dx; dy = ::dy; }
    if (!px) memcpy(hx, x, xb);
    if (!checked(cudaMemcpyAsync(dx, px ? x : hx, xb, cudaMemcpyHostToDevice, stream), "input")) return 0;
    dim3 grid((unsigned)((t->O + 3) / 4), (unsigned)S);
    /* one row (a decode step) gets a one-row tile: fewer registers, more warps in
     * flight; a row's arithmetic is the same in either */
    int ts = S == 1 ? 1 : MIMO_TS;
    dim3 tiles((unsigned)((t->O + 3) / 4), (unsigned)((S + ts - 1) / ts));
    if (tiled_ok && gemm_ok(t)) gemm_launch(t, dx, S, dy, stream);
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
        !checked(cudaMemcpyAsync(py ? y : hy, dy, yb, cudaMemcpyDeviceToHost, stream), "output") ||
        !checked(cudaStreamSynchronize(stream), "synchronize")) {
        cudaStreamSynchronize(stream); fallback++; return 0;
    }
    if (!py) memcpy(y, hy, yb);
    calls++; rows += S;
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
static int packed_tensor(const void *packed, size_t bytes, int I, int O, MimoCudaTensor *t) {
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
        if (!checked(cudaMemcpyAsync(pk_dev, packed, bytes, cudaMemcpyHostToDevice, stream), "packed weights")) return 0;
        pk_host = packed; pk_bytes_up = bytes; pk_uploads++;
    }
    memset(t, 0, sizeof *t);
    uint8_t *d = (uint8_t *)pk_dev;
    t->w = d; t->codes = d + e; t->lo = d + a_lo; t->exc_off = (const unsigned *)(d + a_off); t->exc = d + a_exc;
    t->fmt = 5; t->I = I; t->O = O; t->bytes = bytes;
    return 1;
}
extern "C" int mimo_cuda_matmul_packed(const void *packed, size_t bytes, int I, int O, float *y, const float *x, int S) {
    MimoCudaTensor t;
    return packed_tensor(packed, bytes, I, O, &t) && mimo_cuda_matmul(&t, y, x, S);
}
/* The same for a plain bf16 host matrix [O][I] (a router's), in a buffer of its own. */
static void *bf_dev;
static size_t bf_cap;
static const void *bf_host;
static int bf16_host_tensor(const uint16_t *w, int I, int O, MimoCudaTensor *t) {
    if (!ready || failed || !w || I < 1 || O < 1) return 0;
    size_t bytes = (size_t)O * I * sizeof(uint16_t);
    if (bf_host != w) {
        if (bytes > bf_cap) {
            void *nd = nullptr;
            if (cudaMalloc(&nd, bytes) != cudaSuccess) { cudaGetLastError(); return 0; }
            cudaFree(bf_dev); bf_dev = nd; bf_cap = bytes;
        }
        bf_host = nullptr;
        if (!checked(cudaMemcpyAsync(bf_dev, w, bytes, cudaMemcpyHostToDevice, stream), "bf16 weights")) return 0;
        bf_host = w; pk_uploads++;
    }
    memset(t, 0, sizeof *t);
    t->w = bf_dev; t->fmt = 1; t->I = I; t->O = O; t->bytes = bytes;
    return 1;
}
extern "C" int mimo_cuda_matmul_bf16_host(const uint16_t *w, int I, int O, float *y, const float *x, int S) {
    MimoCudaTensor t;
    return bf16_host_tensor(w, I, O, &t) && mimo_cuda_matmul(&t, y, x, S);
}

/* ---- a prompt block's attention ------------------------------------------------------
 * The attention of n rows at positions p0.. over the keys and values of positions
 * lo..lo+nkeys-1: for each (row, head) the keys of the sliding window of `window` positions
 * that end at the row (window 0: every key from lo to the row), softmax with the head's sink
 * logit in its sum when there is one, the values weighted by it. The CPU's attention() in
 * f32 with an online softmax over tiles of keys instead of a max pass: the same quantity,
 * summed in another order. */
/* The attention as tiled products: a block of 256 threads takes 128 query-heads of one kv head
 * -- 128 / G rows, every head of the group -- and the keys in tiles of 128 at multiples of 128
 * (absolute positions), so a query's tiles, and its bytes, depend on its own keys alone. Per
 * tile: the scores Q K^T in chunks of 16 dimensions through shared memory (two buffers), 8
 * queries x 8 keys a thread; masked to the query's keys (its window or every key before it,
 * those the chunk of keys in memory holds); the tile's max per query, one rescale of its sum and
 * output; P = exp(S - max) through shared memory in two halves of 64 keys, O += P V, 8 queries x
 * VD / 16 dimensions a thread. f32 throughout: the CPU's softmax, summed in tiles. A query's
 * running max and sum (ml) and output (out) wait in device memory between chunks of keys. */
#define FA_Q 128
#define FA_K 128
#define FA_D 16
template<int HD, int VD>
__global__ static void __launch_bounds__(256, 1) attn_flash(
        const float *__restrict__ q, int qstride, const float *__restrict__ k, const float *__restrict__ v,
        const float *__restrict__ sink, float2 *__restrict__ ml, float *__restrict__ out,
        int n, int nh, int kvh, int G, int p0, int lo, int window, int j0, int nk, float scale,
        int first_chunk, int last_chunk) {
    static_assert(HD % FA_D == 0 && VD % 32 == 0 && VD <= 128, "attn_flash: head shapes");
    constexpr int DP = VD / 32;   /* dimensions a thread holds in each half of the output */
    extern __shared__ __align__(16) float fsm[];
    /* the scores' phase: Q and K chunks [2][16][132]; the values' phase (aliased): P [64][132], V [64][VD + 4] */
    float (*qs)[FA_D][FA_Q + 4] = (float (*)[FA_D][FA_Q + 4])fsm;
    float (*ks)[FA_D][FA_K + 4] = (float (*)[FA_D][FA_K + 4])(fsm + 2 * FA_D * (FA_Q + 4));
    float (*ps)[FA_Q + 4] = (float (*)[FA_Q + 4])fsm;
    float (*vs)[VD + 4] = (float (*)[VD + 4])(fsm + 64 * (FA_Q + 4));
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4, kh = blockIdx.y;
    const int R = FA_Q / G, t0 = blockIdx.x * R;
    const float NEG = __int_as_float(0xff800000);
    /* this thread's queries: rows qrow(i) of S and O, their positions and first keys */
    int qt[8], qf[8];
    bool qon[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int qi = i < 4 ? ty * 4 + i : 64 + ty * 4 + i - 4, t = t0 + qi / G;
        qon[i] = t < n;
        qt[i] = p0 + t;
        qf[i] = window ? max(p0 + t - window + 1, lo) : lo;
    }
    float m[8], l[8], o[8][2 * DP];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int qi = i < 4 ? ty * 4 + i : 64 + ty * 4 + i - 4, t = t0 + qi / G, h = kh * G + qi % G;
        m[i] = NEG; l[i] = 0.f;
        #pragma unroll
        for (int j = 0; j < 2 * DP; j++) o[i][j] = 0.f;
        if (!qon[i]) continue;
        if (first_chunk) {
            if (sink) { m[i] = sink[h]; l[i] = 1.f; }
        } else {
            float2 s2 = ml[(size_t)t * nh + h];
            m[i] = s2.x; l[i] = s2.y;
            const float *op = out + ((size_t)t * nh + h) * VD;
            #pragma unroll
            for (int j = 0; j < DP; j++) { o[i][j] = op[tx * DP + j]; o[i][DP + j] = op[VD / 2 + tx * DP + j]; }
        }
    }
    /* the chunk loads: query-head lq (its dimensions lh..lh+7 of the chunk), key lq of the tile */
    const int lq = tid >> 1, lh = (tid & 1) * 8;
    const int lt = t0 + lq / G;
    const float *qrow = lt < n ? q + (size_t)lt * qstride + (size_t)(kh * G + lq % G) * HD + lh : nullptr;
    /* the block's keys in this chunk */
    const int t_last = min(t0 + R, n) - 1;
    int kb = window ? max(p0 + t0 - window + 1, lo) : lo, ke = p0 + t_last;
    if (kb < j0) kb = j0;
    if (ke > j0 + nk - 1) ke = j0 + nk - 1;
    for (int T = kb / FA_K * FA_K; T <= ke; T += FA_K) {
        /* ---- S = Q K^T over HD */
        float s[8][8];
        #pragma unroll
        for (int i = 0; i < 8; i++)
            #pragma unroll
            for (int j = 0; j < 8; j++) s[i][j] = 0.f;
        const int kj = T + lq;
        const float *krow = kj >= kb && kj <= ke ? k + ((size_t)(kj - j0) * kvh + kh) * HD + lh : nullptr;
        float4 rq0, rq1, rk0, rk1;
        auto fetch = [&](int d0) {
            const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
            rq0 = qrow ? *(const float4 *)(qrow + d0) : z; rq1 = qrow ? *(const float4 *)(qrow + d0 + 4) : z;
            rk0 = krow ? *(const float4 *)(krow + d0) : z; rk1 = krow ? *(const float4 *)(krow + d0 + 4) : z;
        };
        auto stash = [&](int b) {
            qs[b][lh + 0][lq] = rq0.x; qs[b][lh + 1][lq] = rq0.y; qs[b][lh + 2][lq] = rq0.z; qs[b][lh + 3][lq] = rq0.w;
            qs[b][lh + 4][lq] = rq1.x; qs[b][lh + 5][lq] = rq1.y; qs[b][lh + 6][lq] = rq1.z; qs[b][lh + 7][lq] = rq1.w;
            ks[b][lh + 0][lq] = rk0.x; ks[b][lh + 1][lq] = rk0.y; ks[b][lh + 2][lq] = rk0.z; ks[b][lh + 3][lq] = rk0.w;
            ks[b][lh + 4][lq] = rk1.x; ks[b][lh + 5][lq] = rk1.y; ks[b][lh + 6][lq] = rk1.z; ks[b][lh + 7][lq] = rk1.w;
        };
        fetch(0);
        stash(0);
        __syncthreads();
        constexpr int NC = HD / FA_D;
        for (int c = 0; c < NC; c++) {
            const int b = c & 1;
            if (c + 1 < NC) fetch((c + 1) * FA_D);
            #pragma unroll
            for (int d = 0; d < FA_D; d++) {
                const float4 a0 = *(const float4 *)&qs[b][d][ty * 4], a1 = *(const float4 *)&qs[b][d][64 + ty * 4];
                const float4 b0 = *(const float4 *)&ks[b][d][tx * 4], b1 = *(const float4 *)&ks[b][d][64 + tx * 4];
                const float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
                const float bv[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
                #pragma unroll
                for (int i = 0; i < 8; i++)
                    #pragma unroll
                    for (int j = 0; j < 8; j++) s[i][j] = fmaf(av[i], bv[j], s[i][j]);
            }
            if (c + 1 < NC) stash(b ^ 1);
            __syncthreads();
        }
        /* ---- the tile's softmax step, per query: max, one rescale, P */
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float mx = NEG;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int pos = T + (j < 4 ? tx * 4 + j : 64 + tx * 4 + j - 4);
                bool ok = qon[i] && pos >= qf[i] && pos <= qt[i] && pos >= kb && pos <= ke;
                s[i][j] = ok ? s[i][j] * scale : NEG;
                mx = fmaxf(mx, s[i][j]);
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 8));
            const float mn = fmaxf(m[i], mx);
            const float cr = mn == m[i] ? 1.f : expf(m[i] - mn);
            float sum = 0.f;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                s[i][j] = s[i][j] == NEG ? 0.f : expf(s[i][j] - mn);
                sum += s[i][j];
            }
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            sum += __shfl_xor_sync(0xffffffffu, sum, 2);
            sum += __shfl_xor_sync(0xffffffffu, sum, 4);
            sum += __shfl_xor_sync(0xffffffffu, sum, 8);
            l[i] = fmaf(l[i], cr, sum);
            m[i] = mn;
            #pragma unroll
            for (int j = 0; j < 2 * DP; j++) o[i][j] *= cr;
        }
        /* ---- O += P V, 64 keys at a time */
        #pragma unroll
        for (int hk = 0; hk < 2; hk++) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                const int qi = i < 4 ? ty * 4 + i : 64 + ty * 4 + i - 4;
                #pragma unroll
                for (int j = 0; j < 4; j++) ps[tx * 4 + j][qi] = s[i][hk * 4 + j];
            }
            for (int idx = tid; idx < 64 * (VD / 4); idx += 256) {
                const int kr = idx / (VD / 4), c4 = idx - kr * (VD / 4), kj2 = T + hk * 64 + kr;
                float4 val = make_float4(0.f, 0.f, 0.f, 0.f);
                if (kj2 >= kb && kj2 <= ke) val = *(const float4 *)(v + ((size_t)(kj2 - j0) * kvh + kh) * VD + c4 * 4);
                *(float4 *)&vs[kr][c4 * 4] = val;
            }
            __syncthreads();
            #pragma unroll 8
            for (int kk = 0; kk < 64; kk++) {
                const float4 a0 = *(const float4 *)&ps[kk][ty * 4], a1 = *(const float4 *)&ps[kk][64 + ty * 4];
                const float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
                float bv[2 * DP];
                if (DP == 4) {
                    const float4 b0 = *(const float4 *)&vs[kk][tx * 4], b1 = *(const float4 *)&vs[kk][VD / 2 + tx * 4];
                    bv[0] = b0.x; bv[1] = b0.y; bv[2] = b0.z; bv[3] = b0.w; bv[DP] = b1.x; bv[DP + 1] = b1.y; bv[DP + 2] = b1.z; bv[DP + 3] = b1.w;
                } else {
                    #pragma unroll
                    for (int j = 0; j < DP; j++) { bv[j] = vs[kk][tx * DP + j]; bv[DP + j] = vs[kk][VD / 2 + tx * DP + j]; }
                }
                #pragma unroll
                for (int i = 0; i < 8; i++)
                    #pragma unroll
                    for (int j = 0; j < 2 * DP; j++) o[i][j] = fmaf(av[i], bv[j], o[i][j]);
            }
            __syncthreads();
        }
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (!qon[i]) continue;
        const int qi = i < 4 ? ty * 4 + i : 64 + ty * 4 + i - 4, t = t0 + qi / G, h = kh * G + qi % G;
        float *op = out + ((size_t)t * nh + h) * VD;
        const float r = last_chunk ? 1.f / l[i] : 1.f;
        #pragma unroll
        for (int j = 0; j < DP; j++) { op[tx * DP + j] = o[i][j] * r; op[VD / 2 + tx * DP + j] = o[i][DP + j] * r; }
        if (!last_chunk && !tx) ml[(size_t)t * nh + h] = make_float2(m[i], l[i]);
    }
}

static float *a_q, *a_k, *a_v, *a_out, *a_sink;
static float2 *a_ml;
static size_t a_qcap, a_kcap, a_vcap, a_outcap, a_mlcap, a_sinkcap;
static unsigned long long attn_calls, attn_rows_n, attn_fallback;
static double attn_seconds;
static int attn_grow(float **d, size_t *cap, size_t bytes) {
    if (bytes <= *cap) return 1;
    float *nd = nullptr;
    if (cudaMalloc(&nd, bytes) != cudaSuccess) { cudaGetLastError(); return 0; }
    cudaFree(*d); *d = nd; *cap = bytes;
    return 1;
}
/* The attention of n rows (queries dq, row t at dq + t * qstride, on the device) over the keys
 * and values of positions lo..lo+nkeys-1: those before lo + nhost from the host (hk, hv: rows
 * of kvh * HD and kvh * VD floats), the others from device rows (dkb, dvb, pitch floats apart:
 * a block's own qkv). They go up in chunks of MIMO_CUDA_ATTN_MB (256) into a_k / a_v, each
 * query's state carried between the chunks in a_ml and dout. Queued on `stream`.
 * tag (a full layer's, within one prompt block): a_k / a_v keep its keys and values for its
 * next sub-block, which then sends up none of the positions they hold -- a 25,000-row block's
 * full layers sent their whole history up again for each of their 25 sub-blocks. */
/* keys a chunk holds: MIMO_CUDA_ATTN_MB (256) of keys and values, a multiple of the tiles' 128 */
static int attn_chunk(int kvh, int hd, int vd) {
    const char *mb = getenv("MIMO_CUDA_ATTN_MB");
    size_t budget = (size_t)((mb && *mb ? atof(mb) : 256.0) * 1048576.0);
    int chunk = (int)(budget / ((size_t)kvh * (hd + vd) * sizeof(float))) / FA_K * FA_K;
    return chunk < FA_K ? FA_K : chunk;
}
/* kb, vb: room for `cap` keys and values (cap >= the chunk, or >= nkeys when one chunk holds
 * them); mlb: n x nh, when the keys go in chunks. keep (the chain's): what kb / vb held for tag. */
template<int HD, int VD>
static int attn_run(const float *dq, int qstride, int n, int nh, int kvh, int lo, int nkeys, int p0, int window,
                    const float *dsink, float scale, float *dout, const float *hk, const float *hv, int nhost,
                    const float *dkb, const float *dvb, size_t pitch, float *kb, float *vb, float2 *mlb, int keep,
                    long long tag) {
    const int G = nh / kvh;
    if (FA_Q % G) return 0;
    const size_t shared = sizeof(float) * (size_t)(4 * FA_D * (FA_Q + 4) > 64 * (FA_Q + 4) + 64 * (VD + 4)
                                                   ? 4 * FA_D * (FA_Q + 4) : 64 * (FA_Q + 4) + 64 * (VD + 4));
    static int attr_set;
    if (!attr_set) {
        if (cudaFuncSetAttribute(attn_flash<HD, VD>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared) != cudaSuccess) {
            cudaGetLastError(); return 0;
        }
        attr_set = 1;
    }
    const size_t kd = (size_t)kvh * HD, vdd = (size_t)kvh * VD;
    /* chunks of keys at multiples of their size (a multiple of the tiles'): where a row's
     * keys split between chunks depends on its positions alone */
    const int chunk = attn_chunk(kvh, HD, VD);
    const int cfirst = lo / chunk, clast = (lo + nkeys - 1) / chunk;
    dim3 grid((unsigned)((n + FA_Q / G - 1) / (FA_Q / G)), (unsigned)kvh);
    const int split = lo + nhost;   /* the first position the device rows hold */
    if (keep && tag && !window && cfirst == clast) {   /* kept: the rows stay where they are */
        const int have = kv_tag == tag && kv_lo == lo && kv_hi == split ? split : lo;
        kv_tag = 0;
        if (split > have &&
            (!checked(cudaMemcpyAsync(kb + (size_t)(have - lo) * kd, hk + (size_t)(have - lo) * kd,
                                      (size_t)(split - have) * kd * sizeof(float), cudaMemcpyHostToDevice, stream), "attention keys") ||
             !checked(cudaMemcpyAsync(vb + (size_t)(have - lo) * vdd, hv + (size_t)(have - lo) * vdd,
                                      (size_t)(split - have) * vdd * sizeof(float), cudaMemcpyHostToDevice, stream), "attention values")))
            return 0;
        if (lo + nkeys > split &&
            (!checked(cudaMemcpy2DAsync(kb + (size_t)(split - lo) * kd, kd * sizeof(float), dkb, pitch * sizeof(float),
                                        kd * sizeof(float), (size_t)(lo + nkeys - split), cudaMemcpyDeviceToDevice, stream), "attention keys") ||
             !checked(cudaMemcpy2DAsync(vb + (size_t)(split - lo) * vdd, vdd * sizeof(float), dvb, pitch * sizeof(float),
                                        vdd * sizeof(float), (size_t)(lo + nkeys - split), cudaMemcpyDeviceToDevice, stream), "attention values")))
            return 0;
        attn_flash<HD, VD><<<grid, 256, shared, stream>>>(dq, qstride, kb, vb, dsink, mlb, dout,
                                                            n, nh, kvh, G, p0, lo, window, lo, nkeys, scale, 1, 1);
        if (!checked(cudaGetLastError(), "attention kernel")) return 0;
        kv_tag = tag; kv_lo = lo; kv_hi = lo + nkeys;
        return 1;
    }
    if (keep) kv_tag = 0;
    for (int c = cfirst; c <= clast; c++) {
        int j0 = c * chunk > lo ? c * chunk : lo, je = (c + 1) * chunk < lo + nkeys ? (c + 1) * chunk : lo + nkeys;
        int nk = je - j0, h1 = je < split ? je : split;
        if (h1 > j0 &&
            (!checked(cudaMemcpyAsync(kb, hk + (size_t)(j0 - lo) * kd, (size_t)(h1 - j0) * kd * sizeof(float),
                                      cudaMemcpyHostToDevice, stream), "attention keys") ||
             !checked(cudaMemcpyAsync(vb, hv + (size_t)(j0 - lo) * vdd, (size_t)(h1 - j0) * vdd * sizeof(float),
                                      cudaMemcpyHostToDevice, stream), "attention values"))) return 0;
        int d0 = j0 > split ? j0 : split;
        if (j0 + nk > d0 &&
            (!checked(cudaMemcpy2DAsync(kb + (size_t)(d0 - j0) * kd, kd * sizeof(float), dkb + (size_t)(d0 - split) * pitch,
                                        pitch * sizeof(float), kd * sizeof(float), (size_t)(j0 + nk - d0),
                                        cudaMemcpyDeviceToDevice, stream), "attention keys") ||
             !checked(cudaMemcpy2DAsync(vb + (size_t)(d0 - j0) * vdd, vdd * sizeof(float), dvb + (size_t)(d0 - split) * pitch,
                                        pitch * sizeof(float), vdd * sizeof(float), (size_t)(j0 + nk - d0),
                                        cudaMemcpyDeviceToDevice, stream), "attention values"))) return 0;
        attn_flash<HD, VD><<<grid, 256, shared, stream>>>(dq, qstride, kb, vb, dsink, mlb, dout,
                                                            n, nh, kvh, G, p0, lo, window, j0, nk, scale,
                                                            c == cfirst, c == clast);
        if (!checked(cudaGetLastError(), "attention kernel")) return 0;
    }
    return 1;
}
template<int HD, int VD>
static int attn_launch(const float *q, int qstride, int n, int nh, int kvh, const float *k, const float *v,
                       int lo, int nkeys, int p0, int window, const float *sink, float scale, float *out) {
    size_t qb = (size_t)n * nh * HD * sizeof(float), ob = (size_t)n * nh * VD * sizeof(float);
    if (!attn_grow(&a_q, &a_qcap, qb) || !attn_grow(&a_out, &a_outcap, ob) ||
        (sink && !attn_grow(&a_sink, &a_sinkcap, (size_t)nh * sizeof(float)))) return 0;
    if (!checked(cudaMemcpy2DAsync(a_q, (size_t)nh * HD * sizeof(float), q, (size_t)qstride * sizeof(float),
                                   (size_t)nh * HD * sizeof(float), (size_t)n, cudaMemcpyHostToDevice, stream), "attention queries") ||
        (sink && !checked(cudaMemcpyAsync(a_sink, sink, (size_t)nh * sizeof(float), cudaMemcpyHostToDevice, stream), "attention sinks")))
        return 0;
    const int chunk = attn_chunk(kvh, HD, VD), held = nkeys < chunk ? nkeys : chunk;
    if (!attn_grow(&a_k, &a_kcap, (size_t)held * kvh * HD * sizeof(float)) ||
        !attn_grow(&a_v, &a_vcap, (size_t)held * kvh * VD * sizeof(float)) ||
        (nkeys > held && !attn_grow((float **)&a_ml, &a_mlcap, (size_t)n * nh * sizeof(float2)))) return 0;
    if (!attn_run<HD, VD>(a_q, nh * HD, n, nh, kvh, lo, nkeys, p0, window, sink ? a_sink : nullptr, scale, a_out,
                          k, v, nkeys, nullptr, nullptr, 0, a_k, a_v, a_ml, 0, 0))
        return 0;
    if (!checked(cudaMemcpyAsync(out, a_out, ob, cudaMemcpyDeviceToHost, stream), "attention output") ||
        !checked(cudaStreamSynchronize(stream), "attention")) return 0;
    return 1;
}
extern "C" int mimo_cuda_attention(const float *q, int qstride, int n, int nh, int kvh, int hd, int vd,
                                   const float *k, const float *v, int lo, int nkeys, int p0, int window,
                                   const float *sink, float scale, float *out) {
    if (!ready || failed || !q || !k || !v || !out || n < 1 || nkeys < 1 || kvh < 1 || nh % kvh || FA_Q % (nh / kvh) ||
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

/* ---- a prompt block's attention, from its normed rows to o_proj's output -----------------
 * mimo_cuda_attn_block keeps a block's rows on the device between the matrices and the
 * attention, where attention() sent them down and up again: the qkv product into c_qkv, RoPE
 * (the host's cos/sin, each row's own) and the values' scale in place, the keys and values the
 * host keeps (its caches) down, the attention reading the queries and the block's own keys
 * and values where they are and the earlier positions' from the host, its output into c_ctx
 * and o_proj from there. Only the rows' input goes up and the caches' rows and o_proj's output
 * come down. */
__global__ static void rope_rows(float *__restrict__ qkv, int rows, int heads, int hd, int nv, int rd,
                                 const float *__restrict__ tab, float vscale) {
    float *r = qkv + (size_t)blockIdx.x * rows;
    const float *cs = tab + (size_t)blockIdx.x * rd, *sn = cs + rd / 2;
    const int half = rd / 2;
    for (int i = threadIdx.x; i < heads * half; i += blockDim.x) {   /* the q heads, then the k heads */
        int h = i / half, d = i - h * half;
        float *v = r + (size_t)h * hd;
        float a = v[d], b = v[d + half];
        v[d] = a * cs[d] - b * sn[d];
        v[d + half] = b * cs[d] + a * sn[d];
    }
    float *vv = r + (size_t)heads * hd;
    for (int i = threadIdx.x; i < nv; i += blockDim.x) vv[i] *= vscale;
}
static float *c_rope, *c_sink;
static size_t c_ropecap, c_sinkcap;
static size_t c_kvoff;   /* where the arena's keys sat for the chain's last call (kept keys move: dropped) */
static cudaEvent_t c_ev[6];   /* a block's start, its rows up, qkv and RoPE, attention, o_proj, its outputs down */
static double c_seconds[6];   /* the device's: rows up, qkv and RoPE, attention (its keys' copies in), o_proj, outputs down; all */
static unsigned long long c_calls, c_rows, c_fallback;
template<int HD, int VD>
static int attn_block(const MimoCudaAttnBlock *a, const MimoCudaTensor *o) {
    const int n = a->n, nh = a->nh, kvh = a->kvh, qd = nh * HD, kd = kvh * HD, vdd = kvh * VD, rows = qd + kd + vdd;
    const int nkeys = a->p0 + n - a->lo, npast = a->p0 - a->lo;
    /* the arena: rows in and o_proj out, qkv, the attention's output, its state in chunks, the
     * keys and values -- those last, from the most rows a call has had, so a layer's kept keys
     * stay where its next sub-block finds them */
    static int ncap;
    if (n > ncap) ncap = n;
    const int chunk = attn_chunk(kvh, HD, VD);
    const size_t bx = ar_align((size_t)ncap * (a->H > o->O ? a->H : o->O) * sizeof(float)),
                 bq = ar_align((size_t)ncap * rows * sizeof(float)), bc = ar_align((size_t)ncap * nh * VD * sizeof(float)),
                 bm = ar_align((size_t)ncap * nh * sizeof(float2)), bk = ar_align((size_t)chunk * kd * sizeof(float)),
                 bv = ar_align((size_t)chunk * vdd * sizeof(float));
    uint8_t *ar = ar_use(AR_ATTN, bx + bq + bc + bm + bk + bv);
    if (!ar || !attn_grow(&c_rope, &c_ropecap, (size_t)n * a->rd * sizeof(float)) ||
        (a->sink && !attn_grow(&c_sink, &c_sinkcap, (size_t)nh * sizeof(float)))) return 0;
    float *c_x = (float *)ar, *c_qkv = (float *)(ar + bx), *c_ctx = (float *)(ar + bx + bq);
    float2 *c_ml = (float2 *)(ar + bx + bq + bc);
    float *c_k = (float *)(ar + bx + bq + bc + bm), *c_v = (float *)(ar + bx + bq + bc + bm + bk);
    if (bx + bq + bc + bm != c_kvoff) { kv_tag = 0; c_kvoff = bx + bq + bc + bm; }
    if (!c_ev[0])
        for (int i = 0; i < 6; i++)
            if (!checked(cudaEventCreate(&c_ev[i]), "attention events")) return 0;
    if (!checked(cudaEventRecord(c_ev[0], stream), "attention events") ||
        !checked(cudaMemcpyAsync(c_x, a->xn, (size_t)n * a->H * sizeof(float), cudaMemcpyHostToDevice, stream), "block rows") ||
        !checked(cudaMemcpyAsync(c_rope, a->rope, (size_t)n * a->rd * sizeof(float), cudaMemcpyHostToDevice, stream), "rope") ||
        (a->sink && !checked(cudaMemcpyAsync(c_sink, a->sink, (size_t)nh * sizeof(float), cudaMemcpyHostToDevice, stream), "sinks")) ||
        !checked(cudaEventRecord(c_ev[1], stream), "attention events")) return 0;
    gemm_launch(a->qkv, c_x, n, c_qkv, stream);
    rope_rows<<<n, 256, 0, stream>>>(c_qkv, rows, nh + kvh, HD, vdd, a->rd, c_rope, a->v_scale);
    if (!checked(cudaGetLastError(), "qkv") || !checked(cudaEventRecord(c_ev[2], stream), "attention events") ||
        (n > a->kv_from &&
         !checked(cudaMemcpy2DAsync(a->kv, (size_t)(kd + vdd) * sizeof(float), c_qkv + (size_t)a->kv_from * rows + qd,
                                    (size_t)rows * sizeof(float), (size_t)(kd + vdd) * sizeof(float), (size_t)(n - a->kv_from),
                                    cudaMemcpyDeviceToHost, stream), "keys and values")))
        return 0;
    if (!attn_run<HD, VD>(c_qkv, rows, n, nh, kvh, a->lo, nkeys, a->p0, a->window, a->sink ? c_sink : nullptr, a->scale,
                          c_ctx, a->kpast, a->vpast, npast, c_qkv + qd, c_qkv + qd + kd, (size_t)rows, c_k, c_v, c_ml, 1,
                          a->kv_tag) ||
        !checked(cudaEventRecord(c_ev[3], stream), "attention events")) return 0;
    gemm_launch(o, c_ctx, n, c_x, stream);
    if (!checked(cudaGetLastError(), "o_proj") || !checked(cudaEventRecord(c_ev[4], stream), "attention events") ||
        !checked(cudaMemcpyAsync(a->out, c_x, (size_t)n * o->O * sizeof(float), cudaMemcpyDeviceToHost, stream), "o_proj output") ||
        !checked(cudaEventRecord(c_ev[5], stream), "attention events")) return 0;
    return 1;
}
/* mimo_cuda_attn_block's work queued, not waited for: kv and out (pinned, for the copies to
 * run beside the caller) are written once mimo_cuda_attn_wait returns 1. One at a time. */
extern "C" int mimo_cuda_attn_wait(void);
extern "C" int mimo_cuda_attn_submit(const MimoCudaAttnBlock *a) {
    if (!ready || failed || !a || !a->qkv || !a->xn || !a->rope || !a->out || !a->kv || a->n < 1 || a->kvh < 1 ||
        a->nh % a->kvh || FA_Q % (a->nh / a->kvh) || a->rd < 2 || a->rd % 2 || a->rd > a->hd || a->p0 < a->lo ||
        a->kv_from < 0 || a->kv_from > a->n || (a->p0 > a->lo && (!a->kpast || !a->vpast)) ||
        !gemm_ok(a->qkv) || a->qkv->I != a->H || a->qkv->O != a->nh * a->hd + a->kvh * (a->hd + a->vd)) return 0;
    MimoCudaTensor pk;
    const MimoCudaTensor *o = a->o;
    if (!o && !(a->o_packed && packed_tensor(a->o_packed, a->o_bytes, a->nh * a->vd, a->H, &pk))) return 0;
    if (!o) o = &pk;
    if (!gemm_ok(o) || o->I != a->nh * a->vd) return 0;
    if (c_pending && !mimo_cuda_attn_wait()) return 0;
    int ok = a->hd == 192 && a->vd == 128 ? attn_block<192, 128>(a, o)
           : a->hd == 48 && a->vd == 32 ? attn_block<48, 32>(a, o)
           : -1;
    if (ok < 0) return 0;
    if (!ok) { cudaStreamSynchronize(stream); c_fallback++; return 0; }
    c_pending = 1; c_calls++; c_rows += (unsigned long long)a->n;
    return 1;
}
extern "C" int mimo_cuda_attn_wait(void) {
    if (!c_pending) return 1;
    c_pending = 0;
    if (!checked(cudaEventSynchronize(c_ev[5]), "attention block")) { cudaStreamSynchronize(stream); c_fallback++; return 0; }
    float ms;
    for (int i = 0; i < 5; i++) { cudaEventElapsedTime(&ms, c_ev[i], c_ev[i + 1]); c_seconds[i] += ms / 1e3; }
    cudaEventElapsedTime(&ms, c_ev[0], c_ev[5]); c_seconds[5] += ms / 1e3;
    return 1;
}
extern "C" int mimo_cuda_attn_block(const MimoCudaAttnBlock *a) {
    return mimo_cuda_attn_submit(a) && mimo_cuda_attn_wait();
}
/* ---- a prompt block's routed experts --------------------------------------------------
 * The MXFP4 experts of a prompt block on the device. mimo_cuda_moe_rows sends the block's
 * normed rows up (or mimo_cuda_moe_begin does), for the router (mimo_cuda_moe_router) and the
 * experts; mimo_cuda_moe_begin hands out a pinned host buffer for every (row, choice) pair's
 * output. For each expert, in any order, the host fills one of MIMO_CUDA_MOE_STAGES (32) pinned
 * staging buffers with its bytes as the expert cache holds them (down, its scales, gate, its
 * scales, up, its scales) -- a cached expert copied in, a missing one read from disk straight
 * into it -- and names it to mimo_cuda_moe_expert_at: the copy stream sends the bytes up into
 * one of MOE_SLOTS device slots, in turn, once the slot's kernels before are done; the slot's own
 * stream runs gate and up over the rows that chose it, silu(gate) * up and down (gemm_tiled: each
 * 32-column group's products of the e2m1 values summed, times its e8m0 scale), down
 * storing its rows' outputs straight into their pairs' places in the pinned buffer;
 * mimo_cuda_moe_end waits for the last. The host then sums each row's pairs in its routing
 * order, as it does the CPU's (moe). The slots run side by side, so a small expert's tiles run
 * beside the next one's. An expert with more rows than a slot's gate * up holds (moe_rcap) goes
 * in pieces. Its buffers share the arena with the attention's (ar_use).
 * The card has one copy engine, and it takes copies in the order they come: an output copy
 * queued behind its expert's kernels held up every later expert's upload (the experts went one
 * at a time, 2.9 ms each). The outputs' stores hold the down kernel's blocks while the link
 * drains them, beside the uploads (1.65 ms an expert, 1.2 ms of the link's: 33 MB); copies of
 * them from the device, by the copy engine once the kernels were done or by a narrow store
 * kernel, made the slots wait on them instead (1.8 - 2.5 ms; tools/bench_mimo_moe.cu). */
#define MOE_SLOTS 12
#define MOE_STAGES_MAX 256
static int moe_rcap = 1024;   /* the rows of an expert computed at once, its buffers' (MIMO_CUDA_MOE_RCAP for tests) */
static cudaStream_t m_copy, m_comp[MOE_SLOTS];
static cudaEvent_t m_copied[MOE_SLOTS], m_computed[MOE_SLOTS], m_xin;
static int m_cur, m_H, m_MI, m_E, m_rmax, m_npairs, m_on, m_rcap;
static size_t m_ebytes, m_metacap, m_pairscap, m_sbytes;
static void *m_stage_mem[MOE_STAGES_MAX];   /* the pinned allocations; m_stage[i] is 4 KiB aligned inside */
static uint8_t *m_stage[MOE_STAGES_MAX];
static cudaEvent_t m_staged[MOE_STAGES_MAX];  /* the copy that read stage i last */
static int m_sused[MOE_STAGES_MAX], m_nstage, m_scur;
static int *m_hrows[MOE_SLOTS], *m_drows[MOE_SLOTS];
static int m_busy[MOE_SLOTS], m_pair0[MOE_SLOTS];   /* a slot's expert queued: its kernels may run */
/* the arena's: the rows, the router's logits, each slot's weights and gate * up */
static float *m_x, *m_logits, *m_g[MOE_SLOTS];
static uint8_t *m_dw[MOE_SLOTS];
static float *m_pairs;
static const float *m_xhost;   /* the rows' host copy, for an arena that had to grow after they went up */
static unsigned long long m_xgen, ar_gen;
static unsigned long long moe_blocks, moe_experts, moe_rows_n, moe_fallback;
static double moe_seconds;
static void moe_free(void) {
    for (int b = 0; b < MOE_SLOTS; b++) {
        if (m_copied[b]) cudaEventDestroy(m_copied[b]);
        if (m_computed[b]) cudaEventDestroy(m_computed[b]);
        if (m_comp[b]) cudaStreamDestroy(m_comp[b]);
        cudaFreeHost(m_hrows[b]); cudaFree(m_drows[b]);
        m_copied[b] = m_computed[b] = nullptr; m_comp[b] = nullptr;
        m_hrows[b] = m_drows[b] = nullptr;
    }
    for (int i = 0; i < m_nstage; i++) {
        if (m_staged[i]) cudaEventDestroy(m_staged[i]);
        cudaFreeHost(m_stage_mem[i]);
        m_staged[i] = nullptr; m_stage_mem[i] = nullptr; m_stage[i] = nullptr;
    }
    m_nstage = 0; m_sbytes = 0;
    if (m_copy) cudaStreamDestroy(m_copy);
    m_copy = nullptr; m_ebytes = m_metacap = 0;
}
static void moe_free_rows(void) {
    if (m_xin) cudaEventDestroy(m_xin);
    m_xin = nullptr;
    cudaFreeHost(m_pairs);
    m_pairs = nullptr; m_pairscap = 0; m_xn = 0;
}
/* The MoE step's buffers for n rows in the arena, with the sizes known so far (the router's
 * E and the experts' from the step before); the rows sent up again if the arena had to grow. */
static int moe_layout(int n) {
    const size_t bx = ar_align((size_t)n * m_xh * sizeof(float)), bl = ar_align((size_t)n * m_E * sizeof(float)),
                 bw = ar_align(m_ebytes), bg = ar_align((size_t)m_rcap * m_MI * sizeof(float));
    uint8_t *before = ar_mem;
    size_t had = ar_cap;
    uint8_t *ar = ar_use(AR_MOE, bx + bl + MOE_SLOTS * (bw + bg));
    if (!ar) return 0;
    if (ar != before || ar_cap != had) ar_gen++;
    m_x = (float *)ar; m_logits = (float *)(ar + bx);
    for (int b = 0; b < MOE_SLOTS; b++) {
        uint8_t *sb = ar + bx + bl + (size_t)b * (bw + bg);
        m_dw[b] = sb; m_g[b] = (float *)(sb + bw);
    }
    if (m_xn == n && m_xgen != ar_gen) {   /* the rows went up into memory since replaced */
        if (!m_xhost || !checked(cudaMemcpyAsync(m_x, m_xhost, (size_t)n * m_xh * sizeof(float), cudaMemcpyHostToDevice, stream), "experts' rows") ||
            !checked(cudaEventRecord(m_xin, stream), "experts' rows")) { m_xn = 0; return 0; }
        m_xgen = ar_gen;
    }
    return 1;
}
/* A prompt block's normed rows for its MoE step, sent up once for its router and its experts
 * (mimo_cuda_moe_begin with no rows takes these); x stays the caller's until the step ends. */
extern "C" int mimo_cuda_moe_rows(const float *x, int n, int H) {
    if (!ready || failed || !x || n < 1 || H < 1) return 0;
    m_xn = 0; m_xh = H;
    if (!m_xin && !checked(cudaEventCreateWithFlags(&m_xin, cudaEventDisableTiming), "experts' rows")) return 0;
    if (!moe_layout(n)) return 0;
    if (!checked(cudaMemcpyAsync(m_x, x, (size_t)n * H * sizeof(float), cudaMemcpyHostToDevice, stream), "experts' rows") ||
        !checked(cudaEventRecord(m_xin, stream), "experts' rows")) return 0;
    m_xn = n; m_xhost = x; m_xgen = ar_gen;
    return 1;
}
/* The router's logits [n][E] of the rows mimo_cuda_moe_rows sent: w [E][H] bf16 on the host,
 * sent up and kept while the next call names the same bytes. gemm_tiled, as a bf16 matrix call
 * of a prompt block computes it. */
extern "C" int mimo_cuda_moe_router(const uint16_t *w, int E, float *logits) {
    MimoCudaTensor t;
    if (!m_xn || !logits || !bf16_host_tensor(w, m_xh, E, &t) || !gemm_ok(&t)) return 0;
    auto started = std::chrono::steady_clock::now();
    m_E = E;
    if (!moe_layout(m_xn)) return 0;
    size_t lb = (size_t)m_xn * E * sizeof(float);
    gemm_launch(&t, m_x, m_xn, m_logits, stream);
    if (!checked(cudaGetLastError(), "router") ||
        !checked(cudaMemcpyAsync(logits, m_logits, lb, cudaMemcpyDeviceToHost, stream), "router logits") ||
        !checked(cudaStreamSynchronize(stream), "router")) { cudaStreamSynchronize(stream); return 0; }
    calls++; rows += (unsigned long long)m_xn;
    seconds[1] += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return 1;
}
extern "C" float *mimo_cuda_moe_begin(const float *x, int n, int H, int MI, size_t ebytes, int npairs, int rmax) {
    if (!ready || failed || n < 1 || npairs < 1 || rmax < 1 || rmax > n || H % GB_N || MI % (GB_N / 2) ||
        H % 32 || MI % 32 || !ebytes || (!x && (m_xn != n || m_xh != H))) return nullptr;
    if (ebytes != m_ebytes) {   /* the streams, events and staging, made once for the model's expert size */
        moe_free();
        if (cudaStreamCreateWithFlags(&m_copy, cudaStreamNonBlocking) != cudaSuccess ||
            (!m_xin && cudaEventCreateWithFlags(&m_xin, cudaEventDisableTiming) != cudaSuccess)) { cudaGetLastError(); moe_free(); return nullptr; }
        for (int b = 0; b < MOE_SLOTS; b++)
            if (cudaStreamCreateWithFlags(&m_comp[b], cudaStreamNonBlocking) != cudaSuccess ||
                cudaEventCreateWithFlags(&m_copied[b], cudaEventDisableTiming) != cudaSuccess ||
                cudaEventCreateWithFlags(&m_computed[b], cudaEventDisableTiming) != cudaSuccess) {
                cudaGetLastError(); moe_free(); return nullptr;
            }
        /* staging: room for a direct read's 4 KiB alignment on both ends (the cache's slots' size) */
        const char *ns = getenv("MIMO_CUDA_MOE_STAGES");
        int want = ns && *ns ? atoi(ns) : 32;
        if (want < 2) want = 2;
        if (want > MOE_STAGES_MAX) want = MOE_STAGES_MAX;
        m_sbytes = (ebytes + 8192 + 4095) & ~(size_t)4095;
        for (int i = 0; i < want; i++) {
            if (cudaMallocHost(&m_stage_mem[i], m_sbytes + 4096) != cudaSuccess ||
                cudaEventCreateWithFlags(&m_staged[i], cudaEventDisableTiming) != cudaSuccess) {
                cudaGetLastError(); cudaFreeHost(m_stage_mem[i]); m_stage_mem[i] = nullptr;
                if (m_staged[i]) { cudaEventDestroy(m_staged[i]); m_staged[i] = nullptr; }
                break;
            }
            m_stage[i] = (uint8_t *)(((uintptr_t)m_stage_mem[i] + 4095) & ~(uintptr_t)4095);
            m_nstage = i + 1;
        }
        if (m_nstage < 2) { moe_free(); return nullptr; }
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
    { const char *e = getenv("MIMO_CUDA_MOE_RCAP"); moe_rcap = e && *e && atoi(e) > 0 ? atoi(e) : 1024; }
    m_rcap = rmax < moe_rcap ? rmax : moe_rcap;
    m_MI = MI; m_xh = H;
    if (x) { m_xn = 0; m_xhost = nullptr; }
    if (!moe_layout(n)) return nullptr;
    size_t pb = (size_t)npairs * H * sizeof(float);
    if (pb > m_pairscap) {
        cudaFreeHost(m_pairs); m_pairs = nullptr; m_pairscap = 0;
        if (cudaMallocHost(&m_pairs, pb) != cudaSuccess) { cudaGetLastError(); return nullptr; }
        m_pairscap = pb;
    }
    if (x && (!checked(cudaMemcpyAsync(m_x, x, (size_t)n * H * sizeof(float), cudaMemcpyHostToDevice, stream), "experts' rows") ||
              !checked(cudaEventRecord(m_xin, stream), "experts' rows"))) return nullptr;
    m_xn = 0; m_xhost = nullptr;   /* the rows are this step's now */
    for (int b = 0; b < MOE_SLOTS; b++) m_busy[b] = 0;
    for (int i = 0; i < m_nstage; i++) m_sused[i] = 0;
    m_cur = 0; m_scur = 0; m_H = H; m_rmax = rmax; m_npairs = npairs; m_on = 1;
    moe_blocks++; moe_rows_n += (unsigned long long)n;
    return m_pairs;
}
/* A piece of slot b's expert: gate * up and down over rows p0.. of its list (pr of them), down
 * storing into the pairs pair0 + p0.. of the pinned buffer. */
static int moe_piece(int b, int p0, int pr) {
    const int H = m_H, MI = m_MI;
    const size_t dq = 0, ds = dq + (size_t)H * (MI / 2), gq = ds + (size_t)H * (MI / 32), gs = gq + (size_t)MI * (H / 2);
    const size_t uq = gs + (size_t)MI * (H / 32), us = uq + (size_t)MI * (H / 2);
    GemmW gu{}, gd{};
    gu.q = m_dw[b] + gq; gu.s8 = m_dw[b] + gs; gu.q2 = m_dw[b] + uq; gu.s82 = m_dw[b] + us;
    gd.q = m_dw[b] + dq; gd.s8 = m_dw[b] + ds;
    cudaStream_t cs = m_comp[b];
    dim3 g1((unsigned)(MI / (GB_N / 2)), (unsigned)((pr + GB_M - 1) / GB_M)), g2((unsigned)(H / GB_N), (unsigned)((pr + GB_M - 1) / GB_M));
    gemm_tiled<GW_PAIR><<<g1, 256, 0, cs>>>(m_x, H, m_drows[b] + p0, pr, H, MI, gu, m_g[b], MI);
    gemm_tiled<GW_MX4><<<g2, 256, 0, cs>>>(m_g[b], MI, nullptr, pr, MI, H, gd, m_pairs + (size_t)(m_pair0[b] + p0) * H, H);
    return checked(cudaGetLastError(), "expert kernels");
}
/* The slots whose kernels are done. Returns the experts still in flight, -1 on error. */
static int moe_poll(void) {
    int busy = 0;
    for (int b = 0; b < MOE_SLOTS; b++) {
        if (!m_busy[b]) continue;
        cudaError_t e = cudaEventQuery(m_computed[b]);
        if (e == cudaErrorNotReady) { busy++; continue; }
        if (!checked(e, "experts")) return -1;
        m_busy[b] = 0;
    }
    return busy;
}
extern "C" int mimo_cuda_moe_poll(void) { return m_on ? moe_poll() : 0; }
/* waits for event e, the slots' outputs going down meanwhile */
static int moe_wait(cudaEvent_t e) {
    for (;;) {
        cudaError_t q = cudaEventQuery(e);
        if (q == cudaSuccess) return 1;
        if (q != cudaErrorNotReady) return checked(q, "experts");
        if (moe_poll() < 0) return 0;
        std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
}
/* The staging buffers: their count, and *bytes each (4 KiB aligned). Valid after begin. */
extern "C" int mimo_cuda_moe_stages(size_t *bytes) {
    if (bytes) *bytes = m_sbytes;
    return m_on ? m_nstage : 0;
}
/* Staging buffer i once the copy that read it last is done; wait 0: NULL while it is not. */
extern "C" void *mimo_cuda_moe_stage(int i, int wait) {
    if (!m_on || i < 0 || i >= m_nstage) return nullptr;
    if (moe_poll() < 0) return nullptr;
    if (m_sused[i]) {
        if (!wait) {
            cudaError_t e = cudaEventQuery(m_staged[i]);
            if (e == cudaErrorNotReady) return nullptr;
            if (!checked(e, "experts' staging")) return nullptr;
        } else if (!moe_wait(m_staged[i])) return nullptr;
        m_sused[i] = 0;
    }
    return m_stage[i];
}
/* The next staging buffer in turn, for a caller that fills them one after the other. */
extern "C" void *mimo_cuda_moe_staging(void) { return mimo_cuda_moe_stage(m_scur, 1); }
static int moe_expert_at(int i, size_t off, const int *rows, int r, int pair0);
extern "C" int mimo_cuda_moe_expert(const int *rows, int r, int pair0) {
    int i = m_scur;
    m_scur = (m_scur + 1) % (m_nstage ? m_nstage : 1);
    return moe_expert_at(i, 0, rows, r, pair0);
}
/* The expert whose bytes (as the cache holds them) start at offset off of staging buffer i,
 * over rows[0..r) of the block; its outputs go to the pairs pair0 .. pair0 + r - 1 of the
 * buffer begin returned. The buffer is the host's again once mimo_cuda_moe_stage gives it. */
extern "C" int mimo_cuda_moe_expert_at(int i, size_t off, const int *rows, int r, int pair0) {
    return moe_expert_at(i, off, rows, r, pair0);
}
static int moe_expert_at(int i, size_t off, const int *rows, int r, int pair0) {
    if (!m_on || i < 0 || i >= m_nstage || off + m_ebytes > m_sbytes || r < 1 || r > m_rmax || pair0 < 0 || pair0 + r > m_npairs) return 0;
    auto started = std::chrono::steady_clock::now();
    const int b = m_cur;
    cudaStream_t cs = m_comp[b];
    /* the rows' list of the slot's expert before has gone up */
    if (m_busy[b] && !moe_wait(m_copied[b])) return 0;
    memcpy(m_hrows[b], rows, (size_t)r * sizeof(int));
    /* copy stream: the weights and rows into slot b once the kernels that read it last are done */
    if ((m_busy[b] && !checked(cudaStreamWaitEvent(m_copy, m_computed[b], 0), "experts' order")) ||
        !checked(cudaMemcpyAsync(m_dw[b], m_stage[i] + off, m_ebytes, cudaMemcpyHostToDevice, m_copy), "expert weights") ||
        !checked(cudaEventRecord(m_staged[i], m_copy), "experts' copies") ||
        !checked(cudaMemcpyAsync(m_drows[b], m_hrows[b], (size_t)r * sizeof(int), cudaMemcpyHostToDevice, m_copy), "expert rows") ||
        !checked(cudaEventRecord(m_copied[b], m_copy), "experts' copies") ||
        !checked(cudaStreamWaitEvent(cs, m_copied[b], 0), "experts' order") ||
        !checked(cudaStreamWaitEvent(cs, m_xin, 0), "experts' order")) return 0;
    m_pair0[b] = pair0;
    for (int p0 = 0; p0 < r; p0 += m_rcap)
        if (!moe_piece(b, p0, r - p0 < m_rcap ? r - p0 : m_rcap)) return 0;
    if (!checked(cudaEventRecord(m_computed[b], cs), "experts' order")) return 0;
    m_busy[b] = 1; m_cur = (b + 1) % MOE_SLOTS; m_sused[i] = 1;
    moe_experts++;
    moe_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return 1;
}
/* Waits until every expert's outputs are in the pairs buffer. */
extern "C" int mimo_cuda_moe_end(void) {
    if (!m_on) return 0;
    auto started = std::chrono::steady_clock::now();
    m_on = 0;
    int ok = checked(cudaStreamSynchronize(m_copy), "experts' copies") && checked(cudaStreamSynchronize(stream), "experts' rows");
    for (int b = 0; b < MOE_SLOTS && ok; b++) ok = checked(cudaStreamSynchronize(m_comp[b]), "experts");
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
    if (ready && (c_calls || c_fallback))
        fprintf(stderr, "[CUDA] mimo prompt attention kept on the device: %llu blocks, %llu rows, %.3fs on the device (rows up %.3fs, "
                        "qkv and RoPE %.3fs, attention %.3fs, o_proj %.3fs, outputs down %.3fs), %llu left to the host\n",
                c_calls, c_rows, c_seconds[5], c_seconds[0], c_seconds[1], c_seconds[2], c_seconds[3], c_seconds[4], c_fallback);
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
    cudaFree(a_q); cudaFree(a_k); cudaFree(a_v); cudaFree(a_out); cudaFree(a_ml); cudaFree(a_sink);
    a_q = a_k = a_v = a_out = a_sink = nullptr; a_ml = nullptr;
    a_qcap = a_kcap = a_vcap = a_outcap = a_mlcap = a_sinkcap = 0; kv_tag = 0;
    attn_calls = attn_rows_n = attn_fallback = 0; attn_seconds = 0;
    moe_free(); moe_free_rows();
    moe_blocks = moe_experts = moe_rows_n = moe_fallback = 0; moe_seconds = 0; m_on = 0;
    cudaFree(pk_dev); pk_dev = nullptr; pk_cap = pk_bytes_up = 0; pk_host = nullptr; pk_uploads = 0;
    cudaFree(c_rope); cudaFree(c_sink);
    c_rope = c_sink = nullptr; c_ropecap = c_sinkcap = 0; c_kvoff = 0;
    cudaFree(ar_mem); ar_mem = nullptr; ar_cap = 0; ar_owner = AR_NONE;
    for (int i = 0; i < 6; i++) { if (c_ev[i]) cudaEventDestroy(c_ev[i]); c_ev[i] = nullptr; c_seconds[i] = 0; }
    c_calls = c_rows = c_fallback = 0; c_pending = 0;
    cudaFree(bf_dev); bf_dev = nullptr; bf_cap = 0; bf_host = nullptr;
    if (stream) cudaStreamDestroy(stream);
    stream = nullptr; dx = dy = hx = hy = nullptr; xcap = ycap = resident = tensor_count = 0;
    calls = rows = fallback = packed_count = packed_exc = 0; packed_bytes = packed_bf16 = 0;
    ready = failed = 0; selected = -1;
    memset(seconds, 0, sizeof seconds);
}
