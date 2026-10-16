/* Native FP8/BF16 dense offload for MiMo. No weight/activation requantization.
 * The FP8 scale table is PER ROW, after MiMo's pre-sharded QKV deinterleave;
 * using the generic backend's 128x128 scale grid here would be incorrect.
 * One warp per output row, four warps per block. Every batch size uses the
 * same reduction, preserving prompt-chunk and prefix-reuse consistency. */
#include "backend_cuda_mimo.h"
#include <cuda_runtime.h>
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
static double seconds[4];

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
    if (!ready || failed || !w || I < 1 || I > INT_MAX - 127 || O < 1 || O > INT_MAX - 3 || fmt < 0 || fmt > 3 ||
        ((fmt == 2 || fmt == 3) && !scales)) return nullptr;
    size_t esz = fmt == 0 ? 4 : fmt == 1 ? 2 : 1;
    if ((size_t)O > SIZE_MAX / (size_t)I / esz) return nullptr;
    size_t wb = (size_t)O * I * esz;
    size_t ns = fmt == 2 ? (size_t)O * (((size_t)I + 127) / 128) : fmt == 3 ? (size_t)O : 0;
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
    if (FMT == 2) {
        double total = 0;
        int nb = (I + 127) / 128;
        for (int b = 0; b < nb; b++) {
            float part = 0;
            for (int k = lane; k < 128 && b * 128 + k < I; k += 32) {
                int i = b * 128 + k;
                part = fmaf(e4m3(((const unsigned char *)weights)[base + i]), xs[i], part);
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
    switch (t->fmt) {
    case 0: dense<0><<<grid,128,0,stream>>>(t->w,t->scales,dx,dy,t->I,t->O); break;
    case 1: dense<1><<<grid,128,0,stream>>>(t->w,t->scales,dx,dy,t->I,t->O); break;
    case 2: dense<2><<<grid,128,0,stream>>>(t->w,t->scales,dx,dy,t->I,t->O); break;
    case 3: dense<3><<<grid,128,0,stream>>>(t->w,t->scales,dx,dy,t->I,t->O); break;
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
    if (ready) fprintf(stderr, "[CUDA] mimo matmul wall time (copies + compute): f32 %.3fs, bf16 %.3fs, fp8 %.3fs, int8 %.3fs\n",
                       seconds[0], seconds[1], seconds[2], seconds[3]);
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
