// Does a copy that waits on an event (cudaStreamWaitEvent on its stream) hold up copies other streams
// queue after it? A one-block kernel spins ~100 ms on stream K; stream C1 waits for it and then copies;
// stream C2 then queues its own copy, which has nothing to wait for. If C2's copy finishes in ~2 ms the
// streams are independent; ~100 ms means the copies run in submission order across streams (WDDM).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
__global__ void spin(long long cycles) {
    long long t0 = clock64();
    while (clock64() - t0 < cycles) { }
}
__global__ void tiny(float *p) { if (p) p[threadIdx.x] += 1.0f; }
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e_)); exit(1); } } while (0)
int main() {
    const size_t n = 50u << 20;
    char *h1, *h2; CK(cudaHostAlloc((void **)&h1, n, cudaHostAllocDefault)); CK(cudaHostAlloc((void **)&h2, n, cudaHostAllocDefault));
    char *pg = (char *)malloc(n); memset(pg, 1, n); memset(h1, 1, n); memset(h2, 1, n);
    void *d1, *d2; CK(cudaMalloc(&d1, n)); CK(cudaMalloc(&d2, n));
    float *dt; CK(cudaMalloc(&dt, 1024 * 4));
    cudaStream_t K, C1, C2; cudaEvent_t eK, e2;
    CK(cudaStreamCreateWithFlags(&K, cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&C1, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&C2, cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&eK, cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&e2, cudaEventDisableTiming));
    int clk; CK(cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0));
    const long long cyc = (long long)clk * 100;   // ~100 ms
    struct T { const char *what; int c1dir; int c2dir; int c2pageable; int c2kernel; } tests[] = {
        {"C1 waits, then D2H; C2 H2D (pinned)", 2, 1, 0, 0},
        {"C1 waits, then H2D; C2 H2D (pinned)", 1, 1, 0, 0},
        {"C1 waits, then D2H; C2 D2H (pinned)", 2, 2, 0, 0},
        {"C1 waits, then H2D; C2 D2H (pinned)", 1, 2, 0, 0},
        {"C1 waits, then H2D; C2 H2D (pageable)", 1, 1, 1, 0},
        {"C1 waits, then D2H; C2 a tiny kernel", 2, 0, 0, 1},
        {"C1 waits, then a tiny kernel; C2 a tiny kernel", 3, 0, 0, 1},
        {"nothing waits (baseline): C2 H2D (pinned)", 0, 1, 0, 0},
    };
    for (int rep = 0; rep < 2; rep++)
    for (auto &t : tests) {
        CK(cudaDeviceSynchronize());
        spin<<<1, 1, 0, K>>>(cyc);
        CK(cudaEventRecord(eK, K));
        if (t.c1dir) {
            CK(cudaStreamWaitEvent(C1, eK, 0));
            if (t.c1dir == 1) CK(cudaMemcpyAsync(d1, h1, n, cudaMemcpyHostToDevice, C1));
            else if (t.c1dir == 2) CK(cudaMemcpyAsync(h1, d1, n, cudaMemcpyDeviceToHost, C1));
            else tiny<<<1, 32, 0, C1>>>(dt);
        }
        const double t0 = now();
        if (t.c2kernel) tiny<<<1, 32, 0, C2>>>(dt + 64);
        else if (t.c2dir == 1) CK(cudaMemcpyAsync(d2, t.c2pageable ? pg : h2, n, cudaMemcpyHostToDevice, C2));
        else CK(cudaMemcpyAsync(h2, d2, n, cudaMemcpyDeviceToHost, C2));
        CK(cudaEventRecord(e2, C2));
        CK(cudaEventSynchronize(e2));
        const double t2 = now() - t0;
        CK(cudaStreamSynchronize(K)); const double tk = now() - t0;
        printf("%-48s C2 done after %6.1f ms (the kernel after %6.1f ms)\n", t.what, t2 * 1e3, tk * 1e3);
    }
    return 0;
}
