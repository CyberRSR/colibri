// Kimi K3.1's GPU indexer kernels (backend_cuda_k3_dsa.cuh) run on the CPU: the .cuh compiled as
// host C++, a std::thread a CUDA thread, a std::barrier for __syncthreads, a block at a time (its
// __shared__ arrays then static). Checked against the indexer's arithmetic spelled out -- for each
// score one chain of fmaf over d, then over the heads, as kimi_k3.c's k3_dsa_scores computes it
// (tests/test_k3_dsa checks that chain against the engine's AVX2 path, bit for bit) -- and the
// pick against a stable sort: the K highest, ties to the lower position. Random rows, rows of
// ties (zero weights: every score +0), rows of a few levels, rows just past K, and NaN scores.
//   g++ -O2 -std=c++20 -pthread tools/check_k3_dsa_kernels.cpp -o check_k3_dsa_kernels && ./check_k3_dsa_kernels
#include <algorithm>
#include <atomic>
#include <barrier>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <thread>
#include <vector>

struct Dim3 { unsigned x = 1, y = 1, z = 1; };
static thread_local Dim3 threadIdx;
static Dim3 blockIdx, blockDim;
static std::barrier<> *g_bar;
#define __global__
#define __device__
#define __forceinline__ inline
#define __launch_bounds__(...)
#define __restrict__ __restrict
#define __shared__ static
static inline void __syncthreads() { g_bar->arrive_and_wait(); }
static inline float __fmaf_rn(float a, float b, float c) { return std::fmaf(a, b, c); }
static inline uint32_t __float_as_uint(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static inline unsigned atomicAdd(unsigned *p, unsigned v) { return __atomic_fetch_add(p, v, __ATOMIC_RELAXED); }
using std::min;
#include "../backend_cuda_k3_dsa.cuh"

// a kernel over a grid, a block at a time, its 256 threads at once
static void launch(unsigned gx, unsigned gy, const std::function<void()> &kernel) {
    blockDim.x = 256;
    for (unsigned by = 0; by < gy; by++)
        for (unsigned bx = 0; bx < gx; bx++) {
            blockIdx.x = bx; blockIdx.y = by;
            std::barrier<> bar(256);
            g_bar = &bar;
            std::vector<std::thread> team;
            for (unsigned t = 0; t < 256; t++)
                team.emplace_back([&, t] { threadIdx.x = t; kernel(); });
            for (auto &th : team) th.join();
        }
}

static uint32_t rng = 20261105u;
static float rnd() { rng = rng * 1664525u + 1013904223u; return (float)(rng >> 9) / 4194304.0f - 1.0f; }

static int run_case(const char *name, int nh, int hd, int K, int pos0, int C, int mode) {
    const int N = pos0 + C, mw = (N + 31) / 32, r0 = K > pos0 ? K - pos0 : 0, R = C - r0;
    if (R <= 0) return 0;
    std::vector<float> Kc((size_t)N * hd), q((size_t)R * nh * hd), w((size_t)R * nh);
    for (auto &v : Kc) v = mode == 2 ? (float)(int)(rnd() * 3) : rnd();
    for (auto &v : q) v = mode == 2 ? (float)(int)(rnd() * 3) : rnd();
    for (auto &v : w) v = mode == 1 ? 0.0f : rnd();
    if (mode == 3) for (int i = 0; i < hd; i++) Kc[(size_t)5 * hd + i] = NAN;   // position 5's scores NaN
    // the kernels
    const long long ldS = ((long long)N + 3) / 4 * 4;
    std::vector<float> S((size_t)R * ldS, 12345.0f);
    std::vector<uint32_t> M((size_t)R * mw, 0xdeadbeefu);
    const int first = pos0 + r0;
    launch((unsigned)((N + DS_T - 1) / DS_T), (unsigned)((R + DS_R - 1) / DS_R),
           [&] { k3_dsa_scores(Kc.data(), q.data(), w.data(), R, first, nh, hd, S.data(), (int)ldS); });
    launch((unsigned)R, 1, [&] { k3_dsa_pick(S.data(), (int)ldS, first, K, M.data(), mw); });
    // the reference
    int bad = 0;
    for (int r = 0; r < R; r++) {
        const int nt = first + r + 1;
        std::vector<float> sc(nt);
        for (int t = 0; t < nt; t++) {
            float s = 0.0f;
            for (int h = 0; h < nh; h++) {
                float a = 0.0f;
                for (int d = 0; d < hd; d++) a = std::fmaf(q[((size_t)r * nh + h) * hd + d], Kc[(size_t)t * hd + d], a);
                s = std::fmaf(w[(size_t)r * nh + h], a > 0 ? a : 0.0f, s);
            }
            if (s != s) s = -INFINITY; else if (s == 0) s = 0.0f;
            sc[t] = s;
            if (std::memcmp(&s, &S[(size_t)r * ldS + t], 4)) { if (bad++ < 5) printf("  %s: row %d position %d: score %a, kernel %a\n", name, r, t, s, S[(size_t)r * ldS + t]); }
        }
        std::vector<int> order(nt);
        for (int t = 0; t < nt; t++) order[t] = t;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return sc[a] > sc[b]; });
        std::vector<uint32_t> want(mw, 0);
        for (int i = 0; i < K && i < nt; i++) want[order[i] >> 5] |= 1u << (order[i] & 31);
        if (std::memcmp(want.data(), &M[(size_t)r * mw], (size_t)mw * 4)) {
            int n = 0; for (int i = 0; i < mw; i++) n += __builtin_popcount(M[(size_t)r * mw + i]);
            if (bad++ < 5) printf("  %s: row %d (nt %d): the mask differs (%d bits set, want %d)\n", name, r, nt, n, K);
        }
    }
    printf("%s: %d heads of %d, K %d, rows %d..%d (%d picking): %s\n", name, nh, hd, K, pos0, N - 1, R, bad ? "FAILED" : "ok");
    return bad != 0;
}

int main() {
    int bad = 0;
    bad += run_case("tiny", 4, 16, 16, 10, 60, 0);            // the K3.1 tiny fixture's widths
    bad += run_case("tiny, zero weights", 4, 16, 16, 0, 90, 1);   // every score +0: ties to the lower position
    bad += run_case("few levels", 3, 8, 20, 30, 70, 2);       // many ties among nonzero scores
    bad += run_case("NaN keys", 4, 16, 16, 20, 40, 3);         // a NaN position: -inf, never picked over a number
    bad += run_case("just past K", 4, 16, 64, 63, 3, 0);       // rows of 64 and 65 positions, a mask of 3 words
    bad += run_case("real widths", 32, 128, 256, 300, 70, 0);  // 32 heads of 128 (K smaller to keep this quick)
    printf(bad ? "FAILED\n" : "ok\n");
    return bad ? 1 : 0;
}
