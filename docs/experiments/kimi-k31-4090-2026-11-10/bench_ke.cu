// The routed experts' kernels at Kimi K3.1's widths (LT 3584, MI 3072): k3_gate_up + k3_down (the old ones)
// against k3_gu_t + k3_dn_t (the tiles) on the same random expert and rows: bytes compared, times.
#include "backend_cuda_k3.cu"   // nvcc -O3 -std=c++17 -arch=sm_89 -ftz=false -I <checkout>/c bench_ke.cu
#include <vector>
#include <random>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
int main(int argc, char **argv) {
    const int LT = 3584, MI = 3072, N = 4096;
    const float b1 = 4.0f, b2 = 25.0f;
    std::mt19937 rng(12345);
    auto rb = [&](size_t n, int lo, int hi) { std::vector<uint8_t> v(n); std::uniform_int_distribution<int> d(lo, hi); for (auto &x : v) x = (uint8_t)d(rng); return v; };
    const size_t n1p = (size_t)MI * LT / 2, n1s = (size_t)MI * LT / 32, n2p = (size_t)LT * MI / 2, n2s = (size_t)LT * MI / 32;
    auto h1p = rb(n1p, 0, 255), h3p = rb(n1p, 0, 255), h2p = rb(n2p, 0, 255);
    auto h1s = rb(n1s, 116, 130), h3s = rb(n1s, 116, 130), h2s = rb(n2s, 118, 132);
    std::vector<int8_t> hzq((size_t)N * LT); { std::uniform_int_distribution<int> d(-127, 127); for (auto &x : hzq) x = (int8_t)d(rng); }
    std::vector<float> hzs((size_t)N * (LT / 32)); { std::uniform_real_distribution<float> d(1e-3f, 3e-2f); for (auto &x : hzs) x = d(rng); }
    std::vector<int> hrows(N); for (int i = 0; i < N; i++) hrows[i] = (int)((i * 2654435761u) % N);
    uint8_t *w1p, *w1s, *w3p, *w3s, *w2p, *w2s; int8_t *zq, *gq0, *gq1; float *zs, *gs0, *gs1, *y0, *y1; int *rows;
    CK(cudaMalloc(&w1p, n1p)); CK(cudaMalloc(&w3p, n1p)); CK(cudaMalloc(&w2p, n2p));
    CK(cudaMalloc(&w1s, n1s)); CK(cudaMalloc(&w3s, n1s)); CK(cudaMalloc(&w2s, n2s));
    CK(cudaMemcpy(w1p, h1p.data(), n1p, cudaMemcpyHostToDevice)); CK(cudaMemcpy(w3p, h3p.data(), n1p, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(w2p, h2p.data(), n2p, cudaMemcpyHostToDevice)); CK(cudaMemcpy(w1s, h1s.data(), n1s, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(w3s, h3s.data(), n1s, cudaMemcpyHostToDevice)); CK(cudaMemcpy(w2s, h2s.data(), n2s, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&zq, hzq.size())); CK(cudaMemcpy(zq, hzq.data(), hzq.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&zs, hzs.size() * 4)); CK(cudaMemcpy(zs, hzs.data(), hzs.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&rows, N * 4)); CK(cudaMemcpy(rows, hrows.data(), N * 4, cudaMemcpyHostToDevice));
    const int RC = 1024;
    CK(cudaMalloc(&gq0, (size_t)RC * MI)); CK(cudaMalloc(&gq1, (size_t)RC * MI));
    CK(cudaMalloc(&gs0, (size_t)RC * MI / 32 * 4)); CK(cudaMalloc(&gs1, (size_t)RC * MI / 32 * 4));
    CK(cudaMalloc(&y0, (size_t)RC * LT * 4)); CK(cudaMalloc(&y1, (size_t)RC * LT * 4));
    CK(cudaMemcpyToSymbol(d_exp_in, k3x_EXP_in, sizeof k3x_EXP_in)); CK(cudaMemcpyToSymbol(d_exp_out, k3x_EXP_out, sizeof k3x_EXP_out));
    CK(cudaMemcpyToSymbol(d_tanh_in, k3x_TANH_in, sizeof k3x_TANH_in)); CK(cudaMemcpyToSymbol(d_tanh_out, k3x_TANH_out, sizeof k3x_TANH_out));
    CK(cudaFuncSetAttribute(k3_gu_t, cudaFuncAttributeMaxDynamicSharedMemorySize, KE_SMEM));
    CK(cudaFuncSetAttribute(k3_dn_t, cudaFuncAttributeMaxDynamicSharedMemorySize, KE_SMEM));
    const size_t s1 = smem_gate_up(LT), s2 = smem_down(MI);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    const int rs[] = {8, 20, 32, 64, 100, 200, 300, 600, 1000};
    printf("rows | old gate_up+down ms (TMAC/s) | tiles ms (TMAC/s) | gq, gs, y differ\n");
    for (int r : rs) {
        const double mac = 3.0 * LT * MI * r;
        auto run_old = [&]() {
            dim3 g1(MI / K3_TO, (r + K3_TP - 1) / K3_TP), g2(LT / K3_TO, (r + K3_TP - 1) / K3_TP);
            k3_gate_up<<<g1, 256, s1>>>(w1p, w1s, w3p, w3s, zq, zs, rows, r, LT, MI, b1, b2, gq0, gs0);
            k3_down<<<g2, 256, s2>>>(w2p, w2s, gq0, gs0, r, MI, LT, y0);
        };
        auto run_new = [&]() {
            dim3 g1(MI / 64, (r + KE_BR - 1) / KE_BR), g2(LT / 128, (r + KE_BR - 1) / KE_BR);
            k3_gu_t<<<g1, 256, KE_SMEM>>>(w1p, w1s, w3p, w3s, zq, zs, rows, r, LT, MI, b1, b2, gq1, gs1);
            k3_dn_t<<<g2, 256, KE_SMEM>>>(w2p, w2s, gq1, gs1, r, MI, LT, y1);
        };
        CK(cudaMemset(gq1, 0x55, (size_t)RC * MI)); CK(cudaMemset(y1, 0xff, (size_t)RC * LT * 4));
        run_old(); run_new(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<int8_t> q0((size_t)r * MI), q1((size_t)r * MI); std::vector<float> c0((size_t)r * MI / 32), c1(c0.size()), v0((size_t)r * LT), v1(v0.size());
        CK(cudaMemcpy(q0.data(), gq0, q0.size(), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(q1.data(), gq1, q1.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(c0.data(), gs0, c0.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(c1.data(), gs1, c1.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(v0.data(), y0, v0.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(v1.data(), y1, v1.size() * 4, cudaMemcpyDeviceToHost));
        size_t dq = 0, dc = 0, dy = 0;
        for (size_t i = 0; i < q0.size(); i++) dq += q0[i] != q1[i];
        for (size_t i = 0; i < c0.size(); i++) dc += memcmp(&c0[i], &c1[i], 4) != 0;
        for (size_t i = 0; i < v0.size(); i++) dy += memcmp(&v0[i], &v1[i], 4) != 0;
        const int reps = r < 100 ? 200 : 40;
        float t0, t1;
        cudaEventRecord(a); for (int k = 0; k < reps; k++) run_old(); cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&t0, a, b);
        cudaEventRecord(a); for (int k = 0; k < reps; k++) run_new(); cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&t1, a, b);
        t0 /= reps; t1 /= reps;
        printf("%5d | %8.3f (%5.1f) | %8.3f (%5.1f) | %zu %zu %zu\n", r, t0, mac / t0 / 1e9, t1, mac / t1 / 1e9, dq, dc, dy);
    }
    return 0;
}
