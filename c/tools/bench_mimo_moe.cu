/* A 25k-token prompt block's MoE step at MiMo-V2.6-Pro's shapes through the DLL's own calls
 * (mimo_cuda_moe_begin / staging / expert / end): 384 experts of 521 rows each, every expert's
 * bytes copied into staging from a host copy as moe_gpu copies them from the cache. Prints the
 * step's time and its parts. Built with the backend itself:
 *   nvcc -O3 -std=c++17 -arch=sm_89 -ftz=false -Xcompiler=/openmp tools/bench_mimo_moe.cu -o tools/bench_mimo_moe.exe
 * usage: bench_mimo_moe [rows=25000] [experts=384] [steps=3] [rows per expert, a file] */
#include "../backend_cuda_mimo.cu"
#include <chrono>
#include <cstdio>
#include <vector>
#include <omp.h>

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(int argc, char **argv) {
    const int n = argc > 1 ? atoi(argv[1]) : 25000, E = argc > 2 ? atoi(argv[2]) : 384, steps = argc > 3 ? atoi(argv[3]) : 3;
    const int H = 6144, MI = 2048, K = 8;
    if (!mimo_cuda_init(0)) return 1;
    size_t eb = (size_t)H * (MI / 2) + (size_t)H * (MI / 32) + 2 * ((size_t)MI * (H / 2) + (size_t)MI * (H / 32));
    const int NB = 16;   /* host copies of experts, cycled */
    std::vector<std::vector<uint8_t>> host(NB, std::vector<uint8_t>(eb));
    uint32_t st = 7;
    for (auto &h : host) for (auto &b : h) { st = st * 1664525u + 1013904223u; b = (uint8_t)(st >> 24); }
    size_t ds = (size_t)H * (MI / 2), gs = ds + (size_t)H * (MI / 32) + (size_t)MI * (H / 2), us = gs + (size_t)MI * (H / 32) + (size_t)MI * (H / 2);
    for (auto &h : host) {   /* sane e8m0 scales */
        for (size_t i = ds; i < ds + (size_t)H * (MI / 32); i++) h[i] = (uint8_t)(118 + h[i] % 6);
        for (size_t i = gs; i < gs + (size_t)MI * (H / 32); i++) h[i] = (uint8_t)(118 + h[i] % 6);
        for (size_t i = us; i < us + (size_t)MI * (H / 32); i++) h[i] = (uint8_t)(118 + h[i] % 6);
    }
    std::vector<float> x((size_t)n * H);
    for (auto &v : x) { st = st * 1664525u + 1013904223u; v = (float)(st >> 8) / 16777216.0f - 0.5f; }
    /* each row's K experts: e = (t * 7 + q * 53) % E, so every expert gets about n K / E rows; or
     * each expert's rows counted from a real step (argv[4]: the counts, one an expert) */
    std::vector<std::vector<int>> rows(E);
    if (argc > 4) {
        FILE *f = fopen(argv[4], "r");
        for (int e = 0; e < E && f; e++) {
            int c = 0;
            if (fscanf(f, "%d", &c) != 1) break;
            for (int k = 0; k < c; k++) rows[e].push_back((int)(((long long)k * n) / (c > n ? n : c)) % n);
        }
        if (f) fclose(f);
    } else
        for (int t = 0; t < n; t++) for (int q = 0; q < K; q++) rows[(t * 7 + q * 53 + q * q) % E].push_back(t);
    int rmax = 0, npairs = 0;
    for (auto &r : rows) { rmax = r.size() > (size_t)rmax ? (int)r.size() : rmax; npairs += (int)r.size(); }
    printf("%d rows, %d experts, %d pairs, up to %d rows an expert, %.1f MB an expert\n", n, E, npairs, rmax, eb / 1e6);
    for (int s = 0; s < steps; s++) {
        double t0 = now(), tstage = 0, tcopy = 0, tenq = 0;
        float *y = mimo_cuda_moe_begin(x.data(), n, H, MI, eb, npairs, rmax);
        if (!y) { puts("begin failed"); return 1; }
        /* as moe_gpu_staged meets them: every staging buffer holds an expert already (read there
         * from disk), and HITS percent of the experts (the cache's) are copied in first */
        size_t sb = 0;
        int ns = mimo_cuda_moe_stages(&sb), hits = getenv("HITS") ? atoi(getenv("HITS")) : 27;
        for (int i = 0; i < ns && s == 0; i++) memcpy(mimo_cuda_moe_stage(i, 1), host[i % NB].data(), eb);
        int pair0 = 0, si = 0;
        for (int e = 0; e < E; e++) {
            if (rows[e].empty()) continue;
            double a = now();
            mimo_cuda_moe_poll();
            uint8_t *stage = (uint8_t *)mimo_cuda_moe_stage(si, 1);
            double b = now();
            if (e % 100 < hits) {
                const uint8_t *src = host[e % NB].data();
                const int64_t CH = 1 << 20, nc = (int64_t)((eb + CH - 1) / CH);
                #pragma omp parallel for schedule(static)
                for (int64_t i = 0; i < nc; i++) {
                    size_t off = (size_t)i * CH, len = eb - off < (size_t)CH ? eb - off : (size_t)CH;
                    memcpy(stage + off, src + off, len);
                }
            }
            double c = now();
            if (!mimo_cuda_moe_expert_at(si, 0, rows[e].data(), (int)rows[e].size(), pair0)) { puts("expert failed"); return 1; }
            si = (si + 1) % ns;
            pair0 += (int)rows[e].size();
            tstage += b - a; tcopy += c - b; tenq += now() - c;
        }
        double te = now();
        if (!mimo_cuda_moe_end()) { puts("end failed"); return 1; }
        double t1 = now();
        printf("step %d: %.3f s (%.2f ms an expert): waiting for staging %.3f s, copying %.3f s, queueing %.3f s, the last outputs %.3f s\n",
               s, t1 - t0, (t1 - t0) * 1e3 / E, tstage, tcopy, tenq, t1 - te);
    }
    mimo_cuda_report();
    return 0;
}
