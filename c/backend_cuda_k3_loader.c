/* coli_cuda_k3.dll beside the executable (Windows: a MinGW host and an MSVC-built CUDA DLL).
 * A missing or incomplete DLL leaves every call failing, and the engine on the CPU. */
#include "backend_cuda_k3.h"
#include <windows.h>
#include <stdio.h>
#include <string.h>
#include <wchar.h>

static HMODULE module;
static int tried;
static int (*p_init)(int);
static size_t (*p_free_bytes)(void);
static void *(*p_host_alloc)(size_t);
static void (*p_host_free)(void *);
static int (*p_exact_eval)(int, uint32_t, uint32_t, uint32_t *);
static int (*p_moe_begin)(const int8_t *, const float *, int, int, int, float, float, size_t, const size_t *, float *, int);
static int (*p_moe_expert)(const uint8_t *, const int *, int, int, int);
static int (*p_moe_poll)(int *, int);
static int (*p_moe_wait)(void);
static int (*p_moe_end)(void);
static int (*p_matmul)(int, float *, const float *, const void *, const float *, int, int, int, int);
static int (*p_mla)(const float *, const float *, int, int, int, const float *, const float *, const int8_t *,
                    const float *, int, int, int, int, int, float, float *);
static int (*p_router)(float *, const float *, const uint16_t *, int, int, int);
static int (*p_shared)(float *, const float *, int, int, int, const uint8_t *, const float *, const uint8_t *,
                       const float *, int, const uint8_t *, const float *, int, float, float);
static int (*p_kda)(const K3cuKda *, const float *, float *);   /* optional: an older DLL lacks it */
static int (*p_mla_layer)(const K3cuMlaLayer *, const float *, float *);   /* optional too */
static int (*p_vision)(const K3cuVision *, const uint8_t *, int, int, float *);   /* optional: pictures (k3_vision.h) */
static int (*p_matmul_dev)(int, float *, const float *, const void *, const float *, int, int, int, int);   /* and this */
static int (*p_mla_mask)(const float *, const float *, int, int, int, const float *, const float *, const int8_t *,
                         const float *, int, int, int, int, int, float, const uint32_t *, int, float *);   /* K3.1: and this */
static int (*p_dsa_mask)(const float *, int, int, int, const float *, const float *, int, int, int, uint32_t *, int);   /* and this */
static int (*p_mla_sel)(const float *, const float *, int, int, int, const float *, const float *, const int8_t *,
                        const float *, int, int, int, int, int, float, const uint32_t *, int, int, float *);   /* and this */
static int (*p_host_register)(void *, size_t);   /* optional: registered rows beside the kernels */
static void (*p_host_unregister)(void *);
static void (*p_report)(void), (*p_shutdown)(void);

int k3cu_init(int device) {
    if (module) return p_init(device);
    if (tried) return 0;
    tried = 1;
    wchar_t path[32768];
    DWORD n = GetModuleFileNameW(NULL, path, 32768);
    if (!n || n >= 32768) return 0;
    wchar_t *slash = wcsrchr(path, L'\\');
    if (!slash || (size_t)(slash - path) + 32 >= 32768) return 0;
    wcscpy(slash + 1, L"coli_cuda_k3.dll");
    HMODULE h = LoadLibraryExW(path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!h) {   /* tests/: the DLL in c/, one up */
        wcscpy(slash + 1, L"..\\coli_cuda_k3.dll");
        h = LoadLibraryExW(path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    }
    if (!h) { fprintf(stderr, "[K3-CUDA] no coli_cuda_k3.dll beside the engine (error %lu): a prompt's experts stay on the CPU\n", GetLastError()); return 0; }
#define BIND(name) do { FARPROC fn = GetProcAddress(h, "k3cu_" #name); \
    if (!fn) { fprintf(stderr, "[K3-CUDA] coli_cuda_k3.dll has no k3cu_" #name ": a prompt's experts stay on the CPU\n"); FreeLibrary(h); return 0; } \
    memcpy(&p_##name, &fn, sizeof fn); } while (0)
    BIND(init); BIND(free_bytes); BIND(host_alloc); BIND(host_free); BIND(exact_eval);
    BIND(moe_begin); BIND(moe_expert); BIND(moe_poll); BIND(moe_wait); BIND(moe_end); BIND(matmul); BIND(mla); BIND(router); BIND(shared); BIND(report); BIND(shutdown);
#undef BIND
    { FARPROC fn = GetProcAddress(h, "k3cu_kda"); if (fn) memcpy(&p_kda, &fn, sizeof fn); else fprintf(stderr, "[K3-CUDA] coli_cuda_k3.dll has no k3cu_kda: KDA on the CPU\n"); }
    { FARPROC fn = GetProcAddress(h, "k3cu_vision"); if (fn) memcpy(&p_vision, &fn, sizeof fn); else fprintf(stderr, "[K3-CUDA] coli_cuda_k3.dll has no k3cu_vision: pictures through the tower on the CPU\n"); }
    { FARPROC fn = GetProcAddress(h, "k3cu_mla_layer"); if (fn) memcpy(&p_mla_layer, &fn, sizeof fn); else fprintf(stderr, "[K3-CUDA] coli_cuda_k3.dll has no k3cu_mla_layer: the gated MLA call by call\n"); }
    { FARPROC fn = GetProcAddress(h, "k3cu_matmul_dev"); if (fn) memcpy(&p_matmul_dev, &fn, sizeof fn); else fprintf(stderr, "[K3-CUDA] coli_cuda_k3.dll has no k3cu_matmul_dev: coli_cuda.dll's kernels\n"); }
    { FARPROC fn = GetProcAddress(h, "k3cu_mla_mask"); if (fn) memcpy(&p_mla_mask, &fn, sizeof fn); else fprintf(stderr, "[K3-CUDA] coli_cuda_k3.dll has no k3cu_mla_mask: K3.1's sparse rows on the CPU\n"); }
    { FARPROC fn = GetProcAddress(h, "k3cu_mla_sel"); if (fn) memcpy(&p_mla_sel, &fn, sizeof fn); else fprintf(stderr, "[K3-CUDA] coli_cuda_k3.dll has no k3cu_mla_sel: K3.1's selecting rows against every position (k3cu_mla_mask), slower as the context grows\n"); }
    { FARPROC fn = GetProcAddress(h, "k3cu_dsa_mask"); if (fn) memcpy(&p_dsa_mask, &fn, sizeof fn); else fprintf(stderr, "[K3-CUDA] coli_cuda_k3.dll has no k3cu_dsa_mask: K3.1's picks on the CPU, a long prompt waits there\n"); }
    { FARPROC fn = GetProcAddress(h, "k3cu_host_register"), fu = GetProcAddress(h, "k3cu_host_unregister");
      if (fn && fu) { memcpy(&p_host_register, &fn, sizeof fn); memcpy(&p_host_unregister, &fu, sizeof fu); } }
    module = h;
    return p_init(device);
}
size_t k3cu_free_bytes(void) { return module ? p_free_bytes() : 0; }
void *k3cu_host_alloc(size_t bytes) { return module ? p_host_alloc(bytes) : NULL; }
void k3cu_host_free(void *p) { if (module && p) p_host_free(p); }
int k3cu_host_register(void *p, size_t bytes) { return module && p_host_register ? p_host_register(p, bytes) : 0; }
void k3cu_host_unregister(void *p) { if (module && p_host_unregister && p) p_host_unregister(p); }
int k3cu_exact_eval(int kind, uint32_t first, uint32_t count, uint32_t *out) { return module ? p_exact_eval(kind, first, count, out) : 0; }
int k3cu_moe_begin(const int8_t *zq, const float *zs, int n, int LT, int MI, float b1, float b2,
                   size_t ebytes, const size_t *off, float *Y, int npairs) {
    return module ? p_moe_begin(zq, zs, n, LT, MI, b1, b2, ebytes, off, Y, npairs) : 0;
}
int k3cu_moe_expert(const uint8_t *bytes, const int *rows, int r, int pair0, int tag) {
    return module ? p_moe_expert(bytes, rows, r, pair0, tag) : 0;
}
int k3cu_moe_poll(int *tags, int cap) { return module ? p_moe_poll(tags, cap) : -1; }
int k3cu_moe_wait(void) { return module ? p_moe_wait() : -1; }
int k3cu_moe_end(void) { return module ? p_moe_end() : 0; }
int k3cu_matmul(int fmt, float *y, const float *x, const void *w, const float *scale, int S, int I, int O, int gs) {
    return module ? p_matmul(fmt, y, x, w, scale, S, I, O, gs) : 0;
}
int k3cu_matmul_dev(int fmt, float *y, const float *x, const void *w, const float *scale, int S, int I, int O, int gs) {
    return module && p_matmul_dev ? p_matmul_dev(fmt, y, x, w, scale, S, I, O, gs) : 0;
}
int k3cu_mla(const float *Lc, const float *Rc, int N, int pos0, int C, const float *qv, const float *gv,
             const int8_t *kvb, const float *kvbs, int H, int qn, int qr, int vh, int kvl, float scale, float *ctx) {
    return module ? p_mla(Lc, Rc, N, pos0, C, qv, gv, kvb, kvbs, H, qn, qr, vh, kvl, scale, ctx) : 0;
}
int k3cu_mla_mask(const float *Lc, const float *Rc, int N, int pos0, int C, const float *qv, const float *gv,
                  const int8_t *kvb, const float *kvbs, int H, int qn, int qr, int vh, int kvl, float scale,
                  const uint32_t *mask, int mw, float *ctx) {
    return module && p_mla_mask ? p_mla_mask(Lc, Rc, N, pos0, C, qv, gv, kvb, kvbs, H, qn, qr, vh, kvl, scale, mask, mw, ctx) : 0;
}
int k3cu_mla_sel(const float *Lc, const float *Rc, int N, int pos0, int C, const float *qv, const float *gv,
                 const int8_t *kvb, const float *kvbs, int H, int qn, int qr, int vh, int kvl, float scale,
                 const uint32_t *mask, int mw, int K, float *ctx) {
    if (module && p_mla_sel) return p_mla_sel(Lc, Rc, N, pos0, C, qv, gv, kvb, kvbs, H, qn, qr, vh, kvl, scale, mask, mw, K, ctx);
    return k3cu_mla_mask(Lc, Rc, N, pos0, C, qv, gv, kvb, kvbs, H, qn, qr, vh, kvl, scale, mask, mw, ctx);
}
int k3cu_dsa_mask(const float *Ic, int N, int pos0, int C, const float *q, const float *w, int nh, int hd, int K,
                  uint32_t *mask, int mw) {
    return module && p_dsa_mask ? p_dsa_mask(Ic, N, pos0, C, q, w, nh, hd, K, mask, mw) : 0;
}
int k3cu_router(float *sco, const float *x, const uint16_t *w, int C, int D, int E) {
    return module ? p_router(sco, x, w, C, D, E) : 0;
}
int k3cu_shared(float *sd, const float *x, int S, int D, int I, const uint8_t *gq, const float *gs,
                const uint8_t *uq, const float *us, int ggs, const uint8_t *dq, const float *ds, int dgs, float b1, float b2) {
    return module ? p_shared(sd, x, S, D, I, gq, gs, uq, us, ggs, dq, ds, dgs, b1, b2) : 0;
}
int k3cu_kda(const K3cuKda *a, const float *x, float *out) { return module && p_kda ? p_kda(a, x, out) : 0; }
int k3cu_mla_layer(const K3cuMlaLayer *a, const float *x, float *out) { return module && p_mla_layer ? p_mla_layer(a, x, out) : 0; }
int k3cu_vision(const K3cuVision *v, const uint8_t *rgb, int gh, int gw, float *rows) { return module && p_vision ? p_vision(v, rgb, gh, gw, rows) : 0; }
void k3cu_report(void) { if (module) p_report(); }
void k3cu_shutdown(void) { if (module) p_shutdown(); }
