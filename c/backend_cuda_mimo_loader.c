/* Windows MinGW host / MSVC CUDA DLL boundary. Only load next to the EXE;
 * missing or incomplete DLL is a clean CPU fallback. */
#include "backend_cuda_mimo.h"
#include <windows.h>
#include <stdio.h>
#include <wchar.h>
#include <string.h>
static HMODULE module;
static int (*p_init)(int);
static size_t (*p_free_bytes)(void);
static MimoCudaTensor *(*p_upload)(const void *, const float *, int, int, int);
static int (*p_matmul)(MimoCudaTensor *, float *, const float *, int);
static void (*p_report)(void), (*p_shutdown)(void);
static size_t (*p_bytes)(const MimoCudaTensor *);   /* optional: an older DLL has none */
static int (*p_attention)(const float *, int, int, int, int, int, int, const float *, const float *,
                          int, int, int, int, const float *, float, float *);   /* optional too */
static int (*p_matmul_packed)(const void *, size_t, int, int, float *, const float *, int);
static float *(*p_moe_begin)(const float *, int, int, int, size_t, int, int);   /* the four: all or none */
static void *(*p_moe_staging)(void);
static int (*p_moe_expert)(const int *, int, int);
static int (*p_moe_end)(void);
int mimo_cuda_init(int device) {
    if (module) return p_init(device);
    wchar_t path[32768];
    DWORD n = GetModuleFileNameW(NULL, path, 32768);
    if (!n || n >= 32768) return 0;
    wchar_t *slash = wcsrchr(path, L'\\');
    if (!slash || (size_t)(slash - path) + 32 >= 32768) return 0;
    wcscpy(slash + 1, L"coli_cuda_mimo.dll");
    module = LoadLibraryExW(path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!module) { fprintf(stderr, "[CUDA] mimo: cannot load coli_cuda_mimo.dll beside the executable (error %lu)\n", GetLastError()); return 0; }
#define BIND(name) do { FARPROC fn = GetProcAddress(module, "mimo_cuda_" #name); \
    if (!fn) goto missing; memcpy(&p_##name, &fn, sizeof fn); } while (0)
    BIND(init); BIND(free_bytes); BIND(upload); BIND(matmul); BIND(report); BIND(shutdown);
#undef BIND
    { FARPROC fn = GetProcAddress(module, "mimo_cuda_bytes"); if (fn) memcpy(&p_bytes, &fn, sizeof fn); }
    { FARPROC fn = GetProcAddress(module, "mimo_cuda_attention"); if (fn) memcpy(&p_attention, &fn, sizeof fn); }
    { FARPROC fn = GetProcAddress(module, "mimo_cuda_matmul_packed"); if (fn) memcpy(&p_matmul_packed, &fn, sizeof fn); }
    {
        FARPROC b = GetProcAddress(module, "mimo_cuda_moe_begin"), s = GetProcAddress(module, "mimo_cuda_moe_staging"),
                e = GetProcAddress(module, "mimo_cuda_moe_expert"), d = GetProcAddress(module, "mimo_cuda_moe_end");
        if (b && s && e && d) {
            memcpy(&p_moe_begin, &b, sizeof b); memcpy(&p_moe_staging, &s, sizeof s);
            memcpy(&p_moe_expert, &e, sizeof e); memcpy(&p_moe_end, &d, sizeof d);
        }
    }
    return p_init(device);
missing:
    fprintf(stderr, "[CUDA] mimo: incomplete DLL; CPU fallback\n");
    FreeLibrary(module); module = NULL;
    return 0;
}
size_t mimo_cuda_free_bytes(void) { return module ? p_free_bytes() : 0; }
MimoCudaTensor *mimo_cuda_upload(const void *w, const float *s, int fmt, int I, int O) {
    return module ? p_upload(w, s, fmt, I, O) : NULL;
}
int mimo_cuda_matmul(MimoCudaTensor *t, float *y, const float *x, int S) {
    return module ? p_matmul(t, y, x, S) : 0;
}
size_t mimo_cuda_bytes(const MimoCudaTensor *t) { return module && p_bytes ? p_bytes(t) : 0; }
int mimo_cuda_attention(const float *q, int qstride, int n, int nh, int kvh, int hd, int vd,
                        const float *k, const float *v, int lo, int nkeys, int p0, int window,
                        const float *sink, float scale, float *out) {
    return module && p_attention ? p_attention(q, qstride, n, nh, kvh, hd, vd, k, v, lo, nkeys, p0, window, sink, scale, out) : 0;
}
int mimo_cuda_matmul_packed(const void *packed, size_t bytes, int I, int O, float *y, const float *x, int S) {
    return module && p_matmul_packed ? p_matmul_packed(packed, bytes, I, O, y, x, S) : 0;
}
float *mimo_cuda_moe_begin(const float *x, int n, int H, int MI, size_t ebytes, int npairs, int rmax) {
    return module && p_moe_begin ? p_moe_begin(x, n, H, MI, ebytes, npairs, rmax) : NULL;
}
void *mimo_cuda_moe_staging(void) { return module && p_moe_staging ? p_moe_staging() : NULL; }
int mimo_cuda_moe_expert(const int *rows, int r, int pair0) {
    return module && p_moe_expert ? p_moe_expert(rows, r, pair0) : 0;
}
int mimo_cuda_moe_end(void) { return module && p_moe_end ? p_moe_end() : 0; }
void mimo_cuda_report(void) { if (module) p_report(); }
void mimo_cuda_shutdown(void) { if (module) { p_shutdown(); FreeLibrary(module); module = NULL; } }
