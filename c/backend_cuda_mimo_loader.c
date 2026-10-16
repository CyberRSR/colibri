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
void mimo_cuda_report(void) { if (module) p_report(); }
void mimo_cuda_shutdown(void) { if (module) { p_shutdown(); FreeLibrary(module); module = NULL; } }
