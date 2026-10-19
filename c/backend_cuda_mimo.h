/* MiMo resident dense matrices. The format numbers are MiMo DW_* values,
 * NOT the generic backend's QT formats, plus 4: int8 with a float scale per
 * (row, 128-column block), MIMO_CUDA_OPROJ_INT8's device form, which no host
 * matrix has, and 5: a bf16 matrix (given as for 1, rows a multiple of 256)
 * packed without loss on the device, 12 bits a weight, computing the bytes 1
 * computes. All activations stay float32.
 * Single host thread; opaque handles never cross an allocator boundary.
 * Every operation returns 1 on success, 0 on failure; matmul leaves y alone
 * on failure so the caller can recompute with its canonical host weights. */
#ifndef COLIBRI_BACKEND_CUDA_MIMO_H
#define COLIBRI_BACKEND_CUDA_MIMO_H
#include <stddef.h>
#if defined(_WIN32) && defined(MIMO_CUDA_BUILDING_DLL)
#define MIMO_CUDA_API __declspec(dllexport)
#else
#define MIMO_CUDA_API
#endif
#ifdef __cplusplus
extern "C" {
#endif
typedef struct MimoCudaTensor MimoCudaTensor;
MIMO_CUDA_API int mimo_cuda_init(int device);
MIMO_CUDA_API size_t mimo_cuda_free_bytes(void);
MIMO_CUDA_API MimoCudaTensor *mimo_cuda_upload(const void *w, const float *scales,
                                             int fmt, int I, int O);
MIMO_CUDA_API int mimo_cuda_matmul(MimoCudaTensor *t, float *y, const float *x, int S);
/* The device bytes a resident matrix holds (0 for none): fmt 5's depend on its weights. */
MIMO_CUDA_API size_t mimo_cuda_bytes(const MimoCudaTensor *t);
MIMO_CUDA_API void mimo_cuda_report(void);
MIMO_CUDA_API void mimo_cuda_shutdown(void);
#ifdef __cplusplus
}
#endif
#endif
