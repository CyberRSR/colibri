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
/* A prompt block's attention: q, row t at q + t * qstride, [nh][hd] with RoPE applied; the
 * keys and values of positions lo..lo+nkeys-1 = p0+n-1, [nkeys][kvh][hd] and
 * [nkeys][kvh][vd]; window 0 for full attention; sink [nh] or NULL. out [n][nh][vd].
 * f32, an online softmax: the CPU's numbers summed in another order. hd/vd 192/128 or
 * 48/32 (the tiny fixture's); 0 for anything else, which the CPU then computes. */
MIMO_CUDA_API int mimo_cuda_attention(const float *q, int qstride, int n, int nh, int kvh, int hd, int vd,
                                      const float *k, const float *v, int lo, int nkeys, int p0, int window,
                                      const float *sink, float scale, float *out);
/* A matrix the device does not hold, for a prompt block: the host's packed bf16 (fmt 5's
 * planes, bytes long), sent up and kept while the next call names the same bytes. */
MIMO_CUDA_API int mimo_cuda_matmul_packed(const void *packed, size_t bytes, int I, int O, float *y,
                                          const float *x, int S);
/* A prompt block's routed experts: begin with its normed rows x [n][H], the number of
 * (row, choice) pairs and the most rows one expert has; it returns a pinned buffer of
 * npairs x H floats. For each expert, in any order: its bytes (down, down scales, gate,
 * gate scales, up, up scales; e2m1 [O][I/2] and e8m0 [O][I/32] each) copied into the
 * buffer staging() returns, then expert() with the rows that chose it, whose outputs go
 * to pairs pair0.. of that buffer; end() waits for them all. NULL or 0: nothing usable,
 * the CPU computes the step. */
MIMO_CUDA_API float *mimo_cuda_moe_begin(const float *x, int n, int H, int MI, size_t ebytes, int npairs, int rmax);
MIMO_CUDA_API void *mimo_cuda_moe_staging(void);
MIMO_CUDA_API int mimo_cuda_moe_expert(const int *rows, int r, int pair0);
MIMO_CUDA_API int mimo_cuda_moe_end(void);
MIMO_CUDA_API void mimo_cuda_report(void);
MIMO_CUDA_API void mimo_cuda_shutdown(void);
#ifdef __cplusplus
}
#endif
#endif
