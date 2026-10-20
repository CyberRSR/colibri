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
#include <stdint.h>
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
/* Pinned host memory (NULL when there is none): the device copies to and from it directly. */
MIMO_CUDA_API void *mimo_cuda_host_alloc(size_t bytes);
MIMO_CUDA_API void mimo_cuda_host_free(void *p);
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
/* A prompt block's attention from its normed rows to o_proj's output, the rows kept on the
 * device in between: qkv (resident) over xn [n][H], RoPE on the first rd dimensions of every
 * q and k head from rope [n][rd] (each row's cos[rd/2], then sin[rd/2]) and the values times
 * v_scale; rows kv_from..n-1 of the block's keys and values into kv ([kvh*hd keys][kvh*vd
 * values] a row, for the caller's caches); the attention over positions lo..p0-1 (kpast
 * [p0-lo][kvh][hd], vpast [p0-lo][kvh][vd], the host's) and the block's own, window 0 for full
 * attention, sink [nh] or NULL; o_proj (resident o, or the host's packed bf16 o_packed of
 * o_bytes) into out [n][H]. 0: not taken, nothing written; the caller computes it. */
typedef struct {
    MimoCudaTensor *qkv, *o;
    const void *o_packed;
    size_t o_bytes;
    const float *xn, *rope, *kpast, *vpast, *sink;
    float *kv, *out;
    int n, H, nh, kvh, hd, vd, rd, p0, lo, window, kv_from;
    float v_scale, scale;
    long long kv_tag;   /* nonzero: a full layer's sub-blocks within one block, one tag; the device keeps their
                         * keys and values for the next sub-block (the same tag at p0 = this one's end) */
} MimoCudaAttnBlock;
MIMO_CUDA_API int mimo_cuda_attn_block(const MimoCudaAttnBlock *a);
/* The same queued (submit), its kv and out written once wait returns 1 -- pinned buffers, so
 * the copies run while the caller works. One in flight: a submit waits for the one before. */
MIMO_CUDA_API int mimo_cuda_attn_submit(const MimoCudaAttnBlock *a);
MIMO_CUDA_API int mimo_cuda_attn_wait(void);
/* A matrix the device does not hold, for a prompt block: the host's packed bf16 (fmt 5's
 * planes, bytes long), sent up and kept while the next call names the same bytes. */
MIMO_CUDA_API int mimo_cuda_matmul_packed(const void *packed, size_t bytes, int I, int O, float *y,
                                          const float *x, int S);
/* The same for a plain bf16 host matrix w [O][I] (a router's), in a buffer of its own. */
MIMO_CUDA_API int mimo_cuda_matmul_bf16_host(const uint16_t *w, int I, int O, float *y, const float *x, int S);
/* A prompt block's routed experts: begin with its normed rows x [n][H], the number of
 * (row, choice) pairs and the most rows one expert has; it returns a pinned buffer of
 * npairs x H floats. For each expert, in any order: its bytes (down, down scales, gate,
 * gate scales, up, up scales; e2m1 [O][I/2] and e8m0 [O][I/32] each) copied into the
 * buffer staging() returns, then expert() with the rows that chose it, whose outputs go
 * to pairs pair0.. of that buffer; end() waits for them all. NULL or 0: nothing usable,
 * the CPU computes the step. */
/* The rows can go up first for the router as well (mimo_cuda_moe_rows; the router's logits
 * [n][E] of them from its bf16 [E][H] on the host with mimo_cuda_moe_router), and begin then
 * takes x NULL. */
MIMO_CUDA_API int mimo_cuda_moe_rows(const float *x, int n, int H);
MIMO_CUDA_API int mimo_cuda_moe_router(const uint16_t *w, int E, float *logits);
MIMO_CUDA_API float *mimo_cuda_moe_begin(const float *x, int n, int H, int MI, size_t ebytes, int npairs, int rmax);
MIMO_CUDA_API void *mimo_cuda_moe_staging(void);
MIMO_CUDA_API int mimo_cuda_moe_expert(const int *rows, int r, int pair0);
MIMO_CUDA_API int mimo_cuda_moe_end(void);
/* The staging buffers by index, for a caller that fills them in any order (a missing expert
 * read from disk straight into one): their count and size (4 KiB aligned, room for a direct
 * read's alignment) after begin; buffer i once the copy that read it last is done (wait 0:
 * NULL while it is not); the expert whose bytes start at offset off of buffer i. */
MIMO_CUDA_API int mimo_cuda_moe_stages(size_t *bytes);
MIMO_CUDA_API void *mimo_cuda_moe_stage(int i, int wait);
MIMO_CUDA_API int mimo_cuda_moe_expert_at(int i, size_t off, const int *rows, int r, int pair0);
/* The experts whose kernels are done: their outputs on their way down (a caller that waits for
 * something else calls it now and then; the waits here do). Returns the experts in flight. */
MIMO_CUDA_API int mimo_cuda_moe_poll(void);
MIMO_CUDA_API void mimo_cuda_report(void);
MIMO_CUDA_API void mimo_cuda_shutdown(void);
#ifdef __cplusplus
}
#endif
#endif
