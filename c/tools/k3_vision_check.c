/* k3_vision_check.c -- Kimi K3.1's vision tower (k3_vision.h) alone, for a picture the gateway
 * would send: its rows, and the steps' rows for tools/k3_vision_ref.py to hold against Moonshot's
 * own modeling code.
 *
 *   k3_vision_check <model_dir> <pixels.u8> <grid_h> <grid_w> <out_dir> [gpu]
 *
 * pixels.u8: the processor's resized and padded picture, RGB uint8 [grid_h*14][grid_w*14][3].
 * Writes pe.f32, b0.f32, enc.f32 and rows.f32 into out_dir (K3_VISION_DUMP); with `gpu` (a build
 * with -DCOLI_K3_CUDA and backend_cuda_k3_loader.c, coli_cuda_k3.dll beside it) the card's rows
 * too, rows_gpu.f32. Only the shards' headers are read, and the tower's two shards' bytes. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../st.h"
#include "../exp_f.h"
#include "../matmul_f32.h"
#include "../k3_vision.h"
#ifdef COLI_K3_CUDA
#include "../backend_cuda_k3.h"
#endif

int main(int argc, char **argv) {
    if (argc != 6 && argc != 7) {
        fprintf(stderr, "usage: %s <model_dir> <pixels.u8> <grid_h> <grid_w> <out_dir> [gpu]\n", argv[0]);
        return 2;
    }
    const int gh = atoi(argv[3]), gw = atoi(argv[4]);
    shards S;
    st_init(&S, argv[1]);
    int pad = -1;
    K3Vision *v = k3v_load(&S, argv[1], &pad);
    if (!v) return 1;
    const size_t n = (size_t)gh * v->patch * gw * v->patch * 3;
    uint8_t *rgb = malloc(n);
    FILE *f = fopen(argv[2], "rb");
    if (!rgb || !f || fread(rgb, 1, n, f) != n) {
        fprintf(stderr, "%s does not hold a %dx%d grid's pixels\n", argv[2], gh, gw);
        return 1;
    }
    fclose(f);
    char path[4200];
    if (argc == 7 && !strcmp(argv[6], "gpu")) {
#ifdef COLI_K3_CUDA
        if (!k3cu_init(0)) { fprintf(stderr, "coli_cuda_k3.dll did not start\n"); return 1; }
        const float **n0 = malloc(v->depth * sizeof *n0), **n1 = malloc(v->depth * sizeof *n1);
        const uint16_t **wq = malloc(v->depth * sizeof *wq), **wo = malloc(v->depth * sizeof *wo);
        const uint16_t **f0 = malloc(v->depth * sizeof *f0), **f1 = malloc(v->depth * sizeof *f1);
        for (int l = 0; l < v->depth; l++) {
            n0[l] = v->b[l].n0; n1[l] = v->b[l].n1; wq[l] = v->b[l].qkv; wo[l] = v->b[l].wo;
            f0[l] = v->b[l].fc0; f1[l] = v->b[l].fc1;
        }
        K3cuVision cv = {v->depth, v->d, v->heads, v->hd, v->qd, v->inter, v->out, v->patch, v->ph, v->pw, v->eps,
                         v->peps, v->patch_w, v->pos, v->p0, v->p2, v->fnorm, v->pnorm, n0, n1, wq, wo, f0, f1};
        const int T = (gh / 2) * (gw / 2);
        float *rows = malloc((size_t)T * v->out * sizeof(float));
        for (int rep = 0; rep < 3; rep++) {   /* the first pays the start; the rest are what a picture costs */
            const double t0 = k3v_now();
            if (!k3cu_vision(&cv, rgb, gh, gw, rows)) { fprintf(stderr, "k3cu_vision failed\n"); return 1; }
            printf("GPU: %d rows in %.3fs\n", T, k3v_now() - t0);
        }
        snprintf(path, sizeof path, "%s/rows_gpu.f32", argv[5]);
        FILE *o = fopen(path, "wb");
        if (o) { fwrite(rows, sizeof(float), (size_t)T * v->out, o); fclose(o); }
        k3cu_report();
        return 0;
#else
        fprintf(stderr, "this build has no GPU (-DCOLI_K3_CUDA with backend_cuda_k3_loader.c)\n");
        return 2;
#endif
    }
    snprintf(path, sizeof path, "K3_VISION_DUMP=%s", argv[5]);
    putenv(path);
    int rows = 0;
    float *r = k3v_run(v, rgb, gh, gw, &rows);
    if (!r) return 1;
    printf("%d rows of %d, pad id %d\n", rows, v->out, pad);
    free(r);
    free(rgb);
    return 0;
}
