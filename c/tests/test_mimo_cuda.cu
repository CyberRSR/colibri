/* Independent dense reference + chunk-invariance + lifecycle on a real GPU.
 * Exit 1 if CUDA is unavailable: explicitly invoking this is a hardware test. */
#include "../backend_cuda_mimo.h"
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>

static float fp8(unsigned v) {
    int a = v & 127, e = a >> 3, m = a & 7;
    float f = e ? std::ldexp(1.f + m / 8.f, e - 7) : m / 512.f;
    return v & 128 ? -f : f;
}
static int check(int fmt, int I, int O, int S) {
    size_t count = (size_t)I * O;
    std::vector<float> w(count), x((size_t)S * I), want((size_t)S * O), got(want.size()), single(want.size());
    std::vector<unsigned char> q(count * 4);
    int blocks = (I + 127) / 128;
    std::vector<float> scales((size_t)O * blocks);
    for (size_t i = 0; i < scales.size(); i++) scales[i] = 0.01f * (1 + i % 11);
    for (size_t i = 0; i < x.size(); i++) x[i] = std::sin((float)(i + 1) * 0.137f);
    for (size_t j = 0; j < count; j++) {
        float f = std::sin((float)(j + 7) * 0.17f) * 0.25f;
        if (fmt == 0) { memcpy(q.data() + 4 * j, &f, 4); w[j] = f; }
        if (fmt == 1) {
            uint32_t u; memcpy(&u, &f, 4); uint16_t b = (uint16_t)(u >> 16);
            memcpy(q.data() + 2 * j, &b, 2); u &= 0xffff0000u; memcpy(&w[j], &u, 4);
        }
        if (fmt == 2) {
            unsigned v = (unsigned)(j * 73 + j / 7) % 256;
            if ((v & 127) == 127) v ^= 1;
            q[j] = (unsigned char)v;
            w[j] = fp8(v) * scales[(j / I) * blocks + (j % I) / 128];
        }
        if (fmt == 3) {
            int8_t v = (int8_t)((int)(j * 37 % 255) - 127);
            memcpy(q.data() + j, &v, 1); w[j] = v * scales[j / I];
        }
        if (fmt == 4) {   /* int8 with a scale per (row, 128-column block) */
            int8_t v = (int8_t)((int)(j * 37 % 255) - 127);
            memcpy(q.data() + j, &v, 1); w[j] = v * scales[(j / I) * blocks + (j % I) / 128];
        }
    }
    for (int s = 0; s < S; s++) for (int o = 0; o < O; o++) {
        double acc = 0;
        for (int i = 0; i < I; i++) acc += (double)x[(size_t)s * I + i] * w[(size_t)o * I + i];
        want[(size_t)s * O + o] = (float)acc;
    }
    MimoCudaTensor *t = mimo_cuda_upload(q.data(), scales.data(), fmt, I, O);
    if (!t || !mimo_cuda_matmul(t, got.data(), x.data(), S)) return 1;
    double max_error = 0;
    for (size_t i = 0; i < want.size(); i++) {
        double err = std::abs((double)got[i] - want[i]);
        if (!std::isfinite(got[i]) || err > 0.00015 + 0.00002 * std::abs(want[i])) {
            printf("FAIL fmt=%d I=%d O=%d S=%d at %zu: got=%g want=%g\n", fmt,I,O,S,i,got[i],want[i]); return 1;
        }
        if (err > max_error) max_error = err;
    }
    for (int s = 0; s < S; s++)
        if (!mimo_cuda_matmul(t, single.data() + (size_t)s * O, x.data() + (size_t)s * I, 1)) return 1;
    if (memcmp(got.data(), single.data(), got.size() * sizeof(float))) { puts("FAIL chunk invariance"); return 1; }
    float sentinel = 123;
    if (mimo_cuda_matmul(t, &sentinel, x.data(), 0) || sentinel != 123 ||
        mimo_cuda_matmul(nullptr, &sentinel, x.data(), 1) || sentinel != 123) return 1;
    printf("ok fmt=%d I=%d O=%d S=%d max_abs=%g; chunks bit-exact\n",fmt,I,O,S,max_error);
    return 0;
}
int main() {
    if (!mimo_cuda_init(0)) return 1;
    float w = 1;
    if (mimo_cuda_upload(&w, nullptr, 2, 1, 1) || mimo_cuda_upload(&w, nullptr, 4, 1, 1) ||
        mimo_cuda_upload(&w, nullptr, 5, 1, 1) ||
        mimo_cuda_upload(&w, nullptr, 0, 0, 1) || mimo_cuda_init(-1)) return 1;
    int rc = 0;
    for (int f = 0; f < 5; f++) {
        rc |= check(f, 1, 1, 1);
        rc |= check(f, 129, 13, 7);
        rc |= check(f, 6144, 33, 3);
        rc |= check(f, 257, 129, 17);
        rc |= check(f, 512, 9, 17);      /* the wide kernels: FP8 and BF16, row tiles of 8 */
        rc |= check(f, 1024, 6, 9);
    }
    mimo_cuda_report(); mimo_cuda_shutdown();
    if (!mimo_cuda_init(0)) return 1;
    rc |= check(2, 128, 7, 3);
    mimo_cuda_shutdown(); mimo_cuda_shutdown();
    return rc;
}
