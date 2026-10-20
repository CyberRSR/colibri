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
/* fmt 5 (bf16 packed without loss) against fmt 1 on the same bf16: the same bytes,
 * for weights like a trained matrix's (kind 0), exponents everywhere so most weights
 * are exceptions (1), and rows of zeros, of one value, of 15 and of 16 exponents (2). */
static uint32_t rng_state = 12345;
static uint32_t rng() { rng_state = rng_state * 1664525u + 1013904223u; return rng_state >> 8; }
static int check_packed(int I, int O, int S, int kind) {
    size_t count = (size_t)I * O;
    std::vector<uint16_t> q(count);
    for (size_t j = 0; j < count; j++) {
        int o = (int)(j / I), i = (int)(j % I);
        uint16_t b;
        if (kind == 0) {   /* about N(0, 0.02): a sum of uniforms */
            float f = 0;
            for (int k = 0; k < 4; k++) f += (float)(rng() & 0xffff) / 65536.0f - 0.5f;
            f *= 0.035f;
            uint32_t u; memcpy(&u, &f, 4); b = (uint16_t)(u >> 16);
        } else if (kind == 1) {   /* any sign and mantissa, exponents 0..140 */
            b = (uint16_t)((rng() & 0x807f) | ((rng() % 141) << 7));
        } else {
            switch (o % 5) {
            case 0: b = (uint16_t)(i % 3 ? 0x0000 : 0x8000); break;            /* zeros of both signs */
            case 1: b = 0x3f80; break;                                          /* one value */
            case 2: b = (uint16_t)((rng() & 0x807f) | ((100 + i % 15) << 7)); break;   /* exactly 15 exponents */
            case 3: b = (uint16_t)((rng() & 0x807f) | ((100 + i % 16) << 7)); break;   /* 16: one is an exception */
            default: b = (uint16_t)((rng() & 0x807f) | ((i % 7 ? 120 : rng() % 3) << 7)); break; /* subnormals */
            }
        }
        q[j] = b;
    }
    std::vector<float> x((size_t)S * I), plain((size_t)S * O), got(plain.size()), single(plain.size());
    for (size_t i = 0; i < x.size(); i++) x[i] = std::sin((float)(i + 1) * 0.137f);
    MimoCudaTensor *a = mimo_cuda_upload(q.data(), nullptr, 1, I, O), *t = mimo_cuda_upload(q.data(), nullptr, 5, I, O);
    if (!a || !t || !mimo_cuda_matmul(a, plain.data(), x.data(), S) || !mimo_cuda_matmul(t, got.data(), x.data(), S)) {
        printf("FAIL packed I=%d O=%d S=%d kind=%d: upload or matmul\n", I, O, S, kind); return 1;
    }
    if (memcmp(got.data(), plain.data(), got.size() * sizeof(float))) {
        for (size_t i = 0; i < got.size(); i++)
            if (memcmp(&got[i], &plain[i], 4)) { printf("FAIL packed I=%d O=%d S=%d kind=%d at %zu: %a vs bf16 %a\n", I, O, S, kind, i, got[i], plain[i]); break; }
        return 1;
    }
    for (int s = 0; s < S; s++)
        if (!mimo_cuda_matmul(t, single.data() + (size_t)s * O, x.data() + (size_t)s * I, 1)) return 1;
    if (memcmp(got.data(), single.data(), got.size() * sizeof(float))) { puts("FAIL packed chunk invariance"); return 1; }
    size_t pb = mimo_cuda_bytes(t), bb = mimo_cuda_bytes(a);
    if (bb != (size_t)I * O * 2 || !pb || (kind == 0 && pb * 4 > bb * 3 + (size_t)O * 8 + 4096)) {
        printf("FAIL packed I=%d O=%d kind=%d: %zu bytes packed, %zu plain\n", I, O, kind, pb, bb); return 1;
    }
    printf("ok packed I=%d O=%d S=%d kind=%d: the bytes of bf16 from %zu bytes, not %zu\n", I, O, S, kind, pb, bb);
    return 0;
}
/* A prompt block's tiled kernels (S >= 64 rows; MIMO_CUDA_TILED_ROWS): FP8 (fmt 2), plain bf16
 * (fmt 1) and packed bf16 (fmt 5, the generators above, exceptions included) against a double
 * reference, a row's bytes the same in a block of S rows and in two of 64 and S - 64, and every
 * output the bytes of its sums in the kernels' order: FP8 each 128-column scale group's products
 * summed apart, then fma(sum, scale, output), as the CPU groups them; bf16 one chain. (A kernel
 * with FP8's scales folded into its weights, one chain, passed the reference's bound here but
 * moved the model's argmax: these bytes catch it.) */
static int check_tiled(int fmt, int I, int O, int S, int kind) {
    size_t count = (size_t)I * O;
    int blocks = I / 128;
    std::vector<unsigned char> q8(count);
    std::vector<uint16_t> q16(count);
    std::vector<float> w(count), scales((size_t)O * (blocks ? blocks : 1));
    for (size_t i = 0; i < scales.size(); i++) scales[i] = 0.01f * (1 + i % 11);
    for (size_t j = 0; j < count; j++) {
        if (fmt == 2) {
            unsigned v = (unsigned)(j * 73 + j / 7) % 256;
            if ((v & 127) == 127) v ^= 1;
            q8[j] = (unsigned char)v;
            w[j] = fp8(v) * scales[(j / I) * blocks + (j % I) / 128];
        } else {
            int o = (int)(j / I), i = (int)(j % I);
            uint16_t b;
            if (kind == 0) { float f = std::sin((float)(j + 7) * 0.17f) * 0.03f; uint32_t u; memcpy(&u, &f, 4); b = (uint16_t)(u >> 16); }
            else if (kind == 1) b = (uint16_t)((rng() & 0x807f) | ((rng() % 141) << 7));
            else b = (uint16_t)((rng() & 0x807f) | ((o % 2 ? 100 + i % 16 : 120) << 7));
            q16[j] = b;
            uint32_t u = (uint32_t)b << 16; memcpy(&w[j], &u, 4);
        }
    }
    std::vector<float> x((size_t)S * I), got((size_t)S * O), split(got.size());
    for (size_t i = 0; i < x.size(); i++) x[i] = std::sin((float)(i + 1) * 0.137f);
    MimoCudaTensor *t = fmt == 2 ? mimo_cuda_upload(q8.data(), scales.data(), 2, I, O) : mimo_cuda_upload(q16.data(), nullptr, fmt, I, O);
    if (!t || !mimo_cuda_matmul(t, got.data(), x.data(), S) || !mimo_cuda_matmul(t, split.data(), x.data(), 64) ||
        !mimo_cuda_matmul(t, split.data() + (size_t)64 * O, x.data() + (size_t)64 * I, S - 64)) {
        printf("FAIL tiled fmt=%d I=%d O=%d S=%d: upload or matmul\n", fmt, I, O, S); return 1;
    }
    double max_rel = 0;
    for (int s = 0; s < S; s++) for (int o = 0; o < O; o++) {
        double acc = 0, mag = 0;
        float sum = 0, part = 0;
        for (int i = 0; i < I; i++) {
            const float xv = x[(size_t)s * I + i];
            double p = (double)xv * w[(size_t)o * I + i];
            acc += p; mag += std::abs(p);
            if (fmt == 2) {
                part = std::fmaf(xv, fp8(q8[(size_t)o * I + i]), part);
                if (i % 128 == 127) { sum = std::fmaf(part, scales[(size_t)o * blocks + i / 128], sum); part = 0; }
            } else sum = std::fmaf(xv, w[(size_t)o * I + i], sum);
        }
        float g = got[(size_t)s * O + o];
        double err = std::abs(g - acc);
        if (!std::isfinite(g) || err > 1e-6 * mag + 1e-30) {
            printf("FAIL tiled fmt=%d kind=%d I=%d O=%d S=%d at row %d out %d: got=%g want=%g\n", fmt, kind, I, O, S, s, o, g, acc);
            return 1;
        }
        if (memcmp(&g, &sum, sizeof g)) {
            printf("FAIL tiled fmt=%d kind=%d I=%d O=%d S=%d at row %d out %d: %a, its sums in order %a\n", fmt, kind, I, O, S, s, o, g, sum);
            return 1;
        }
        if (mag > 0 && err / mag > max_rel) max_rel = err / mag;
    }
    if (memcmp(got.data(), split.data(), got.size() * sizeof(float))) { printf("FAIL tiled fmt=%d: rows differ by block\n", fmt); return 1; }
    printf("ok tiled fmt=%d kind=%d I=%d O=%d S=%d: max error %.2g of the products' magnitude; the sums' bytes; blocks bit-exact\n", fmt, kind, I, O, S, max_rel);
    return 0;
}
int main() {
    if (!mimo_cuda_init(0)) return 1;
    float w = 1;
    if (mimo_cuda_upload(&w, nullptr, 2, 1, 1) || mimo_cuda_upload(&w, nullptr, 4, 1, 1) ||
        mimo_cuda_upload(&w, nullptr, 5, 1, 1) || mimo_cuda_upload(&w, nullptr, 6, 1, 1) ||
        mimo_cuda_upload(&w, nullptr, 0, 0, 1) || mimo_cuda_init(-1) || mimo_cuda_bytes(nullptr)) return 1;
    int rc = 0;
    for (int f = 0; f < 5; f++) {
        rc |= check(f, 1, 1, 1);
        rc |= check(f, 129, 13, 7);
        rc |= check(f, 6144, 33, 3);
        rc |= check(f, 257, 129, 17);
        rc |= check(f, 512, 9, 17);      /* the wide kernels: FP8 and BF16, row tiles of 8 */
        rc |= check(f, 1024, 6, 9);
    }
    for (int kind = 0; kind < 3; kind++) {
        rc |= check_packed(256, 5, 1, kind);
        rc |= check_packed(512, 9, 17, kind);
        rc |= check_packed(6144, 33, 3, kind);
        rc |= check_packed(16384, 7, 9, kind);   /* Pro's o_proj rows */
        rc |= check_packed(1024, 130, 8, kind);
        rc |= check_tiled(5, 1024, 128, 130, kind);
        rc |= check_tiled(5, 16384, 64, 66 + 64, kind);   /* Pro's o_proj rows */
    }
    rc |= check_tiled(1, 6144, 384, 130, 0);   /* a router: plain bf16 */
    rc |= check_tiled(1, 1024, 128, 150, 1);
    rc |= check_tiled(2, 512, 64, 130, 0);
    rc |= check_tiled(2, 6144, 192, 200, 0);   /* Pro's qkv rows */
    mimo_cuda_report(); mimo_cuda_shutdown();
    if (!mimo_cuda_init(0)) return 1;
    rc |= check(2, 128, 7, 3);
    mimo_cuda_shutdown(); mimo_cuda_shutdown();
    return rc;
}
