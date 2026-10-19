"""MIMO_MTP's speculative sampling draws sample()'s distribution, with sample()'s draws.

mimo.c verifies an MTP draft against the trunk's own next-token distribution with
sample_draft(): the draft stands with the probability sample() gives it, otherwise the
token is drawn from the nucleus without it. This cuts sample(), sample_draft() and their
helpers out of mimo.c, compiles them with gcc beside a driver, and compares the tokens'
frequencies over many draws with sample()'s, for drafts inside and outside the nucleus,
at several temperatures and top_p values; greedy keeps the argmax.

Both sort only the entries that could be in top_p's nucleus, and the whole vocabulary
only where the order qsort gives entries of equal p decides a draw: the second test
holds them to the draws of the sampler that sorted the whole vocabulary every time
(REFERENCE, the code before), token, kept and the rand() state after, on rows full of
such ties.
"""
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "mimo.c"

HEADER = r'''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "exp_f.h"
static void *xmalloc(size_t n, const char *what) { void *p = malloc(n); if (!p) { fprintf(stderr, "OOM %s\n", what); exit(1); } return p; }
'''

DRIVER = HEADER + r'''
@CODE@
int main(void) {
    enum { V = 40, N = 400000 };
    float lg[V];
    srand(7);
    for (int i = 0; i < V; i++) lg[i] = (float)(rand() % 1000) / 150.0f - 3.0f;
    float temps[3] = {0.7f, 1.0f, 0.3f}, tops[3] = {0.9f, 1.0f, 0.5f};
    int bad = 0;
    for (int c = 0; c < 3; c++) {
        int top = argmax(lg, V);
        int drafts[3] = {top, (top + 7) % V, (top + 23) % V};
        static double ref[V];
        memset(ref, 0, sizeof ref);
        for (int k = 0; k < N; k++) ref[sample(lg, V, temps[c], tops[c])] += 1.0 / N;
        for (int q = 0; q < 3; q++) {
            static double got[V];
            memset(got, 0, sizeof got);
            int kept_n = 0;
            for (int k = 0; k < N; k++) {
                int kept, t = sample_draft(lg, V, temps[c], tops[c], drafts[q], &kept);
                if (kept && t != drafts[q]) bad = 1;
                kept_n += kept;
                got[t] += 1.0 / N;
            }
            double err = 0;
            for (int i = 0; i < V; i++) if (fabs(got[i] - ref[i]) > err) err = fabs(got[i] - ref[i]);
            printf("T=%.1f top_p=%.1f draft %2d: p %.3f, kept %.3f, max |freq - sample()'s| %.4f\n",
                   temps[c], tops[c], drafts[q], ref[drafts[q]], (double)kept_n / N, err);
            if (err > 0.006 || fabs((double)kept_n / N - ref[drafts[q]]) > 0.01) bad = 1;
        }
    }
    int kept, y = sample_draft(lg, V, 0.0f, 1.0f, argmax(lg, V), &kept);
    if (!kept || y != argmax(lg, V)) bad = 1;
    y = sample_draft(lg, V, 0.0f, 1.0f, (argmax(lg, V) + 1) % V, &kept);
    if (kept || y != argmax(lg, V)) bad = 1;
    puts(bad ? "FAIL" : "ok");
    return bad;
}
'''

# The sampler before it sorted only the nucleus' candidates: the whole vocabulary, every draw.
REFERENCE = r'''
static int ref_sample(const float *logits, int vocab, float temperature, float top_p) {
    if (temperature <= 0.0f) return argmax(logits, vocab);
    SampleProb *rank = xmalloc((size_t)vocab * sizeof(SampleProb), "sampling");
    float top = logits[0];
    for (int i = 1; i < vocab; i++) if (logits[i] > top) top = logits[i];
    double total = 0.0;
    for (int i = 0; i < vocab; i++) {
        float p = expf((logits[i] - top) / temperature);
        total += p; rank[i].p = p; rank[i].id = i;
    }
    qsort(rank, (size_t)vocab, sizeof(SampleProb), sample_desc);
    double cut = (top_p > 0.0f && top_p < 1.0f) ? top_p * total : total, kept = 0.0;
    int n = 0;
    while (n < vocab && kept < cut) kept += rank[n++].p;
    double draw = ((double)rand() / RAND_MAX) * kept, acc = 0.0;
    int pick = rank[0].id;
    for (int i = 0; i < n; i++) { acc += rank[i].p; if (acc >= draw) { pick = rank[i].id; break; } }
    free(rank);
    return pick;
}

static int ref_sample_draft(const float *logits, int vocab, float temperature, float top_p, int d, int *kept) {
    if (temperature <= 0.0f) { int y = argmax(logits, vocab); *kept = y == d; return y; }
    SampleProb *rank = xmalloc((size_t)vocab * sizeof(SampleProb), "sampling");
    float top = logits[0];
    for (int i = 1; i < vocab; i++) if (logits[i] > top) top = logits[i];
    double total = 0.0;
    for (int i = 0; i < vocab; i++) {
        float p = expf((logits[i] - top) / temperature);
        total += p; rank[i].p = p; rank[i].id = i;
    }
    qsort(rank, (size_t)vocab, sizeof(SampleProb), sample_desc);
    double cut = (top_p > 0.0f && top_p < 1.0f) ? top_p * total : total, nucleus = 0.0, pd = 0.0;
    int n = 0;
    while (n < vocab && nucleus < cut) nucleus += rank[n++].p;
    for (int i = 0; i < n; i++) if (rank[i].id == d) { pd = rank[i].p; break; }
    int pick = d;
    *kept = ((double)rand() / RAND_MAX) * nucleus < pd;
    if (!*kept) {
        double draw = ((double)rand() / RAND_MAX) * (nucleus - pd), acc = 0.0;
        for (int i = 0; i < n; i++) {
            if (rank[i].id == d) continue;
            pick = rank[i].id;
            acc += rank[i].p;
            if (acc >= draw) break;
        }
    }
    free(rank);
    return pick;
}
'''

EXACT = HEADER + r'''
@CODE@
@REFERENCE@
static unsigned long long rng = 88172645463325252ull;
static double urand(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (rng >> 11) * (1.0 / 9007199254740992.0); }
static double nrand(void) { double u = urand() + 1e-300, v = urand(); return sqrt(-2 * log(u)) * cos(6.283185307179586 * v); }

int main(void) {
    enum { V = 2000, ROWS = 96, SEEDS = 40 };
    static float lg[V];
    static int ord[V];
    float temps[4] = {0.7f, 1.0f, 0.3f, 1.5f}, tops[5] = {0.9f, 0.5f, 0.95f, 1.0f, 0.0f};
    long long checks = 0, bad = 0;
    for (int row = 0; row < ROWS; row++) {
        /* logits: continuous, or on a grid of 1/2, 1/8, 1/64 (ties everywhere, the nucleus'
         * edge too), a few tied with the max, or all the same */
        int grid[4] = {0, 2, 8, 64}, g = grid[row % 4];
        double scale = 1.0 + 3.0 * urand();
        for (int i = 0; i < V; i++) {
            double v = scale * nrand();
            lg[i] = (float)(g ? round(v * g) / g : v);
        }
        int top = argmax(lg, V);
        if (row % 3 == 0) lg[top] += (float)(2.0 * scale);
        if (row % 5 == 1) for (int k = 1; k <= 3; k++) lg[(top + 101 * k) % V] = lg[top];
        if (row % 16 == 15) for (int i = 0; i < V; i++) lg[i] = 0.5f;
        float T = temps[row % 4], P = tops[(row / 4) % 5];
        /* drafts: the most likely, the next ones, the last of the nucleus and the entries
         * just past it (by the reference's own sort), and any entry */
        float m = lg[argmax(lg, V)];
        double total = 0.0;
        static double p[V];
        for (int i = 0; i < V; i++) { p[i] = expf((lg[i] - m) / T); total += (float)p[i]; ord[i] = i; }
        for (int i = 1; i < V; i++) {   /* insertion sort by p, descending */
            int x = ord[i], j = i - 1;
            while (j >= 0 && p[ord[j]] < p[x]) { ord[j + 1] = ord[j]; j--; }
            ord[j + 1] = x;
        }
        double cut = (P > 0.0f && P < 1.0f) ? P * total : total, acc = 0.0;
        int n = 0;
        while (n < V && acc < cut) acc += (float)p[ord[n++]];
        int drafts[8] = {ord[0], ord[1], ord[2], ord[n - 1], ord[n < V ? n : V - 1], ord[n + 1 < V ? n + 1 : V - 1],
                         ord[n / 2], (int)(urand() * V)};
        for (int q = 0; q < 8; q++)
            for (int s = 0; s < SEEDS; s++) {
                unsigned seed = 7919u * (unsigned)row + (unsigned)s;
                int ka, kb;
                srand(seed); int a = ref_sample_draft(lg, V, T, P, drafts[q], &ka); int ra = rand();
                srand(seed); int b = sample_draft(lg, V, T, P, drafts[q], &kb); int rb = rand();
                checks++;
                if (a != b || ka != kb || ra != rb) {
                    if (bad++ < 10) printf("row %d T %.1f top_p %.2f draft %d seed %u: %d/%d against %d/%d\n", row, T, P, drafts[q], seed, b, kb, a, ka);
                }
                if (q == 0) {
                    srand(seed); a = ref_sample(lg, V, T, P); ra = rand();
                    srand(seed); b = sample(lg, V, T, P); rb = rand();
                    checks++;
                    if (a != b || ra != rb) {
                        if (bad++ < 10) printf("row %d T %.1f top_p %.2f seed %u: sample %d against %d\n", row, T, P, seed, b, a);
                    }
                }
            }
    }
    printf("%lld draws, %lld not the reference's\n", checks, bad);
    puts(bad ? "FAIL" : "ok");
    return bad != 0;
}
'''


def cut(src, start):
    i = src.index(start)
    return src[i:src.index("\n}\n", i) + 3]


def sampler():
    """sample(), sample_draft() and what they call, as mimo.c has them."""
    src = SRC.read_text(encoding="utf-8").replace("\r\n", "\n")
    return "\n".join([
        cut(src, "static int argmax(const float *x, int n) {"),
        re.search(r"typedef struct \{[^}]*\} SampleProb;", src).group(0),
        cut(src, "static int sample_desc(const void *a, const void *b) {"),
        re.search(r"typedef struct \{[^}]*\} SampleRank;", src).group(0),
        cut(src, "static void sample_sort(SampleRank *k, int vocab, double least) {"),
        cut(src, "static void sample_rank(SampleRank *k, const float *logits, int vocab, float temperature, float top_p) {"),
        cut(src, "static int sample_tie(const SampleRank *k, int j, int skip) {"),
        cut(src, "static int sample(const float *logits, int vocab, float temperature, float top_p) {"),
        cut(src, "static int sample_walk(const SampleProb *r, int n, int d, double draw) {"),
        cut(src, "static int sample_draft(const float *logits, int vocab, float temperature, float top_p, int d, int *kept) {"),
    ])


class SampleDraft(unittest.TestCase):
    def run_driver(self, driver):
        gcc = shutil.which("gcc")
        if not gcc:
            self.skipTest("gcc not on PATH")
        with tempfile.TemporaryDirectory() as tmp:
            c_file, exe = Path(tmp) / "draft.c", Path(tmp) / "draft.exe"
            c_file.write_text(driver.replace("@CODE@", sampler()).replace("@REFERENCE@", REFERENCE), encoding="utf-8")
            subprocess.run([gcc, "-O2", "-I", str(SRC.parent), str(c_file), "-o", str(exe), "-lm"], check=True)
            run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=600)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn("ok", run.stdout.splitlines()[-1])

    def test_distribution(self):
        self.run_driver(DRIVER)

    def test_same_draws_as_the_whole_sort(self):
        self.run_driver(EXACT)


if __name__ == "__main__":
    unittest.main()
