"""MIMO_MTP's speculative sampling draws sample()'s distribution.

mimo.c verifies an MTP draft against the trunk's own next-token distribution with
sample_draft(): the draft stands with the probability sample() gives it, otherwise the
token is drawn from the nucleus without it. This cuts sample(), sample_draft() and their
helpers out of mimo.c, compiles them with gcc beside a driver, and compares the tokens'
frequencies over many draws with sample()'s, for drafts inside and outside the nucleus,
at several temperatures and top_p values; greedy keeps the argmax.
"""
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "mimo.c"

DRIVER = r'''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
static void *xmalloc(size_t n, const char *what) { void *p = malloc(n); if (!p) { fprintf(stderr, "OOM %s\n", what); exit(1); } return p; }
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


def cut(src, start):
    i = src.index(start)
    return src[i:src.index("\n}\n", i) + 3]


class SampleDraft(unittest.TestCase):
    def test_distribution(self):
        gcc = shutil.which("gcc")
        if not gcc:
            self.skipTest("gcc not on PATH")
        src = SRC.read_text(encoding="utf-8").replace("\r\n", "\n")
        code = "\n".join([
            cut(src, "static int argmax(const float *x, int n) {"),
            re.search(r"typedef struct \{[^}]*\} SampleProb;", src).group(0),
            cut(src, "static int sample_desc(const void *a, const void *b) {"),
            cut(src, "static int sample(const float *logits, int vocab, float temperature, float top_p) {"),
            cut(src, "static int sample_draft(const float *logits, int vocab, float temperature, float top_p, int d, int *kept) {"),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            c_file, exe = Path(tmp) / "draft.c", Path(tmp) / "draft.exe"
            c_file.write_text(DRIVER.replace("@CODE@", code), encoding="utf-8")
            subprocess.run([gcc, "-O2", str(c_file), "-o", str(exe), "-lm"], check=True)
            run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=600)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn("ok", run.stdout.splitlines()[-1])


if __name__ == "__main__":
    unittest.main()
