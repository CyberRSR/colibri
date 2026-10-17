"""Write bench.c: ede829c7's sample()/sample_draft() (old_*) beside the working tree's, a driver
that replays rows (dumped by a timed engine, or synthetic), checks that both give the same
token, kept flag and rand() state for the same seed, and times both per call."""
import re
import subprocess
from pathlib import Path

here = Path(__file__).resolve().parent
repo = here.parents[2]
old_src = subprocess.run(["git", "-C", str(repo), "show", "ede829c7:c/mimo.c"], capture_output=True, text=True,
                         encoding="utf-8", check=True).stdout.replace("\r\n", "\n")
new_src = (repo / "c" / "mimo.c").read_text(encoding="utf-8").replace("\r\n", "\n")


def cut(src, start):
    i = src.index(start)
    return src[i:src.index("\n}\n", i) + 3]


SIG_S = "static int sample(const float *logits, int vocab, float temperature, float top_p) {"
SIG_D = "static int sample_draft(const float *logits, int vocab, float temperature, float top_p, int d, int *kept) {"
shared = "\n".join([
    cut(new_src, "static int argmax(const float *x, int n) {"),
    re.search(r"typedef struct \{[^}]*\} SampleProb;", new_src).group(0),
    cut(new_src, "static int sample_desc(const void *a, const void *b) {"),
])
old = "\n".join([cut(old_src, SIG_S), cut(old_src, SIG_D)])
old = old.replace("static int sample(", "static int old_sample(").replace("static int sample_draft(", "static int old_sample_draft(")
new = "\n".join([
    re.search(r"typedef struct \{[^}]*\} SampleRank;", new_src, re.S).group(0),
    cut(new_src, "static void sample_sort("),
    cut(new_src, "static void sample_rank("),
    cut(new_src, "static int sample_tie("),
    cut(new_src, SIG_S),
    cut(new_src, "static int sample_walk("),
    cut(new_src, SIG_D),
])
# count the full sorts the new code falls back to
new = new.replace("static void sample_sort(SampleRank *k, int vocab, double least) {",
                  "static long long g_full, g_sorts, g_cand;\nstatic void sample_sort(SampleRank *k, int vocab, double least) {\n    g_sorts++;")
new = new.replace("    k->m = m; k->all = m == vocab;", "    k->m = m; k->all = m == vocab;\n    if (m == vocab) g_full++; else g_cand += m;")

driver = (here / "bench_driver.c").read_text(encoding="utf-8")
(here / "bench.c").write_text(driver.replace("@SHARED@", shared).replace("@OLD@", old).replace("@NEW@", new), encoding="utf-8")
print("wrote", here / "bench.c")
