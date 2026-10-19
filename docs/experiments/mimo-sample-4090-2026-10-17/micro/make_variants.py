"""Engine sources for the sampler experiment (scratch only, never committed):
  before/mimo.c        ede829c7's c/mimo.c
  before-timed/mimo.c  ede829c7's, each sample()/sample_draft() call at temperature > 0 timed
                       ([sample-time] lines on stderr); MIMO_SAMPLE_DUMP=file writes every
                       row (bench_driver.c's Rec, then the logits)
  after-timed/mimo.c   the working tree's c/mimo.c, timed the same way."""
import subprocess
from pathlib import Path

here = Path(__file__).resolve().parent
repo = here.parents[2]
head = subprocess.run(["git", "-C", str(repo), "show", "ede829c7:c/mimo.c"], capture_output=True, text=True,
                      encoding="utf-8", check=True).stdout.replace("\r\n", "\n")
work = (repo / "c" / "mimo.c").read_text(encoding="utf-8").replace("\r\n", "\n")

SIG_S = "static int sample(const float *logits, int vocab, float temperature, float top_p) {"
SIG_D = "static int sample_draft(const float *logits, int vocab, float temperature, float top_p, int d, int *kept) {"
NL = "\\n"
WRAP = f"""
/* TIMING (scratch build): each draw at temperature > 0 timed; MIMO_SAMPLE_DUMP writes the rows */
static void sd_record(int kind, const float *logits, int vocab, float T, float top_p, int d, int y, int kept, double t) {{
    static FILE *dump; static int init; static double tt[2]; static long long nn[2];
    if (T <= 0.0f) return;
    if (!init) {{ init = 1; const char *f = getenv("MIMO_SAMPLE_DUMP"); if (f && *f) dump = fopen(f, "wb"); }}
    tt[kind] += t; nn[kind]++;
    if ((nn[0] + nn[1]) % 25 == 0)
        fprintf(stderr, "[sample-time] sample %lld calls %.3f ms avg, sample_draft %lld calls %.3f ms avg{NL}",
                nn[0], nn[0] ? 1e3 * tt[0] / nn[0] : 0.0, nn[1], nn[1] ? 1e3 * tt[1] / nn[1] : 0.0);
    if (dump) {{
        struct {{ int32_t kind, vocab, d, y, kept, pad; float T, top_p; double t; }} h = {{kind, vocab, d, y, kept, 0, T, top_p, t}};
        fwrite(&h, sizeof h, 1, dump); fwrite(logits, sizeof(float), (size_t)vocab, dump); fflush(dump);
    }}
}}
static int sample(const float *logits, int vocab, float temperature, float top_p) {{
    double t0 = now_s();
    int y = sample_orig(logits, vocab, temperature, top_p);
    sd_record(0, logits, vocab, temperature, top_p, -1, y, 0, now_s() - t0);
    return y;
}}
static int sample_draft(const float *logits, int vocab, float temperature, float top_p, int d, int *kept) {{
    double t0 = now_s();
    int y = sample_draft_orig(logits, vocab, temperature, top_p, d, kept);
    sd_record(1, logits, vocab, temperature, top_p, d, y, *kept, now_s() - t0);
    return y;
}}
"""


def timed(src):
    assert src.count(SIG_S) == 1 and src.count(SIG_D) == 1
    src = src.replace(SIG_S, SIG_S.replace("sample(", "sample_orig(")).replace(SIG_D, SIG_D.replace("sample_draft(", "sample_draft_orig("))
    i = src.index(SIG_D.replace("sample_draft(", "sample_draft_orig("))
    end = src.index("\n}\n", i) + 3
    return src[:end] + WRAP + src[end:]


for name, src in (("before", head), ("before-timed", timed(head)), ("after-timed", timed(work))):
    d = here / name
    d.mkdir(exist_ok=True)
    (d / "mimo.c").write_text(src, encoding="utf-8", newline="\n")
    print("wrote", d / "mimo.c")
