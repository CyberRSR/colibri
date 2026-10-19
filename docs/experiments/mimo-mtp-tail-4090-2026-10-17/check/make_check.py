"""Write mimo_check.c: c/mimo.c with two checks (scratch only, never committed).
- every draft's p both ways: the chunked sum the engine now uses, and the old serial one;
  the float bits compared, every difference and a running count on stderr;
- dense_mlp's threaded silu against the serial loop on the same activations."""
from pathlib import Path

here = Path(__file__).resolve().parent
src = (here.parents[2] / "mimo.c").read_text(encoding="utf-8")
NL = "\\n"

old = """        d = 0;
        for (int i = 1; i < c->vocab; i++) if (lg[i] > lg[d]) d = i;
        *p = (float)(1.0 / exp_sum(lg, c->vocab, lg[d]));
        free(lg);"""
new = f"""        d = 0;
        for (int i = 1; i < c->vocab; i++) if (lg[i] > lg[d]) d = i;
        *p = (float)(1.0 / exp_sum(lg, c->vocab, lg[d]));
        {{   /* CHECK: the serial sum the engine had */
            static long long drafts, differ;
            double s = 0;
            for (int i = 0; i < c->vocab; i++) s += expf(lg[i] - lg[d]);
            float q = (float)(1.0 / s);
            drafts++;
            if (memcmp(&q, p, sizeof q)) {{ differ++; fprintf(stderr, "[check] block %d draft %d: p %a (new) != %a (old){NL}", k, d, *p, q); }}
            if (drafts % 100 == 0) fprintf(stderr, "[check] %lld drafts, p bits differ in %lld{NL}", drafts, differ);
        }}
        free(lg);"""
assert src.count(old) == 1
src = src.replace(old, new)

old = """    #pragma omp parallel for schedule(dynamic, 512)
    for (int64_t i = 0; i < (int64_t)n * F; i++) g[i] = silu(g[i]) * u[i];"""
new = f"""    float *chk = xmalloc((size_t)n * F * sizeof(float), "check");   /* CHECK */
    for (size_t i = 0; i < (size_t)n * F; i++) chk[i] = silu(g[i]) * u[i];
    #pragma omp parallel for schedule(dynamic, 512)
    for (int64_t i = 0; i < (int64_t)n * F; i++) g[i] = silu(g[i]) * u[i];
    {{
        static long long calls, elems, differ;
        calls++; elems += (long long)n * F;
        if (memcmp(chk, g, (size_t)n * F * sizeof(float))) {{ differ++; fprintf(stderr, "[check] silu call %lld (n %d) differs{NL}", calls, n); }}
        if (calls % 200 == 0) fprintf(stderr, "[check] silu %lld calls, %lld elements, %lld calls differ{NL}", calls, elems, differ);
        free(chk);
    }}"""
assert src.count(old) == 1
src = src.replace(old, new)
(here / "mimo_check.c").write_text(src, encoding="utf-8")
print("wrote", here / "mimo_check.c")
