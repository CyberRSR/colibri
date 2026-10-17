"""summary.json for the OpenMP spin shim's re-measure on d5acaa4d: the ABA series' arms
(c/build/omp-spin2/series), with each arm's phases, streaming counters and MTP drafts."""
import json, re, statistics, sys
from pathlib import Path

S = Path(sys.argv[1] if len(sys.argv) > 1 else "c/build/omp-spin2/series")
GOOD = "0305abcddf3160f6d54f739703f5c711ecb335d22a76e9dac2397e22221a06be"
REF = json.loads(Path("docs/experiments/mimo-mtp-4090-2026-10-17/varied-prompts/mtp.json").read_text(encoding="utf-8"))
MTP = re.compile(r"\[mimo\] MTP: (\d+) steps of up to (\d+) drafts, (\d+) kept \(([^)]*)\), ([\d.]+) tokens a step, ([\d.]+)s drafting")
GEN = re.compile(r"prompt ([\d.]+)s \(stalled ([\d.]+)s, experts ([\d.]+)s, attention ([\d.]+)s, (\d+) demand reads\); "
                 r"generation ([\d.]+)s \(stalled ([\d.]+)s, experts ([\d.]+)s, attention ([\d.]+)s, (\d+) demand reads, (\d+) prefetched\)")
STR = re.compile(r"streaming: (\d+) demand reads, (\d+) prefetched \((\d+) used, (\d+) unused(?:, (\d+) (?:of them )?cancelled unread)?\), "
                 r"predicted ([\d.]+)% of (\d+) choices, ([\d.]+) GB read, ([\d.]+)s stalled")
SPIN = "[OMP] spinning team"


def streaming(m):
    return {"demand_reads": int(m[0]), "prefetched": int(m[1]), "used": int(m[2]), "unused": int(m[3]),
            "cancelled_unread": int(m[4] or 0), "predicted_pct": float(m[5]), "gb_read": float(m[7]), "stalled_s": float(m[8])}


def phases(g):
    p = dict(zip(("prompt_s", "prompt_stalled_s", "prompt_experts_s", "prompt_attention_s", "prompt_demand_reads",
                  "generation_s", "stalled_s", "experts_s", "attention_s", "demand_reads", "prefetched"), map(float, g)))
    p["generation_minus_stalled_s"] = round(p["generation_s"] - p["stalled_s"], 2)
    return p


def bench(d):
    r = json.loads((d / "result.json").read_text())
    log = (d / "engine.log").read_text(errors="replace")
    meas = [s for s in r["samples"] if not s.get("warmup")]
    mtp, gen, st = MTP.findall(log), GEN.findall(log), STR.findall(log)
    return {"engine_sha256": r["engine_sha256"], "spinning_team": SPIN in log,
            "median_tok_s": round(r["median_tok_s"], 4), "median_decode_tok_s": round(r["median_decode_tok_s"], 4),
            "tok_s": [round(s["tok_s"], 4) for s in meas], "decode_tok_s": [round(s["decode_tok_s"], 4) for s in meas],
            "replies_0305abcd": all(s["reply_sha256"] == GOOD for s in r["samples"]),
            "measured_generation_minus_stalled_s": [phases(g)["generation_minus_stalled_s"] for g in gen[1:]],
            "measured_experts_s": [float(g[7]) for g in gen[1:]], "measured_attention_s": [float(g[8]) for g in gen[1:]],
            "last_request": dict(phases(gen[-1]), streaming=streaming(st[-1]), mtp_steps=int(mtp[-1][0]),
                                 kept=mtp[-1][3], drafting_s=float(mtp[-1][5]))}


def varied(name, ref):
    recs = json.loads((S / f"{name}.json").read_text(encoding="utf-8"))
    log = (S / f"{name}-engine.txt").read_text(errors="replace")
    kept = []
    for m in MTP.findall(log):
        for i, (a, b) in enumerate(re.findall(r"(\d+) of (\d+)", m[3])):
            while len(kept) <= i: kept.append([0, 0])
            kept[i][0] += int(a); kept[i][1] += int(b)
    tok, sec = sum(r["tokens"] for r in recs), sum(r["seconds"] for r in recs)
    gen, st = [phases(g) for g in GEN.findall(log)], [streaming(m) for m in STR.findall(log)]
    tot = lambda k: round(sum(g[k] for g in gen), 2)
    return {"spinning_team": SPIN in log,
            "overall_tok_s": round(tok / sec, 4), "mean_decode_tok_s": round(statistics.mean(r["decode_tok_s"] for r in recs), 4),
            "decode_tok_s": [round(r["decode_tok_s"], 3) for r in recs], "seconds": round(sec, 2), "tokens": tok,
            "generation_s": tot("generation_s"), "stalled_s": tot("stalled_s"),
            "generation_minus_stalled_s": round(tot("generation_s") - tot("stalled_s"), 2),
            "experts_s": tot("experts_s"), "attention_s": tot("attention_s"),
            "prompt_s": tot("prompt_s"), "prompt_stalled_s": tot("prompt_stalled_s"),
            "reads": {k: sum(s[k] for s in st) for k in ("demand_reads", "prefetched", "used", "unused", "cancelled_unread")},
            "gb_read": round(sum(s["gb_read"] for s in st), 2),
            "replies_equal_reference": sum(a["text"] == b["text"] for a, b in zip(recs, ref)),
            "drafting_s": round(sum(float(m[5]) for m in MTP.findall(log)), 2), "kept": kept}


def pct(a, b):
    return round(100 * (a / b - 1), 2)


def summary():
    g = {n: varied(n, REF) for n in ("g-before-1", "g-after", "g-before-2")}
    b = {n: bench(S / n) for n in ("b-before-1", "b-after", "b-before-2")}
    mb = lambda d, a, k: statistics.mean([d[a + "-before-1"][k], d[a + "-before-2"][k]])
    gl = {n: b[n]["last_request"]["generation_minus_stalled_s"] for n in b}
    return {"order": ["g-before-1", "g-after", "g-before-2", "b-before-1", "b-after", "b-before-2"],
            "varied_greedy": g,
            "varied_greedy_after_vs_mean_before_pct": {
                "overall": pct(g["g-after"]["overall_tok_s"], mb(g, "g", "overall_tok_s")),
                "mean_decode": pct(g["g-after"]["mean_decode_tok_s"], mb(g, "g", "mean_decode_tok_s")),
                "generation_minus_stalled": pct(g["g-after"]["generation_minus_stalled_s"], mb(g, "g", "generation_minus_stalled_s"))},
            "benchmark": b,
            "benchmark_after_vs_mean_before_pct": {
                "overall": pct(b["b-after"]["median_tok_s"], mb(b, "b", "median_tok_s")),
                "decode": pct(b["b-after"]["median_decode_tok_s"], mb(b, "b", "median_decode_tok_s")),
                "last_generation_minus_stalled": pct(gl["b-after"], statistics.mean([gl["b-before-1"], gl["b-before-2"]]))}}


if __name__ == "__main__":
    print(json.dumps(summary(), indent=2))
