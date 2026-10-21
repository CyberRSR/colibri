#!/usr/bin/env python3
"""A turns.py run's resumed turns against the same prompts computed cold.

usage: compare.py RUN_DIR [RUN_DIR ...]

For each request named X with a request X-cold in the same run: the logits after the two prompts
(the same bytes, or the largest difference), the replies (the same bytes), the prompt's time and
how it started ([PREFIX]/[PIN]). Prints JSON and writes RUN_DIR/compare.json."""
import hashlib, json, pathlib, struct, sys

VOCAB = 152576


def main():
    for run in map(pathlib.Path, sys.argv[1:]):
        results = json.loads((run / "results.json").read_text(encoding="utf-8"))
        data = (run / "logits.f32").read_bytes()
        width = 4 * VOCAB
        assert len(data) == width * len(results), "one logits record a request"
        rec = {r["name"]: (i, r) for i, r in enumerate(results)}
        out = []
        for name, (i, r) in rec.items():
            if name + "-cold" not in rec:
                continue
            j, c = rec[name + "-cold"]
            a, b = data[i * width:(i + 1) * width], data[j * width:(j + 1) * width]
            row = {"turn": name, "prompt_tokens": r["prompt_tokens"],
                   "resumed": [l for l in r["engine"] if l.startswith("[PIN] resumed")],
                   "prompt_s": r["ttft_s"], "cold_prompt_s": c["ttft_s"],
                   "logits_same_bytes": a == b, "logits_sha256": hashlib.sha256(a).hexdigest(),
                   "reply_same_bytes": r["reply"] == c["reply"],
                   "reply_sha256": hashlib.sha256(r["reply"].encode()).hexdigest()}
            if a != b:
                fa, fb = struct.unpack(f"{VOCAB}f", a), struct.unpack(f"{VOCAB}f", b)
                row["max_abs_logit_difference"] = max(abs(x - y) for x, y in zip(fa, fb))
                row["argmax_same"] = fa.index(max(fa)) == fb.index(max(fb))
            out.append(row)
        (run / "compare.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        print(run.name, json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
