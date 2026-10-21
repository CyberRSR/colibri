#!/usr/bin/env python3
"""The agent-loop runs (api_run.py) of the series folders: each turn's time to the first streamed
token and to the end, its prompt tokens, how its prompt started (the engine's [PREFIX] and [PIN]
lines in the gateway's log), the engine's prompt line, and a hash of what the turn returned
(reasoning, content and tool calls), into summary.json beside the series and a table on stdout.

usage: summarize.py SERIES_DIR [SERIES_DIR ...]"""
import hashlib, json, pathlib, sys

rows = []
for series in map(pathlib.Path, sys.argv[1:]):
    for run in sorted(p for p in series.iterdir() if (p / "turns.json").exists()):
        turns = json.loads((run / "turns.json").read_text(encoding="utf-8"))["turns"]
        log = (run / "gateway.log").read_text(encoding="utf-8", errors="replace").splitlines()
        # one [PREFIX] line and one "[mimo] prompt" line a request, in order; [PIN] lines between
        starts, prompts, k = [[] for _ in turns], [], -1
        for line in log:
            if line.startswith("[PREFIX]") and "prompt tokens" in line:
                k += 1
            if 0 <= k < len(turns) and line.startswith(("[PREFIX]", "[PIN]")):
                starts[k].append(line.strip())
            if line.startswith("[mimo] prompt ") and "(stalled" in line:   # not the gateway's lines interleaved
                prompts.append(line.strip())
        for k, t in enumerate(turns):
            returned = t["reasoning"] + "\x00" + t["content"] + "\x00" + json.dumps(t["tool_calls"])
            rows.append({"series": series.name, "run": run.name, "turn": t["name"], "ttft_s": t["ttft_s"],
                         "total_s": t["total_s"], "prompt_tokens": (t.get("usage") or {}).get("prompt_tokens"),
                         "completion_tokens": (t.get("usage") or {}).get("completion_tokens"),
                         "started": starts[k], "engine_prompt": prompts[k] if k < len(prompts) else None,
                         "returned_sha256": hashlib.sha256(returned.encode()).hexdigest()})
out = pathlib.Path(sys.argv[1]).resolve().parent / "summary.json"
out.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
for r in rows:
    how = next((s[6:] for s in r["started"] if s.startswith("[PIN] resumed")), None) or \
          next((s[9:] for s in r["started"] if s.startswith("[PREFIX]")), "")
    print(f"{r['series']:8} {r['run']:14} {r['turn']:14} first token {r['ttft_s']:7.2f} s  end {r['total_s']:7.2f} s  "
          f"prompt {r['prompt_tokens']:>6}  {r['returned_sha256'][:12]}  {how}")
