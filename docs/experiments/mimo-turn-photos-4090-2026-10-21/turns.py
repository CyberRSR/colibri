#!/usr/bin/env python3
"""Chat turns through mimo.exe's SERVE pipe on the RTX 4090 profile, one engine for a whole plan.

usage: turns.py --engine E --plan PLAN.json --out DIR [--env K=V ...]

A plan is {"requests": [{"name", "messages" | "text", "thinking": true, "tokens": N}, ...]}:
messages are rendered as the gateway renders them (openai_server.render_chat_mimo), with the
generation cue, so a request is the prompt an API client would cause. Each request's reply
(greedy), its time to the first token and in all, the engine's DONE line and the engine's lines
while it ran ([PREFIX], [PIN], [mimo] ...) go to DIR/results.json; the logits after each prompt
to DIR/logits.f32 (MIMO_PROMPT_LOGITS: one record of the vocabulary's width a request)."""
import argparse, json, os, pathlib, queue, subprocess, sys, threading, time

C = pathlib.Path(__file__).resolve().parents[3] / "c"
sys.path.insert(0, str(C))
from openai_server import render_chat_mimo  # noqa: E402


def reader(pipe, q):
    try:
        while line := pipe.readline():
            f = line.decode("utf-8", "replace").strip().split()
            payload = b""
            if f and f[0] in ("DATA", "ECHO", "TOOL", "DECISION"):
                n = int(f[2]); chunks = []
                while n:
                    b = pipe.read(n); chunks.append(b); n -= len(b)
                payload = b"".join(chunks); pipe.read(1)
            q.put((time.perf_counter(), f, payload))
    finally:
        q.put((time.perf_counter(), ["EOF"], b""))


def render(req):
    if "text" in req:
        return req["text"]
    return render_chat_mimo(req["messages"], enable_thinking=req.get("thinking", True),
                            tools=req.get("tools"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", default="D:/AI/Models/MiMo-V2.6-Pro-MOPD")
    ap.add_argument("--shards", default="C:/AI/MiMoPro-Shards")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(pathlib.Path(a.plan).read_text(encoding="utf-8"))
    logits = out / "logits.f32"
    if logits.exists():
        logits.unlink()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("COLI_", "MIMO_", "CUDA_"))}
    # the RTX 4090 profile (c/tools/start_mimo_cuda_4090.ps1), prefix reuse on as served
    env.update({"SNAP": a.model, "SERVE": "1", "KV_SLOTS": "1", "CTX": str(a.ctx), "COLI_NO_OMP_TUNE": "1",
                "OMP_NUM_THREADS": "16", "MIMO_IO_PER_DRIVE": "2", "MIMO_PREFETCH": "1",
                "MIMO_DENSE_BITS": "0", "MIMO_IDOT": "0", "MIMO_DIRECT": "1", "MIMO_DIRS": a.shards,
                "MIMO_CUDA": "1", "MIMO_CUDA_DENSE_GB": "20", "MIMO_CUDA_RESERVE_GB": "2",
                "MIMO_CUDA_RELEASE_HOST": "1", "MIMO_CUDA_OPROJ_INT8": "0", "COLI_VULKAN": "0", "COLI_GPU": "0",
                "COLI_PREFIX_LOG": "1", "MIMO_PROMPT_LOGITS": str(logits)})
    for kv in a.env:
        k, _, v = kv.partition("=")
        if v:
            env[k] = v
        else:
            env.pop(k, None)
    log_path = out / "engine.log"
    log = open(log_path, "wb")
    p = subprocess.Popen([a.engine, "104"], env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                         bufsize=0)
    q = queue.Queue(); threading.Thread(target=reader, args=(p.stdout, q), daemon=True).start()
    while True:
        _, f, _ = q.get()
        if f[0] == "EOF": sys.exit("engine died before READY")
        if "READY" in f[0]: break
    results = []
    for i, req in enumerate(plan["requests"]):
        body = render(req).encode("utf-8")
        at = log_path.stat().st_size
        t0 = time.perf_counter()
        p.stdin.write(f"SUBMIT {i+1} 0 {len(body)} {req.get('tokens', 16)} 0 1\n".encode() + body + b"\n")
        p.stdin.flush()
        first = None; chunks = []; done = None; accept = None
        while True:
            t, f, pl = q.get()
            if f[0] == "EOF": sys.exit("engine died:\n" + log_path.read_text(errors="replace")[-3000:])
            if f[0] == "ACCEPT":
                accept = int(f[2])
            elif f[0] == "DATA":
                first = first or t; chunks.append(pl)
            elif f[0] in ("DONE", "ERROR"):
                done = (t, f); break
        time.sleep(0.2)   # the request's report lines follow its DONE
        lines = log_path.read_bytes()[at:].decode("utf-8", "replace").splitlines()
        reply = b"".join(chunks).decode("utf-8", "replace")
        r = {"name": req["name"], "prompt_tokens": accept, "ttft_s": round((first or done[0]) - t0, 3),
             "total_s": round(done[0] - t0, 3), "end": done[1], "reply": reply,
             "engine": [l for l in lines if l.startswith(("[PREFIX]", "[PIN]", "[mimo] prompt", "[mimo] MTP",
                                                           "[mimo] streaming", "[mimo] experts"))]}
        results.append(r)
        print(json.dumps({k: r[k] for k in ("name", "prompt_tokens", "ttft_s", "total_s")}),
              *[l for l in r["engine"] if l.startswith(("[PREFIX]", "[PIN]", "[mimo] prompt"))], sep="\n  ",
              flush=True)
    p.stdin.close(); p.wait(timeout=600)
    log.close()
    (out / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
