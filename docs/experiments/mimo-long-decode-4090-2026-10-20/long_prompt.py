#!/usr/bin/env python3
"""One long prompt through mimo.exe's SERVE pipe on the RTX 4090 profile: time to the first
token, the reply, and the engine's lines. usage: long_prompt.py --engine E --prompt FILE
--tokens N --out DIR [--env K=V ...]"""
import argparse, json, os, pathlib, queue, subprocess, sys, threading, time

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", default="D:/AI/Models/MiMo-V2.6-Pro-MOPD")
    ap.add_argument("--shards", default="C:/AI/MiMoPro-Shards")
    ap.add_argument("--prompt", required=True, action="append")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("COLI_", "MIMO_", "CUDA_"))}
    env.update({"SNAP": a.model, "SERVE": "1", "KV_SLOTS": "1", "CTX": str(a.ctx), "COLI_NO_OMP_TUNE": "1",
                "COLI_KV_PREFIX": "0", "OMP_NUM_THREADS": "16", "MIMO_IO_PER_DRIVE": "2", "MIMO_PREFETCH": "1",
                "MIMO_DENSE_BITS": "0", "MIMO_IDOT": "0", "MIMO_DIRECT": "1", "MIMO_CHUNK": "512",
                "MIMO_DIRS": a.shards, "MIMO_CUDA": "1", "MIMO_CUDA_DENSE_GB": "20", "MIMO_CUDA_RESERVE_GB": "2",
                "MIMO_CUDA_RELEASE_HOST": "1", "MIMO_CUDA_OPROJ_INT8": "0", "COLI_VULKAN": "0", "COLI_GPU": "0"})
    for kv in a.env:
        k, _, v = kv.partition("="); env[k] = v
    log = open(out / "engine.log", "wb")
    p = subprocess.Popen([a.engine, "104"], env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, bufsize=0)
    q = queue.Queue(); threading.Thread(target=reader, args=(p.stdout, q), daemon=True).start()
    while True:
        _, f, _ = q.get()
        if f[0] == "EOF": sys.exit("engine died")
        if "READY" in f[0]: break
    results = []
    for i, path in enumerate(a.prompt):
        text = open(path, encoding="utf-8").read()
        body = (f"<|im_start|>user\n{text}<|im_end|><|im_start|>assistant\n<think></think>").encode()
        t0 = time.perf_counter()
        p.stdin.write(f"SUBMIT {i+1} 0 {len(body)} {a.tokens} 0 1\n".encode() + body + b"\n"); p.stdin.flush()
        first = None; chunks = []; done = None
        while True:
            t, f, pl = q.get()
            if f[0] == "EOF": sys.exit("engine died")
            if f[0] == "DATA":
                first = first or t; chunks.append(pl)
            elif f[0] in ("DONE", "END", "ERROR") or (f and f[0].startswith("DONE")):
                done = (t, f); break
        reply = b"".join(chunks).decode("utf-8", "replace")
        r = {"prompt": path, "ttft_s": (first or done[0]) - t0, "total_s": done[0] - t0, "end": done[1], "reply": reply}
        results.append(r)
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k != "reply"}), flush=True)
    p.stdin.close(); p.wait(timeout=600)
    log.close()
    (out / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    lines = [l for l in open(out / "engine.log", encoding="utf-8", errors="replace") if l.startswith(("[mimo]", "[CUDA]"))]
    (out / "engine.txt").write_text("".join(lines), encoding="utf-8")
    for l in lines:
        if any(x in l for x in ("prompt ", "prefill", "prompt attention", "time:")): print(l.rstrip())

if __name__ == "__main__":
    main()
