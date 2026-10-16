#!/usr/bin/env python3
"""Drive mimo.exe over its SERVE pipe with a list of prompts; print per-request timing.

usage: drive_mimo.py --engine E --model M --shards S --cap N --tokens T [--env K=V ...] prompts.txt
prompts.txt: one user message per line (rendered with the ChatML non-thinking cue)."""
import argparse, json, os, subprocess, sys, threading, time, queue

def render(msg):
    return f"<|im_start|>user\n{msg}<|im_end|><|im_start|>assistant\n<think></think>"

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
    ap.add_argument("--engine", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--shards", default=""); ap.add_argument("--cap", type=int, default=104)
    ap.add_argument("--tokens", type=int, default=64); ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--env", action="append", default=[]); ap.add_argument("--log", default="engine.log")
    ap.add_argument("--json", default=""); ap.add_argument("prompts")
    ap.add_argument("--temperature", type=float, default=0.0); ap.add_argument("--top-p", type=float, default=1.0)
    a = ap.parse_args()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("COLI_", "MIMO_", "CUDA_"))}
    env.update({"SNAP": a.model, "SERVE": "1", "KV_SLOTS": "1", "CTX": "16384", "COLI_NO_OMP_TUNE": "1",
                "COLI_KV_PREFIX": "0", "OMP_NUM_THREADS": str(a.threads), "MIMO_READ_THREADS": "16",
                "MIMO_DENSE_BITS": "0", "MIMO_IDOT": "0", "MIMO_DIRECT": "1", "MIMO_CHUNK": "64",
                "MIMO_DIRS": a.shards, "MIMO_CUDA": "1", "MIMO_CUDA_DENSE_GB": "20", "MIMO_CUDA_RESERVE_GB": "2",
                "MIMO_CUDA_RELEASE_HOST": "1", "COLI_VULKAN": "0"})
    for kv in a.env:
        k, _, v = kv.partition("="); env[k] = v
    prompts = [l.strip() for l in open(a.prompts, encoding="utf-8") if l.strip()]
    log = open(a.log, "wb")
    p = subprocess.Popen([a.engine, str(a.cap)], env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, bufsize=0)
    q = queue.Queue(); threading.Thread(target=reader, args=(p.stdout, q), daemon=True).start()
    while True:
        _, f, _ = q.get()
        if f[0] == "EOF": sys.exit("engine died")
        if "READY" in f[0]: break
    out = []
    for i, msg in enumerate(prompts):
        body = render(msg).encode()
        t0 = time.perf_counter()
        p.stdin.write(f"SUBMIT {i+1} 0 {len(body)} {a.tokens} {a.temperature:.8g} {a.top_p:.8g}\n".encode() + body + b"\n"); p.stdin.flush()
        first = None; chunks = []; done = prof = None
        while True:
            t, f, pl = q.get()
            if f[0] == "EOF": sys.exit("engine died")
            if f[0] == "DATA":
                first = first or t; chunks.append(pl)
            elif f[0] == "DONE": done = f; tend = t
            elif f[0] == "PROF" and done: prof = f; break
            elif f[0] == "ERROR": sys.exit(" ".join(f))
        n = int(done[3])
        rec = {"i": i, "tokens": n, "seconds": tend - t0, "ttft": first - t0,
               "decode_tok_s": (n - 1) / (tend - first) if n > 1 else 0, "hit": float(done[5]),
               "prof": prof[1:], "text": b"".join(chunks).decode("utf-8", "replace")}
        out.append(rec)
        print(f"{i}: {n} tok in {rec['seconds']:.1f}s (ttft {rec['ttft']:.1f}s, decode {rec['decode_tok_s']:.3f} tok/s, hit {rec['hit']}%) "
              f"disk {prof[4]} expert {prof[6]} attn {prof[7]}", flush=True)
    p.stdin.close(); p.wait(timeout=120); log.close()
    if a.json: json.dump(out, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    tot_tok = sum(r["tokens"] for r in out); tot_s = sum(r["seconds"] for r in out)
    dec = [r["decode_tok_s"] for r in out]
    print(f"overall {tot_tok/tot_s:.4f} tok/s, mean decode {sum(dec)/len(dec):.4f} tok/s over {len(out)} prompts")

if __name__ == "__main__":
    main()
