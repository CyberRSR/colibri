#!/usr/bin/env python3
"""One engine behind the gateway on the RTX 4090 profile, as START-MIMO serves it, and agent_loop.py
against it; then the gateway is stopped.

usage: api_run.py --engine E --mode drop|keep --out DIR [--tokens N] [--env K=V ...]

The gateway is this checkout's (c/coli serve, c/openai_server.py) with COLI_ENGINE=E and the
environment of c/tools/start_mimo_cuda_4090.ps1, plus COLI_PREFIX_LOG=1. DIR gets the gateway's
log (the engine's lines in it), turns.json and the card's telemetry every 5 s (gpu.csv)."""
import argparse, os, pathlib, subprocess, sys, time, urllib.request

HERE = pathlib.Path(__file__).resolve().parent
C = HERE.parents[2] / "c"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--mode", choices=("drop", "keep"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--env", action="append", default=[])
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("COLI_", "MIMO_", "CUDA_"))}
    env.update({"MIMO_DIRS": r"C:\AI\MiMoPro-Shards", "MIMO_CUDA": "1", "MIMO_CUDA_DENSE_GB": "20",
                "MIMO_CUDA_OPROJ_INT8": "0", "MIMO_CUDA_RESERVE_GB": "2", "MIMO_CUDA_RELEASE_HOST": "1",
                "MIMO_CTX": "1048576", "MIMO_DENSE_BITS": "0", "MIMO_IDOT": "0", "MIMO_DIRECT": "1",
                "MIMO_IO_PER_DRIVE": "2", "MIMO_PREFETCH": "1", "OMP_NUM_THREADS": "16", "COLI_NO_OMP_TUNE": "1",
                "COLI_VULKAN": "0", "COLI_GPU": "0", "COLI_ENGINE": str(pathlib.Path(a.engine).resolve()),
                "COLI_PREFIX_LOG": "1"})
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v
    log = open(out / "gateway.log", "wb")
    gw = subprocess.Popen([sys.executable, str(C / "coli"), "serve", "--model", r"D:\AI\Models\MiMo-V2.6-Pro-MOPD",
                           "--cap", "104", "--ram", "174", "--ctx", "1048576", "--ngen", "65536",
                           "--host", "127.0.0.1", "--port", str(a.port), "--model-id", "mimo-v2.6-pro-mopd",
                           "--kv-slots", "1", "--queue-timeout", "7200", "--max-queue", "8"],
                          env=env, stdout=log, stderr=subprocess.STDOUT)
    gpu = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,clocks.sm,power.draw,utilization.gpu,memory.used",
                            "--format=csv", "-l", "5"], stdout=open(out / "gpu.csv", "wb"))
    try:
        deadline = time.time() + 900
        while True:
            if gw.poll() is not None:
                sys.exit("the gateway exited:\n" + (out / "gateway.log").read_text(errors="replace")[-3000:])
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2) as r:
                    if b'"ok"' in r.read():
                        break
            except Exception:
                pass
            if time.time() > deadline:
                sys.exit("the gateway never became healthy")
            time.sleep(2)
        code = subprocess.call([sys.executable, str(HERE / "agent_loop.py"), "--mode", a.mode, "--tokens",
                                str(a.tokens), "--url", f"http://127.0.0.1:{a.port}/v1/chat/completions",
                                "--out", str(out / "turns.json")])
        if code:
            sys.exit(f"agent_loop.py exited {code}")
    finally:
        gpu.terminate()
        # the engine is the gateway's child: it ends when its stdin closes with the gateway; its
        # pid is taken first so that the next run never starts beside it
        kids = subprocess.run(["powershell", "-NoProfile", "-Command",
                               f"(Get-CimInstance Win32_Process -Filter 'ParentProcessId={gw.pid}').ProcessId"],
                              capture_output=True, text=True).stdout.split()
        gw.terminate()
        try:
            gw.wait(timeout=120)
        except subprocess.TimeoutExpired:
            gw.kill()
        for pid in kids:
            for _ in range(60):
                alive = subprocess.run(["powershell", "-NoProfile", "-Command", f"Get-Process -Id {pid} -ErrorAction SilentlyContinue"],
                                       capture_output=True, text=True).stdout.strip()
                if not alive:
                    break
                time.sleep(2)
            else:
                subprocess.call(["taskkill", "/F", "/PID", pid], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log.close()


if __name__ == "__main__":
    main()
