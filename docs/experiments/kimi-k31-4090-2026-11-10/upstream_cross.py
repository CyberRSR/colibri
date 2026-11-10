"""Our engine's per-layer hidden states on the tiny K3.1 fixture against UPSTREAM's numpy reference
(JustVugg/colibri dev, tools/k3_ref.py: its indexer written from the definition, apart from both engines)."""
import json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import k3_ref_upstream as ref
binary, fixture = sys.argv[1], Path(sys.argv[2])
cfg = json.loads((fixture / "config.json").read_text()); tc = cfg.get("text_config", cfg)
D, L, K = tc["hidden_size"], tc["num_hidden_layers"], tc["dsa_config"]["index_topk"]
T = 40
x0 = np.random.default_rng(31).standard_normal((T + 1, D)).astype(np.float32)
work = Path(tempfile.mkdtemp(prefix="k31-cross-"))
x0.tofile(work / "x0.bin")
env = dict(os.environ, K3_BITS="32", K3_MLA_BITS="32", K3_HEAD_BITS="32", K3_IDOT="0", COLI_TEMP="0",
           K3_X0=str(work / "x0.bin"), K3_TRACE=str(work / "trace.bin"), AUTOPIN="0", USAGE_SAVE="0")
r = subprocess.run([binary, str(fixture), "--ids", " ".join(["5"] * T), "--ngen", "1"], env=env, capture_output=True, text=True)
assert r.returncode == 0, r.stderr[-800:]
c = np.fromfile(work / "trace.bin", dtype=np.float32)[:T * (L + 1) * D].reshape(T, L + 1, D)[:, :L]
ref.K3Ref(str(fixture), L).run(x0[:T], str(work / "ref.bin"))
u = np.fromfile(work / "ref.bin", dtype=np.float32).reshape(T, L, D)
worst = 0.0
for li in range(L):
    rel = float(np.linalg.norm(c[:, li] - u[:, li]) / np.linalg.norm(u[:, li])); worst = max(worst, rel)
    print(f"layer {li}: rel-L2 {rel:.2e}")
per = np.linalg.norm(c[:, L - 1] - u[:, L - 1], axis=1) / np.linalg.norm(u[:, L - 1], axis=1)
print(f"worst layer rel-L2 {worst:.2e}; last layer, rows past index_topk={K}: max {per[K:].max():.2e}")
