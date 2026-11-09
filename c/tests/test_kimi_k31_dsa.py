#!/usr/bin/env python3
"""Kimi K3.1's sparse attention (dsa_config) in the Kimi K3 engine, on the tiny fixtures.

The K3.1 fixture (tools/make_kimi_k3_tiny.py --dsa) is K3's tiny model byte for byte
plus a lightning indexer on every gated MLA layer, index_topk 16. Checked:

  - oracle: the engine's hidden state after every layer (K3_TRACE, inputs injected
    with K3_X0) against tools/k3_ref.py's independent numpy model, which picks each
    row's rows in float64 and attends them with a mask; rel-L2 below 3e-4 a layer.
    The same reference with dense attention must differ past index_topk positions
    (the comparison bites) and agree before them.
  - steps: a 72-token prompt's logits (K3_LOGITS) are the same bytes in prompt steps
    of 5, 16, 17, 32 and 72 rows (a step of 1 row differs on K3's fixture too: the
    one-row and the many-row kernels sum in other orders, before K3.1 as after).
  - K3_DSA=0 on the K3.1 fixture gives K3's logits, and so does K3.1 itself on a
    prompt of at most index_topk tokens (every row attends all of its positions).
  - a dsa_config the engine does not implement is refused at load.
  - determinism: two runs, the same tokens.

Needs numpy (the oracle), as the fixture generator does.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

ENV = {"K3_BITS": "32", "K3_MLA_BITS": "32", "K3_HEAD_BITS": "32", "K3_IDOT": "0", "COLI_TEMP": "0"}


def run(binary, fixture, ids, ngen, extra=None, check=True):
    env = dict(os.environ, **ENV)
    env.update(extra or {})
    r = subprocess.run([binary, str(fixture), "--ids", " ".join(map(str, ids)), "--ngen", str(ngen)],
                       env=env, capture_output=True, text=True, timeout=600)
    if check and r.returncode != 0:
        raise AssertionError(f"engine exited {r.returncode}:\n{r.stderr[-2000:]}")
    return r


def tokens(r):
    return [int(t) for t in r.stdout.split("TUNE")[0].split()]


def logits(binary, fixture, ids, extra=None):
    with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as tmp:
        path = tmp.name
    try:
        run(binary, fixture, ids, 0, dict(extra or {}, K3_LOGITS=path))
        return Path(path).read_bytes()
    finally:
        os.unlink(path)


def text_config(fixture):
    cfg = json.loads((Path(fixture) / "config.json").read_text())
    return cfg.get("text_config", cfg)


def check_oracle(binary, fixture, work):
    import numpy as np   # here, not at the top: unittest discovery imports this file
    sys.path.insert(0, str(HERE.parent / "tools"))
    import k3_ref
    tc = text_config(fixture)
    D, L, K = tc["hidden_size"], tc["num_hidden_layers"], tc["dsa_config"]["index_topk"]
    T = 40
    rng = np.random.default_rng(31)
    x0 = rng.standard_normal((T + 1, D)).astype(np.float32)   # one row more: the decode step's
    x0_path = work / "x0.bin"
    x0.tofile(x0_path)
    trace = work / "trace.bin"
    run(binary, fixture, [5] * T, 1, {"K3_X0": str(x0_path), "K3_TRACE": str(trace)})
    # chunk forced to 1: per token its L layer rows, then the final norm's row
    c = np.fromfile(trace, dtype=np.float32)[:T * (L + 1) * D].reshape(T, L + 1, D)[:, :L]
    refs = {}
    for name, dense in (("sparse", False), ("dense", True)):
        out = work / f"ref_{name}.bin"
        k3_ref.K3Ref(str(fixture), L, dense=dense).run(x0[:T], str(out))
        refs[name] = np.fromfile(out, dtype=np.float32).reshape(T, L, D)
    worst = 0.0
    for li in range(L):
        r = refs["sparse"][:, li]
        rel = np.linalg.norm(c[:, li] - r) / np.linalg.norm(r)
        worst = max(worst, rel)
        if rel > 3e-4:
            raise AssertionError(f"layer {li}: rel-L2 {rel:.3e} against the sparse reference")
    r = refs["dense"][:, L - 1]
    per = np.linalg.norm(c[:, L - 1] - r, axis=1) / np.linalg.norm(r, axis=1)
    if per[:K].max() > 3e-4:
        raise AssertionError(f"before {K} positions the dense reference differs: {per[:K].max():.3e}")
    if per[K:].max() < 1e-2:
        raise AssertionError(f"past {K} positions the dense reference agrees ({per[K:].max():.3e}): "
                             "the selection is not exercised")
    print(f"ok oracle: {L} layers, worst rel-L2 {worst:.2e}; the dense reference differs by up to "
          f"{per[K:].max():.0%} past position {K}")


def check_steps(binary, fixture):
    ids = [(5 + 7 * i) % 300 + 2 for i in range(72)]
    got = {ch: logits(binary, fixture, ids, {"K3_CHUNK": str(ch)}) for ch in (5, 16, 17, 32, 72)}
    first = got[5]
    for ch, raw in got.items():
        if raw != first:
            raise AssertionError(f"prompt steps of {ch} rows give other logits than steps of 5")
    print(f"ok steps: 72 positions, the same logits in steps of {sorted(got)}")
    return ids


def check_dense_equivalence(binary, fixture, k3_fixture, ids):
    if logits(binary, fixture, ids, {"K3_DSA": "0"}) != logits(binary, k3_fixture, ids):
        raise AssertionError("K3_DSA=0 on the K3.1 fixture is not K3's model")
    K = text_config(fixture)["dsa_config"]["index_topk"]
    short = ids[:K]
    if logits(binary, fixture, short) != logits(binary, k3_fixture, short):
        raise AssertionError(f"a prompt of {K} tokens: K3.1's logits are not K3's")
    if logits(binary, fixture, ids) == logits(binary, k3_fixture, ids):
        raise AssertionError("a prompt past index_topk: K3.1's logits are K3's (no selection?)")
    print(f"ok dense: K3_DSA=0 is K3's model; up to {K} positions K3.1 is K3's to the bit, past them not")


def check_refusals(binary, fixture, work):
    cases = [("index_rope_dim", 64, "index_rope_dim"),
             ("index_k_norm_type", "rmsnorm", "index_k_norm_type"),
             ("index_topk_freq", 2, "index_topk_freq"),
             ("indexer_use_fp8", True, "indexer_use_fp8"),
             ("index_unknown_knob", 1, "index_unknown_knob"),
             ("dsa_layers", [1, 2], "dsa_layers")]   # layer 1 is a KDA layer
    for key, value, word in cases:
        bad = work / f"bad_{key}"
        shutil.copytree(fixture, bad)
        cfg = json.loads((bad / "config.json").read_text())
        cfg["text_config"]["dsa_config"][key] = value
        (bad / "config.json").write_text(json.dumps(cfg))
        r = run(binary, bad, [5, 7, 9], 1, check=False)
        if r.returncode == 0 or word not in r.stderr:
            raise AssertionError(f"dsa_config.{key}={value!r}: exit {r.returncode}, stderr {r.stderr[-300:]!r}")
        shutil.rmtree(bad)
    print(f"ok refusals: {len(cases)} dsa_config values the engine does not implement stop the load")


def check_determinism(binary, fixture, ids):
    a = tokens(run(binary, fixture, ids, 8))
    b = tokens(run(binary, fixture, ids, 8))
    if a != b or len(a) != 8:
        raise AssertionError(f"two runs: {a} vs {b}")
    print("ok determinism: two runs, the same 8 tokens")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--binary", required=True)
    ap.add_argument("--fixture", required=True, help="the K3.1 tiny fixture (make_kimi_k3_tiny.py --dsa)")
    ap.add_argument("--k3-fixture", required=True, help="K3's tiny fixture")
    a = ap.parse_args()
    work = Path(tempfile.mkdtemp(prefix="kimi-k31-dsa-"))
    try:
        check_oracle(a.binary, Path(a.fixture), work)
        ids = check_steps(a.binary, a.fixture)
        check_dense_equivalence(a.binary, a.fixture, a.k3_fixture, ids)
        check_refusals(a.binary, Path(a.fixture), work)
        check_determinism(a.binary, a.fixture, ids)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("kimi-k31-dsa: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
