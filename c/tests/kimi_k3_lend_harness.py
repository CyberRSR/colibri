#!/usr/bin/env python3
"""What the expert cache lends (kimi_k3.c, k3_cache_lent_sync), on a tiny fixture.

The RAM plan gives the routed experts what the weights, a small step's activations and the
caches of K3_KV_MIN positions leave. A conversation's rows past that and a large prompt
step's activations take their bytes out of the cache: that many slots are retired, the
coldest of every layer in turn, and a step's come back when it ends. None of it may move a
number -- which expert is in RAM only decides what is read again. Checked:

  - rows: a 40-token prompt and 300 tokens of decode with the caches grown from the
    prompt's positions (K3_KV_GEN=1: they double three times; the plan counted 64,
    K3_KV_MIN) and a cache of 8 slots a layer: the logits of every position and the
    tokens are those of the whole caches allocated at once with nothing lent
    (K3_KV_GROW=0, K3_LEND=0); the cache lent more slots as the rows grew, and K3_CHECK
    found every expert's bytes where it left them.
  - step: a 1,200-token prompt in one step (past 512 rows a step borrows its activations):
    the same logits and tokens as with K3_LEND=0, and the step said what it borrowed.
  - back: the same prompt in two steps of 600 rows: both borrow the same number of slots,
    so the first step's came back when it ended.

Run by path (not a test_*.py: unittest discovery would import it):
  python tests/kimi_k3_lend_harness.py --binary ./kimi_k3 --fixture ./kimi_k31_tiny
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ENV = {"COLI_TEMP": "0", "AUTOPIN": "0", "USAGE_SAVE": "0", "K3_MAXT": "4096", "K3_EXPERT_GB": "0.002"}


def go(binary, fixture, ids, ngen, extra):
    env = dict(os.environ, **ENV)
    env.update(extra)
    with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as tmp:
        path = tmp.name
    try:
        env["K3_LOGITS"] = path
        r = subprocess.run([binary, str(fixture), "--ids", " ".join(map(str, ids)), "--ngen", str(ngen)],
                           env=env, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            raise AssertionError(f"engine exited {r.returncode}:\n{r.stderr[-2000:]}")
        return Path(path).read_bytes(), r.stdout.split("TUNE")[0].split(), r.stderr
    finally:
        os.unlink(path)


def same(what, got, ref):
    if got[0] != ref[0]:
        raise AssertionError(f"{what}: the logits are not the reference's ({len(got[0])} against {len(ref[0])} bytes)")
    if got[1] != ref[1] or not got[1]:
        raise AssertionError(f"{what}: tokens {got[1]} against {ref[1]}")


def check_rows(binary, fixture, vocab):
    ids = [(5 + 7 * i) % (vocab - 10) + 3 for i in range(40)]
    ref = go(binary, fixture, ids, 300, {"K3_CHUNK": "24", "K3_KV_GROW": "0", "K3_LEND": "0"})
    got = go(binary, fixture, ids, 300, {"K3_CHUNK": "24", "K3_KV_MIN": "64", "K3_KV_GEN": "1", "K3_CHECK": "1"})
    same("the caches grown from the prompt's positions", got, ref)
    grows = re.findall(r"\[K3\] KV: (\d+) of the \d+ positions", got[2])
    lent = [int(n) for n in re.findall(r"the expert cache lends (\d+) slots for them", got[2])]
    if len(grows) < 3 or len(lent) < 2 or lent != sorted(lent) or lent[0] < 1 or lent[-1] <= lent[0]:
        raise AssertionError(f"grows {grows}, lent {lent}:\n{got[2][-1500:]}")
    if "[K3-CHECK] on" not in got[2] or re.search(r"\[K3-CHECK\] [0-9.]+s ", got[2]):
        raise AssertionError(f"K3_CHECK found a slot whose bytes changed:\n{got[2][-1500:]}")
    if "lends" in ref[2]:
        raise AssertionError("K3_LEND=0 lent slots")
    print(f"ok rows: 340 positions' logits and 300 tokens the same with the caches grown to {grows[-1]} positions "
          f"in {len(grows)} steps, {lent} slots lent for them, K3_CHECK clean")


def borrowed(err):
    return [int(n) for n in re.findall(r"a step of \d+ rows borrows [0-9.]+ GB for its activations: the expert cache "
                                       r"lends (\d+) slots", err)]


def check_step(binary, fixture, vocab):
    ids = [(11 + 5 * i) % (vocab - 10) + 3 for i in range(1200)]
    ref = go(binary, fixture, ids, 8, {"K3_CHUNK": "2048", "K3_LEND": "0"})
    got = go(binary, fixture, ids, 8, {"K3_CHUNK": "2048"})
    same("a step of 1,200 rows", got, ref)
    n = borrowed(got[2])
    if len(n) != 1 or n[0] < 1 or borrowed(ref[2]):
        raise AssertionError(f"borrowed {n}, with K3_LEND=0 {borrowed(ref[2])}:\n{got[2][-1500:]}")
    print(f"ok step: a 1,200-row step borrowed {n[0]} slots; the logits and the 8 tokens after it are K3_LEND=0's")
    two = go(binary, fixture, ids, 8, {"K3_CHUNK": "600"})
    n2 = borrowed(two[2])
    if len(n2) != 2 or n2[0] < 1 or n2[0] != n2[1]:
        raise AssertionError(f"two steps of 600 rows borrowed {n2}: the first step's slots did not come back")
    if two[1] != ref[1]:
        raise AssertionError(f"two steps of 600 rows: tokens {two[1]} against {ref[1]}")
    print(f"ok back: two steps of 600 rows borrowed {n2[0]} slots each (the first step's came back), the same tokens")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--binary", required=True)
    ap.add_argument("--fixture", required=True, help="a tiny fixture (K3's or K3.1's)")
    a = ap.parse_args()
    cfg = json.loads((Path(a.fixture) / "config.json").read_text())
    vocab = cfg.get("text_config", cfg)["vocab_size"]
    check_rows(a.binary, a.fixture, vocab)
    check_step(a.binary, a.fixture, vocab)
    print("kimi-k3-lend: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
