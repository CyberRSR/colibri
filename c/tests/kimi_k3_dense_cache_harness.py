#!/usr/bin/env python3
"""K3_DENSE_CACHE (kimi_k3.c, k3dc_*): the load-time quantization kept in a file, on a tiny fixture.

The first start writes every matrix it quantized (int4-g64 / int8 and their scales) to one
file; the next ones read that file instead of the checkpoint. What the model computes may
not depend on which of the two happened. Checked, each against the logits of a start
without the file:

  - write, then read: the first start says it wrote N matrices, the second that it read
    the same N, both give the reference's logits;
  - other bits: K3_BITS=8 with a file written at the default bits takes nothing from it
    (the records' formats are not this start's), drops it, and the start after writes it
    again;
  - a byte changed inside the file: the matrix whose sum no longer holds is quantized from
    the checkpoint and the file is dropped;
  - the checkpoint touched (its time of writing moved): the file is not trusted, and is
    written again by the same start;
  - the file cut short: the same.

Run by path (not a test_*.py: unittest discovery would import it):
  python tests/kimi_k3_dense_cache_harness.py --binary ./kimi_k3 --fixture ./kimi_k3_tiny
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ENV = {"COLI_TEMP": "0", "AUTOPIN": "0", "USAGE_SAVE": "0", "K3_CHUNK": "32"}


def go(binary, fixture, ids, extra):
    env = dict(os.environ, **ENV)
    env.pop("K3_DENSE_CACHE", None)
    env.update(extra)
    with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as tmp:
        path = tmp.name
    try:
        env["K3_LOGITS"] = path
        r = subprocess.run([binary, str(fixture), "--ids", " ".join(map(str, ids)), "--ngen", "4"],
                           env=env, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            raise AssertionError(f"engine exited {r.returncode}:\n{r.stderr[-2000:]}")
        return Path(path).read_bytes(), r.stderr
    finally:
        os.unlink(path)


def said(err, pattern):
    m = re.search(pattern, err)
    return m


def expect(what, got, ref, pattern):
    if got[0] != ref:
        raise AssertionError(f"{what}: the logits are not those of a start without the file")
    m = said(got[1], pattern)
    if not m:
        lines = [l for l in got[1].splitlines() if "dense cache" in l]
        raise AssertionError(f"{what}: no line matching {pattern!r}; the engine said {lines}")
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--binary", required=True)
    ap.add_argument("--fixture", required=True, help="a tiny fixture (K3's or K3.1's)")
    a = ap.parse_args()
    work = Path(tempfile.mkdtemp(prefix="kimi-k3-dc-"))
    try:
        fixture = work / "model"
        shutil.copytree(a.fixture, fixture)          # its own copy: one check moves the shard's time
        cfg = json.loads((fixture / "config.json").read_text())
        vocab = cfg.get("text_config", cfg)["vocab_size"]
        ids = [(5 + 7 * i) % (vocab - 10) + 3 for i in range(96)]
        dc = work / "dense.bin"
        on = {"K3_DENSE_CACHE": str(dc)}
        wrote, read = r"dense cache: (\d+) matrices, [0-9.]+ GB written", r"dense cache: (\d+) matrices, [0-9.]+ GB read"

        ref = go(a.binary, fixture, ids, {})[0]
        if "dense cache" in go(a.binary, fixture, ids, {"K3_DENSE_CACHE": "0"})[1] or dc.exists():
            raise AssertionError("K3_DENSE_CACHE=0 touched the file")
        n = int(expect("the start that writes the file", go(a.binary, fixture, ids, on), ref, wrote).group(1))
        if not dc.exists() or Path(str(dc) + ".tmp").exists():
            raise AssertionError("the file is not under its name after the load")
        got = go(a.binary, fixture, ids, on)
        m = int(expect("the start that reads the file", got, ref, read).group(1))
        if m != n or n < 8 or "written" in got[1]:
            raise AssertionError(f"wrote {n} matrices, read {m}")
        print(f"ok write, then read: {n} matrices written by one start and read by the next, the same logits")

        ref8 = go(a.binary, fixture, ids, {"K3_BITS": "8"})[0]
        if ref8 == ref:
            raise AssertionError("K3_BITS=8 gives the default bits' logits: the check would not bite")
        got = go(a.binary, fixture, ids, dict(on, K3_BITS="8"))
        expect("K3_BITS=8 over a file of the default bits", got, ref8, r"were not in it as this start wants them")
        if dc.exists():
            raise AssertionError("a file of other bits was kept")
        expect("K3_BITS=8, the start after", go(a.binary, fixture, ids, dict(on, K3_BITS="8")), ref8, wrote)
        expect("K3_BITS=8, read", go(a.binary, fixture, ids, dict(on, K3_BITS="8")), ref8, read)
        dc.unlink()
        print("ok other bits: a file of the default bits gives K3_BITS=8 nothing, is dropped and written again")

        expect("write", go(a.binary, fixture, ids, on), ref, wrote)
        raw = bytearray(dc.read_bytes())
        raw[4096 + 100] ^= 0x10                      # inside the first matrix
        dc.write_bytes(raw)
        expect("a byte changed", go(a.binary, fixture, ids, on), ref, r"does not read back as it was written")
        if dc.exists():
            raise AssertionError("a file with a changed byte was kept")
        print("ok a changed byte: the matrix is quantized from the checkpoint, the file dropped, the same logits")

        expect("write", go(a.binary, fixture, ids, on), ref, wrote)
        shard = next(fixture.glob("*.safetensors"))
        st = shard.stat()
        os.utime(shard, (st.st_atime, st.st_mtime + 7))
        got = go(a.binary, fixture, ids, on)
        expect("the checkpoint touched", got, ref, r"the checkpoint changed since it was written")
        expect("the checkpoint touched, the file again", got, ref, wrote)
        expect("read after it", go(a.binary, fixture, ids, on), ref, read)
        print("ok the checkpoint touched: the file is not trusted and the same start writes it again")

        raw = dc.read_bytes()
        dc.write_bytes(raw[:len(raw) - 5000])
        got = go(a.binary, fixture, ids, on)
        expect("the file cut short", got, ref, r"(the file ends inside a matrix|the file is not whole|a record does not read)")
        expect("the file cut short, written again", got, ref, wrote)
        print("ok the file cut short: not trusted, written again, the same logits")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("kimi-k3-dense-cache: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
