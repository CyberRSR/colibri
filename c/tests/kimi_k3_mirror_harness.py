#!/usr/bin/env python3
"""The weights whole on two drives (kimi_k3.c, k3_mirror.h: K3_MIRROR, K3_FILL), on a tiny fixture.

The tiny model is laid out as the published checkpoint is (a shard per layer, an index) in a model
folder, and whole again in a second folder that stands for the other drive (K3_DIRS; K3_MIRROR_ANY=1
lets a folder on the same drive count as one, with readers and queues of its own). Then:

  - mirror: every expert's read is cut in two, a piece from each folder -- at the default share, at
    25% and at 90%, unbuffered and buffered, and with the readers slowed so that the queues fill and
    prefetches are promoted and cancelled; or read whole from the folder whose queue is shorter
    (K3_MIRROR_ROWS: every step's, or the prompt's steps' while the decode's are cut): the logits of
    every position and the tokens are the model folder's alone, and both "drives" read.
  - a copy missing, or differing (another header): that file is read from the model's folder only,
    the others are still cut; the same logits.
  - fill: the model's folder lacks two shards, holds an interrupted copy (<name>.k3fill) and a
    shortened one (its header whole, its end cut off): K3_FILL=1 copies the two, copies the short one
    again and removes the leftover, and the folder's files are then the other's byte for byte; the
    same logits. With K3_FILL_KEEP_GB above the drive's free space nothing is copied and the missing
    shards are read from the other folder: the same logits. A file that differs is left alone.
  - K3_MIRROR=0: no copy is used.

Run by path (not a test_*.py: unittest discovery would import it):
  python tests/kimi_k3_mirror_harness.py --binary ./kimi_k3 --fixture ./kimi_k31_tiny
"""

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

# a cache of 2 experts a layer (the tiny model has 8, of about 1.4 KB): the decode keeps reading
ENV = {"COLI_TEMP": "0", "AUTOPIN": "0", "USAGE_SAVE": "0", "K3_MAXT": "4096", "K3_EXPERT_GB": "0.00002",
       "K3_LOGITS_DECODE": "1", "K3_CHUNK": "16", "K3_PIPE": "1", "K3_IO_PER_DRIVE": "2", "K3_PREFETCH": "1"}
CLEAR = ("K3_DIRS", "K3_MIRROR", "K3_MIRROR_SPLIT", "K3_MIRROR_ANY", "K3_FILL", "K3_FILL_KEEP_GB", "K3_DIRECT",
         "K3_IO_DELAY_US", "K3_IO_THREADS", "K3_LOAD_THREADS", "K3_DRAFT", "K3_DRAFT_ORACLE")
DTYPE_ORDER = ["BOOL", "U8", "I8", "F8_E5M2", "F8_E4M3", "I16", "U16", "F16", "BF16",
               "I32", "U32", "F32", "F64", "I64", "U64"]


def read_st(path):
    with open(path, "rb") as handle:
        (n,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(n))
        header.pop("__metadata__", None)
        base = 8 + n
        out = {}
        for name, spec in header.items():
            a, b = spec["data_offsets"]
            handle.seek(base + a)
            out[name] = (spec["dtype"], spec["shape"], handle.read(b - a))
    return out


def write_st(path, tensors, meta=None):
    """safetensors as the published shards are written: dtype rank descending, then name."""
    order = sorted(tensors, key=lambda t: (-DTYPE_ORDER.index(tensors[t][0]), t))
    header, at = ({"__metadata__": meta} if meta else {}), 0
    for name in order:
        dtype, shape, data = tensors[name]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [at, at + len(data)]}
        at += len(data)
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(text)) + text)
        for name in order:
            handle.write(tensors[name][2])


def lay_out(fixture, root):
    """the fixture as a shard a layer (and one for the rest), an index and the config"""
    tensors = read_st(fixture / "model.safetensors")
    cfg = json.loads((fixture / "config.json").read_text())
    layers = (cfg.get("text_config") or cfg)["num_hidden_layers"]
    n = layers + 1
    name = lambda i: f"model-{i:05d}-of-{n:05d}.safetensors"  # noqa: E731
    groups = {i: {} for i in range(1, n + 1)}
    for t, v in tensors.items():
        groups[int(t.split(".")[2]) + 1 if t.startswith("model.layers.") else n][t] = v
    root.mkdir(parents=True)
    weight_map = {}
    for i, group in groups.items():
        write_st(root / name(i), group)
        weight_map.update({t: name(i) for t in group})
    (root / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}))
    shutil.copy(fixture / "config.json", root / "config.json")
    return [name(i) for i in groups]


def copy_dir(src, dst, skip=()):
    dst.mkdir(parents=True)
    for f in src.iterdir():
        if f.name not in skip:
            shutil.copy(f, dst / f.name)


def run(binary, model, ids, ngen, extra):
    env = {k: v for k, v in os.environ.items() if k not in CLEAR}
    env.update(ENV)
    env.update(extra)
    with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as tmp:
        path = tmp.name
    try:
        env["K3_LOGITS"] = path
        r = subprocess.run([binary, str(model), "--ids", " ".join(map(str, ids)), "--ngen", str(ngen)],
                           env=env, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            raise AssertionError(f"engine exited {r.returncode}:\n{r.stderr[-2000:]}")
        return Path(path).read_bytes(), [int(t) for t in r.stdout.split("TUNE")[0].split()], r.stderr
    finally:
        os.unlink(path)


def same(what, got, ref):
    if got[1] != ref[1] or not got[1]:
        raise AssertionError(f"{what}: tokens {got[1]} against {ref[1]}")
    if got[0] != ref[0]:
        n = next((i for i, (a, b) in enumerate(zip(got[0], ref[0])) if a != b), min(len(got[0]), len(ref[0])))
        raise AssertionError(f"{what}: the logits are not the model folder's alone ({len(got[0])} against "
                             f"{len(ref[0])} bytes, the first difference at byte {n})\n{got[2][-2000:]}")


def mirrored(what, err, want, of, full=True):
    """want of `of` files with a copy; full: every expert file has one, so a decode step's reads are cut in two
    (else every read of a file with a copy goes whole to the shorter queue)"""
    m = re.search(r"\[K3-MIRROR\] (\d+) of (\d+) shard files", err)
    if not m or (int(m.group(1)), int(m.group(2))) != (want, of):
        raise AssertionError(f"{what}: expected {want} of {of} shard files with a copy:\n{err[-2000:]}")
    said = "cut in two" if full else "some files have no copy"
    if said not in err:
        raise AssertionError(f"{what}: the readers' line does not say '{said}':\n{err[-2000:]}")
    # every device the decode read from, and its bytes
    line = re.search(r"\[K3\] reads \(the decode\):(.*)", err)
    if not line:
        raise AssertionError(f"{what}: no '[K3] reads (the decode)' line:\n{err[-2000:]}")
    drives = re.findall(r"(\S+) ([\d.]+) GB in (\d+) reads", line.group(1))
    if len(drives) != 2 or any(int(n) == 0 for _, _, n in drives):
        raise AssertionError(f"{what}: the decode read from {drives}, not from both folders")
    return drives


def check(binary, fixture, vocab, root):
    ids = [(5 + 7 * i) % (vocab - 10) + 3 for i in range(40)]
    ngen = 48
    a = root / "drive-a" / "k3"
    names = lay_out(fixture, a)
    b = root / "drive-b" / "k3"
    copy_dir(a, b)
    ref = run(binary, a, ids, ngen, {})
    if len(ref[1]) < 16 or "K3-MIRROR" in ref[2]:
        raise AssertionError(f"the plain run: {len(ref[1])} tokens{', mentions a copy' if 'K3-MIRROR' in ref[2] else ''}")
    n = len(names)
    two = {"K3_DIRS": str(b), "K3_MIRROR_ANY": "1"}

    got = run(binary, a, ids, ngen, two)
    same("every file on both", got, ref)
    drives = mirrored("every file on both", got[2], n, n)
    print(f"ok mirror: {n} of {n} files with a copy, the decode read from {drives}; the same logits and tokens")
    for what, extra in (("a 25% share", {"K3_MIRROR_SPLIT": "0.25"}), ("a 90% share", {"K3_MIRROR_SPLIT": "0.9"}),
                        ("buffered reads", {"K3_DIRECT": "0"}),
                        ("whole reads by the shorter queue", {"K3_MIRROR_ROWS": "0"}),
                        ("the prompt's steps whole, the decode's cut", {"K3_MIRROR_ROWS": "4"}),
                        ("whole reads, buffered", {"K3_MIRROR_ROWS": "0", "K3_DIRECT": "0"}),
                        ("slow readers", {"K3_IO_DELAY_US": "3000", "K3_IO_PER_DRIVE": "1"}),
                        ("slow readers, whole reads", {"K3_IO_DELAY_US": "3000", "K3_IO_PER_DRIVE": "1", "K3_MIRROR_ROWS": "0"})):
        got = run(binary, a, ids, ngen, dict(two, **extra))
        same(what, got, ref)
        mirrored(what, got[2], n, n)
        print(f"ok mirror, {what}: the same logits and tokens")
    if not re.search(r"prefetch \(run\): \d+ experts read ahead", got[2]):
        raise AssertionError(f"slow readers: no prefetch line:\n{got[2][-1500:]}")

    got = run(binary, a, ids, ngen, dict(two, K3_MIRROR="0"))
    same("K3_MIRROR=0", got, ref)
    if "K3-MIRROR" in got[2] or "cut in two" in got[2]:
        raise AssertionError(f"K3_MIRROR=0 used a copy:\n{got[2][-1500:]}")
    print("ok K3_MIRROR=0: no copy used, the same logits and tokens")

    partial = root / "drive-b" / "partial"
    copy_dir(b, partial, skip={names[2]})
    differ = names[3]
    t = read_st(partial / differ)
    write_st(partial / differ, t, meta={"note": "another header"})
    got = run(binary, a, ids, ngen, {"K3_DIRS": str(partial), "K3_MIRROR_ANY": "1"})
    same("a copy missing, one differing", got, ref)
    mirrored("a copy missing, one differing", got[2], n - 2, n, full=False)
    if not re.search(r"\[K3-MIRROR\] 1 file\(s\) of the same name on another drive differ", got[2]):
        raise AssertionError(f"the differing copy was not named:\n{got[2][-1500:]}")
    print(f"ok partial: {n - 2} of {n} files with a copy (one missing, one differing): every read of a file with one "
          f"whole, by the shorter queue; the same logits and tokens")

    experts = root / "drive-b" / "experts"   # copies of the files with routed experts only (layers 2-5)
    copy_dir(b, experts, skip={names[0], names[1], names[-1]})
    got = run(binary, a, ids, ngen, {"K3_DIRS": str(experts), "K3_MIRROR_ANY": "1"})
    same("copies of the expert files only", got, ref)
    mirrored("copies of the expert files only", got[2], n - 3, n, full=True)
    print(f"ok expert files only: {n - 3} of {n} files with a copy, all of those with experts: the reads cut in two; "
          f"the same logits and tokens")

    fill = root / "drive-a" / "fill"
    lacks = [names[1], names[4]]
    copy_dir(a, fill, skip=set(lacks))
    (fill / (names[0] + ".k3fill")).write_bytes(b"left by an interrupted copy")
    short = names[5]
    data = (fill / short).read_bytes()
    (fill / short).write_bytes(data[:len(data) - 100])
    got = run(binary, fill, ids, ngen, dict(two, K3_FILL="1"))
    same("fill", got, ref)
    for want in [f"copying {x} " for x in lacks] + [f"copying again {short} ", "an interrupted copy"]:
        if want not in got[2]:
            raise AssertionError(f"fill: no '{want}' line:\n{got[2][-2500:]}")
    for x in names:
        if (fill / x).read_bytes() != (b / x).read_bytes():
            raise AssertionError(f"fill: {x} is not the other folder's byte for byte")
    left = list(fill.glob("*.k3fill"))
    if left:
        raise AssertionError(f"fill: left {left}")
    mirrored("fill", got[2], n, n)
    print(f"ok fill: {len(lacks)} shards copied, a short one copied again, a leftover removed; the folder is the other's "
          f"byte for byte; the same logits and tokens")
    got = run(binary, fill, ids, ngen, dict(two, K3_FILL="1"))
    if "copying" in got[2] or not re.search(rf"\[K3-FILL\] .*: {n} of the other drive's {n} model file\(s\) here", got[2]):
        raise AssertionError(f"fill again: expected nothing to copy:\n{got[2][-1500:]}")
    print("ok fill again: nothing to copy")

    got = run(binary, a, ids, ngen, {"K3_DIRS": str(partial), "K3_MIRROR_ANY": "1", "K3_FILL": "1"})
    same("fill beside a differing copy", got, ref)
    if not re.search(rf"\[K3-FILL\] .*{re.escape(differ)} differs from", got[2]) or "[K3-FILL] copying" in got[2]:
        raise AssertionError(f"fill beside a differing copy: not named, or copied:\n{got[2][-1500:]}")
    print("ok fill beside a differing copy: left alone and named")

    keep = root / "drive-a" / "keep"
    copy_dir(a, keep, skip=set(lacks))
    got = run(binary, keep, ids, ngen, dict(two, K3_FILL="1", K3_FILL_KEEP_GB="1000000000"))
    same("no room to fill", got, ref)
    if "not copied" not in got[2] or any((keep / x).exists() for x in lacks):
        raise AssertionError(f"no room to fill: copied anyway, or did not say so:\n{got[2][-1500:]}")
    mirrored("no room to fill", got[2], n - len(lacks), n, full=False)
    print(f"ok no room: nothing copied, {len(lacks)} shards read from the other folder alone; the same logits and tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--fixture", required=True)
    args = ap.parse_args()
    fixture = Path(args.fixture)
    cfg = json.loads((fixture / "config.json").read_text())
    vocab = int((cfg.get("text_config") or cfg)["vocab_size"])
    with tempfile.TemporaryDirectory(prefix="k3mirror-") as tmp:
        check(os.path.abspath(args.binary), fixture, vocab, Path(tmp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
