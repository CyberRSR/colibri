#!/usr/bin/env python3
"""The photo of the system prompt's end (kimi_k3.c, K3_ROOT_PHOTO), on a tiny fixture.

Every conversation of one client begins with the same instructions. The turns' photos
(COLI_K3_CKPT) are all past the first user message, where two conversations already differ,
so a new conversation read the instructions again. With K3_ROOT_PHOTO (the default once
photos are on) the state at the end of the leading system records is photographed inside
the step that crosses it -- the KDA layers' calls made in two there -- and kept. Checked:

  - cut: a prompt's logits (K3_LOGITS) are the same bytes with the KDA layers' calls of a
    step cut at several rows (K3_SNAP_TEST) as uncut, in one step and in steps of 32.
  - root: through the serve protocol, chat payloads with a system message. The first
    conversation says where it photographed the system prompt's end; a second conversation
    with the same system message restores exactly that position and computes only what
    follows; so does a third, after the turns' photos have gone round the other slots.
  - another client's system message takes nothing from that photo and gets its own; the
    first client's next conversation then reads its system message once more (the other
    conversation took the caches' rows: one system prompt at a time) and answers the same.
  - identity: every reply is the bytes of the same session with K3_ROOT_PHOTO=0, and of one
    with no photos at all.

Run by path (not a test_*.py: unittest discovery would import it):
  python tests/kimi_k3_root_harness.py --binary ./kimi_k3 --fixture ./kimi_k3_tiny
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_kimi_k3_ckpt import Serve   # noqa: E402  the SUBMIT/ACCEPT/DATA/DONE client

SYS_A = ("You are a careful assistant. Answer briefly and precisely. " * 6 +
         "Use the tools only when asked. Never repeat the question. " * 5).strip()
SYS_B = ("You translate between languages and say nothing else. " * 9).strip()
MAX_NEW = 24


def chat(system, turns, thinking=0):
    """K3CHAT1: M <role> <bytes>\\n<text> records, then the generation cue."""
    out = b"K3CHAT1\n"
    for role, text in [("system", system)] + turns:
        raw = text.encode("utf-8")
        out += f"M {role} {len(raw)}\n".encode("ascii") + raw
    return (out + f"G {thinking}\n".encode("ascii")).decode("utf-8")


def session(binary, model, env, stderr_path):
    serve = Serve(binary, model, env, stderr_path)
    try:
        r = []
        r.append(serve.submit(chat(SYS_A, [("user", "Name one colour.")]), MAX_NEW))                 # 1: conversation A
        r.append(serve.submit(chat(SYS_A, [("user", "Name one metal, please.")]), MAX_NEW))          # 2: a new one, the same system
        r.append(serve.submit(chat(SYS_A, [("user", "Name one metal, please."),                      # 3: its next turn
                                           ("assistant", "Iron."), ("user", "And one more?")]), MAX_NEW))
        r.append(serve.submit(chat(SYS_A, [("user", "What is seven plus five?")]), MAX_NEW))         # 4: a third conversation
        r.append(serve.submit(chat(SYS_B, [("user", "Good morning to you.")]), MAX_NEW))             # 5: another client's system
        r.append(serve.submit(chat(SYS_A, [("user", "Name one river.")]), MAX_NEW))                  # 6: the first client again
    finally:
        serve.close()
    return r, Path(stderr_path).read_text(errors="replace")


def check_cut(binary, fixture, vocab):
    ids = [(5 + 7 * i) % (vocab - 10) + 3 for i in range(400)]
    env = dict(os.environ, COLI_TEMP="0", AUTOPIN="0", USAGE_SAVE="0", K3_EXPERT_GB="0.002")

    def logits(extra):
        with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as tmp:
            path = tmp.name
        try:
            r = subprocess.run([binary, str(fixture), "--ids", " ".join(map(str, ids)), "--ngen", "0"],
                               env=dict(env, K3_LOGITS=path, **extra), capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                raise AssertionError(f"engine exited {r.returncode}:\n{r.stderr[-1500:]}")
            return Path(path).read_bytes(), r.stderr
        finally:
            os.unlink(path)

    cuts = 0
    for chunk in ("4096", "32"):
        ref = logits({"K3_CHUNK": chunk})[0]
        for at in (2, 37, 150, 257, 398):
            got, err = logits({"K3_CHUNK": chunk, "K3_SNAP_TEST": str(at)})
            if got != ref:
                raise AssertionError(f"steps of {chunk}: the KDA calls cut at position {at} give other logits")
            cuts += "K3_SNAP_TEST: the KDA layers' calls" in err
    if cuts < 6:
        raise AssertionError(f"only {cuts} of the 10 runs cut a step: the check does not bite")
    print(f"ok cut: 400 positions' logits the same with the KDA layers' calls cut at 5 positions, "
          f"in one step and in steps of 32 ({cuts} runs cut a step)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--binary", required=True)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--tokenizer", default=str(HERE / "tok_kimi_tiny.json"))
    a = ap.parse_args()
    base = Path(a.fixture)
    cfg = json.loads((base / "config.json").read_text())
    check_cut(a.binary, base, cfg.get("text_config", cfg)["vocab_size"])
    work = Path(tempfile.mkdtemp(prefix="kimi-k3-root-"))
    try:
        model = work / "fixture"
        model.mkdir()
        shutil.copy(base / "config.json", model / "config.json")
        shutil.copy(base / "model.safetensors", model / "model.safetensors")
        shutil.copy(a.tokenizer, model / "tokenizer.json")
        common = {"K3_CHUNK": "4096", "K3_ROOT_MIN": "8"}
        none, _ = session(a.binary, model, common, work / "none.err")
        off, err_off = session(a.binary, model, dict(common, COLI_K3_CKPT="4", K3_ROOT_PHOTO="0"), work / "off.err")
        on, err_on = session(a.binary, model, dict(common, COLI_K3_CKPT="4"), work / "on.err")
        for i, (x, y, z) in enumerate(zip(none, off, on), 1):
            if x != y or x != z:
                raise AssertionError(f"request {i}: no photos {x!r}, K3_ROOT_PHOTO=0 {y!r}, with it {z!r}")
        if not any(x[0] for x in none):
            raise AssertionError(f"no reply has any text to compare: {[x[0] for x in none]}")
        if "the system prompt's end photographed" in err_off:
            raise AssertionError("K3_ROOT_PHOTO=0 photographed a system prompt")
        roots = [int(n) for n in re.findall(r"the system prompt's end photographed at (\d+) tokens", err_on)]
        rest = [(int(p), int(t), int(c)) for p, t, c in re.findall(
            r"\[K3-CKPT\] restored pos=(\d+) of (\d+) prompt tokens \(recompute (\d+)\)", err_on)]
        # the other client's conversation takes the caches' rows: the first client's next conversation
        # reads its system prompt once more, and may photograph it again
        if (len(roots) < 2 or roots[0] < 8 or roots[1] < 8 or roots[0] == roots[1]
                or any(r != roots[0] for r in roots[2:])):
            raise AssertionError(f"expected the two system prompts photographed in turn, saw {roots}:\n{err_on[-3000:]}")
        ra = roots[0]
        by_np = {t: (p, c) for p, t, c in rest}
        for turn in (2, 4):
            np_turn = on[turn - 1][1]
            if by_np.get(np_turn) != (ra, np_turn - ra):
                raise AssertionError(f"request {turn} ({np_turn} tokens): restored {by_np.get(np_turn)}, expected the "
                                     f"system prompt's {ra} and {np_turn - ra} to compute; restores {rest}")
        off_rest = re.findall(r"\[K3-CKPT\] restored pos=(\d+) of (\d+)", err_off)
        if any(int(p) == ra and int(t) in (on[1][1], on[3][1]) for p, t in off_rest):
            raise AssertionError("K3_ROOT_PHOTO=0 restored the system prompt's end: the scenario does not need the photo")
        if any(p == roots[0] and t == on[4][1] for p, t, _ in rest):
            raise AssertionError("another system prompt was served from the first one's photo")
        print(f"ok root: the system prompt's end photographed at {ra} tokens inside the first prompt's step; the "
              f"second and the fourth request, new conversations, restored it and computed "
              f"{on[1][1] - ra} and {on[3][1] - ra} tokens instead of {on[1][1]} and {on[3][1]}")
        print(f"ok another system: its own photo at {roots[1]} tokens, nothing taken from the first one's")
        print("ok identity: all 6 replies the same with the photo, with K3_ROOT_PHOTO=0 and with no photos")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("kimi-k3-root: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
