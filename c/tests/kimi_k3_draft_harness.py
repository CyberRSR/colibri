#!/usr/bin/env python3
"""A decode step that carries drafts (kimi_k3.c, k3_draft.h), on a tiny fixture.

With a draft model (K3_DRAFT) a decode step also feeds the tokens it guesses behind the one it
must feed, and a guess the model then picks costs no step of its own; the rows of the guesses it
does not pick are taken back -- the KDA layers' state by a swap of pointers, their conv windows
rebuilt, the record of the tokens cut. None of it may move a number. Here the guesses come from
a file (K3_DRAFT_ORACLE: no draft model and no device needed), so a test chooses which stand:

  - right: every guess is the token the plain run produced. Up to 3 a step: the logits of every
    position and the tokens are the plain run's, every draft stood, and the reply took about a
    quarter of the steps.
  - wrong: every guess is another token. Every block is taken back whole: the same logits and
    tokens, no draft stood.
  - mixed: up to 7 a step with every 3rd, 4th or 5th guess wrong, so blocks end on every row
    from the first to the last: the same logits and tokens.
  - sampled: the same at a temperature, where a draft stands when the sampler draws it.
  - short: a reply that ends inside what a block could have carried (--ngen 5 with 7 guesses on
    offer): no row past the reply's end is computed, the same logits and tokens.

Run by path (not a test_*.py: unittest discovery would import it):
  python tests/kimi_k3_draft_harness.py --binary ./kimi_k3 --fixture ./kimi_k31_tiny
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ENV = {"COLI_TEMP": "0", "AUTOPIN": "0", "USAGE_SAVE": "0", "K3_MAXT": "4096", "K3_EXPERT_GB": "0.002",
       "K3_LOGITS_DECODE": "1", "K3_CHUNK": "16"}


def go(binary, fixture, ids, ngen, extra, guesses=None):
    env = dict(os.environ, **ENV)
    env.pop("K3_DRAFT", None)
    env.update(extra)
    made = []
    with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as tmp:
        path = tmp.name
    made.append(path)
    try:
        env["K3_LOGITS"] = path
        if guesses is not None:
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
                tmp.write("".join(f"{p} {t}\n" for p, t in guesses))
                made.append(tmp.name)
            env["K3_DRAFT_ORACLE"] = tmp.name
        r = subprocess.run([binary, str(fixture), "--ids", " ".join(map(str, ids)), "--ngen", str(ngen)],
                           env=env, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            raise AssertionError(f"engine exited {r.returncode}:\n{r.stderr[-2000:]}")
        return Path(path).read_bytes(), [int(t) for t in r.stdout.split("TUNE")[0].split()], r.stderr
    finally:
        for p in made:
            os.unlink(p)


def same(what, got, ref):
    if got[1] != ref[1] or not got[1]:
        raise AssertionError(f"{what}: tokens {got[1]} against {ref[1]}")
    if got[0] != ref[0]:
        n = next((i for i, (a, b) in enumerate(zip(got[0], ref[0])) if a != b), min(len(got[0]), len(ref[0])))
        raise AssertionError(f"{what}: the logits are not the plain run's ({len(got[0])} against {len(ref[0])} bytes, "
                             f"the first difference at byte {n})\n{got[2][-1500:]}")


def counts(err):
    m = re.search(r"\[K3-DRAFT\] run: (\d+) tokens in (\d+) steps .*?; (\d+) of (\d+) drafts stood", err)
    if not m:
        raise AssertionError(f"no [K3-DRAFT] run line:\n{err[-1500:]}")
    return tuple(int(x) for x in m.groups())


def ends(err):
    return sorted(set((int(a), int(b)) for a, b in re.findall(r"\[K3-DRAFT\] at \d+: (\d+) of (\d+) drafts stood", err)))


def check(binary, fixture, vocab):
    ids = [(5 + 7 * i) % (vocab - 10) + 3 for i in range(40)]
    np_, ngen = len(ids), 96
    ref = go(binary, fixture, ids, ngen, {})
    toks = ref[1]
    if len(toks) < 24:
        raise AssertionError(f"the plain run produced {len(toks)} tokens: too few to test with")
    if len(ref[0]) not in ((np_ + len(toks)) * vocab * 4, (np_ + len(toks) - 1) * vocab * 4):   # an eos is not fed
        raise AssertionError(f"the plain run's logits: {len(ref[0])} bytes for {np_} + {len(toks)} positions of {vocab}")
    if "K3-DRAFT" in ref[2]:
        raise AssertionError("the plain run mentions a draft model")
    at = lambda j: np_ + j          # the reply's token j is the one at this position
    other = lambda t: (t + 1) % vocab

    right = [(at(j), toks[j]) for j in range(1, len(toks))]
    got = go(binary, fixture, ids, ngen, {"K3_DRAFT_MAX": "3"}, right)
    same("every guess right", got, ref)
    n, steps, stood, drafts = counts(got[2])
    if n != len(toks) or stood != drafts or stood < len(toks) // 2 or steps > len(toks) // 2:
        raise AssertionError(f"every guess right: {n} tokens in {steps} steps, {stood} of {drafts} drafts stood")
    print(f"ok right: {n} tokens in {steps} steps, {stood} of {drafts} drafts stood; the logits of {np_} + {n} positions "
          f"and the tokens are the plain run's")

    wrong = [(p, other(t)) for p, t in right]
    got = go(binary, fixture, ids, ngen, {"K3_DRAFT_MAX": "3", "K3_DRAFT_LOG": "1"}, wrong)
    same("every guess wrong", got, ref)
    n, steps, stood, drafts = counts(got[2])
    if stood != 0 or drafts < len(toks) or steps < len(toks) - 1:
        raise AssertionError(f"every guess wrong: {n} tokens in {steps} steps, {stood} of {drafts} drafts stood")
    print(f"ok wrong: {drafts} drafts, none stood, every block taken back whole; the same logits and tokens")

    seen = set()
    for what, bad in (("every 3rd guess wrong", lambda j: j % 3 == 0), ("every 4th guess wrong", lambda j: j % 4 == 0),
                      ("two of every 7 guesses wrong", lambda j: j % 7 in (0, 1)), ("every guess but each 9th wrong", lambda j: j % 9 != 4)):
        mixed = [(p, other(t) if bad(p - np_) else t) for p, t in right]
        got = go(binary, fixture, ids, ngen, {"K3_DRAFT_MAX": "7", "K3_DRAFT_LOG": "1"}, mixed)
        same(what, got, ref)
        n, steps, stood, drafts = counts(got[2])
        if not 0 < stood < drafts:
            raise AssertionError(f"{what}: {stood} of {drafts} drafts stood")
        seen |= set(ends(got[2]))
    kept = sorted(set(j for j, _ in seen))
    if kept[0] != 0 or kept[-1] < 4 or len(kept) < 5:
        raise AssertionError(f"mixed guesses: the blocks ended with {kept} drafts standing only ({sorted(seen)})")
    print(f"ok mixed: blocks of up to 7 drafts ended with {kept} of them standing; the same logits and tokens every time")

    hot = {"COLI_TEMP": "0.9"}
    sref = go(binary, fixture, ids, ngen, hot)
    st = sref[1]
    if st == toks:
        raise AssertionError("the sampled run produced the greedy run's tokens: nothing was sampled")
    sguess = [(at(j), other(st[j]) if j % 4 == 0 else st[j]) for j in range(1, len(st))]
    got = go(binary, fixture, ids, ngen, dict(hot, K3_DRAFT_MAX="5"), sguess)
    same("sampled at 0.9", got, sref)
    n, steps, stood, drafts = counts(got[2])
    if not 0 < stood < drafts:
        raise AssertionError(f"sampled: {stood} of {drafts} drafts stood")
    print(f"ok sampled: at temperature 0.9, {stood} of {drafts} drafts stood in {steps} steps; the same logits and tokens")

    sh = go(binary, fixture, ids, 5, {})
    got = go(binary, fixture, ids, 5, {"K3_DRAFT_MAX": "7", "K3_DRAFT_LOG": "1"}, right)
    same("a reply of 5 tokens", got, sh)
    n, steps, stood, drafts = counts(got[2])
    if n != len(sh[1]) or drafts > 3 or stood != drafts:
        raise AssertionError(f"a reply of 5 tokens: {n} tokens, {stood} of {drafts} drafts")
    print(f"ok short: a reply of {n} tokens carried {drafts} drafts, none past its end; the same logits and tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--fixture", required=True)
    args = ap.parse_args()
    fixture = Path(args.fixture)
    cfg = json.loads((fixture / "config.json").read_text())
    vocab = int((cfg.get("text_config") or cfg)["vocab_size"])
    check(args.binary, fixture, vocab)
    return 0


if __name__ == "__main__":
    sys.exit(main())
