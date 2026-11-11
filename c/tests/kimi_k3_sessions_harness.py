#!/usr/bin/env python3
"""Several conversations kept beside the live one, in RAM and in files (kimi_k3.c, k3_sessions.h).

A request of another conversation used to write its rows over the live one's, and the first
conversation's next turn read every token again. With K3_SESSIONS the live conversation goes
aside whole and comes back when a request continues it; with K3_SESSION_DIR it is also written
to a file, read back after a restart. On a tiny fixture, through the serve protocol:

  - alone: each of three conversations (A and C share a system prompt, B has its own) served
    by a server of its own, its turns one after the other: the replies every other run must
    give.
  - interleaved, no sessions: A1 B1 A2 C1 B2 A3 C2 on one server. A2 reads A again (the
    check below bites).
  - interleaved, kept in RAM (K3_SESSIONS=4): every reply is the alone run's; A2, B2, A3 and
    C2 compute only what follows their conversation's previous prompt; C1 starts after the
    system prompt it shares with A, its rows copied from A.
  - interleaved, in files (K3_SESSION_DIR, nothing in RAM), the server restarted between B2
    and A3: the same replies; the second server finds the three conversations' files, and A3
    and C2 are read from them.
  - a file cut short: the second server deletes it and C2 reads its conversation again.
  - RAM too small and no files (K3_SESSION_GB tiny): the kept conversations are dropped and
    their turns read them again; every request is answered.
  - the count of the caches' bytes: three new conversations in a row lend the expert cache's
    slots for one conversation's rows, not for the three (model_state_reset took a dropped
    conversation's rows off the count only from this change on).

Run by path (not a test_*.py: unittest discovery would import it):
  python tests/kimi_k3_sessions_harness.py --binary ./kimi_k3 --fixture ./kimi_k31_tiny
"""

import argparse
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_kimi_k3_ckpt import Serve   # noqa: E402  the SUBMIT/ACCEPT/DATA/DONE client

SYS_A = ("You are a careful assistant. Answer briefly and precisely. " * 6 +
         "Use the tools only when asked. Never repeat the question. " * 5).strip()
SYS_B = ("You translate between languages and say nothing else. " * 9).strip()
MAX_NEW = 24
CONV = {"A": (SYS_A, ["Name one colour.", "And a metal?", "Which is heavier?"]),
        "B": (SYS_B, ["Good morning to you.", "See you tomorrow."]),
        "C": (SYS_A, ["What is seven plus five?", "And times two?"])}
ORDER = ["A", "B", "A", "C", "B", "A", "C"]   # the interleaved session: A1 B1 A2 C1 B2 A3 C2
CLOSED = []                                  # how long the restarted servers took to exit once stdin closed


def chat(system, turns):
    """K3CHAT1: M <role> <bytes>\\n<text> records, then the generation cue."""
    out = b"K3CHAT1\n"
    for role, text in [("system", system)] + turns:
        raw = text.encode("utf-8")
        out += f"M {role} {len(raw)}\n".encode("ascii") + raw
    return (out + b"G 0\n").decode("utf-8")


class Run:
    """One server; each request's stderr lines kept apart."""

    def __init__(self, binary, model, env, err):
        self.err = Path(err)
        self.serve = Serve(binary, model, env, err)
        self.seen = 0

    def ask(self, prompt):
        reply, np_ = self.serve.submit(prompt, MAX_NEW)
        text = self.err.read_text(errors="replace")
        lines, self.seen = text[self.seen:], len(text)
        return reply, np_, lines

    def start_log(self):
        return self.err.read_text(errors="replace")

    def close(self):
        self.serve.close()
        text = self.err.read_text(errors="replace")
        return text[self.seen:]


def reuse_of(lines):
    m = re.search(r"\[PREFIX\] reusing (\d+) of (\d+) prompt tokens", lines)
    return int(m.group(1)) if m else 0


class Conversations:
    """The histories as each run's own replies make them."""

    def __init__(self):
        self.hist = {k: [] for k in CONV}
        self.turn = {k: 0 for k in CONV}

    def next_prompt(self, k):
        system, users = CONV[k]
        return chat(system, self.hist[k] + [("user", users[self.turn[k]])])

    def answered(self, k, reply):
        system, users = CONV[k]
        self.hist[k] += [("user", users[self.turn[k]]), ("assistant", reply.decode("utf-8", "replace"))]
        self.turn[k] += 1


def alone(binary, model, env, work):
    out = {}
    for k in CONV:
        run, conv = Run(binary, model, env, work / f"alone_{k}.err"), Conversations()
        try:
            out[k] = []
            for _ in CONV[k][1]:
                reply, np_, _ = run.ask(conv.next_prompt(k))
                out[k].append(reply)
                conv.answered(k, reply)
        finally:
            run.close()
    return out


def interleaved(binary, model, env, work, name, restart_at=None, between=None):
    """ORDER on one server, or on two with the second started before request `restart_at`
    (`between` runs in the pause). Returns [(conversation, reply, np, its lines)], the start logs."""
    conv, res, starts = Conversations(), [], []
    run = Run(binary, model, env, work / f"{name}_1.err")
    try:
        for i, k in enumerate(ORDER):
            if restart_at is not None and i == restart_at:
                t0 = time.time()
                run.close()
                CLOSED.append(time.time() - t0)
                starts.append(run.start_log())
                if between:
                    between()
                run = Run(binary, model, env, work / f"{name}_2.err")
            reply, np_, lines = run.ask(conv.next_prompt(k))
            res.append((k, reply, np_, lines))
            conv.answered(k, reply)
    finally:
        run.close()
        starts.append(run.start_log())
    return res, starts


def check_against(ref, res, what):
    seen = {k: 0 for k in CONV}
    for i, (k, reply, _, _) in enumerate(res, 1):
        want = ref[k][seen[k]]
        if reply != want:
            raise AssertionError(f"{what}: request {i} ({k}{seen[k] + 1}) answered {reply!r}, alone {want!r}")
        seen[k] += 1


def check_reuse(res, what, files=False):
    """A2 B2 A3 C2 compute only what follows their previous prompt; C1 starts after SYS_A."""
    prev = {}
    for i, (k, _, np_, lines) in enumerate(res):
        got = reuse_of(lines)
        if k in prev and got < prev[k]:
            raise AssertionError(f"{what}: request {i + 1} ({k}, {np_} tokens) reused {got}, its conversation's "
                                 f"previous prompt was {prev[k]}:\n{lines[-2500:]}")
        if k == "C" and k not in prev:
            if "the new conversation starts after the" not in lines or got < 8:
                raise AssertionError(f"{what}: C1 did not start after the system prompt it shares with A "
                                     f"(reused {got}):\n{lines[-2500:]}")
        prev[k] = np_
    if files:
        for i in (5, 6):
            if "read from its file" not in res[i][3]:
                raise AssertionError(f"{what}: request {i + 1} ({res[i][0]}) was not read from its file:\n"
                                     f"{res[i][3][-2500:]}")


def check_leak(binary, model, work):
    """Three new conversations of one length in a row: the slots lent for the caches' rows are one
    conversation's. The plan holds 30 positions less than one conversation, so the first lends a
    few slots; a count that kept the dropped conversations' rows lends more for the second (3 and
    6 slots with release r3's engine on this fixture)."""
    text = " ".join(f"Tell me about rivers number {i} and their names." for i in range(14))

    def lent_after(kv_min):
        env = {"K3_MAXT": "4096", "K3_EXPERT_GB": "0.002", "K3_KV_MIN": str(kv_min), "K3_KV_GEN": "64",
               "K3_CHUNK": "4096", "AUTOPIN": "0", "USAGE_SAVE": "0"}
        run = Run(binary, model, env, work / "leak.err")
        lent, caps = [], []
        try:
            for n in (1, 2, 3):
                _, _, lines = run.ask(chat(f"You answer questions, set {n}.", [("user", text)]))
                caps += [int(x) for x in re.findall(r"\[PREFIX\] no reuse: held=\d+ cap=(\d+)", lines)]
                got = [int(x) for x in re.findall(r"the expert cache lends (\d+) slots for them", lines)]
                lent.append(got[-1] if got else (lent[-1] if lent else 0))
        finally:
            run.close()
        return lent, caps

    _, caps = lent_after(4096)
    if len(set(caps)) != 1:
        raise AssertionError(f"the three conversations' caches are not of one size: {caps}")
    lent, _ = lent_after(caps[0] - 30)
    if lent[0] < 1:
        raise AssertionError(f"the first conversation lent no slot: the check does not bite ({lent})")
    if max(lent) != lent[0]:
        raise AssertionError(f"slots lent after one, two, three new conversations: {lent} -- the dropped "
                             f"conversations' rows are still counted")
    print(f"ok count: three new conversations of {caps[0]} positions each lend {lent} slots of the expert cache "
          f"(the plan holds {caps[0] - 30}): the dropped conversations' rows are not counted")


def check_floor(binary, base, tokenizer, work):
    """The history's pins and a conversation's rows past the plan: a layer lent down to its floor
    keeps slots no pin holds for a token's experts and its read-ahead. Release r3's engine counted
    the pins among them, and the first token after the prompt waited for a slot forever."""
    fx = work / "floor_fixture"
    fx.mkdir()
    shutil.copy(base / "config.json", fx / "config.json")
    shutil.copy(base / "model.safetensors", fx / "model.safetensors")
    shutil.copy(tokenizer, fx / "tokenizer.json")
    text = " ".join(f"Tell me about rivers number {i} and their names." for i in range(40))
    prompt = chat("You answer questions about rivers.", [("user", text)])
    env = {"K3_MAXT": "4096", "K3_EXPERT_GB": "0.02", "K3_CHUNK": "4096"}
    run = Run(binary, fx, env, work / "floor_history.err")   # the expert history, saved at the exit:
    try:                                                      # enough selections for two pins a layer
        for n in range(10):
            run.ask(chat(f"You answer questions, set {n}.", [("user", text)]))
    finally:
        run.close()
    run = Run(binary, fx, dict(env, K3_KV_MIN="64", USAGE_SAVE="0"), work / "floor.err")
    got = []
    t = threading.Thread(target=lambda: got.append(run.ask(prompt)), daemon=True)
    t.start()
    t.join(120)
    log = run.err.read_text(errors="replace")
    if not got:
        run.serve.process.kill()
        raise AssertionError(f"no answer in 120 s with the history's pins and the rows past the plan:\n{log[-2000:]}")
    run.close()
    pins = re.search(r"\[PIN\] (\d+) experts pinned", log)
    lent = re.findall(r"the expert cache lends (\d+) slots for them \(\d+ more, [0-9.]+ GB of experts out; (\d+)-(\d+)", log)
    if not pins or int(pins.group(1)) < 1 or not lent:
        raise AssertionError(f"no pins or nothing lent: the check does not bite:\n{log[-2000:]}")
    print(f"ok floor: {pins.group(1)} experts pinned from the history and {lent[-1][0]} slots lent "
          f"({lent[-1][1]}-{lent[-1][2]} a layer left): the request is answered")


def check_resume(binary, model, work):
    """A prompt of several steps cancelled in its third step keeps the state after its first two;
    sent again, it resumes there and answers what a server that never saw the cancel answers.
    K3_CANCEL_TEST=N cancels at a step's N-th poll (one a layer: six a step on this fixture)."""
    text = " ".join(f"Tell me about rivers number {i} and their names." for i in range(12))
    prompt = chat("You answer questions about rivers.", [("user", text)])
    env = {"K3_CHUNK": "128", "AUTOPIN": "0", "USAGE_SAVE": "0"}
    ref_run = Run(binary, model, env, work / "resume_ref.err")
    try:
        ref, np_ref, _ = ref_run.ask(prompt)
    finally:
        ref_run.close()
    run = Run(binary, model, dict(env, K3_CANCEL_TEST="15"), work / "resume.err")
    try:
        try:
            run.ask(prompt)
            raise RuntimeError("the prompt was not cancelled")
        except AssertionError as e:
            if "CANCELLED" not in str(e):
                raise
        reply, np_, lines = run.ask(prompt)
    finally:
        run.close()
    log = run.err.read_text(errors="replace")
    m = re.search(r"the prompt of (\d+) tokens was cancelled: the state after its first (\d+) is kept", log)
    if not m:
        raise AssertionError(f"the cancelled prompt kept nothing:\n{log[-2500:]}")
    kept, got = int(m.group(2)), reuse_of(lines)
    if kept < 200 or got != kept:
        raise AssertionError(f"kept {kept} tokens of {m.group(1)}, the prompt sent again reused {got}")
    if reply != ref:
        raise AssertionError(f"after the resume: {reply!r}, a server that never saw the cancel: {ref!r}")
    print(f"ok resume: a {np_}-token prompt cancelled in its third step kept its first {kept} tokens; sent again it "
          f"computed {np_ - kept} and answered as a server that never saw the cancel")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--binary", required=True)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--tokenizer", default=str(HERE / "tok_kimi_tiny.json"))
    a = ap.parse_args()
    base = Path(a.fixture)
    work = Path(tempfile.mkdtemp(prefix="kimi-k3-sess-"))
    try:
        model = work / "fixture"
        model.mkdir()
        shutil.copy(base / "config.json", model / "config.json")
        shutil.copy(base / "model.safetensors", model / "model.safetensors")
        shutil.copy(a.tokenizer, model / "tokenizer.json")
        check_leak(a.binary, model, work)
        check_floor(a.binary, base, a.tokenizer, work)
        check_resume(a.binary, model, work)
        common = {"K3_CHUNK": "4096", "K3_ROOT_MIN": "8", "COLI_K3_CKPT": "4", "K3_SESSION_MIN": "8"}

        ref = alone(a.binary, model, common, work)
        if not any(r for v in ref.values() for r in v):
            raise AssertionError(f"no reply has any text to compare: {ref}")

        ctl, _ = interleaved(a.binary, model, common, work, "none")
        if reuse_of(ctl[2][3]) >= ctl[0][2]:
            raise AssertionError("without K3_SESSIONS A2 still reused A1's prompt: the scenario does not need them")

        ram, _ = interleaved(a.binary, model, dict(common, K3_SESSIONS="4"), work, "ram")
        check_against(ref, ram, "kept in RAM")
        check_reuse(ram, "kept in RAM")
        print(f"ok RAM: A1 B1 A2 C1 B2 A3 C2 answered as each conversation alone; each turn computed only what "
              f"follows its previous prompt (A2 reused {reuse_of(ram[2][3])} of {ram[2][2]}, without K3_SESSIONS "
              f"{reuse_of(ctl[2][3])}); C1 started after the {reuse_of(ram[3][3])} tokens it shares with A")

        sdir = work / "sessions"
        fenv = dict(common, K3_SESSIONS="4", K3_SESSION_DIR=str(sdir))
        snap = work / "sessions_at_restart"
        files, starts = interleaved(a.binary, model, fenv, work, "files", restart_at=5,
                                    between=lambda: shutil.copytree(sdir, snap))
        check_against(ref, files, "in files")
        check_reuse(files, "in files", files=True)
        kept = sorted(p.name for p in snap.iterdir())
        if len([n for n in kept if n.endswith(".k3s")]) != 3 or any(n.endswith(".tmp") for n in kept):
            raise AssertionError(f"after the first server: {kept}, expected three conversations' files")
        m = re.search(r"(\d+) conversations found there", starts[1])
        if not m or int(m.group(1)) != 3:
            raise AssertionError(f"the second server's start: {starts[1][-1500:]}")
        if "leaves RAM" in starts[0] + starts[1] or "in RAM and in its file" in starts[0]:
            raise AssertionError("K3_SESSION_DIR alone kept a conversation in RAM")
        if "written to its file (the server stops)" not in starts[0] or CLOSED[-1] > 15:
            raise AssertionError(f"the first server took {CLOSED[-1]:.1f} s to exit once its input closed, or did "
                                 f"not write the live conversation then:\n{starts[0][-1500:]}")
        print(f"ok files: the same replies with nothing kept in RAM and the server restarted before A3; the second "
              f"server found the 3 conversations' files ({len(kept)} files) and read A3's and C2's from them")

        # a file cut short: the restarted server deletes it and C2 reads its conversation again
        cut = work / "sessions_cut"
        shutil.copytree(snap, cut)
        rows = sorted(cut.glob("*.k3r"))
        by_len = {}
        for meta in cut.glob("*.k3s"):
            by_len[meta.stem] = meta.stat().st_size
        victim = sorted(by_len)[-1]   # the newest number: C's (kept last, at B2)
        r = cut / f"{victim}.k3r"
        r.write_bytes(r.read_bytes()[:-7])
        conv = Conversations()
        for k, reply, _, _ in files[:5]:
            conv.next_prompt(k)
            conv.answered(k, reply)
        run = Run(a.binary, model, dict(fenv, K3_SESSION_DIR=str(cut)), work / "cut.err")
        try:
            _, _, lines = run.ask(conv.next_prompt("C"))
        finally:
            run.close()
        start = run.start_log()
        if "not a whole conversation of this model, deleted" not in start or (cut / f"{victim}.k3r").exists():
            raise AssertionError(f"the cut file was not refused:\n{start[-2000:]}")
        if "read from its file" in lines:
            raise AssertionError("C2 was read from a file cut short")
        if not re.search(r"2 conversations found there", start):
            raise AssertionError(f"expected the two whole conversations: {start[-1500:]}")
        print(f"ok cut: a rows file 7 bytes short is deleted at the start ({len(rows)} rows files there) and its "
              f"conversation read again")

        tight, _ = interleaved(a.binary, model, dict(common, K3_SESSIONS="4", K3_SESSION_GB="0.000001"), work, "tight")
        if "is dropped (the live conversation and the kept ones are past K3_SESSION_GB): it has no file" not in \
                "".join(x[3] for x in tight):
            raise AssertionError("a tiny K3_SESSION_GB without files dropped nothing")
        print(f"ok tight: with no files and K3_SESSION_GB tiny the kept conversations are dropped, and all "
              f"{len(tight)} requests are answered")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("kimi-k3-sessions: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
