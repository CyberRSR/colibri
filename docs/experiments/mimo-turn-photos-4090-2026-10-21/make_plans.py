#!/usr/bin/env python3
"""The plans turns.py runs here, from the GPU prompt rounds' documents (prompts/ of
../mimo-gpu-prompt-4090-2026-10-20). Writes plan-*.json next to itself."""
import json, pathlib

HERE = pathlib.Path(__file__).resolve().parent
DOCS = HERE.parent / "mimo-gpu-prompt-4090-2026-10-20" / "prompts"
p4k = (DOCS / "p4k.txt").read_text(encoding="utf-8")
p25k = (DOCS / "p25k.txt").read_text(encoding="utf-8")


def write(name, requests):
    (HERE / f"plan-{name}.json").write_text(json.dumps({"requests": requests}, indent=1, ensure_ascii=False),
                                            encoding="utf-8")


# exact: a turn resumed from the photo of the turn before against the same prompt computed cold
# (after an unrelated prompt, which leaves no photo standing) -- a short tail (the reply and a
# question, under the GPU's 64-row prompt kernels) and a long one
first = [{"role": "user", "content": p4k + "\n\nWhat is this text about? One sentence."}]
reply = {"role": "assistant", "content": "It describes MiMo's native dense CUDA offload on an RTX 4090."}
short = first + [reply, {"role": "user", "content": "And in three words?"}]
long_ = first + [reply, {"role": "user", "content": p25k[:6000] + "\n\nHow does this relate to the text before?"}]
other = [{"role": "user", "content": "Say hello in one word."}]
write("exact", [
    {"name": "t1", "messages": first, "tokens": 32},
    {"name": "t2-short", "messages": short, "tokens": 32},
    {"name": "other", "messages": other, "tokens": 4},
    {"name": "t2-short-cold", "messages": short, "tokens": 32},
    {"name": "t1-again", "messages": first, "tokens": 32},
    {"name": "t2-long", "messages": long_, "tokens": 32},
    {"name": "other-2", "messages": other, "tokens": 4},
    {"name": "t2-long-cold", "messages": long_, "tokens": 32},
])

# tail: the short turn alone after the first, for its time with each kernel choice
write("tail", [
    {"name": "t1", "messages": first, "tokens": 32},
    {"name": "t2-short", "messages": short, "tokens": 32},
])
