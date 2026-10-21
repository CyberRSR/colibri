#!/usr/bin/env python3
"""An agent-like loop against a running gateway, thinking on, greedy, as an OpenAI client sends it.

usage: agent_loop.py --mode drop|keep --out FILE [--url URL] [--tokens N]

Turn 1 is the 24,957-token prompt of the GPU prompt rounds (p25k.txt: this fork's documents and a
question) behind a short system prompt and a tool; turn 2 answers a read_file call with a tool
result; turns 3 and 4 are user follow-ups. Every turn resends the whole conversation, the
assistant's turns as the client keeps them: `drop` sends back the content and the tool calls only
(what most OpenAI-compatible agent clients send), `keep` also the `reasoning_content` the gateway
returned. Each reply is cut at --tokens, so the turns' times are the prompt's and a short reply's.
The assistant's turn 1 is the model's reply with a read_file call added, so turn 2 is a tool round
trip whatever the model chose to do in its first tokens. Per turn: the time to the first streamed
token (reasoning or content), the time to the end, the reply and the usage, into FILE (JSON)."""
import argparse, json, pathlib, time, urllib.request

HERE = pathlib.Path(__file__).resolve().parent
DOCS = HERE.parent / "mimo-gpu-prompt-4090-2026-10-20" / "prompts"
REPO = HERE.parents[2]

TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a file of this repository.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
SYSTEM = "You are a coding agent working in this repository. Use the tools when you need a file."


def ask(url, messages, tokens):
    body = {"model": "mimo-v2.6-pro-mopd", "stream": True, "max_tokens": tokens, "temperature": 0,
            "messages": messages, "tools": TOOLS, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    reasoning, content, usage, finish = [], [], None, None
    calls = {}
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                d = choice.get("delta", {})
                if (d.get("reasoning_content") or d.get("content")) and first is None:
                    first = time.perf_counter() - t0
                reasoning.append(d.get("reasoning_content") or "")
                content.append(d.get("content") or "")
                for c in d.get("tool_calls") or []:
                    slot = calls.setdefault(c.get("index", 0), {"id": None, "name": "", "arguments": ""})
                    slot["id"] = c.get("id") or slot["id"]
                    fn = c.get("function") or {}
                    slot["name"] += fn.get("name") or ""
                    slot["arguments"] += fn.get("arguments") or ""
                finish = choice.get("finish_reason") or finish
    return {"ttft_s": round(first if first is not None else time.perf_counter() - t0, 2),
            "total_s": round(time.perf_counter() - t0, 2), "reasoning": "".join(reasoning),
            "content": "".join(content), "tool_calls": [calls[k] for k in sorted(calls)], "usage": usage,
            "finish": finish}


def assistant(reply, mode, extra_calls=()):
    """The assistant's turn as the client sends it back."""
    calls = [{"id": c["id"] or f"call_{i}", "type": "function",
              "function": {"name": c["name"], "arguments": c["arguments"]}}
             for i, c in enumerate(reply["tool_calls"])] + list(extra_calls)
    turn = {"role": "assistant", "content": reply["content"]}
    if calls:
        turn["tool_calls"] = calls
    if mode == "keep":
        turn["reasoning_content"] = reply["reasoning"]
    return turn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("drop", "keep"), required=True)
    ap.add_argument("--url", default="http://127.0.0.1:18080/v1/chat/completions")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    task = (DOCS / "p25k.txt").read_text(encoding="utf-8")
    tool_text = (REPO / "docs" / "mimo-cuda.md").read_text(encoding="utf-8")[:6000]
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
    turns = []

    def turn(name):
        reply = ask(a.url, messages, a.tokens)
        reply["name"] = name
        turns.append(reply)
        print(json.dumps({k: reply[k] for k in ("name", "ttft_s", "total_s", "usage", "finish")}), flush=True)
        return reply

    r1 = turn("1 documents")
    call = {"id": "call_read_1", "type": "function",
            "function": {"name": "read_file", "arguments": json.dumps({"path": "docs/mimo-cuda.md"})}}
    messages.append(assistant(r1, a.mode, [call] if not r1["tool_calls"] else []))
    first_call = messages[-1]["tool_calls"][0]
    messages.append({"role": "tool", "tool_call_id": first_call["id"], "content": tool_text})
    r2 = turn("2 tool result")
    messages.append(assistant(r2, a.mode))
    messages.append({"role": "user", "content": "Which of the two numbers is the time to the first token?"})
    r3 = turn("3 follow-up")
    messages.append(assistant(r3, a.mode))
    messages.append({"role": "user", "content": "Thanks. Answer in one sentence."})
    turn("4 follow-up")
    pathlib.Path(a.out).write_text(json.dumps({"mode": a.mode, "tokens": a.tokens, "turns": turns}, indent=1,
                                              ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
