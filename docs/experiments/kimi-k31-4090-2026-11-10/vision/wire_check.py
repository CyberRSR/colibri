"""K3CHAT1 with pictures (render_chat_kimi) through kimi_k3.exe --wire-test, against Moonshot's
encoding_k3.build_chat_segments tokenized segment by segment, its one <|media_pad|> a picture
expanded to the picture's rows."""
import base64, io, json, os, subprocess, sys
C = r"C:/colibri/dev/.claude/worktrees/kimi-k3-main/c"
K31 = r"C:/AI/Kimi-K3.1-Pro"
sys.path.insert(0, C)
import openai_server as gw
gw.ARCH = "kimi"
sys.path.insert(0, K31)
import importlib.util
spec = importlib.util.spec_from_file_location("encoding_k3", os.path.join(K31, "encoding_k3.py"))
enc = importlib.util.module_from_spec(spec); sys.modules["encoding_k3"] = enc; spec.loader.exec_module(enc)
from tokenizers import Tokenizer
tok = Tokenizer.from_file(os.path.join(K31, "tokenizer.json"))
def url(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()
small, big = url("small/shot.png"), url("big/shot.png")
tools = [{"type": "function", "function": {"name": "get_window_state", "description": "The state of a window.",
          "parameters": {"type": "object", "properties": {"window": {"type": "string"}}, "required": ["window"]}}}]
messages = [
    {"role": "system", "content": "You are a helpful agent."},
    {"role": "user", "content": [{"type": "text", "text": "Look at this:"}, {"type": "image_url", "image_url": {"url": small}},
                                 {"type": "text", "text": "and tell me the baud rate."}]},
    {"role": "assistant", "content": "", "reasoning_content": "Check the window.", "tool_calls": [{"id": "call_1", "type": "function",
        "function": {"name": "get_window_state", "arguments": json.dumps({"window": "LookRS232"})}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "text", "text": "{title: 'LookRS232'}"},
        {"type": "image_url", "image_url": {"url": big}}, {"type": "text", "text": "<extras/>"}]},
    {"role": "user", "content": "So?"},
]
expanded, images = gw.expand_kimi_images(messages, K31)
wire = gw.render_chat_kimi(expanded, enable_thinking=True, tools=tools)
open("wire.bin", "wb").write(wire.encode("utf-8"))
out = subprocess.run([os.path.join(C, "kimi_k3.exe"), K31, "--wire-test", "wire.bin"], capture_output=True, text=True,
                     env={**os.environ, "SNAP": ""})
ours = [int(x) for x in out.stdout.split()]
if not ours:
    print(out.stderr[-2000:]); sys.exit(1)
# the reference: the processor's image prompts (the original sizes), the tokenizer's segments
prompts = []
for m in messages:
    if isinstance(m["content"], list):
        for p in m["content"]:
            if p["type"] == "image_url":
                from PIL import Image
                data = base64.b64decode(p["image_url"]["url"].split(",", 1)[1])
                w, h = Image.open(io.BytesIO(data)).size
                prompts.append(f"<|media_begin|>image {w}x{h}<|media_content|><|media_pad|><|media_end|>")
ref_msgs = json.loads(json.dumps(messages))
segs = enc.build_chat_segments(ref_msgs, tools=tools, add_generation_prompt=True, thinking=True, image_prompts=prompts)
pad = tok.token_to_id("<|media_pad|>")
rows = [(gh // 2) * (gw_ // 2) for (_, gh, gw_) in images]
text = "".join(s_.text for s_ in segs)
parts = text.split("<|media_pad|>")
assert len(parts) == len(rows) + 1
want = parts[0] + "".join("<|media_pad|>" * r + parts[i + 1] for i, r in enumerate(rows))
got = tok.decode(ours, skip_special_tokens=False)
print("pictures", [(gh, gw_) for (_, gh, gw_) in images], "rows", rows, "ids", len(ours))
print("decoded text", "SAME" if got == want else "DIFF")
if got != want:
    for i, (a, b) in enumerate(zip(got, want)):
        if a != b:
            print("first diff at", i, repr(got[max(0, i - 80):i + 80]), "-----", repr(want[max(0, i - 80):i + 80])); break
print(repr(got[got.index("Look"):got.index("Look") + 60]), "...", got.count("<|media_pad|>"), "pads")
print(repr(got[got.index('role="tool"'):got.index('role="tool"') + 120]))
