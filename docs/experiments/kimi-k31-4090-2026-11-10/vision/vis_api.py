"""vis_api.py PORT TEST... -- pictures through the gateway to Kimi K3.1 (kimi_k3.c's tower)."""
import base64, json, sys, time, urllib.request, urllib.error
port = int(sys.argv[1]); base = f"http://127.0.0.1:{port}/v1"
def url(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()
def chat(messages, max_tokens=400, tools=None):
    body = {"model": "kimi-k3.1-pro", "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    if tools: body["tools"] = tools
    req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=7200) as r:
            out = json.load(r)
    except urllib.error.HTTPError as e:
        return {"error": e.code, "body": e.read().decode()[:600], "s": round(time.time() - t0, 1)}
    m = out["choices"][0]["message"]
    return {"s": round(time.time() - t0, 1), "usage": out.get("usage"), "reasoning": (m.get("reasoning_content") or "")[:1500],
            "content": m.get("content"), "tool_calls": m.get("tool_calls")}
t0 = time.time()
while True:
    try:
        urllib.request.urlopen(base + "/models", timeout=5).read(); break
    except Exception:
        if time.time() - t0 > 1200: raise SystemExit("no server")
        time.sleep(2)
print("server up after", round(time.time() - t0), "s", flush=True)
Q = "What is written in the title bar of this window, what baud rate does it show, and what are the three buttons at the bottom? Answer briefly."
tools = [{"type": "function", "function": {"name": "get_window_state", "description": "The state of a window: its title, controls and a screenshot.",
          "parameters": {"type": "object", "properties": {"window": {"type": "string"}}, "required": ["window"]}}}]
for test in sys.argv[2:]:
    if test == "user":
        r = chat([{"role": "user", "content": [{"type": "text", "text": Q}, {"type": "image_url", "image_url": {"url": url("big/shot.png")}}]}])
    elif test == "small":
        r = chat([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url("small/shot.png")}}, {"type": "text", "text": "Describe this picture in two sentences: the shapes, their colours and the text."}]}])
    elif test == "tool":
        r = chat([{"role": "user", "content": "What are the buttons at the bottom of the LookRS232 window? Use the tool, then answer in one sentence."},
                  {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_window_state", "arguments": json.dumps({"window": "LookRS232"})}}]},
                  {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "text", "text": "{ app: 'LookRS232.exe', screenshots: [{ url: '[provided as MCP image content]' }] }"},
                                                                        {"type": "image_url", "image_url": {"url": url("big/shot.png")}}]}], tools=tools)
    print(f"== {test}: {json.dumps(r, ensure_ascii=False, indent=1)}", flush=True)
