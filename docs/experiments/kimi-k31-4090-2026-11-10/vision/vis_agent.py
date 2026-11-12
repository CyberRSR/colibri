"""vis_agent.py PORT -- a Kimi Code-shaped conversation: two tool results with screenshots, the second turn
continuing the first (its state reused over the first picture)."""
import base64, json, sys, time, urllib.request
port = int(sys.argv[1]); base = f"http://127.0.0.1:{port}/v1"
url = lambda p: "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()
tools = [{"type": "function", "function": {"name": "get_window_state", "description": "The state of a window: its title, controls and a screenshot.",
          "parameters": {"type": "object", "properties": {"window": {"type": "string"}}, "required": ["window"]}}}]
def chat(messages, max_tokens):
    body = {"model": "kimi-k3.1-pro", "messages": messages, "max_tokens": max_tokens, "temperature": 0, "tools": tools}
    req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=7200) as r:
        out = json.load(r)
    m = out["choices"][0]["message"]
    return round(time.time() - t0, 1), out.get("usage"), m.get("content"), m.get("tool_calls")
t0 = time.time()
while True:
    try: urllib.request.urlopen(base + "/models", timeout=5).read(); break
    except Exception:
        if time.time() - t0 > 1200: raise SystemExit("no server")
        time.sleep(2)
print("server up after", round(time.time() - t0), "s", flush=True)
tc = lambda i, w: {"role": "assistant", "content": "", "tool_calls": [{"id": f"call_{i}", "type": "function",
                   "function": {"name": "get_window_state", "arguments": json.dumps({"window": w})}}]}
res = lambda i, app, shot: {"role": "tool", "tool_call_id": f"call_{i}", "content": [
    {"type": "text", "text": "{ app: '%s', screenshots: [{ url: '[provided as MCP image content]' }] }" % app},
    {"type": "image_url", "image_url": {"url": url(shot)}}]}
conv = [{"role": "user", "content": "Look at the LookRS232 window and at its serial port settings window (use the tool for each), then tell me in one sentence whether their baud rates agree."},
        tc(1, "LookRS232"), res(1, "LookRS232.exe", "big/shot.png")]
r = chat(conv, 150); print("== A", json.dumps(r, ensure_ascii=False), flush=True)
conv += [tc(2, "Settings - Serial port"), res(2, "LookRS232.exe", "shot2/shot.png")]
r = chat(conv, 200); print("== B", json.dumps(r, ensure_ascii=False), flush=True)
