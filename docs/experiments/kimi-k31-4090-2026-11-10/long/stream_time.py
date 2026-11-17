#!/usr/bin/env python3
"""stream_time.py <request.json> <out prefix> [port]: one request streamed to 127.0.0.1:<port> (18081),
`stream_options.include_usage` on, and the time every event with text arrived (the gateway sends a token's
text as its own event; an empty keep-alive delta during the prompt is not counted).

Writes <out>.json (status, usage, finish_reason, every event's time, kind and length) and <out>.txt (the
reasoning, a line, the answer), and prints: the time to the first token, the reply's tokens and seconds after
it, tokens a second, and the pace by windows of 256 events (an event is a token but where a character's bytes
span tokens: the windows are scaled by tokens / events).
"""
import http.client
import json
import sys
import time

WIN = 256


def main():
    req, out = sys.argv[1], sys.argv[2]
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 18081
    with open(req, encoding="utf-8") as f:
        body = json.load(f)
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=7200)
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", body=data,
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    events, usage, finish, err = [], None, None, None
    parts = {"r": [], "c": []}
    if resp.status != 200:
        err = resp.read().decode("utf-8", "replace")[:2000]
    else:
        while True:
            line = resp.readline()
            if not line:
                break
            t = time.perf_counter() - t0
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            e = json.loads(payload)
            if e.get("usage"):
                usage = e["usage"]
            if e.get("error"):
                err = e["error"]
            for ch in e.get("choices") or []:
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                d = ch.get("delta") or {}
                for key, k in (("reasoning_content", "r"), ("content", "c")):
                    s = d.get(key)
                    if s:
                        events.append([round(t, 4), k, len(s)])
                        parts[k].append(s)
                if d.get("tool_calls"):
                    events.append([round(t, 4), "t", 0])
    total = time.perf_counter() - t0
    reasoning, answer = "".join(parts["r"]), "".join(parts["c"])
    with open(out + ".txt", "w", encoding="utf-8") as f:
        f.write(reasoning + "\n----- answer -----\n" + answer)
    rec = {"request": req, "status": resp.status, "error": err, "usage": usage, "finish_reason": finish,
           "total_s": round(total, 3), "events": events}
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False)
    print(f"HTTP {resp.status} in {total:.1f} s; usage {usage}; finish {finish}")
    if err:
        print(f"error: {err}")
    if not events:
        return
    ntok = (usage or {}).get("completion_tokens") or len(events)
    n = len(events)
    scale = ntok / n
    first, last = events[0][0], events[-1][0]
    rate = (ntok - 1) / (last - first) if last > first else 0.0
    kinds = {k: sum(1 for e in events if e[1] == k) for k in "rct"}
    print(f"first token after {first:.1f} s; then {ntok - 1} tokens in {last - first:.1f} s: "
          f"{rate:.3f} tokens/s, {1000.0 / rate if rate else 0:.0f} ms a token; "
          f"{n} events ({kinds['r']} reasoning, {kinds['c']} answer, {kinds['t']} tool), "
          f"{scale:.3f} tokens an event")
    pts = list(range(0, n, WIN))
    if pts[-1] != n - 1:
        pts.append(n - 1)
    row = []
    for a, b in zip(pts, pts[1:]):
        dt = events[b][0] - events[a][0]
        row.append(f"{a * scale:.0f}-{b * scale:.0f} {(b - a) * scale / dt:.3f}" if dt > 0 else f"{a}-{b} -")
    print("by window (tokens: tokens/s): " + "; ".join(row))


if __name__ == "__main__":
    main()
