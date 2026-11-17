#!/usr/bin/env python3
"""summarize.py [runs folder]: long_answer.sh's runs side by side -- per run the reply's pace as the client saw
it (stream_time.py's events) and as the engine counted it (its lines of the turn), the pace by windows of 256
tokens, and whether the draft model's run and the plain one wrote the same text."""
import glob
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WIN = 256


def engine_lines(log):
    out = {}
    try:
        text = open(log, encoding="utf-8", errors="replace").read()
    except OSError:
        return out
    m = re.search(r"expert cache (\d+)/layer", text)
    if m:
        out["cache"] = int(m.group(1))
    m = re.search(r"\[K3\] prompt of (\d+) tokens \((\d+) reused\) in ([\d.]+)s", text)
    if m:
        out["prompt_s"] = float(m.group(3))
    m = re.search(r"\[K3\] reads \(the turn's reply\): .*together ([\d.]+) GB/s over ([\d.]+) s", text)
    if m:
        out["reply_gbs"], out["reply_s"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"\[K3-DRAFT\] turn: (\d+) tokens in (\d+) steps \(([\d.]+) a step\); (\d+) of (\d+) drafts stood", text)
    if m:
        out["draft"] = dict(tokens=int(m.group(1)), steps=int(m.group(2)), per=float(m.group(3)),
                            kept=int(m.group(4)), drafts=int(m.group(5)))
    m = re.search(r"\[K3\] prefetch \(turn\): .*?([\d.]+)% of the routed experts were predicted", text)
    if m:
        out["predicted"] = float(m.group(1))
    return out


def windows(events, ntok):
    n = len(events)
    if n < 2:
        return []
    scale = ntok / n
    pts = list(range(0, n, WIN))
    if pts[-1] != n - 1:
        pts.append(n - 1)
    out = []
    for a, b in zip(pts, pts[1:]):
        dt = events[b][0] - events[a][0]
        out.append(((b - a) * scale / dt) if dt > 0 else 0.0)
    return out


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "runs")
    runs = []
    for d in sorted(glob.glob(os.path.join(root, "[0-9][0-9]_*"))):
        js = os.path.join(d, "reply.json")
        if not os.path.exists(js):
            continue
        rec = json.load(open(js, encoding="utf-8"))
        name = os.path.basename(d)
        _, cfg, req = name.split("_", 2)
        ev = rec["events"]
        usage = rec.get("usage") or {}
        ntok = usage.get("completion_tokens") or len(ev)
        text = open(os.path.join(d, "reply.txt"), encoding="utf-8").read()
        first, last = (ev[0][0], ev[-1][0]) if ev else (0.0, 0.0)
        runs.append(dict(name=name, cfg=cfg, req=req, ptok=usage.get("prompt_tokens"), ntok=ntok,
                         first=first, dec=last - first, rate=(ntok - 1) / (last - first) if last > first else 0.0,
                         finish=rec.get("finish_reason"), win=windows(ev, ntok),
                         sha=hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
                         eng=engine_lines(os.path.join(d, "serve.log"))))
    print("| run | prompt | first token | reply | its time | tokens/s | ms a token | engine: tokens/s | "
          "a step carried | drafts stood | drives during the reply | text |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in runs:
        e = r["eng"]
        d = e.get("draft")
        erate = r["ntok"] / e["reply_s"] if e.get("reply_s") else 0.0
        per = d["per"] if d else 1.0
        stood = "%d/%d" % (d["kept"], d["drafts"]) if d else "-"
        print(f"| {r['name']} | {r['ptok']} | {r['first']:.1f} s | {r['ntok']} ({r['finish']}) | {r['dec']:.1f} s | "
              f"{r['rate']:.3f} | {1000 / r['rate'] if r['rate'] else 0:.0f} | {erate:.3f} | "
              f"{per:.2f} | {stood} | {e.get('reply_gbs', 0):.1f} GB/s | {r['sha']} |")
    print()
    print(f"Tokens a second by windows of {WIN} tokens of the reply:")
    for r in runs:
        print(f"  {r['name']:<24} " + " ".join(f"{x:.2f}" for x in r["win"]))
    print()
    by = {}
    for r in runs:
        by.setdefault(r["req"], {}).setdefault(r["cfg"], []).append(r)
    for req, cfgs in by.items():
        if "draft" in cfgs and "nodraft" in cfgs:
            a, b = cfgs["draft"][0], cfgs["nodraft"][0]
            same = "the same text" if a["sha"] == b["sha"] else "DIFFERENT texts"
            print(f"{req}: with the draft model {a['rate']:.3f} tokens/s, without {b['rate']:.3f}: "
                  f"x{a['rate'] / b['rate']:.3f}; {same}")


if __name__ == "__main__":
    main()
