#!/usr/bin/env python3
"""The engine's draft model against a PyTorch reading of it, from the same rows.

kimi_k3 with K3_DRAFT (k3_draft.h, backend_cuda_k3_draft.cuh) runs Inferact/Kimi-K3.1-Pro-DSpark
on the card. With K3_DRAFT_DUMP=<prefix> it writes what the card was given and what it answered:

  <prefix>.taps.f32      the context rows, [5 * hidden] f32 each (a position's five tapped rows)
  <prefix>.events.txt    in order:  R pos0 n               n rows from position pos0 on went up
                                    D pos tk lo id p ...   a pass after token tk at position pos over
                                                           the rows lo .. pos-1: its 7 picks, each
                                                           with its probability

This file computes every pass again -- speculators' DSparkDraftModel as its code reads (fc and
hidden_norm over a position's rows; each layer's k_proj with k_norm and RoPE, and v_proj; the
block [token, mask x 6] through five pre-norm layers, 96 query heads over 16 key heads, causal
inside the block, the context before it within the window; norm, the target's head, the Markov
bias left to right) -- with the matrices rounded to int8 rows as the card holds them, and
compares the picks. Nothing of speculators or vLLM is imported.

  python tools/k3_draft_ref.py <prefix> --draft C:\\AI\\Kimi-K3.1-Pro-DSpark --model C:\\AI\\Kimi-K3.1-Pro

A pick may differ where two tokens are within the two orders of the sums of each other (about
one in several hundred, at probabilities far below any gate); the picks after it in that pass
then differ too and are not compared. Needs torch with CUDA and about 9 GB of VRAM.
"""
import argparse
import glob
import json
import math
import os
import struct

import numpy as np
import torch


def st_open(path):
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    return f, json.loads(f.read(n)), 8 + n


def st_tensor(f, hdr, base, name, device):
    m = hdr[name]
    if m["dtype"] != "BF16":
        raise SystemExit(f"{name}: {m['dtype']}, not BF16")
    a, b = m["data_offsets"]
    f.seek(base + a)
    buf = np.frombuffer(f.read(b - a), dtype=np.int16).reshape(m["shape"])
    return torch.from_numpy(buf.copy()).view(torch.bfloat16).to(device)


def find_tensor(model_dir, name):
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors")), reverse=True):
        f, hdr, base = st_open(path)
        if name in hdr:
            return f, hdr, base
        f.close()
    raise SystemExit(f"{name} is in no safetensors file of {model_dir}")


def rounded(w):
    """int8 rows with a scale each, |max| / 127: the engine's w_load and the card's kd_quant"""
    f = w.float()
    sc = f.abs().amax(dim=-1, keepdim=True).clamp_min(1e-20) / 127.0
    return torch.clamp(torch.round(f / sc), -127, 127) * sc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefix", help="K3_DRAFT_DUMP's prefix")
    ap.add_argument("--draft", required=True, help="the draft model's folder (config.json, model.safetensors)")
    ap.add_argument("--model", required=True, help="the target's folder (its embedding and head)")
    ap.add_argument("--max", type=int, default=0, help="compare this many passes at most")
    args = ap.parse_args()
    dev = torch.device("cuda")
    cfg = json.load(open(os.path.join(args.draft, "config.json")))
    tl = cfg["transformer_layer_config"]
    D, H, G, HD = tl["hidden_size"], tl["num_attention_heads"], tl["num_key_value_heads"], tl["head_dim"]
    BLOCK, MASK, EPS, WIN = cfg["block_size"], cfg["mask_token_id"], tl["rms_norm_eps"], tl["sliding_window"]
    T, NL, REP = len(cfg["aux_hidden_state_layer_ids"]), tl["num_hidden_layers"], H // G
    theta = tl["rope_parameters"]["rope_theta"]

    f, hdr, base = st_open(os.path.join(args.draft, "model.safetensors"))
    W = {k: st_tensor(f, hdr, base, k, dev) for k in hdr if k != "__metadata__"}
    f.close()
    for k in list(W):
        w = W[k]
        W[k] = rounded(w).half() if w.dim() == 2 and w.shape[0] >= 64 and "markov_w1" not in k else w.float()
    f, hdr, base = find_tensor(args.model, "language_model.lm_head.weight")
    lm_head = st_tensor(f, hdr, base, "language_model.lm_head.weight", dev)
    embed = st_tensor(f, hdr, base, "language_model.model.embed_tokens.weight", "cpu")
    f.close()
    head = torch.empty(lm_head.shape, dtype=torch.float16, device=dev)
    for s in range(0, lm_head.shape[0], 16384):
        head[s:s + 16384] = rounded(lm_head[s:s + 16384]).half()
    del lm_head

    lin = lambda x, w: x.float() @ w.float().t()
    rms = lambda x, w: w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS))
    inv = (1.0 / (theta ** (torch.arange(0, HD, 2, dtype=torch.float32) / HD))).to(dev)

    def rope(x, positions):   # pairs (i, i + HD/2), the angle the position times inv[i]
        ang = positions.float().unsqueeze(-1).unsqueeze(-1) * inv
        cos, sin = ang.cos(), ang.sin()
        x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)

    names = (("in_ln", "input_layernorm.weight"), ("post_ln", "post_attention_layernorm.weight"), ("q", "self_attn.q_proj.weight"),
             ("k", "self_attn.k_proj.weight"), ("v", "self_attn.v_proj.weight"), ("o", "self_attn.o_proj.weight"),
             ("qn", "self_attn.q_norm.weight"), ("kn", "self_attn.k_norm.weight"), ("gate", "mlp.gate_proj.weight"),
             ("up", "mlp.up_proj.weight"), ("down", "mlp.down_proj.weight"))
    L = [{k: W[f"layers.{i}.{n}"] for k, n in names} for i in range(NL)]
    scale = 1.0 / math.sqrt(HD)
    taps = np.memmap(args.prefix + ".taps.f32", dtype=np.float32, mode="r").reshape(-1, T * D)
    Kc, Vc = {}, {}   # position -> its keys and values of every layer

    @torch.inference_mode()
    def context(pos0, n, r0):
        for s in range(0, n, 512):
            e = min(n, s + 512)
            x = torch.from_numpy(np.ascontiguousarray(taps[r0 + s:r0 + e])).to(dev)
            ctx = rms(lin(x, W["fc.weight"]), W["hidden_norm.weight"])
            ps = torch.arange(pos0 + s, pos0 + e, device=dev)
            ks = torch.stack([rope(rms(lin(ctx, l["k"]).view(-1, G, HD), l["kn"]), ps) for l in L], 1)
            vs = torch.stack([lin(ctx, l["v"]).view(-1, G, HD) for l in L], 1)
            for j in range(e - s):
                Kc[pos0 + s + j], Vc[pos0 + s + j] = ks[j], vs[j]

    @torch.inference_mode()
    def draft(a, tk, lo):
        first = max(lo, a - (WIN - 1))
        nc = a - first
        kc = torch.stack([Kc[p] for p in range(first, a)], 0)
        vc = torch.stack([Vc[p] for p in range(first, a)], 0)
        tok = torch.full((BLOCK,), MASK, dtype=torch.int64)
        tok[0] = tk
        h = embed[tok].to(dev).float()
        pos = a + torch.arange(BLOCK, device=dev)
        slot = torch.arange(BLOCK, device=dev).view(BLOCK, 1, 1, 1)
        hide = (first + torch.arange(nc, device=dev)).view(1, 1, 1, nc) <= a + slot - WIN   # seen: p > a + i - WIN
        causal = torch.triu(torch.ones(BLOCK, BLOCK, dtype=torch.bool, device=dev), 1).view(BLOCK, 1, 1, BLOCK)
        for li, l in enumerate(L):
            x = rms(h, l["in_ln"])
            q = rope(rms(lin(x, l["q"]).view(BLOCK, H, HD), l["qn"]), pos).view(BLOCK, G, REP, HD)
            kb = rope(rms(lin(x, l["k"]).view(BLOCK, G, HD), l["kn"]), pos)
            vb = lin(x, l["v"]).view(BLOCK, G, HD)
            sc = torch.einsum("igrd,wgd->igrw", q, kc[:, li]).masked_fill(hide, float("-inf"))
            sb = torch.einsum("igrd,jgd->igrj", q, kb).masked_fill(causal, float("-inf"))
            p = torch.softmax(torch.cat([sc, sb], dim=-1) * scale, dim=-1)
            att = torch.einsum("igrw,wgd->igrd", p[..., :nc], vc[:, li]) + torch.einsum("igrj,jgd->igrd", p[..., nc:], vb)
            h = h + lin(att.reshape(BLOCK, H * HD), l["o"])
            x = rms(h, l["post_ln"])
            h = h + lin(torch.nn.functional.silu(lin(x, l["gate"])) * lin(x, l["up"]), l["down"])
        logits = lin(rms(h, W["norm.weight"]), head)
        prev, ids, ps = tk, [], []
        for i in range(BLOCK):
            lg = logits[i] + lin(W["markov_head.markov_w1.weight"][prev], W["markov_head.markov_w2.weight"])
            pm, d = torch.softmax(lg, dim=-1).max(dim=-1)
            ids.append(int(d)); ps.append(float(pm)); prev = int(d)
        return ids, ps

    nrow = nd = whole = 0
    same_at, cnt_at, maxdp, first_bad = [0] * BLOCK, [0] * BLOCK, 0.0, None
    for line in open(args.prefix + ".events.txt"):
        v = line.split()
        if not v:
            continue
        if v[0] == "R":
            context(int(v[1]), int(v[2]), nrow)
            nrow += int(v[2])
        elif v[0] == "D" and not (args.max and nd >= args.max):
            pos, tk, lo = int(v[1]), int(v[2]), int(v[3])
            eng_ids, eng_p = [int(x) for x in v[4::2]], [float(x) for x in v[5::2]]
            ids, ps = draft(pos, tk, lo)
            nd += 1
            ok = True
            for i in range(min(len(eng_ids), BLOCK)):
                cnt_at[i] += 1
                if eng_ids[i] != ids[i]:
                    ok = False
                    if first_bad is None:
                        first_bad = (pos, i, eng_ids, ids, eng_p, ps)
                    break
                same_at[i] += 1
                maxdp = max(maxdp, abs(eng_p[i] - ps[i]))
            whole += ok
    print(f"{nd} passes; all {BLOCK} picks the same in {whole}; by place: " + " ".join(f"{a}/{b}" for a, b in zip(same_at, cnt_at)) +
          f"; the largest difference of a probability where the picks agree: {maxdp:.2e}")
    if first_bad:
        pos, i, e, r, ep, rp = first_bad
        print(f"the first difference: the pass at {pos}, pick {i + 1}: the engine {e[i]} at {ep[i]:.4f}, here {r[i]} at {rp[i]:.4f}")
    return 0 if nd and sum(same_at) >= 0.98 * sum(cnt_at) else 1


if __name__ == "__main__":
    raise SystemExit(main())
