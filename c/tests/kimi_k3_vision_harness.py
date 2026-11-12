#!/usr/bin/env python3
"""Kimi K3.1's pictures in kimi_k3.c (k3_vision.h) on a tiny fixture, through the serve protocol.

The tiny K3.1 checkpoint gets a tiny MoonViT tower of its own (two blocks of 32, two heads of 32,
an 8 x 8 position table, patches of 14, the patchmergerv2 projector to the language model's
width), drawn from its own seed, the media tokens in its tokenizer and media_placeholder_token_id
in its config. Then:

  - the handshake says CAPS vision=1, and vision=0 with K3_VISION=0;
  - the chat builder (--wire-test) places a picture as make_image_prompt does:
    <|media_begin|>image WxH<|media_content|>, one <|media_pad|> a row, <|media_end|>, in a user
    turn (S record) and in a tool result (R record);
  - the tower's rows (K3_VISION_DUMP) are a numpy reading of Moonshot's modeling_kimi_k3.py
    (written apart from the C: patches, bilinear positions, RMSNorm with FLT_EPSILON, 2-D RoPE
    on interleaved pairs, softmax attention, gelu-tanh MLP, the 2 x 2 merge, the projector with
    erf GELU and its RMSNorm) within 1e-4;
  - two pictures in one conversation, in their places;
  - the same conversation again reuses its state over the picture (a photo of the prompt's end:
    the tower does not run); the same words with another picture reuse nothing past the
    picture's place (its ids are its bytes' hash) and the tower runs;
  - refused by name: more pictures placed than IMAGE frames, a frame whose bytes are not its
    grid's, a picture for an engine without the tower.

Run by path:  python tests/kimi_k3_vision_harness.py --binary ./kimi_k3 --fixture ./kimi_k31_tiny
"""

import argparse
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEED = 20261112
VD, VH, VQ, VI, VDEPTH, PH, PW, P = 32, 2, 64, 64, 2, 8, 8, 14
MEDIA = {"<|media_begin|>": 280, "<|media_content|>": 281, "<|media_end|>": 282, "<|media_pad|>": 283}


def bf16(a):
    """f32 -> bf16 bits (round to nearest even) and the f32 those bits are."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    return r, (r.astype(np.uint32) << 16).view(np.float32)


def write_safetensors(path, tensors):
    header, blobs, off = {}, [], 0
    for name, (bits, shape) in tensors.items():
        raw = bits.tobytes()
        header[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(header).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def make_fixture(src, dst, tokenizer):
    """The tiny K3.1 checkpoint plus its tower; the f32 values of the tower's weights."""
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.suffix in (".safetensors", ".json") and f.name != "ref.json":
            shutil.copy(f, dst / f.name)
    cfg = json.loads((dst / "config.json").read_text())
    text = cfg.get("text_config", cfg)
    out = int(text["hidden_size"])
    cfg["media_placeholder_token_id"] = MEDIA["<|media_pad|>"]
    cfg["vision_config"] = {
        "activation_func": "gelu_pytorch_tanh", "attn_bias": False, "init_pos_emb_height": PH,
        "init_pos_emb_time": 4, "init_pos_emb_width": PW, "linear_bias": False, "merge_kernel_size": [2, 2],
        "merge_type": "sd2_tpool", "mlp_type": "mlp2", "mm_hidden_size": VD, "mm_projector_type": "patchmergerv2",
        "norm_type": "rmsnorm", "patch_embed_proj_bias": False, "patch_size": P, "pos_emb_interpolation_mode": "bilinear",
        "pos_emb_type": "divided_fixed", "projector_hidden_act": "gelu", "projector_ln_eps": 1e-5,
        "qkv_hidden_size": VQ, "text_hidden_size": out, "vt_hidden_size": VD, "vt_intermediate_size": VI,
        "vt_num_attention_heads": VH, "vt_num_hidden_layers": VDEPTH}
    (dst / "config.json").write_text(json.dumps(cfg, indent=1))
    tok = json.loads(Path(tokenizer).read_text(encoding="utf-8"))
    for content, i in MEDIA.items():
        tok["added_tokens"].append({"id": i, "content": content, "single_word": False, "lstrip": False,
                                    "rstrip": False, "normalized": False, "special": True})
    (dst / "tokenizer.json").write_text(json.dumps(tok), encoding="utf-8")
    rng = np.random.default_rng(SEED)
    W, T = {}, {}

    def put(name, shape, scale, base=0.0):
        bits, val = bf16(base + scale * rng.standard_normal(shape).astype(np.float32))
        T[name] = (bits, shape)
        W[name] = val.reshape(shape)

    put("vision_tower.patch_embed.proj.weight", (VD, 3, P, P), 0.05)
    put("vision_tower.patch_embed.pos_emb.weight", (PH, PW, VD), 0.5)
    for b in range(VDEPTH):
        pre = f"vision_tower.encoder.blocks.{b}."
        put(pre + "norm0.weight", (VD,), 0.1, 1.0)
        put(pre + "norm1.weight", (VD,), 0.1, 1.0)
        put(pre + "wqkv.weight", (3 * VQ, VD), 0.2)
        put(pre + "wo.weight", (VD, VQ), 0.1)
        put(pre + "mlp.fc0.weight", (VI, VD), 0.2)
        put(pre + "mlp.fc1.weight", (VD, VI), 0.1)
    put("vision_tower.encoder.final_layernorm.weight", (VD,), 0.1, 1.0)
    put("mm_projector.proj.0.weight", (4 * VD, 4 * VD), 0.1)
    put("mm_projector.proj.2.weight", (out, 4 * VD), 0.1)
    put("mm_projector.post_norm.weight", (out,), 0.1, 1.0)
    write_safetensors(dst / "model-vision.safetensors", T)
    return W, out


# ---- the reference: modeling_kimi_k3.py's MoonViT3dPretrainedModel and PatchMergerMLPV2, in numpy --
def rms(x, w, eps):
    x = x.astype(np.float32)
    return (x / np.sqrt((x * x).mean(-1, keepdims=True) + np.float32(eps))) * w


def interp_bilinear(t, oh, ow):
    """F.interpolate(mode='bilinear', align_corners=False) of a [h][w][d] table."""
    ih, iw, _ = t.shape

    def axis(n_in, n_out):
        s = np.float32(n_in) / np.float32(n_out)
        src = np.maximum(s * (np.arange(n_out, dtype=np.float32) + np.float32(0.5)) - np.float32(0.5), 0)
        i0 = src.astype(np.int64)
        i1 = np.minimum(i0 + 1, n_in - 1)
        l1 = (src - i0).astype(np.float32)
        return i0, i1, (1 - l1).astype(np.float32), l1
    y0, y1, ly0, ly1 = axis(ih, oh)
    x0, x1, lx0, lx1 = axis(iw, ow)
    a = t[y0][:, x0] * lx0[None, :, None] + t[y0][:, x1] * lx1[None, :, None]
    b = t[y1][:, x0] * lx0[None, :, None] + t[y1][:, x1] * lx1[None, :, None]
    return a * ly0[:, None, None] + b * ly1[:, None, None]


def tower_ref(W, rgb, gh, gw, out):
    eps = np.finfo(np.float32).eps
    x = (rgb.astype(np.float64) / 255.0).astype(np.float32)
    x = (x - np.float32(0.5)) * np.float32(2.0)
    pv = x.reshape(gh, P, gw, P, 3).transpose(0, 2, 4, 1, 3).reshape(gh * gw, 3 * P * P)
    h = pv @ W["vision_tower.patch_embed.proj.weight"].reshape(VD, -1).T
    pos = W["vision_tower.patch_embed.pos_emb.weight"]
    h = h + (pos if (gh, gw) == (PH, PW) else interp_bilinear(pos, gh, gw)).reshape(gh * gw, VD)
    hd = VQ // VH
    freqs = (1.0 / (10000.0 ** (np.arange(0, hd, 4)[: hd // 4].astype(np.float32) / hd))).astype(np.float32)
    ys, xs = np.divmod(np.arange(gh * gw), gw)
    ang = np.stack([np.outer(xs.astype(np.float32), freqs), np.outer(ys.astype(np.float32), freqs)],
                   -1).reshape(gh * gw, hd // 2)
    cis = np.cos(ang) + 1j * np.sin(ang)
    for b in range(VDEPTH):
        pre = f"vision_tower.encoder.blocks.{b}."
        q, k, v = (rms(h, W[pre + "norm0.weight"], eps) @ W[pre + "wqkv.weight"].T).reshape(-1, 3, VH, hd).transpose(1, 0, 2, 3)
        rot = lambda t: (t.reshape(*t.shape[:-1], hd // 2, 2) @ np.array([1, 1j])) * cis[:, None, :]
        q = np.stack([rot(q).real, rot(q).imag], -1).reshape(q.shape).astype(np.float32)
        k = np.stack([rot(k).real, rot(k).imag], -1).reshape(k.shape).astype(np.float32)
        s = np.einsum("nhe,mhe->hnm", q, k) / np.float32(math.sqrt(hd))
        s = np.exp(s - s.max(-1, keepdims=True))
        s = s / s.sum(-1, keepdims=True)
        ctx = np.einsum("hnm,mhe->nhe", s, v).reshape(-1, VQ)
        h = h + ctx @ W[pre + "wo.weight"].T
        u = rms(h, W[pre + "norm1.weight"], eps) @ W[pre + "mlp.fc0.weight"].T
        u = 0.5 * u * (1 + np.tanh(np.float32(math.sqrt(2 / math.pi)) * (u + np.float32(0.044715) * u ** 3)))
        h = h + u @ W[pre + "mlp.fc1.weight"].T
    h = rms(h, W["vision_tower.encoder.final_layernorm.weight"], eps)
    m = h.reshape(gh // 2, 2, gw // 2, 2, VD).transpose(0, 2, 1, 3, 4).reshape(-1, 4 * VD)
    m = m @ W["mm_projector.proj.0.weight"].T
    m = 0.5 * m * (1 + np.vectorize(math.erf)(m / math.sqrt(2)).astype(np.float32))
    m = m @ W["mm_projector.proj.2.weight"].T
    return rms(m, W["mm_projector.post_norm.weight"], 1e-5).astype(np.float32)


# ---- the serve protocol, with pictures ---------------------------------------------------------
class Serve:
    def __init__(self, binary, model, env, err):
        e = dict(os.environ, SERVE="1", SNAP=str(model), K3_PREFIX_LOG="1")
        e.update(env)
        self.err_path = Path(err)
        self.err = open(err, "wb")
        self.p = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, env=e)
        self.n = 0
        self.caps = {}
        while True:   # READY, then CAPS (the handshake)
            line = self.p.stdout.readline()
            if not line:
                raise AssertionError("the engine closed stdout before READY")
            f = line.decode("utf-8", "replace").split()
            if f and f[0] == "CAPS":
                self.caps.update(x.split("=", 1) for x in f[1:] if "=" in x)
            if f and f[0] == "STAT" or (f and f[0] == "CAPS"):
                break
        self.seen = 0

    def submit(self, prompt, max_tokens, images=()):
        """(generated bytes, ACCEPT np, ERROR text or None, this request's stderr)."""
        self.n += 1
        rid = f"r{self.n}"
        for rgb, gh, gw in images:
            blob = rgb.tobytes() if hasattr(rgb, "tobytes") else rgb
            self.p.stdin.write(f"IMAGE {rid} {len(blob)} {gh} {gw}\n".encode() + blob + b"\n")
        payload = prompt.encode("utf-8")
        self.p.stdin.write(f"SUBMIT {rid} 0 {len(payload)} {max_tokens} 0.0 1.0\n".encode() + payload + b"\n")
        self.p.stdin.flush()
        pieces, np_, error = [], -1, None
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise AssertionError(f"the engine closed stdout during {rid}")
            f = line.decode("utf-8", "replace").split()
            if not f:
                continue
            if f[0] == "ERROR":
                error = line.decode("utf-8", "replace")
                break
            if f[0] == "ACCEPT" and len(f) >= 3:
                np_ = int(f[2])
            elif f[0] == "DATA" and len(f) == 3:
                pieces.append(self.p.stdout.read(int(f[2])))
                self.p.stdout.read(1)
            elif f[0] == "DONE":
                break
        self.err.flush()
        text = self.err_path.read_text(errors="replace")
        lines, self.seen = text[self.seen:], len(text)
        return b"".join(pieces), np_, error, lines

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=60)
        except Exception:
            self.p.kill()
        self.err.close()


def picture(seed, gh, gw):
    return np.random.default_rng(seed).integers(0, 256, (gh * P, gw * P, 3), dtype=np.uint8)


def wire(records):
    out = "K3CHAT1\n"
    for r in records:
        out += r
    return out + "G 0\n"


def text_piece(t):
    return f"T {len(t.encode())}\n{t}"


FAILS = []


def check(cond, what):
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS.append(what)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--tokenizer", default=str(HERE / "tok_kimi_tiny.json"))
    a = ap.parse_args()
    work = Path(tempfile.mkdtemp(prefix="k3vision_"))
    try:
        model = work / "model"
        W, out = make_fixture(Path(a.fixture), model, a.tokenizer)
        env = {"K3_BITS": "32", "K3_MLA_BITS": "32", "K3_HEAD_BITS": "32", "K3_IDOT": "0", "K3_MAXT": "4096",
               "COLI_K3_CKPT": "2", "K3_VISION_DUMP": str(work)}
        p1, p2 = picture(1, 4, 6), picture(2, 6, 4)          # 56x84 and 84x56: 6 rows each
        user1 = "S user 3\n" + text_piece("look at ") + "I 84 56 6\n" + text_piece(" please")

        # the builder: a picture's tokens, in a user turn and in a tool result
        (work / "w.bin").write_bytes(wire([user1, "R 1 4 2\nshot" + "I 84 56 6\n" + text_piece("done")]).encode())
        r = subprocess.run([a.binary, str(model), "--wire-test", str(work / "w.bin")], capture_output=True, text=True,
                           env=dict(os.environ, SNAP=""))
        ids = [int(x) for x in r.stdout.split()]
        begins = [i for i, t in enumerate(ids) if t == MEDIA["<|media_begin|>"]]
        ok = len(begins) == 2
        for b in begins:
            c = ids.index(MEDIA["<|media_content|>"], b)
            ok = ok and ids[c + 1:c + 7] == [MEDIA["<|media_pad|>"]] * 6 and ids[c + 7] == MEDIA["<|media_end|>"]
        check(ok, "the builder places two pictures (S and R records): begin, image WxH, content, 6 pads, end")
        (work / "b.bin").write_bytes(b"K3CHAT1\nS user 1\nI 84 56\nG 0\n")
        bad = subprocess.run([a.binary, str(model), "--wire-test", str(work / "b.bin")], capture_output=True, text=True,
                             env=dict(os.environ, SNAP=""))
        check(bad.returncode != 0, "a malformed picture piece is refused")

        s = Serve(a.binary, model, env, work / "serve.err")
        try:
            check(s.caps.get("vision") == "1", "CAPS vision=1 with the tower")
            rep, np1, err, log = s.submit(wire([user1]), 4, [(p1, 4, 6)])
            check(err is None and np1 > 6, f"a picture is served (ACCEPT {np1} tokens)")
            check("through the tower" in log and "1 through the tower" in log, "the tower ran once")
            got = np.fromfile(work / "rows.f32", dtype=np.float32).reshape(-1, out)
            ref = tower_ref(W, p1, 4, 6, out)
            rel = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
            check(got.shape == ref.shape and rel < 1e-4, f"the tower's rows are the reference's (rel {rel:.2e})")

            rep2, np2, err, log = s.submit(wire([user1]), 4, [(p1, 4, 6)])
            check(err is None and rep2 == rep and np2 == np1, "the same conversation again: the same reply")
            check("through the tower" not in log, "... and the tower did not run again (state or cache)")
            m = re.search(r"\[PREFIX\] reusing (\d+) of (\d+)", log)
            check(m is not None and int(m.group(1)) == np1 or "nothing to compute" in log,
                  "... its state reused over the picture")

            rep3, np3, err, log = s.submit(wire([user1]), 4, [(picture(3, 4, 6), 4, 6)])
            m = re.search(r"\[PREFIX\] reusing (\d+) of (\d+)", log)
            start = ids.index(MEDIA["<|media_pad|>"]) + 1   # BOS + the builder's ids
            check(err is None and "1 through the tower" in log, "another picture, the same words: the tower runs")
            check(m is None or int(m.group(1)) <= start, f"... and no state past the picture's place is reused ({m.group(1) if m else 0} <= {start})")

            two = "S user 4\n" + text_piece("compare ") + "I 84 56 6\n" + text_piece(" and ") + "I 56 84 6\n"
            rep4, np4, err, log = s.submit(wire([two]), 4, [(p1, 4, 6), (p2, 6, 4)])
            check(err is None and np4 > 12, "two pictures in one turn")
            check("1 through the tower, 1 from the cache" in log, "... the first from the cache, the second through the tower")

            _r, _n, err, _l = s.submit(wire([two]), 4, [(p1, 4, 6)])
            check(err is not None and "BAD_IMAGE" in err and "2 pictures" in err, "two places, one frame: refused by name")
            _r, _n, err, _l = s.submit(wire([user1]), 4, [(p1[:-1], 4, 6)])
            check(err is not None and "BAD_IMAGE" in err, "a frame whose bytes are not its grid's: refused")
            _r, _n, err, _l = s.submit(wire([user1]), 4, [(picture(5, 4, 4), 4, 4)])   # 4 rows for 6 places
            check(err is not None and "BAD_IMAGE" in err, "a picture whose rows are not its place's: refused")
            rep5, _n, err, _l = s.submit(wire(["M user 5\nhello"]), 4)
            check(err is None, "text after the refusals is served")
        finally:
            s.close()

        s = Serve(a.binary, model, dict(env, K3_VISION="0"), work / "serve0.err")
        try:
            check(s.caps.get("vision") == "0", "CAPS vision=0 with K3_VISION=0")
            _r, _n, err, _l = s.submit(wire([user1]), 4, [(p1, 4, 6)])
            check(err is not None and "no vision tower" in err, "a picture without the tower: refused by name")
        finally:
            s.close()
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("kimi_k3_vision_harness: " + (f"{len(FAILS)} failure(s)" if FAILS else "ok"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
