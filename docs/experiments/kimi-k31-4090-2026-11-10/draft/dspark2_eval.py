"""dspark2_eval.py <dump prefix> <anchors.npz> <out.npz> [max anchors]
Inferact/Kimi-K3.1-Pro-DSpark (the checkpoint after the preview; speculators' DSparkDraftModel, a
Qwen3-style stack) run on Kimi K3.1 Pro's own hidden rows (kimi_k3's K3_HID_DUMP), at the anchors
of anchors.npz (tid, pos). Its aux_hidden_state_layer_ids 76, 80, 84, 88, 92 are vLLM's captures
after the 0-based layers 75, 79, 83, 87, 91 (DSPARK_LAYERS names others, to test the convention).
  context row p (p < a):  hidden_norm(fc(cat of the 5 rows)), then each layer's k_proj (per-head
      RMSNorm k_norm, RoPE at p) and v_proj: 16 heads of 64
  block [x_a, MASK x 6] at positions a..a+6: 5 pre-norm layers, 96 query heads over the 16 key
      heads (6 each), q_norm and RoPE (theta 10000, halves), causal inside the block, the context
      positions before a that are within 2,048 of the query (speculators' mask: p > a + i - 2048)
  norm, the target's head, the Markov bias w2(w1[prev]) left to right from x_a, the confidence
      sigmoid(proj(cat[hidden, markov embed]))
Written after reading speculators' dflash/dspark core and vLLM's qwen3_dflash.py; nothing of
theirs is run. Saves each anchor's 7 drafts, confidences and top probabilities.
DSPARK_BITS=8|4: the draft's matrices rounded as kimi_k3 holds a matrix (8: a scale a row, 4: a
scale a group of 64 along a row). DSPARK_HEAD_BITS=8: the target's head rounded to int8 rows too."""
import json, math, os, struct, sys, time
import numpy as np
import torch

DUMP, ANCH, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
MAXA = int(sys.argv[4]) if len(sys.argv) > 4 else 0
DRAFT = os.environ.get('DSPARK_MODEL', 'C:/AI/Kimi-K3.1-Pro-DSpark/model.safetensors')
SHARD = 'C:/AI/Kimi-K3.1-Pro/model-00094-of-000096.safetensors'
LAYERS = [int(x) for x in os.environ.get('DSPARK_LAYERS', '75,79,83,87,91').split(',')]
WIN = int(os.environ.get('DSPARK_WINDOW', '2048'))
D, H, G, HD, BLOCK, MASK, EPS, RANK = 7168, 96, 16, 64, 7, 163839, 1e-5, 512
REP = H // G
dev = torch.device('cuda')
torch.backends.cuda.matmul.allow_tf32 = False


def st_open(path):
    f = open(path, 'rb')
    n = struct.unpack('<Q', f.read(8))[0]
    return f, json.loads(f.read(n)), 8 + n


def st_tensor(f, hdr, base, name, device=dev):
    m = hdr[name]
    assert m['dtype'] == 'BF16', (name, m['dtype'])
    a, b = m['data_offsets']
    f.seek(base + a)
    buf = np.frombuffer(f.read(b - a), dtype=np.int16).reshape(m['shape'])
    return torch.from_numpy(buf.copy()).view(torch.bfloat16).to(device)


def rounded(w, bits):
    """kimi_k3's w_load: 8 bits a scale a row (|max| / 127), 4 bits a scale a group of 64 (|max| / 7)"""
    f = w.float()
    if bits >= 8:
        sc = f.abs().amax(dim=-1, keepdim=True).clamp_min(1e-20) / 127.0
        return (torch.clamp(torch.round(f / sc), -127, 127) * sc).to(torch.bfloat16)
    g = f.view(f.shape[0], -1, 64)
    sc = g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-20) / 7.0
    return (torch.clamp(torch.round(g / sc), -8, 7) * sc).view(f.shape).to(torch.bfloat16)


t0 = time.time()
f, hdr, base = st_open(DRAFT)
W = {k: st_tensor(f, hdr, base, k) for k in hdr if k != '__metadata__'}
f.close()
BITS = int(os.environ.get('DSPARK_BITS', '0'))
if BITS:
    n = 0
    for k in list(W):
        w = W[k]
        if w.dim() != 2 or w.shape[1] % 64 or w.shape[0] < 64 or 'markov_w1' in k: continue   # w1 is a lookup
        W[k] = rounded(w, BITS); n += 1
    print(f'{n} draft matrices rounded to {BITS} bits', flush=True)
f, hdr, base = st_open(SHARD)
lm_head = st_tensor(f, hdr, base, 'language_model.lm_head.weight')
embed = st_tensor(f, hdr, base, 'language_model.model.embed_tokens.weight', 'cpu')
f.close()
if int(os.environ.get('DSPARK_HEAD_BITS', '0')) == 8:
    for s in range(0, lm_head.shape[0], 16384): lm_head[s:s + 16384] = rounded(lm_head[s:s + 16384], 8)
    print('the head rounded to 8 bits', flush=True)
print(f'weights: draft {sum(v.numel() for v in W.values()) / 1e9:.2f} B, head {tuple(lm_head.shape)}, layers {LAYERS}, window {WIN}; {time.time() - t0:.0f} s', flush=True)


def lin(x, w):
    return (x.to(torch.bfloat16) @ w.t()).float()


def rms(x, w):
    x = x.float()
    return w.float() * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS))


INV = (1.0 / (10000.0 ** (torch.arange(0, HD, 2, dtype=torch.float32) / HD))).to(dev)   # 32 angles


def rope(x, positions):   # x [..., heads, 64], positions broadcastable to x's dims before the heads
    ang = positions.float().unsqueeze(-1).unsqueeze(-1) * INV     # [..., 1, 32]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


L = []
for i in range(5):
    p = f'layers.{i}.'
    L.append(dict(in_ln=W[p + 'input_layernorm.weight'], post_ln=W[p + 'post_attention_layernorm.weight'],
                  q=W[p + 'self_attn.q_proj.weight'], k=W[p + 'self_attn.k_proj.weight'], v=W[p + 'self_attn.v_proj.weight'],
                  o=W[p + 'self_attn.o_proj.weight'], qn=W[p + 'self_attn.q_norm.weight'], kn=W[p + 'self_attn.k_norm.weight'],
                  gate=W[p + 'mlp.gate_proj.weight'], up=W[p + 'mlp.up_proj.weight'], down=W[p + 'mlp.down_proj.weight']))
SCALE = 1.0 / math.sqrt(HD)

steps = [tuple(map(int, l.split())) for l in open(DUMP + '.steps.txt') if l.strip()]
ids_all = np.fromfile(DUMP + '.ids.i32', dtype=np.int32)
hid = [np.memmap(f'{DUMP}.L{l:02d}.f32', dtype=np.float32, mode='r').reshape(-1, D) for l in LAYERS]
trans, row = [], 0
for pos0, C in steps:
    if pos0 == 0: trans.append([])
    assert pos0 == len(trans[-1]), 'a step that does not continue its transcript'
    trans[-1].extend(range(row, row + C)); row += C
assert row == len(ids_all) == hid[0].shape[0], (row, len(ids_all), hid[0].shape)
print('transcripts:', [len(t) for t in trans], flush=True)

A = np.load(ANCH)
a_tid, a_pos = A['tid'], A['pos']
if MAXA and len(a_pos) > MAXA:
    pick = np.sort(np.random.default_rng(1).choice(len(a_pos), MAXA, replace=False))
    a_tid, a_pos = a_tid[pick], a_pos[pick]
res = dict(tid=[], pos=[], draft=[], conf=[], pmax=[])
GOLD = os.environ.get('DSPARK_GOLD', '')     # a file of the first batch's tensors, for the engine's port


@torch.inference_mode()
def run_transcript(tid):
    rows = np.asarray(trans[tid]); N = len(rows)
    r0 = int(rows[0]); assert (rows == np.arange(r0, r0 + N)).all()
    ids = torch.from_numpy(ids_all[r0:r0 + N].astype(np.int64))
    # the context rows' keys and values of every layer: 16 heads of 64, f16 on the card
    Kc = [torch.empty((N, G, HD), dtype=torch.float16, device=dev) for _ in L]
    Vc = [torch.empty((N, G, HD), dtype=torch.float16, device=dev) for _ in L]
    for s in range(0, N, 2048):
        e = min(N, s + 2048)
        x = torch.cat([torch.from_numpy(np.ascontiguousarray(h[r0 + s:r0 + e])).to(dev) for h in hid], dim=-1)
        ctx = rms(lin(x, W['fc.weight']), W['hidden_norm.weight'])
        ps = torch.arange(s, e, device=dev)
        for li, l in enumerate(L):
            Kc[li][s:e] = rope(rms(lin(ctx, l['k']).view(-1, G, HD), l['kn']), ps).half()
            Vc[li][s:e] = lin(ctx, l['v']).view(-1, G, HD).half()
    anchors = np.sort(a_pos[a_tid == tid])
    anchors = anchors[(anchors >= 1) & (anchors + BLOCK < N)]
    done, t1, nb = 0, time.time(), 0
    WC = WIN - 1                                                    # context rows a block's first slot may see
    causal = torch.triu(torch.ones(BLOCK, BLOCK, dtype=torch.bool, device=dev), 1).view(1, BLOCK, 1, 1, BLOCK)
    slot = torch.arange(BLOCK, device=dev).view(1, BLOCK, 1, 1, 1)
    while done < len(anchors):
        a = torch.from_numpy(anchors[done:done + 24].astype(np.int64)); B = len(a)
        ad = a.to(dev)
        tok = torch.full((B, BLOCK), MASK, dtype=torch.int64); tok[:, 0] = ids[a]
        h = embed[tok].to(dev).float()                              # [B, 7, D]
        pos = (ad.unsqueeze(1) + torch.arange(BLOCK, device=dev))   # [B, 7]
        # window column w is context position a - WC + w; slot i sees w >= i (p > a + i - WIN) of those at 0 or later
        cpos = ad.view(B, 1) - WC + torch.arange(WC, device=dev).view(1, WC)          # [B, WC]
        gidx = cpos.clamp_min(0)
        hide = (cpos.view(B, 1, 1, 1, WC) < 0) | (torch.arange(WC, device=dev).view(1, 1, 1, 1, WC) < slot)
        gold = {}
        for li, l in enumerate(L):
            x = rms(h, l['in_ln'])
            q = rope(rms(lin(x, l['q']).view(B, BLOCK, H, HD), l['qn']), pos).view(B, BLOCK, G, REP, HD)
            kb = rope(rms(lin(x, l['k']).view(B, BLOCK, G, HD), l['kn']), pos)
            vb = lin(x, l['v']).view(B, BLOCK, G, HD)
            kc, vc = Kc[li][gidx].float(), Vc[li][gidx].float()     # [B, WC, G, HD]
            sc = torch.einsum('bigrd,bwgd->bigrw', q, kc).masked_fill(hide, float('-inf'))
            sb = torch.einsum('bigrd,bjgd->bigrj', q, kb).masked_fill(causal, float('-inf'))
            p = torch.softmax(torch.cat([sc, sb], dim=-1) * SCALE, dim=-1)
            att = torch.einsum('bigrw,bwgd->bigrd', p[..., :WC], vc) + torch.einsum('bigrj,bjgd->bigrd', p[..., WC:], vb)
            h = h + lin(att.reshape(B, BLOCK, H * HD), l['o'])
            x = rms(h, l['post_ln'])
            h = h + lin(torch.nn.functional.silu(lin(x, l['gate'])) * lin(x, l['up']), l['down'])
            if GOLD and nb == 0: gold[f'h{li}'] = h[0].cpu().numpy()
        hh = rms(h, W['norm.weight'])
        logits = lin(hh, lm_head)
        prev = ids[a].to(dev)
        drafts, confs, pmaxs = [], [], []
        for i in range(BLOCK):
            me = W['markov_head.markov_w1.weight'][prev]
            lg = logits[:, i] + (me @ W['markov_head.markov_w2.weight'].t()).float()
            pm, d = torch.softmax(lg, dim=-1).max(dim=-1)
            cf = torch.sigmoid((torch.cat([hh[:, i], me.float()], dim=-1) @ W['confidence_head.proj.weight'].float().t()).squeeze(-1)
                               + W['confidence_head.proj.bias'].float())
            drafts.append(d); confs.append(cf); pmaxs.append(pm); prev = d
        res['tid'].append(np.full(B, tid)); res['pos'].append(a.numpy())
        res['draft'].append(torch.stack(drafts, 1).cpu().numpy()); res['conf'].append(torch.stack(confs, 1).cpu().numpy())
        res['pmax'].append(torch.stack(pmaxs, 1).cpu().numpy())
        if GOLD and nb == 0:
            a0 = int(a[0])
            gold.update(tid=tid, pos=a0, ids=ids[max(0, a0 - WC):a0 + 1].numpy(), hh=hh[0].cpu().numpy(),
                        draft=res['draft'][-1][0], conf=res['conf'][-1][0], pmax=res['pmax'][-1][0],
                        top5=torch.topk(logits[0], 5, dim=-1).indices.cpu().numpy())
            np.savez(GOLD, **gold); print('gold tensors of the first anchor to', GOLD, flush=True)
        done += B; nb += 1
        if nb % 40 == 0: print(f'  transcript {tid}: {done}/{len(anchors)} anchors, {time.time() - t1:.0f} s', flush=True)
    del Kc, Vc
    torch.cuda.empty_cache()


for tid in range(len(trans)):
    if (a_tid == tid).any(): run_transcript(tid)
np.savez(OUT, **{k: np.concatenate(v) for k, v in res.items()})
print('saved', OUT, sum(len(x) for x in res['pos']), 'anchors;', f'{time.time() - t0:.0f} s', flush=True)
