# Kimi K3 engine (`c/kimi_k3.c`)

A sibling engine for [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3)
(2.8T parameters, 104B active, 93 layers), following the one-engine-per-family
pattern (`colibri.c` = GLM-5.2, `olmoe.c`, `inkling.c`). It shares `st.h`,
`json.h`, `tok.h`, `quant.h` and touches nothing in the GLM engine.

```
make kimi_k3
./kimi_k3 <model_dir> "prompt" --ngen 64
```

`<model_dir>` is either the plain HF snapshot (config.json +
`model-*-of-000096.safetensors`) or a repacked container (below). Text-only:
the vision tower (shards 95–96) is never read.

## Architecture notes

K3 is not a DeepSeek-shaped model; four pieces are new relative to everything
else in this repo:

- **Hybrid attention, NoPE.** 69 KDA (Kimi Delta Attention) layers + 24 gated
  MLA layers (every 4th, plus the final layer). There is no positional
  encoding anywhere — position lives in KDA's convolution and decay. The MLA
  block has DeepSeek dims (q_lora 1536, kv_lora 512, nope 128 + 64 uncached
  extra dims, v 128, 96 heads) and both attention types multiply their output
  by a full-rank sigmoid gate `σ(W_g x)` before `o_proj`.
- **KDA** per head (dim 128, 96 heads): `q,k,v = SiLU(causal-conv4(W x))`,
  q/k L2-normalized (ε=1e-6 inside the sqrt), q scaled by 128^-1/2; per-channel
  decay `gk = -5·σ(exp(A_log[h])·(W_fb W_fa x + dt_bias))`; the delta-rule
  state update `S = (I − βkkᵀ)·Diag(e^gk)·S + βkvᵀ`; output
  `W_o[σ(W_g x) ⊙ RMSNorm_head(Sᵀq)]`. The checkpoint stores `A_log` as
  `[128]`: that is the per-head `[96]` parameter zero-padded (verified —
  entries 96..127 are exactly 0). Decode state: 96×128×128 f32 per layer.
- **AttnRes replaces the residual stream.** Each layer keeps a running
  `prefix_sum`; at layers 0, 12, 24, …, 84 it is snapshotted into a block
  list. Twice per layer (before attention, before the MLP) and once at the
  end, the hidden state is REPLACED by a softmax mix over
  `[snapshots…, prefix_sum]`, scored by `(v · (res_norm.w ⊙ res_proj.w)) /
  rms(v)` and mixing the raw (un-normalized) entries in fp32.
- **Stable LatentMoE.** Sigmoid router `[896, 7168]` + score-correction bias,
  top-16 selected on biased scores, weights are the raw scores renormalized.
  The token is projected 7168→3584, the 16 experts run in that latent space
  (GLU, inter 3072), the aggregate is RMSNorm-ed and projected back
  3584→7168; two fused full-width shared experts (inter 6144) are added.
  Activation is **SiTU-GLU**: `4·tanh(g/4)·σ(g) · 25·tanh(u/25)`.

### Weights

Only the routed experts (`experts.N.w{1,2,3}`) are quantized in the
checkpoint — **MXFP4** (compressed-tensors `mxfp4-pack-quantized`,
quantization-aware trained): e2m1 nibbles packed two per byte (low nibble =
even column, bit 3 = sign) with a ue8m0 power-of-two scale per 32 columns,
`w = v · 2^(scale-127)`. 17.55 MB per expert, 82,432 experts ≈ 1.45 TB.
`quant.h` gained `matmul_mxfp4` (scalar + AVX2) that computes on this layout
directly — expert bytes are **never converted**; QAT weights are exactly the
trained values and any re-encode only adds error.

Everything else is BF16 and is quantized into RAM **at load time** (int8
per-row / int4-g64), or read pre-quantized from a repacked container.

## Streaming design

Experts are streamed straight from the safetensors shards through a per-layer
LRU (`K3_EXPERT_GB`). Measured layout facts the engine exploits: the six
tensors of an expert are stored back-to-back (one expert = one `pread`), and
experts are *not* id-ordered inside a shard but by their ids as text
("0", "1", "10", "100", ...: safetensors sorts tensor names), so loads run in
disk-offset order, and a step adds its experts in that order wherever the experts
are (one shard, or a shard cut over drives by `tools/k3_download.py`). Reads are
by default **O_DIRECT** (`K3_DIRECT=0` for buffered): the resident
weights leave little page-cache headroom, and flat routing means cached reads
mostly cannot be reused anyway — measured on the 93-layer model this took
expert reads from ~1.8 to ~6.3 GB/s (drive ceiling 7.1) and decode from ~21
to ~9.4 s/token. The LRU slot floor is 1 (experts are consumed one at a time),
so `K3_EXPERT_GB` is honored even at tiny budgets.

**Readers per drive** (`K3_PIPE`, default on): the reads run on reader threads,
so expert j's matmuls overlap the reads of the experts after it. The shards'
storage devices are told apart, each gets `K3_IO_PER_DRIVE` readers (2) and two
queues, a step's demands before the next layer's prefetches; on Windows each
reader opens its own unbuffered handle per shard, because reads through one
synchronous handle are serialized (the four loader threads this replaced read one
expert of a layer's shard at a time there). With the checkpoint cut over two
drives every layer's reads go to both at once: on the RTX 4090 machine
(docs/kimi_k3-4090.md) 17.5 MB reads at three readers give 10.3 GB/s on `C:`
and 6.6 on `D:` together, 16.9 GB/s.

**Next-layer prefetch** (`K3_PREFETCH`, default on in decode steps of up to
`K3_PREFETCH_ROWS`, 4, rows): after a step's attention in layer i, layer i+1's
router applied to the blocks mixed with i+1's own AttnRes weights (`K3_PREFETCH_MIX=1`;
0: layer i's MLP input), normed by i+1's post-attention norm, predicts the experts i+1
will ask for, and the `K3_PREFETCH_TOP` (12) best of them that its cache lacks are read
ahead, queued behind layer i's own reads; when i+1's routing comes, the predicted experts
it did not choose leave the queue unread. A wrong prediction costs a read and a
slot that is reused first, never a number. The run's last line says how many were
read ahead and used and what share of the routed experts the predictions held. On the
real model (docs/experiments/kimi-k3-4090-2026-10-31) the next layer's mix predicts
66.9% of the routed experts against 63.2% from layer i's input, and 12 a row instead of
the routing's 16 read 12.2 instead of 13.8 GB a decode token for the same speed.

**Waves** (`K3_WAVES`, default on): a MoE step gives every expert its cache lacks a slot
at once and queues the reads on their drives, interleaved, right after the routing; the
latent projection runs while they arrive, then whatever has arrived is computed as one
wave (gate and up of every arrived expert over all workers, then down), and the shared
experts run while no read is in. Each (row, choice) pair writes its own row and the
weighted sum runs at the end in the union's order, so the bytes are the one-expert-at-a-
time loop's (`K3_WAVES=0`); a decode token on the 4090 machine went from 1.62 to 1.50 s,
the 35-token prompt from 57.0 to 41.3 s (its reads had gone to one drive at a time).
`K3_IO_DELAY_US` slows every read for the tests that exercise the queues. None of
this moves a bit: the logits are those of the old loader in every configuration
tested (cache sizes, chunk sizes, one or two folders on two drives, buffered,
slow disks).

`K3_IDOT` (default on) computes the expert matmuls with
per-32-group int8-quantized activations via integer dots (the e2m1 doubled
values are exact int8 — same trick as `dot_i4i8`), and `K3_TOPP` optionally
drops the low-weight tail of the top-16 (renormalized; quality-gate any
setting with a `K3_LOGITS` A/B first). Note that K3's router was trained with Quantile Balancing (deliberately
flat expert usage), so LRU hit rates are structurally lower than on models
with skewed routing — expect the expert tier to be bandwidth-bound.

## Repacked container (`tools/k3_repack.py`)

```
python3 tools/k3_repack.py <hf_src> <dst> --bits 8 [--mla-bits 8] [--head-bits 8]
```

One streaming read of the source (so a disk-to-disk copy can *be* the
conversion): experts pass through byte-identical (id-ordered, single-pread
layout), the big BF16 matrices are quantized with exactly the engine's
load-time algorithm and stored as `U8` + `<name>.qs` f32 scales, the small /
sensitive tensors (norms, router, conv taps, `dt_bias`, `A_log`, `f_a/f_b`,
`b_proj`, embeddings) pass through, vision is dropped. Output is spec-valid
safetensors (`model-XXXXX-of-000094.safetensors`) plus a regenerated index;
interrupted runs resume per shard. On every run, including disjoint `--shards`
passes, the index is rebuilt atomically from all completed output shards in the
destination; its `total_size` therefore includes both new and previously
converted data. Duplicate tensor names abort the index update instead of
publishing an ambiguous artifact. `--verify-full` re-reads every expert byte
and compares against the source.

The engine auto-detects container tensors (dtype U8 + `.qs` sidecar) and skips
load-time quantization. Setting `K3_BITS=4` **explicitly** on an int8 container
downcasts the int8 matrices to int4-g64 at load (~35 vs ~57 GiB resident on
the 93-layer model — fits next to a desktop session; the int8 grid is 16x
finer than int4, so the double-quant noise ~ direct int4). Unset `K3_BITS`
keeps the container's own bits. Startup: measured on two layers, init drops **30.4 s → 0.6 s**;
on the full model this removes a 10–15 minute quantization pass. Quantized
values are bit-identical to the load-time path (verified by hidden-state
trace), so the two formats are numerically interchangeable.

`K3_MMAP=1` is an experimental CPU-only residency option for a fully repacked
container. It maps each prepared U8 matrix and its F32 scale sidecar read-only
instead of copying both into private heap. The mapped pages remain file-backed
and reclaimable; this changes commit/accounting, not the active working set, so
memory pressure can become page faults during inference. On Linux, process
monitors may still show the touched mapping (about 33.8 GB for the full model)
in RSS; it is clean file-backed memory, not private heap, so use `/proc` smaps
accounting to distinguish file-backed RSS from private anonymous memory. The
option has no fallback: tensors that need load-time conversion, non-F32 scale
sidecars, and enabled Vulkan/CUDA backends are refused: `K3_MMAP=1` runs with
`COLI_VULKAN` unset (or `K3_VK=0`) and `K3_CUDA` off.

Sizes: source 1.56 TB → ≈1.50 TB (`--bits 8`) / ≈1.48 TB (`--bits 4`). The
experts (93 % of bytes) are already at 4.25 bits/weight and cannot shrink
losslessly; sub-4-bit expert re-encodes (fmt=5/6) would be double quantization
of QAT weights and are deliberately not offered here.

## Tokenizer

The HF repo ships only a raw tiktoken vocab (`tiktoken.model`).
`tools/k3_tokenizer.py` writes a `tokenizer.json` for it, and `tok.h` gained:

- a **kimi pre-tokenizer family** (sniffed via `\p{Han}` in the Split regex):
  the o200k case-aware rules plus a leading Han-run rule, Han excluded from
  the letter classes, and no `/` tail in the punctuation rule;
- a **rank-BPE mode**, used when `model.merges` is empty: merge the adjacent
  pair whose concatenation has the lowest vocab id — which is tiktoken's own
  algorithm, so encoding is exact by construction. (Merge-list recovery from
  ranks is provably lossy: the standard reconstruction mis-merges e.g.
  "newlines"; that is why the generator emits no merges.)

`tools/k3_tokenizer.py --ctest tests/test_tok_kimi` cross-checks the C
tokenizer against tiktoken; it matches on all 18 corpus cases (Chinese,
kanji-vs-kana, Korean, CRLF, contractions, emoji, mixed-script boundaries).

## Validation

`tools/k3_ref.py` is an independent numpy implementation of the full layer
stack (KDA recurrence, gated MLA, AttnRes, LatentMoE with MXFP4 dequant). With
injected inputs (`K3_X0`, bypassing the embedding) and f32 weights
(`K3_BITS=32`), the C engine matches it on real checkpoint weights across all
four layer types to **rel-L2 ≤ 2.2e-6** (float32 noise).

One property worth knowing: the AttnRes mix logits sit close together
(margins ~0.02), so the mix leverages *weight* noise strongly even though it
does not amplify *input* perturbations (a 1e-4 input perturbation stays 1e-4
through the stack). Load-time int8 produces a hidden-state drift growing to
~14 % over four layers on synthetic inputs — per-tensor quantization SNR is a
uniform ~1 %, so this is architectural sensitivity, not an outlier problem.
Judge quantization choices on real-text logits, not synthetic-vector norms.

## Environment variables

| var | default | meaning |
|---|---|---|
| `K3_BITS` | 4 | load-time bits for KDA/latent/shared/dense mats (4, 8, 32=f32) |
| `K3_MLA_BITS` | 8 | load-time bits for MLA projections |
| `K3_HEAD_BITS` | 8 | load-time bits for lm_head |
| `K3_MMAP` | 0 | map fully prepared U8/F32 weights read-only (experimental, CPU-only, no conversion fallback) |
| `K3_EXPERT_GB` | 8 | routed-expert LRU budget |
| `COLI_VULKAN` | 0 | `make VK=1 kimi_k3`: routed experts on the shared Vulkan expert tier, shared experts where `COLI_VK_DENSE` puts the dense matrices (see [Vulkan](#vulkan-make-vk1-kimi_k3)) |
| `K3_VK` | unset | old switch, an alias: 1 = `COLI_VULKAN=1`, 0 = never open the device |
| `K3_VK_GB` | unset | old switch, read as `COLI_VK_TIER_GB` when that is unset |
| `K3_VK_UP` | unset | old switch, a number read as `COLI_VK_TIER_RATE` when that is unset; `auto` does nothing |
| `K3_DIRECT` | 1 | O_DIRECT expert reads (0 = buffered + WILLNEED) |
| `K3_IDOT` | 1 | int8-activation expert matmuls (0 = exact-float kernel) |
| `K3_PIPE` | 1 | expert reads on reader threads, beside the compute |
| `K3_IO_PER_DRIVE` | 2 | readers per storage device the shards are on |
| `K3_IO_THREADS` | — | that many readers on one queue instead (`K3_LOAD_THREADS`, the old loader's size, is read as this); 0 = reads on the caller |
| `K3_LOAD_PARALLEL` | 4 prepared / 1 BF16 | the dense weights of this many layers read at once while the model loads: 4 where each matrix is one straight read (a repacked U8 or GGUF container), 1 where the load loop is busy quantizing a BF16 checkpoint on every core instead (the layers then go one after the other, each matrix over the whole team); `K3_MMAP=1` forces it |
| `K3_DENSE_CACHE` | off | `1`: `<model dir>/k3_dense_cache.bin`, or a path. The quantized dense matrices written by the first start and read by the next ones instead of reading and quantizing a BF16 checkpoint (Kimi K3.1 Pro: 33.9 GB; the start 28 -> 9.7 s on the RTX 4090 machine). A matrix comes from the file only when its name, shape, bits and its source's place in the checkpoint are the record's and its bytes sum to what was written ([the start](#kimi-k31)) |
| `K3_PREFETCH` | 1 | read the next MoE layer's predicted experts ahead (decode steps) |
| `K3_PREFETCH_ROWS` | 4 | the most rows a step may have for its prediction |
| `K3_PREFETCH_LIMIT` | 64 | prefetches queued at once |
| `K3_PREFETCH_TOP` | 12 | the experts a row's prediction reads ahead (16: the routing's top-k) |
| `K3_PREFETCH_MIX` | 1 | predict from the next layer's own mix of the blocks (0: this layer's MLP input) |
| `K3_WAVES` | 1 | a MoE step's experts computed in waves as their reads arrive (0: one at a time) |
| `K3_IO_DELAY_US` | 0 | tests: every read waits this long first |
| `K3_WAIT_LAYERS` | 0 | measurement: a decode token's wait for reads and its demand reads, by layer (last lines) |
| `K3_REUSE` | -- | measurement: every MoE layer's LRU stack distances and predicted share, written to this file at exit |
| `K3_TOPP` | 0 | keep routed experts to cumulative weight p (0 = off) |
| `K3_DSA` | 1 | Kimi K3.1: 0 runs the MLA layers dense, the indexer neither loaded nor run (exact up to `index_topk` positions, another attention past them; see [Kimi K3.1](#kimi-k31)) |
| `K3_DSA_BITS` | `K3_MLA_BITS` | Kimi K3.1: load-time bits of the indexer's three matrices |
| `K3_DSA_GPU` | 1 | Kimi K3.1, a CUDA build: a prompt step's scores and picks on the GPU (`k3cu_dsa_mask`); 0 = on the CPU (the same bytes) |
| `K3_MLA_SEL` | 1 | Kimi K3.1, a CUDA build: a prompt step's selecting rows over the lists of their picks (`k3cu_mla_sel`); 0 = against every position with the unpicked ones weighing 0 (`k3cu_mla_mask`: the same bytes, a cost that grows with the context) |
| `K3_CHUNK` | 32 | the most prompt tokens a prefill step takes, at most 16384 (1 = token-at-a-time; forced 1 under `K3_TRACE`). A prompt's steps are of one size and none is short: up to an eighth past `K3_CHUNK` is one step, more is equal steps. Past 512 rows a step holds 1.4 MB of activations a row, borrowed from the expert cache while it runs (`K3_LEND`) |
| `K3_THINK` | 1 | chat mode: open the structural think channel (0 = response-only) |
| `K3_DIRS` | — | extra shard directories (multi-drive split, no duplication; `tools/k3_download.py` cuts every layer over them) |
| `K3_MAXT` | prompt+ngen | context capacity |
| `K3_KV_GROW` | 1 | the MLA caches and the index keys grow with the conversation: a request allocates its prompt's positions plus `K3_KV_GEN` of the answer's and doubles as it writes them; what they hold past `K3_KV_MIN` positions comes out of the expert cache for as long as they hold it (`K3_LEND`) (0 = the whole `K3_MAXT` at the first request; off under `KV_SLOTS>1`) |
| `K3_KV_GEN` | 1024 | answer positions allocated with the prompt's, before the caches start doubling |
| `K3_KV_MIN` | 4096 | positions the RAM plan projects the caches for while the session is still idle — what it does not project, the expert cache is allowed to hold |
| `K3_LEND` | 1 | the expert cache lends slots to what the RAM plan did not count -- the conversation's rows past `K3_KV_MIN` (they stay) and a prompt step's activations past 512 rows (until the step ends): a lent slot is retired, its bytes freed and nothing read into it, and comes back with the bytes (0 = the plan reserves a whole `K3_CHUNK` step's activations instead; off under `KV_SLOTS>1`) |
| `K3_ROOT_PHOTO` | 1 | with `COLI_K3_CKPT` of 2 or more: the state at the end of a chat's leading system records photographed inside the step that crosses it and kept, so a new conversation with the same system prompt starts after it (0 = off) |
| `K3_ROOT_MIN` | 1024 | the fewest tokens of leading system records worth that photo |
| `K3_SESSIONS` | 0 | serve: up to N conversations kept beside the live one; a request of another conversation puts the live one aside instead of writing over it (`k3_sessions.h`) |
| `K3_SESSION_DIR` | unset | ... and in files there (the launcher: `<model folder>\k3_sessions`), read back after a restart |
| `K3_SESSION_GB` | 8, 0 with files | RAM for the kept conversations together with the live one's rows past the plan; the longest unused leave first |
| `K3_SESSION_DISK_GB` | 64 | the files' bytes at most (the launcher: 128) |
| `K3_SESSION_MIN` | 1024 | the fewest tokens of a conversation worth keeping |
| `K3_SESSION_IDLE` | 30 | seconds with no request before the live conversation is written |
| `K3_SHARED_PAR` | 1 | a prompt step's shared experts on the card from a thread of their own, beside the routed experts' waves (0: on the step's thread first) |
| `K3_DC_AHEAD` | 1 | the dense cache read ahead by threads of their own (0: each matrix read as the load asks for it) |
| `K3_LOGITS_DECODE` | 0 | the CLI's `K3_LOGITS` file keeps the decode's rows after the prompt's |
| `K3_CANCEL_TEST` | 0 | tests: the N-th cancel poll of the serve's first prompt cancels it |
| `K3_LAYERS` | all | truncate the stack (validation) |
| `K3_TRACE` | — | dump f32 hidden state after every layer (validation) |
| `K3_X0` | — | inject input rows, bypass embedding (validation) |
| `COLI_TEMP` | 0 | 0 = greedy, else softmax temperature |

## Chunked prefill

Prefill processes the prompt in chunks of `K3_CHUNK` tokens, layer-major:
every dense matmul batches over the chunk (each weight matrix streams from
RAM once per chunk instead of once per token), the MoE loads each unique
expert of the chunk once (measured at C=32 on real text: 2.7x dedup —
neighbouring tokens share experts far more than QB-flat routing suggests —
9.6 instead of 25.8 GB/token), and the lm_head runs only on the chunk's last
token. Sequential state (KDA recurrence, MLA cache, AttnRes bookkeeping)
advances per token inside each layer in the original order, so chunked
results are **bit-identical** to token-at-a-time (verified: 125-position
teacher-forced logit streams at C=32 vs C=1 match exactly). Measured prefill:
~5.3 -> **2.0 s/token** at C=32. The KDA state-update sweeps are AVX2
(scalar fallback for head dims not divisible by 8).

**A long prompt's CPU work** (the RTX 4090 machine, docs/experiments/kimi-k3-4090-2026-10-31:
a 2,637-token prompt took 1,401 s in 512-token steps, 549 s on the CPU in steps of 2,048,
137 s in one step of 4,096 with its experts, MLA attention and routers on the GPU (below:
the CPU's bytes there too), 113 s with the card's own matrices in `quant_matmul_blocked` and
**106.5 s** with the exact dense kernel's lanes in one thread (`k3_dense_b`), the logits the same
bytes; the drive with 61% of every layer's experts reads 92% of the time; a 5,862-token prompt
173 s in one step of 8,192 then, 110 s with the layers and the card's matrices below). Each rewrite below keeps
every row's and position's arithmetic:
- the gated MLA's scores were one chain of 576 adds a position (GCC -O3 multiplies 16 at
  a time, then adds the products in order); `k3_mla_scores_g` keeps that arithmetic (an
  opaque product, then the add, as mimo.c's `mul_apart`) with 16 positions in the lanes,
  the cache's rows transposed in 8x8 blocks once for six heads;
- kv_b's rows times the attention's latent (`w_rowdot`, which GCC left scalar) 8 rows at
  a time (`k3_rowdot8_i8`); AttnRes' `res_mix` its entries' double chains side by side, a
  prompt step's rows over every worker;
- `matmul_i4_grouped` (the shared experts and the latent projections on the CPU) decodes
  16 weights once for 8 rows and keeps 16 rows of input in L2 (1.8x at 512 rows),
  `matmul_mxfp4_i8_pre` (the routed experts) an output's weights once for 8 rows and
  their 8 group scales in one multiply (the routed experts 255 -> 216 s), and both sum a
  group's 8 rows' lanes together (`hsum256_x8`, hsum256's pairs); the shared experts'
  SiTU runs over every worker;
- `K3_CHUNK` may be up to 16384: a step reads every expert it needs once from the drives and
  once from RAM for all its rows (2,048 against 512: 5,784 -> 2,455 GB read for that
  prompt, 845 -> 697 s; a step of 4,096 or more reads nearly every expert, 1.4 TB: a
  5,862-token prompt 254 s in two steps of 4,096, 200 s in one of 8,192, the logits of every
  position the same bytes). The GPU's calls take a step's rows 1,024 at a time
  (`K3_GPU_ROWS`; the MLA core the whole step, in chunks whose scores fit in 128 MB), so
  their device buffers fit beside the 20 GiB of resident matrices: a whole 4,096-row step's
  did not (Windows moved the card's
  memory to RAM without an error, minutes at 100% GPU with the drives idle), and with
  2,048-row pieces 150 MB more of the desktop's slowed coli_cuda_k3.dll's calls 2-6x;
- KDA's short convolutions and heads go over a step's tokens channel by channel and head
  by head (one parallel region each where there were four a token; `k3_kda_head` is the
  one head step both orders call, the conv's taps fused adds in order as compiled before);
- `matmul_q` (int8: the MLA projections the GPU does not hold) and `matmul` (f32, every
  engine's, matmul_f32.h) take a step's rows like `matmul_i4_grouped`: 16 rows of input
  in L2, each weight loaded once for 8 (4) rows, every row's lanes and tree its own;
- coli_cuda.dll's `quant_matmul_tiled` (a step's rows on the GPU) ends quant_matmul's tree
  with five levels of warp shuffles, two outputs a round: 20 block barriers where there
  were 80, the same pairs added (10-20% sooner);
- the routers' BF16 rows take a step's rows the same way (the routing 31 -> 23 s);
- a step's big buffers stay allocated from one layer to the next (`k3_balloc`, given back at
  the step's end): a 2,048-row step's 2 GB a layer had cost a page fault every 4 KiB every
  layer (94 -> 41 million faults for that prompt; the rest is the expert cache's first fill).

## Chat, API, and Web

```sh
# standalone one-turn diagnostic
./kimi_k3 <model_dir> --chat "your question" [--system "..."] --ngen 300

# first-class multi-turn interfaces (config.json auto-detects Kimi K3)
COLI_MODEL=<model_dir> ./coli chat
COLI_MODEL=<model_dir> ./coli serve
COLI_MODEL=<model_dir> ./coli web
```

K3's chat format ("XTML", from the checkpoint's own `encoding_k3.py`) uses
only four special tokens — `<|open|>`, `<|close|>`, `<|sep|>`,
`<|end_of_msg|>` — around ordinary-text tag names and attributes:

```
<|open|>message role="user"<|sep|>TEXT<|close|>message<|sep|><|end_of_msg|>
<|open|>message role="assistant"<|sep|><|open|>think<|sep|>        <- generation prompt
```

The assistant's thinking is a *structural* channel and is preserved when an
OpenAI-compatible client sends prior `reasoning_content`; `enable_thinking=false`
opens `<response>` directly. The gateway does not flatten XTML into a string:
it sends length-framed messages to the C engine, which builds every structural
and ordinary-text segment at the tokenizer boundary required by K3's rank-BPE.
We checked our text handling against Moonshot's own: their `encoding_k3.py`
turns a conversation into the numbers the model actually reads, and ours
produced **identical numbers on all 77 test conversations** — multi-turn
histories with system, user and assistant messages, including non-ASCII text.
So the engine is not subtly mangling anything before the model sees it.

`coli chat` starts a private local server for Kimi and keeps the 2.8T model
loaded for the whole terminal session. `coli serve` exposes streaming and
non-streaming `/v1/chat/completions`; `coli web` uses that same API. Reasoning
is returned as `reasoning_content`, response text as `content`, and
`<|end_of_msg|>` remains the model-owned stop token. `STOP` and `CANCEL` are
honoured between generated tokens. Long prefill also polls `CANCEL` between
layers and drops the unpublished partial state before serving another request.

Long agent sessions can opt into recurrent-state checkpoints (`COLI_K3_CKPT=N`
slots in RAM, or parked on disk with `COLI_K3_CKPT_DIR`): an edited or follow-up
prompt restores the deepest surviving checkpoint and re-prefills only the tail,
instead of replaying the whole conversation through the SSM layers.

## Vulkan (`make VK=1 kimi_k3`)

Run with `COLI_VULKAN=1`, a `VK=1` build puts the routed experts on the shared Vulkan
expert tier ([vulkan.md](vulkan.md#the-routed-expert-tier-vk_tierc)): the shards'
MXFP4 bytes as they sit in RAM (e2m1 nibbles, the ue8m0 group exponents widened to
the f32 scales the shader reads, never re-encoded; the fmt 7 kernel against
`matmul_mxfp4`: rel_l2 2.2e-07 on an RX 9070/RADV, 2.6e-07 on llvmpipe,
`tests/test_vk_mxfp4.c`), gate, SiTU-GLU and down on the device, in the latent
space (the tier's hidden is the latent, 3584, its intermediate `moe_inter`, 3072).

- **What is resident adapts.** At startup the tier fills its budget from the expert
  history (`<snap>/.coli_usage`, the hottest first, read in parallel); after that
  every expert the CPU computes is a candidate, and a hotter one displaces the
  coldest resident. The engine's own tier before this one filled once and never
  let go.
- **The device and the CPU work at once, prefill included.** Each MoE step sends
  the (position, rank) pairs whose expert is resident to the device as one batch;
  the CPU reads and computes the other experts of the chunk's union and the shared
  experts meanwhile. An expert the device took entirely is not read from disk.
- **The sum keeps the CPU run's order.** Every pair joins its position in the
  union's disk-offset order, the order a CPU-only run adds them, the device's rows
  and the CPU's alike: which experts were resident never changes the order, and an
  expert computed on the CPU keeps the CPU run's bits.
- **The shared experts** go to the device where `COLI_VK_DENSE` puts the dense
  matrices (int8, int4-g64 or f32, as `K3_BITS` made them; one row at a time, so
  decode): on a discrete GPU the device, on an integrated GPU or Lavapipe the CPU
  while the tier runs. KDA, MLA, the latent projections, the router and the head stay
  on the CPU. With `COLI_VK_TIER=0` the shared experts alone go to the device.

The device computes with f32 activations: its experts match `K3_IDOT=0`, the
configuration of the vendor oracle, while the CPU's default kernel rounds
activations to int8 per 32. With `K3_IDOT=0` a run with the tier gives the CPU's
tokens on the tiny fixture and passes Moonshot's oracle, the eviction and warm-start
cases included (`tests/vulkan_engines.sh kimi`). A build with both backends runs on
Vulkan when CUDA was asked for (`K3_CUDA=1`, which `coli` sets for `COLI_CUDA=1`) and
neither `COLI_VULKAN` nor `K3_VK` was set: the CUDA path uploads every routed expert's
weights at every call (an MXFP4 expert of 7168 x 2048, 2.46 ms on an RTX PRO 4000, against
72 us for the tier's resident ones) and keeps the dense matrices on the CPU. With
`COLI_VULKAN=0` and `K3_CUDA=1`, CUDA wins and the Vulkan tier stays off. Each run and serve turn ends with
`[VK] kimi_k3: N matmuls on the GPU` and the tier's `[VK] tier kimi_k3` line.

The old switches still work, read as the shared tier's (an explicit shared one
wins): `K3_VK=1` opens the device as `COLI_VULKAN=1` does and `K3_VK=0` keeps it
closed, `K3_VK_GB` is the tier's budget (`COLI_VK_TIER_GB`), a numeric `K3_VK_UP`
its promotions per token (`COLI_VK_TIER_RATE`); `K3_VK_UP=auto` and
`K3_VK_FILL_FRAC` do nothing, since the shared tier uploads on a thread of its own
and never holds a step. One default changed: a `VK=1` build used to open the device
without being asked (`K3_VK` defaulted to 1); it now waits for `COLI_VULKAN=1` or
`K3_VK=1`, like every engine, and without either it is the CPU build's bytes.

K3's Quantile-Balancing-flat routing caps what any cache tier can do — the
tier's value scales with how long the server lives and with the measured
short-term reuse (temporal locality), not with marginal expert heat.

## CUDA: the dense matrices on the GPU (`K3_CUDA_DENSE=1`)

A CUDA build (`make kimi_k3 CUDA=1`, or on Windows `mingw32-make kimi_k3 CUDA_DLL=1`
with `coli_cuda.dll` from `build_cuda.bat` beside the engine) puts the largest of the
matrices that are not routed experts on the card with `K3_CUDA_DENSE=1`: the head,
the gated MLA's `o_proj` and gate, KDA's projections, then smaller ones, in the int8
rows and int4 groups `K3_BITS` made, until `K3_CUDA_DENSE_GB` (20) or the card's free
memory less `K3_CUDA_RESERVE_GB` (2) is used. `w_matmul` runs a placed matrix there
for every row count. With `coli_cuda_k3.dll` beside the engine (a `CUDA_DLL=1` build's)
that is the CPU's arithmetic, `k3cu_matmul_dev` (below): from 64 rows `k3_dense_s4` and
`k3_dense_o8`, up to 4 rows (a decode step) `k3_mv4` and `k3_mv8`, `k3_dense` between,
and only the int8 rows and int4 groups those kernels take go up (all of K3's at the default
bits). Without it, coli_cuda.dll's own kernels: one row through `coli_cuda_matmul`
(`quant_matvec_warp`), a prompt step's rows through `coli_cuda_matmul_tiled` (at the
trunk's shapes `quant_matmul_blocked`, 16 outputs by 8 rows a thread), quant_matmul's
sums. Every token reads all of these matrices (about 35 GiB at the
default bits), so on a machine whose RAM bandwidth is shared with the experts' reads
this takes the largest part of the dense work off it: on an RTX 4090 a 12288 x 7168
int4 projection takes 0.05 ms a decode row against 0.90 ms on 16 Zen 4 cores, the
163840 x 7168 int8 head 1.2 ms against 18.4 (a whole call with its copies; a decode
token's 369 held matrices about 24 ms, 24-29 in coli_cuda.dll's kernels).
`kv_b` stays on the CPU (the MLA absorb reads its rows), and so do the routers and
KDA's small f32 projections, so the routing is the CPU's arithmetic on its input (a prompt
step with `coli_cuda_k3.dll` multiplies them on the card in that same arithmetic, below). The
routers stay in the checkpoint's BF16 (`Moe.router16`): `k3_dot_bf16w` widens each weight
to its f32 in the f32 kernel's 32 lanes, fused adds and halving tree, so the scores are
the f32 copy's bits (the tiny model's logits, routers stored as BF16 or as the same values
in F32: the same bytes) from half the RAM: a decode token reads every router twice (the
routing and the next layer's prediction), 2.4 GB less a token, 94 -> 58 ms on the 4090.

With `coli_cuda_k3.dll` these are the CPU's bytes: matmul_q's and matmul_i4_grouped's
lanes, sums and roundings, so the GPU's logits are the CPU build's (Kimi K3's 2,637- and
5,862-token prompts at every position against runs on the CPU alone, 35 tokens fed one a step
through the decode kernels, 48 tokens of text: the same bytes). coli_cuda.dll's own kernels
add the same products in another order, which a long prompt carries a long way (routing flips,
the recurrent state): against the CPU's logits the top token at 5-6% of the positions, KL 0.014
on average; the prompts' own perplexity 13.01 against 13.09 and 10.53 against 10.56, within 1-1.6
standard errors (`docs/experiments/kimi-k3-4090-2026-10-31`). `K3_CUDA_RELEASE_HOST=1` frees the placed matrices' host
copies before the expert cache is sized, so the cache gets that RAM; if a GPU call
fails, the CPU computes every matrix from then on, reading a released one back from
the checkpoint (`COLI_GPU_FAIL_AFTER=N` tests it: the same tokens).

`K3_CUDA=1` is another thing: it uploads each routed expert to the card for every
decode row, 17.5 MB over PCIe an expert, which costs more than computing it where it
was read; keep it off when the experts stream from disk.

| var | default | meaning |
|---|---|---|
| `K3_CUDA_DENSE` | 0 | the dense matrices on the GPU (a CUDA build) |
| `K3_CUDA_DENSE_GB` | 20 | their budget, GiB |
| `K3_CUDA_RESERVE_GB` | 2 | VRAM left free beside them |
| `K3_CUDA_RELEASE_HOST` | 0 | free the placed matrices' host copies (reread from disk if the GPU fails) |
| `COLI_GPU` | 0 | the CUDA device |

## CUDA: a prompt step on the GPU, the CPU's bytes (`coli_cuda_k3.dll`)

A Windows CUDA build (`mingw32-make kimi_k3 CUDA_DLL=1`) also loads `coli_cuda_k3.dll`
(`c\build_k3_cuda.bat`, `backend_cuda_k3.cu`) when it sits beside the engine, and then
computes a prompt step's heavy parts on the card in the CPU's own arithmetic, to the bit:
every kernel there reproduces the loop it replaces, so the logits are the CPU build's bytes
whatever the step, and a step may split its experts between the card and the CPU in any
way.

- **The routed experts.** `matmul_mxfp4_i8_pre`'s output is 8 lanes, each the integer sum
  of four products of the doubled e2m1 values and the int8 inputs (dp4a gives it), times
  the group's scale (e8m0 times 0.5, times the row's: two float multiplies), fused into the
  lane group after group, then hsum256's pairs (three xor shuffles). SiTU calls exp_f and
  MinGW's tanhf through `k3_exact.h`: e^x and tanh in double within about 2^-50, rounded
  to float from 2^-44 below and from 2^-44 above; where the two roundings agree every
  number between them rounds the same way, and the 731 and 415 floats where they differ
  carry MinGW's bits in `k3_exact_tables.h` (`tools/k3_exact_tables.c` writes it and checks
  every one of the 2^32 floats; `tests/test_k3_cuda_experts.c --all-floats` runs the
  device's code on all of them). exp_f, which every kernel here calls, first tries float
  pairs, since the card runs doubles at 1/64 of its FP32 rate: e^x within about 2^-46 as a
  float and a remainder, the float taken where every number within 2^-42 of the pair rounds
  to it, the double path elsewhere (2,681 of the 2.24 billion floats in [-87, 88], and the
  floats past them): 17 times as fast, exp_f's bits on all 2^32 floats. Then `mxfp4_qx`'s
  quantization and down, written straight into one pinned buffer, a warp a row's 128 bytes
  (pieces of 16 bytes over PCIe slowed the expert copies the other way from 26 to 20 GB/s).
  A feeder thread hands the card every expert whose bytes are in
  RAM as the reads bring them, frees its slot and queues the step's next read into it; the
  CPU computes the shared experts meanwhile, then waves of `K3_GPU_CPU_WAVE` experts
  itself; the weighted sums run as before, in the union's order, a token a worker (its pairs
  listed by j, then q: each element's adds in the one loop's order).
- **The dense matrices the GPU does not hold** (`k3cu_matmul`): `matmul_i4_grouped`'s
  lanes, group sums and fused scales, `matmul_q`'s lanes and scale; the shared experts as
  one call (gate, up, SiTU, down: `k3cu_shared`). From 64 rows int4 groups of 64 take
  `k3_dense_s4` and int8 rows `k3_dense_o8`: a block decodes a stage's 64 columns of weights
  once into shared memory as floats (a nibble one mask and one fused add, a byte one permute
  and one add) and brings its rows of x there by column, the next stage's going up while this
  one's products run. In `k3_dense_s4` a thread holds an output's 8 lanes for 4 rows by 8
  outputs and runs them one at a time, 0, 4, 2, 6, 1, 5, 3, 7, so each finished lane joins
  hsum256's tree at once (three kept an output); in `k3_dense_o8` 8 threads hold a tile of 8
  rows by 16 outputs, thread l lane l over the whole row (matmul_q's lanes run the whole row),
  and add the lanes by shuffles at the end. 16-21 and 17-23 TMAC/s at K3's shapes (their
  predecessors `k3_dense_b` and `k3_dense_b8`, each thread decoding its own weights: 13-15 and
  11-14; `k3_dense` 3.1-3.6, `K3_GPU_DENSE_B=0`), the same bytes; the card's shared-memory pipe
  is their limit (a load of any width takes 2 of its cycles). Up to 4 rows (a decode step)
  `k3_mv4` (a 64-column group a thread, four outputs a task) and `k3_mv8` (an output's 8 lanes
  on 8 threads, the warp's rows staged through its shared memory).
- **The dense matrices the card holds** (`k3cu_matmul_dev`, every row count `w_matmul` asks
  for, a decode step's one too): the same kernels on coli_cuda.dll's copies
  (`coli_cuda_tensor_dev_ptrs`; its upload flips each int4 nibble's top bit, the kernels flip
  it back as the words come in), so the CPU's bytes there as well.
- **The gated MLA's attention core** (`k3cu_mla`): mla_forward's loop -- the absorbed
  query (fused, w_addrow's), the scores (products rounded, added in order from zero,
  then scaled), `softmax_` (its sum in order), the values (fused over the positions in
  order), kv_b's rows (`k3_rowdot8_i8`) and the sigmoid gate. The four products run in
  tiles of 128 by 128 outputs, 8 by 8 a thread, every output adding its terms in order (a
  block of scores whose positions all lie past its rows' does not run); softmax_ takes its
  max from the score blocks' own, a warp adds 32 rows' exps in order while its block
  computes the next 128 positions', and the division happens as the values load them. The
  step goes in chunks of up to 256 rows whose scores fit in 128 MB, a chunk's rows going up
  and its outputs down while the next one computes. It grows with the square of the
  context: at 5,862 tokens 4.0 s, 11.6 s in the 64-by-64 tiles of 4 by 4 a thread before.
- **The gated MLA layers whole** (`k3cu_mla_layer`) in those chunks: x up, then qa, its
  RMSNorm (its squares summed in double in order, as `rmsnorm_`) and qb, kva into the cache
  rows (RMSNorm on the latent, the rope part as it is, written back to the model's cache),
  the gate, the core and o_proj, the output down. A matrix the card holds runs on
  coli_cuda.dll's copy, the others from their host weights, all in `k3cu_matmul`'s kernels;
  with `K3_GPU_MLA=0` the layer goes call by call around the core as before, every matrix's
  input up and output back. At 5,862 tokens the layers 11.2
  -> 9.0 s (the prompt 122.2 -> 121.0 s), at 2,637 tokens 4.5 -> 3.2 s (C:'s reads its floor).
- **The routers** (`k3cu_router`): the bf16 router's 32 lanes and halving tree, a warp an
  output.
- **The KDA layers** (`k3cu_kda`): kda_forward's by-step path in pieces of
  `K3_GPU_KDA_ROWS` (512) rows, the layer's input up once and its output back. The
  projections the card holds run on coli_cuda.dll's own copies (`coli_cuda_tensor_dev_ptrs`)
  in `k3cu_matmul_dev`'s kernels, the others are computed as before and handed in; then the conv
  (its taps fused in order, SiLU through exp_f), the f32 projections (matmul_f32.h's 32 lanes),
  k3_kda_head's norms (the squares rounded and added in order, GCC's in-order reduction),
  decay, recurrence (a thread a column of the head's state, in registers, the two sweeps'
  fused adds in order) and gated RMSNorm (its sum in double), and o_proj. The state and the
  conv windows go back to the model's memory at the end of the layer. The pieces share one
  device buffer with the MLA core, which never runs at the same time. A 5,862-token prompt
  173.7 -> 149.2 s on the RTX 4090 (KDA 50.4 -> 25.2 s); `K3_GPU_KDA=0` keeps the layers on
  the CPU.

The rest of a prompt step stays on the CPU, faster too: the conv's and gates' exp_f 8 at a
time through `exp_f8` where KDA runs there (decode, `K3_GPU_KDA=0`), and the routing's rows
over every worker (the same picks). `tests/test_k3_cuda_experts.c` checks every kernel against the
code it replaces, byte for byte; the tiny model's logits at steps of 1 to 2,048 (with
`K3_GPU_MIN_ROWS=1`, every step on the card) and the real model's are the CPU build's.

**Read-ahead.** A step of many rows needs nearly every expert of every MoE layer, one read
each, and the per-layer cache holds a tenth of them: the drives read a layer's experts
only while that layer's MoE step runs. A pool of its own slots (`K3_READAHEAD_GB`, 20 GB
when the card takes a prompt's experts) reads the next MoE layer's experts the cache lacks
behind the step's own reads, while the rest of the layer and the next one's attention
compute; that layer's step takes them as cached ones and frees each as it is done, the
next layer's reads taking its place. The same bytes, read sooner: the pool's RAM comes out
of the expert cache (on the 4090 machine 94 -> 82 slots a layer, decode 0.83 -> 0.81
tokens a second; a 2,637-token prompt 170 -> 137 s). The cache's slots come from pinned
memory up to `K3_GPU_PIN_GB` (64: Windows takes VRAM for pinned memory past about half
the RAM).

| var | default | meaning |
|---|---|---|
| `K3_GPU_EXPERTS` | 1 | a prompt step's experts and dense parts on the card too (0: the CPU's alone) |
| `K3_GPU_MIN_ROWS` | 32 | the rows a step needs for it (1 in tests: every step) |
| `K3_GPU_CPU_WAVE` | 4 | the experts the CPU takes at a time beside the card |
| `K3_GPU_PIN_GB` | 64 | the expert slots' pinned memory, GiB |
| `K3_GPU_DENSE_B` | 1 | the dense matrices' and shared experts' int4 groups of 64 through `k3_dense_s4` and int8 rows through `k3_dense_o8` from 64 rows, `k3_mv4`/`k3_mv8` up to 4 (0: `k3_dense`) |
| `K3_GPU_KDA` | 1 | a prompt step's KDA layers on the card, `k3cu_kda` (0: the CPU's) |
| `K3_GPU_MLA` | 1 | a prompt step's gated MLA layers whole on the card, `k3cu_mla_layer` (0: call by call around the core) |
| `K3_GPU_KDA_ROWS` | 512 | the rows of a KDA layer's piece on the card |
| `K3_READAHEAD_GB` | 20 with the card, else 0 | the read-ahead pool, counted in the RAM budget |

## Kimi K3.1

[moonshotai/Kimi-K3.1-Pro](https://huggingface.co/moonshotai/Kimi-K3.1-Pro) is Kimi K3
post-trained, and not only that: each of its 24 gated MLA layers carries a DeepSeek-V3.2
style **lightning indexer** (`text_config.dsa_config`; the wrapper's `model_type` is
`kimi_k31`, its class `KimiK31ForConditionalGeneration`). The checkpoint ships no modeling
code; the semantics here follow the one implementation published with it,
[TokenSpeed's `KimiK31Indexer`](https://github.com/lightseekorg/tokenspeed/pull/3015)
(`python/tokenspeed/runtime/models/kimi_k31.py`), and its config parser's checks:

| | Kimi K3 | Kimi K3.1 |
|---|---|---|
| MLA layer | gated NoPE MLA over every earlier row | the same projections, over the `index_topk` (2,048) rows its indexer picks; a row with no more positions attends all of them |
| new tensors a layer | -- | `self_attn.indexer.wq_b` [32*128, 1536], `.wk` [128, 7168], `.weights_proj` [32, 7168], `.k_norm.{weight,bias}` [128], BF16 |
| config | -- | `dsa_config`: `dsa_layers` (every MLA layer), `index_topk` 2048, `index_head_dim` 128, `index_num_heads` 32, `index_rope_dim` 0, `index_k_norm_type` layernorm, `index_topk_freq` 1 |
| everything else | | Kimi K3's: KDA, AttnRes, LatentMoE, MXFP4 experts, the tokenizer and the chat format |

For the query row at position t and an earlier row s,

    I(t,s) = d^-0.5 H^-0.5 * sum_h w_h(t) relu(q_h(t) . k(s))
    q(t) = wq_b(c_q(t))           c_q: the RMSNorm'd q latent (qa after q_a_layernorm)
    k(s) = LayerNorm(wk(x(s)))    eps 1e-6, no RoPE, no Hadamard rotation
    w(t) = weights_proj(x(t))     x: the attention input

and the row attends the `index_topk` rows of the highest I (all heads share the pick).

**In the engine.** `load_cfg` reads `dsa_config` and refuses any value the code does not
implement (an unknown key, a RoPE part, an RMSNorm key norm, FP8 indexer weights, a pick
shared between layers). Every MLA step writes its rows' keys into the layer's `Ic` cache
(f32, `index_head_dim` floats a position, beside `Lc`/`Rc`: photos, parking, the
multiplexed slots and the segment sessions carry it). A row past `index_topk` positions
gets its pick -- in a decode step on the CPU (`k3_dsa_scores`: 16 positions a pass, 8x8
blocks of keys transposed in registers; `k3_dsa_pick`: quickselect, ties to the lower
position, the list ascending), in a prompt step of a CUDA build on the GPU (`k3cu_dsa_mask`,
`backend_cuda_k3_dsa.cuh`: the same chain of fused multiply-adds a score, a radix select a
row, so the same bits as the CPU's pick); then its attention runs over the picked rows only
-- on the CPU's loop (the scores and the latent sum over the list) or, for a prompt step, on
the GPU's core over the same list (`k3cu_mla_sel` in `coli_cuda_k3.dll`: the pick's bits
become each row's ascending list on the card, and the scores, the exps with their in-order
sum and the latent's fused sum run over the listed rows of the cache -- the CPU loop's sums,
term for term). `k3cu_mla_mask`, the form before it (`K3_MLA_SEL=0`), computes a row against
every position and weighs the unpicked ones exactly 0, which changes neither the softmax's
sum nor the latent's: the same bytes, at a cost that grows with the context. A step all of
whose rows have at most `index_topk` positions is K3's to the bit, the GPU's whole-layer
path included (the keys then on the CPU). `K3_DSA=0` runs a K3.1 checkpoint dense.

**What it costs.** Up to 2,048 positions: one 128 x 7168 matrix a row and a layer (the
keys), nothing else. Past them, for each row and layer: the projections (32 x 128 by 1536,
32 by 7168), 4,096 multiply-adds a position for the scores, a select, and then the attention
over 2,048 rows instead of all of them. A prompt's scores grow with the square of its length
(a 32,768-token prompt: 44 T multiply-adds over the 24 layers): on the CPU alone that kept
the first K3.1 build's prompts there for minutes with the drives and the card idle, hence the
GPU's pick -- on the RTX 4090 machine a 16,511-token prompt's 347,112 selecting rows are
scored and picked in 2.1 s (3.2 G positions), 4.4 s with the keys and the projections. The
attention itself no longer grows that way once it runs over the lists: the same prompt's MLA
core took 40.9 s with the masks (every row against every position) and takes 9.6 s over the
lists; at 5,862 tokens 4.3 and 3.0 s. A decode row's pick stays on the CPU (4,096
multiply-adds a position: about 25 ms a token at 16,000 positions over the 24 layers), and
its attention reads 2,048 rows whatever the context. The keys add 12 KB a position to the
55 KB of the MLA caches. The prompt's selections are reported (`[K3-DSA] the prompt: ...`)
and the GPU's steps (`[K3-CUDA] the gated MLA's attention: ... K3.1's selections, N rows
of M steps over their lists`).

**Both binaries, or none of the card.** The GPU's pick is in `coli_cuda_k3.dll`, not in
`kimi_k3.exe`, and the loader binds `k3cu_dsa_mask` as an *optional* export: a DLL older than
`dsa_config` loads without a word and every MLA layer then scores its prompt on the CPU. The
bytes are the same, the time is not -- that is what a CPU at 100 % with idle drives and an idle
card for minutes looks like, and the engine's one line about it is easy to miss in a loading
log. So `tools/start_kimi_k31_pro_4090.ps1` checks the DLL for the export and refuses to start
instead of running quietly at a fraction of its speed (`-AllowCpuIndexer` starts it anyway),
the loader names every optional kernel a DLL lacks, and the launcher's banner says where the
picks will run. After a prompt, `[K3-DSA] the prompt: ... (N on the GPU)` says it outright:
`0` there means the CPU did them.

**And the C runtime the engine was built against.** `coli_cuda_k3.dll`'s kernels reproduce
*MinGW's* `exp_f` and `tanhf` bit for bit, so an engine linked against the UCRT -- MSYS2's
`ucrt64` gcc, which is what `mingw-w64-ucrt-x86_64-gcc` installs -- computes those two
differently and is refused the card entirely: not the prompt's routed experts, nor the masked
MLA layers, nor the indexer's picks, and neither `K3_GPU_EXPERTS=1` nor the `coli_cuda.dll`
dense tier changes that. The whole prompt then runs on the CPU's threads, which is minutes per
8,192 rows. The engine says so at startup, with the reason:

    [K3-CUDA] experts: off (the engine was built against a C library whose expf and tanhf are
    not MinGW's (build it with an msvcrt MinGW)) -- a prompt step's routed experts, its gated
    MLA and KDA layers and its index picks stay on the CPU

`deploy\kimi-k31\build-windows.ps1` asks gcc which C runtime it targets and stops on the UCRT
(`-AllowUcrt` builds that engine on purpose), the install's workflow builds with an msvcrt
MinGW-w64, and BUILD-INFO-K31.txt names the runtime the shipped `kimi_k3.exe` was built on.

**Checked** (`make kimi-k31-tiny-check`): `tools/make_kimi_k3_tiny.py --dsa` writes the
tiny fixture with an indexer on every MLA layer (its other weights K3's tiny ones byte for
byte, `index_topk` 16). `tests/test_kimi_k31_dsa.py` runs the engine against
`tools/k3_ref.py`'s independent numpy indexer and sparse MLA on injected inputs (every
layer, rel-L2 below 3e-4; the dense reference differs by 10-100% past position 16, so the
comparison bites), checks that prompt steps of any size give the same logits, that
`K3_DSA=0` gives K3's, and that a config the engine does not implement is refused. The
photos, parking, dashboard and multiplexed-slot checks pass on it as on K3's fixture.
`tests/test_k3_dsa` checks the scores (bit for bit against the fused chain), the pick and the
LayerNorm; `tools/check_k3_dsa_kernels.cpp` runs the GPU's indexer kernels on the CPU (a thread
a CUDA thread) against the same chain and a stable sort, ties and NaN included.
`tests/test_k3_cuda_experts --dsa` checks `k3cu_dsa_mask`, `k3cu_mla_mask` and `k3cu_mla_sel`
on a card against the CPU, bit for bit: K3.1's widths at 3,000 and at 30,000 positions, rows
on both sides of `index_topk`, a pick that is no multiple of 8, the tiny fixture's shapes.

**On the real model** (Kimi K3.1 Pro, the RTX 4090 machine, 2026-11-10;
[the record](experiments/kimi-k31-4090-2026-11-10/README.md)). A 5,862-token prompt's logits
at every position are the same bytes with the launcher's profile -- the routed experts, the
KDA and MLA layers, the index picks and the selecting rows' lists all on the card -- as with
no GPU path at all (`K3_GPU_EXPERTS=0 K3_CUDA_DENSE=0`, 1,861 s on the CPU against 110 s),
in three runs of the release build and with `K3_CHECK=1` (no finding). A 16,511-token
prompt's are the same bytes in one step over the lists as in steps of 8,192 with the masks,
and under `RAM_GB=110`, where the step's activations take the expert cache down to its
floor of 32 slots a layer and the process stays at 105.6 GB. Upstream's K3.1 reference (JustVugg/colibri#2713, `tools/k3_ref.py` there: an indexer
written from the definition, apart from both engines) agrees with this engine on the tiny
fixture to float32 noise (rel-L2 1.2e-5 a layer, 4e-5 past `index_topk`).

**Context.** The keys and the MLA caches take 67 KB a position. They used to be counted
against the RAM whole before the expert cache got the rest, so `-Context 1048576` reserved
71 GB whether or not a conversation ever reached them, leaving the 4090 machine's cache about
33 experts a layer instead of about 85 at the profile's 131,072. Now the caches grow with the
conversation (`K3_KV_GROW=1`, the default): the RAM plan projects `K3_KV_MIN` positions, the
rest of an idle machine goes to experts, and a request takes its positions as it writes
them, doubling. What the caches hold past the plan is lent by the expert cache (`K3_LEND`):
that many slots are retired -- the coldest of every layer in turn, their bytes freed and
nothing read into them again -- so the process stays under `RAM_GB` at any length of
conversation, and the line says what it took:

    [K3][RAM] the conversation's caches hold 1.19 GB, 0.91 GB past the plan's: the expert
    cache lends 52 slots for them (52 more, 0.00 GB of experts out; 84-85 slots a layer now)

(The first form of this freed the coldest experts' bytes and left their slots in place; the
next misses read into those same slots, and a minute later the process stood above `RAM_GB`
by everything the conversation had grown.) A large prompt step borrows the same way for as
long as it runs -- 1.4 MB of activations a row past 512, 22.4 GB for a step of 16,511 rows --
and gives the slots back at its end, so a step's size costs decode nothing: the plan no
longer reserves a `K3_CHUNK` step's activations (85 slots a layer instead of 78 with the
launcher's old 8,192-row steps). A slot a reader is filling, one a step is computing, a
waiting prefetch and the card's pinned staging are never lent, and a layer keeps 32 slots
(two tokens' experts); when the layers cannot give all that is asked the line says so.
`K3_KV_GROW=0` and `K3_LEND=0` are the old behaviours, and a muxed serve (`KV_SLOTS>1`)
keeps them. `tests/kimi_k3_lend_harness.py`: the logits and tokens with the caches grown and
slots lent are those of the whole caches with nothing lent, `K3_CHECK` clean.

**The start.** An HF checkpoint's dense matrices are 115 GB of BF16, read and quantized at
every start (int4-g64 and int8, 31 GiB in RAM). Three things about that, measured on the
4090 machine: (1) the layer loop's `parallel for ... if(par>1)` was a region to omp_spin.c
even with one thread, so `w_load`'s own regions ran as teams of one and the whole load on one
core: 138 s; with the loop outside a region it is 28 s (every core, C: at 4 GB/s). (2)
`K3_DENSE_CACHE` keeps the quantized bytes in a file (33.9 GB), written by the first start
(18 s more, once) and read by the next ones unbuffered, in 2 MiB pieces over every worker:
5.6 s, and the start 9.7 s. A matrix comes from the file only when its name, shape and bits
and its source's place in the checkpoint (the shard's size and time of writing, the tensor's
offset, bytes and type) are the record's and its bytes sum to what was written; anything else
is quantized as before and the file written again (`tests/kimi_k3_dense_cache_harness.py`:
other bits, a changed byte, a touched checkpoint, a cut file -- the same logits every time).
A start from the file also leaves the page cache alone, and the first prompt after it is
sooner (28 against 34 s for 35 tokens). (3) A serve seeds half of every layer's cache from
the experts the history used most (`AUTOPIN`). They are read by the drives' readers now (68
GB in 7.2 s, 9.4 GB/s, where one after the other they came at 4.8) -- but on this model they
are worth little: K3's router spreads its use evenly, a prompt finds 5.7% of its experts among the
pinned 4.7%, and the half of the cache they hold is taken from the one that follows the
conversation: decode 0.78 against 0.85 tok/s. The 4090 launcher starts without them
(`-Pins` to have them back).

**A prompt's steps.** A step reads the experts its rows route to once for all of them, and
a few rows route to most of a layer's: the 127 rows that a 16,511-token prompt leaves after
two steps of 8,192 read 850 GB for themselves, 77 s where the same rows inside a step cost
2. So a prompt's steps are of one size and none is short (`k3_prompt_step`): up to an eighth
past `K3_CHUNK` is one step, more is equal steps of at most that. With the lists and one
step of 16,384 the 16,511-token prompt takes 221 s where three steps with the masks took
375 (the launcher's old 8,192), the same logits at every position.

**The same request again** (a regenerated reply, a client's retry). The photo taken at a
prompt's end now keeps that position's logits, and a prompt that is exactly a photographed
one is restored whole: nothing is computed and the first token is sampled from the kept
logits. Before, at least one token had to be prefilled, which meant the deepest photo before
the prompt's end -- the previous reply's -- and the 90 tokens after it: 27 s on Kimi K3.1 Pro
for a request the engine had just answered (`tests/test_kimi_k3_ckpt.py`: turns sent again
restore their own end, recompute 0 and answer the same; through the API on the real model
the same 16,695-token request again starts its reply in 0.27 s and says the same).

**A new conversation.** Every conversation of one client begins with the same system
records -- an agent's instructions and tool declarations, 10-20 thousand tokens -- and the
turns' photos are all past the first user message, where two conversations already differ:
a new conversation read the instructions again, a whole pass over the drives. The state at
the end of the leading system records is now photographed once and kept pinned
(`K3_ROOT_PHOTO`, on with two photo slots or more; from `K3_ROOT_MIN`, 1,024, tokens). It is
taken inside the step that crosses the position, at no cost: nothing of a step carries
from row to row but the KDA layers' state, so each KDA layer's call is made in two there
-- the rows before the position, then the rest -- and its state copied between them. A row
is computed alone in any call of two rows or more, so the two calls give the one call's
bytes (`K3_SNAP_TEST`: a 5,862-token prompt with the calls cut at row 3,000 on the card
has the CPU's logits at every position). On Kimi K3.1 Pro, a 16,501-token system message
and a question: the first conversation 222.6 s to its first token as before, each new one
after it **19.9 s** (33 tokens to compute) where it took 219.2 s, the same replies. One
system prompt at a time: another client's conversation takes the caches' rows, and the
first client's next conversation reads its instructions once more
(`tests/kimi_k3_root_harness.py`).

**Several conversations** (`K3_SESSIONS`, `c/k3_sessions.h`). The engine holds one
conversation: the KDA layers' state and the MLA layers' rows. A request of another one -- a
second chat, an agent's sub-agent with instructions of its own, a client's summary request --
wrote its rows over the first one's, whose next turn then read every token again. Now the
conversation held goes aside whole (the Model's pointers to its arrays traded for empty ones:
nothing is copied) and comes back when a request continues it. Which conversation a request
is: a prompt that continues a conversation, or parts from it only past the end of its first
record that is not the system's (a regenerated reply, an edited turn), is that conversation's;
one that parts earlier is another, which a kept conversation can only lend the rows of what
they share up to a photo of the pool (the system prompt's end: their rows are copied). With
`K3_SESSION_DIR` a conversation that goes aside is written to two files -- its rows a position
after the other (a later save appends from where the file's rows stop being its own), and its
tokens, the KDA state at its end and the photo of its last prompt's end -- and leaves RAM at
once: a kept conversation's RAM would come out of the expert cache, and every token of every
reply would be a little slower for it, where reading it back takes a second or a few. Without
files the kept ones and the live one's rows past the plan share `K3_SESSION_GB` (8), the
longest unused leaving first as the live conversation grows: past 8 GB of its rows (120
thousand positions) none is left. Nothing of this runs per token, so a long reasoning is the
same with it; the live conversation is written when it goes aside, after `K3_SESSION_IDLE`
(30) seconds with no request, and when the input ends, never while a reply is computed. The
files are read at the start: a server restarted takes its conversations back, and a file is
trusted only whole and only for the model, the quantization and the format it was written by.
On Kimi K3.1 Pro (the launcher's 16 conversations in files beside the model), a 16,501-token
system message and a question, another conversation of 2,956 tokens, then the first one's next
turn: it went to its file in 1.5 s (2.07 GB) and came back in 4.3 s, **26.6 s** to its first
token where it read its 16,592 tokens again before (about 220 s); after a restart of the
server, its next turn **31.9 s** (read back in 0.5 s, 35 tokens computed; 25 s of the rest the
fresh start's first expert reads). The expert cache is the size it is without them (82-83
slots a layer), and a 5,862-token prompt's logits are the CPU's at every position
(`tests/kimi_k3_sessions_harness.py`: three conversations interleaved, in RAM and in files
across a restart, answer as each one alone; a file cut short is deleted; no files and too
little RAM drops them).

Two faults of releases r2 and r3 found on the way. A new conversation's rows were
counted on top of the last one's (`model_state_reset` freed them and left them in the count
the expert cache lends for): the cache shrank by a conversation's rows at every new one until
the process restarted -- 1.12 GB counted 2.24 after the second 16.5k-token conversation. With
the history's pins on (`AUTOPIN`, off in the launcher) a layer lent down to its floor of
2 x topk slots counted the pins among them, and the first token after the prompt waited for a
slot forever; at that floor only a pin goes now. (The idle write also needs the end of the
input seen by its poll: `PeekNamedPipe` fails on a pipe whose writer has closed it, and the
gateway, which closes the engine's input to stop it, kills it 30 s later.)

**A cancelled prompt resumes.** A prompt of several steps keeps a photo of the KDA state at
the end of the last step it finished (the rows of the steps done stand where they are), and a
cancelled prompt -- a client that gave up waiting, a user who pressed stop: the gateway cancels
a request whose client left -- keeps the state of the steps it finished instead of nothing. The
same prompt sent again resumes after them, where it started over before, so a client whose
timeout is shorter than a long prompt never got an answer. The photo is one at a time, at every
step's end (16,384 tokens in the launcher); photos inside a step would serve only a later prompt
that parts from this one there, and the places where agents and chats part -- the end of the
system prompt, of each prompt and of each reply -- have theirs. Through the API on the real model
(a 2,962-token prompt in steps of 1,024, the client gone after 150 s, in the second step): the
state after 988 tokens kept, the prompt sent again 187 s where the whole of it took 290, the same
reply (`tests/kimi_k3_sessions_harness.py`, `K3_CANCEL_TEST`).

**A decode row's attention** (the CPU's, past 2,048 positions): the indexer's scores in blocks
every worker gets (blocks of 2,048 positions left fourteen of sixteen idle at 4,096), its pick of
the 2,048 highest over the team past 32,768 positions (a radix select on the scores' keys, each
worker's picks written at its place: the same list; one thread took 0.9 ms a layer at 131,072
positions, the team 0.08), softmax's exps eight at a time (`exp_f8`, exp_f's bits: half a
millisecond a layer for 96 heads x 2,048 rows) and the MLA core's latent sums with each cached
row read once for a group's heads. The same bytes: a 16,511-token prompt and 16 tokens of its
reply have every logit of release r4's build. A decode token's attention 271 -> 253 ms at
16,500 positions; at 131,072 the indexer's row 2.6 -> 1.8 ms a layer and the MLA core 1.9 ->
1.5 ms (`tests/test_k3_dsa bench`, `tests/bench_k3_mla`) -- the rest of a token is the drives'.

**A large step on the card.** A prompt step's shared experts go to the card from a thread of
their own (`K3_SHARED_PAR`): the step's thread waited 43 s of a 16,511-token prompt for them
before it took a wave of routed experts for the CPU, and takes them from the start now. 220.7
-> 217.7 s, the same logits: the card itself is the limit -- its exact kernels for the KDA
layers, the shared and the routed experts, at about half of what the tile kernels reach alone,
are what a faster large step needs.

**The dense cache read ahead** (`K3_DC_AHEAD`): its pieces in order by 12 threads of their own,
up to 4 GB ahead of the load, where each matrix was read over the team one after the other
(the scales a read of their own): 33.9 GB in 4.6 s instead of 5.6, the engine ready in 7.3-7.5 s
instead of 9.4.

Download and run on the 4090 machine: `tools/download_kimi_k31_pro_4090.ps1` (folders
`C:\AI\Kimi-K3.1-Pro`, `D:\AI\Models\Kimi-K3.1-Pro`) and
`tools/start_kimi_k31_pro_4090.ps1` (API model `kimi-k3.1-pro`; it refuses a `kimi_k3.exe` that
predates `dsa_config` and a `coli_cuda_k3.dll` that lacks `k3cu_dsa_mask`). Both come from one
build: `deploy/kimi-k31/build-windows.ps1` compiles the engine and the DLL together and runs
`tests/test_k3_cuda_experts.exe --dsa` against the card before copying them into the install.

## Current limitations

- Decode is single-token (no speculative decoding — K3 has no MTP head).
- Tool declarations/calls and image content are not exposed through the shared
  gateway yet; unsupported requests fail explicitly.
- CPU + optional Vulkan (the shared expert tier) or CUDA for the dense matrices
  (`K3_CUDA_DENSE`, above) and a prompt step's experts, MLA attention and routers
  (`coli_cuda_k3.dll`, Windows builds: its tables hold MinGW's exp_f and tanhf bits); no
  Metal. KDA's recurrence and convolution stay on the CPU, and so does a decode step's
  every expert.
- The protocol, tokenizer, gateway, TUI, and Web client paths are locally
  testable without the 1.5 TB checkpoint. A release claim still requires one
  full-model multi-turn TUI/Web run on a host that owns the complete snapshot.
