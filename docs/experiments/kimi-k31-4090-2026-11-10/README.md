# Kimi K3.1 Pro on the RTX 4090 machine, 2026-11-10: the first measurements, and what they changed

[moonshotai/Kimi-K3.1-Pro](https://huggingface.co/moonshotai/Kimi-K3.1-Pro) laid out by
`tools/download_kimi_k31_pro_4090.ps1` (61.4% of every MoE layer's experts on `C:`, the rest
on `D:`), on the Ryzen 9 7950X3D, 192 GB DDR5-5400, RTX 4090. The engine is `c/kimi_k3.c`
with `coli_cuda_k3.dll`, built with MinGW-w64 GCC 14.2 (msvcrt) and CUDA 13.4 for sm_89.
Until this day the K3.1 engine (branch `claude/kimi-k31-dsa-gpu-indexer-fix`) had been
compiled and tested on tiny fixtures only: nothing in it had run on the model, and its GPU
tests had not run on a card.

Each run is `k31run.sh NAME NGEN [VAR=value ...]` (this folder): a fresh `kimi_k3.exe` with
the 4090 launcher's settings (`K3_CUDA_DENSE=1` 20 GiB with the host copies released,
`K3_GPU_EXPERTS=1`, `RAM_GB=174`, `K3_MAXT=131072`, two readers a drive, 16 threads,
`AUTOPIN=0` unless a row says otherwise), `monitor.ps1` sampling the drives, the CPU, the
GPU and the engine's memory every 2 s (`raw/*.csv`). Prompts: the chat question "What is the
capital of France? Answer in one short sentence." (35 tokens; every run answered "The capital
of France is Paris."), `long6k_prompt.txt` (5,862 tokens) and `p15k.txt` (16,511 tokens of
this repository's own documentation and a request to summarize it). The 5,862 tokens are what
Git Bash hands over when the text is an argument (it rewrites a few path-like words);
`--prompt-file` reads the file itself, 5,847 tokens, and is what `p15k.txt` goes through.

## What was wrong, in the order it was found

| | found | now |
|---|---|---|
| the load | 138.5 s for 93 layers on one core (`raw/base1`): the layer loop was `#pragma omp parallel for ... if(par>1)`, and a region asked for one thread is still a region to omp_spin.c, so `w_load`'s own regions inside it were teams of one | the loop outside a region unless several layers load at once: 28 s (`raw/load1`, `f_dec_nodc*`) |
| a grow's eviction | the conversation's rows past the plan freed the coldest experts' bytes but left their slots, which the next misses filled again: the process ended above `RAM_GB` by everything the caches had grown | the cache lends slots (`K3_LEND`): retired for as long as the bytes are held. Under `RAM_GB=110` a 16,511-row step takes the cache to its floor and the process stays at 105.6 GB (`raw/f_15k_ram110`) |
| a prompt's last step | 127 rows after two steps of 8,192 read 850 GB for themselves: 77 s (`raw/p15k_8k`: 154.7 + 171.5 + 77.0 s) | steps of one size, none short (`k3_prompt_step`); the launcher's step 16,384 |
| the selecting rows' attention | every row against every position, the unpicked ones weighing 0 (`k3cu_mla_mask`): 40.9 s of a 16,511-token prompt, growing with the square of the context | over the lists of the picks (`k3cu_mla_sel`): 9.6 s, the same bytes |
| the start, every time | 115 GB of BF16 read and quantized | `K3_DENSE_CACHE`: the quantized bytes in a file, 5.6 s to read: the start 9.7 s |
| the history's pins (a serve's default) | half of every layer's cache seeded from the history, read one expert after the other (4.8 GB/s) | read by the drives' readers (9.4 GB/s) -- and off in the launcher: they cost decode 8% (below) |
| the same request again | at least one token had to be prefilled: the photo before the prompt's end and the 90 tokens after it, 27 s | the photo of a prompt's end keeps its logits: 0.27 s |
| a new conversation (2026-11-11, below) | the system prompt read again: every photo is past the first user message | the system prompt's end photographed inside the first prompt's step and kept: 219 -> 20 s to the first token after 16,501 tokens of instructions |

Two of this day's own changes were wrong at first and are in the table as fixed, not found:
the pins read through the readers picked one expert for every slot of a layer (the pick's
"taken" test asked for a completed read: a prompt's cache hits 0.2%, `raw/serve1.log`; with
the index asked instead 5.7%, and `pin_seed` now checks every pinned slot against the
index), and a stale dense-cache file was deleted before its replacement was written, the
drive trimming 34 GB under the load (the start that rewrote it took 134 s, `raw/serve1.log`;
the old file now stays until the new one takes its name).

## The same bytes

Logits at every position of a prompt (`K3_LOGITS`), compared as files.

| prompt | run | against | |
|---|---|---|---|
| 5,862 tokens | the launcher's profile, the first build (`g6k_1`, 124.0 s) | no GPU path at all, `K3_GPU_EXPERTS=0 K3_CUDA_DENSE=0` (`g6k_cpu`, 1,861.0 s) | the same 3,841,720,320 bytes |
| 5,862 tokens | with the lending (`n6k_logits2`), with the lists (`s2_6k`) | the CPU alone | the same |
| 5,862 tokens | the release build with `K3_CHECK=1` (`f_6k_ck`: no finding), and twice more (`f_6k_2`, `f_6k_3`) | the CPU alone | the same, three times |
| 16,511 tokens | one step of 16,384 and 127 rows, masks (`p15k_16k`) | steps of 8,192, masks (`p15k_8k`) | the same 10,820,648,960 bytes |
| 16,511 tokens | one step over the lists (`s2_15k`; the release build `f_15k`) | steps of 8,192, masks | the same |
| 16,511 tokens | the release build under `RAM_GB=110` (`f_15k_ram110`) | steps of 8,192, masks, unpressed | the same |
| 35 tokens + 20 of decode | the start that writes the dense cache (`s2_dcB`), the start that reads it (`s2_dcC`) | a start without it (`s2_dcA`) | the same |

On a card, `tests/test_k3_cuda_experts.exe` (every case, `--dsa` among them: the picks, the
masked attention and the attention over the lists against the CPU's loop at 3,000 and 30,000
positions) reports 0 differing values. On the tiny K3.1 fixture the engine's hidden states
agree with upstream's independent numpy reference (JustVugg/colibri#2713's
`tools/k3_ref.py`, run here as `upstream_cross.py`) to rel-L2 1.2e-5 a layer and 4.0e-5 on the
rows past `index_topk`.

## The start

| | init | of which |
|---|---:|---|
| the branch as it was (`base1`) | 138.5 s | the layers 135.1 s on one core, C: at 0.75 GB/s |
| the layer loop outside a region (`load1`, `f_dec_nodc1/2`) | 28.0-29.6 s | the layers 24.5-27.7 s, C: at 3.7-4.4 GB/s, 13 cores busy |
| `K3_DENSE_CACHE=1`, the start that writes the file (`s2_dcB`) | 48.0 s | 18.3 s writing 33.9 GB |
| `K3_DENSE_CACHE=1`, the starts after (`f_dec_dc1/2`, `f_t15k`, ...) | 9.6-9.9 s | 5.3-5.8 s reading 33.9 GB (2 MiB pieces; 6.3-7.0 s with 8 MiB ones, `s2_dcC`) |
| a serve with the history's pins, on top (`serve2.log`) | +7.2 s | 3,864 experts, 67.8 GB, by the drives' readers |

Through the API (`coli serve`, the launcher's settings, `raw/serve3*`): the server answers
`/v1/models` 11.3 s after its start.

## Prompts

| prompt | | prefill |
|---|---|---:|
| 35 tokens | after a start that quantized the checkpoint (`f_dec_nodc1/2`) | 33.4-34.8 s |
| 35 tokens | after a start from the dense cache (`f_dec_dc1/2`: the page cache was left alone) | 28.3-28.7 s |
| 5,862 tokens, one step | the release build (`f_t6k`) | 109.5 s |
| 16,511 tokens | steps of 8,192, masks: the build before the lists at the launcher's old step (`s2_t15k_old`) | 375.1 s |
| 16,511 tokens | one step over the lists: the release build (`s2_t15k_new`, `f_t15k`) | 221.5, 220.8 s |

Where the 16,511-token step's 220.8 s go: attention 66.9 s (KDA 41.6, MLA 25.3 of which the
core over the lists 9.5 and the index keys and picks 4.4), MoE 133.4 s (the routed experts
53.9, the shared experts 42.8, the latent projections 13.2, the routers 8.4, the weighted
sums 7.1). It reads 1,446 GB, which the two drives deliver in about 105 s: a step of this
size waits for the card's exact kernels, not for the drives (a 5,862-token step, 109.5 s, is
at the drives' pace). A step's activations are borrowed from the expert cache while it runs:
22.4 GB for 16,511 rows, 1,276 slots lent and returned.

## Decode

Each pair alternated twice; 58 tokens of the 35-token prompt's reply.

| | a token | tok/s | cache hit | read a token |
|---|---:|---:|---:|---:|
| the plan that reserved a step's activations, 78 slots a layer (`dec_old1/2`) | 1,113 / 1,110 ms | 0.90 | 65.2% | 12.3 GB |
| the lending, 85 slots a layer (`dec_new1/2`) | 1,095 / 1,096 ms | 0.91 | 65.8% | 11.9 GB |
| the history's pins, `AUTOPIN=1` (`pin_on1/2`) | 1,278 / 1,279 ms | 0.78 | 63.7% | 14.0 GB |
| none, `AUTOPIN=0` (`pin_off1/2`) | 1,174 / 1,177 ms | 0.85 | 65.8% | 11.9 GB |
| the dense cache read at the start (`f_dec_dc1/2`) | 1,181 / 1,173 ms | 0.85 | 65.8% | 11.9 GB |
| the checkpoint quantized at the start (`f_dec_nodc1/2`) | 1,171 / 1,174 ms | 0.85 | 65.8% | 11.9 GB |

The first two pairs ran in the morning and the others in the evening: the machine's own
drift between them is 7% (the same build, 1,095 and 1,174 ms), so only the rows of a pair
compare. Past `index_topk` positions a decode row's attention is the CPU's loop over 2,048
picked rows in every MLA layer plus the index scores: 224 ms a token at 5,900 positions
(`f_t6k`) and 244 at 16,500 (`f_15k_ram110`) against 120 at 35.

## Through the API

`api_session.py` against `coli serve` with the launcher's settings (`raw/serve3.log`, the
engine's own lines; `raw/serve3_session.json`), `max_tokens` 48:

| turn | prompt | reused | to the first token | the whole turn |
|---|---:|---:|---:|---:|
| a short question | 35 | 0 | 28.7 s | 45.0 s |
| a follow-up, the history sent back | 78 | 48 | 15.4 s | 32.9 s |
| a 16,511-token message appended | 16,602 | 91 | 228.6 s | 289.1 s |
| a follow-up past `index_topk` | 16,695 | 16,649 | 27.6 s | 86.1 s |
| the same request again | 16,695 | 16,695 (the photo of its end) | **0.27 s** | 58.2 s |

The same session against the build before the photo kept its logits (`raw/serve2*`): the
last turn 27.0 s to the first token. Both gave the same texts. A conversation longer than the
served context (`api_overlong.py`, `-Context 8192`: 10,186 tokens, and 16,511, past the
framing buffer too) is answered HTTP 400 `context_length_exceeded` in 0.2 s, twice, and the
short question after it is answered: no restart (`raw/serve4.log`).

## A new conversation with the same system prompt (2026-11-11)

`api_root.py` against `coli serve` with the launcher's settings (`serve_root.sh`): each
conversation is `[system, user]`, the system message `p15k.txt` (16,501 tokens with its
framing), sent as a fresh conversation; `max_tokens` 32.

| conversation | `K3_ROOT_PHOTO=0` (`raw/root_off*`) | with the photo (`raw/root_on*`) | reply |
|---|---:|---:|---|
| 1 | 221.3 s to the first token | 222.6 s (the photo taken inside the step, at 16,501 tokens) | the same |
| 2, a new one | 219.2 s (16,534 tokens computed) | **19.9 s** (33 computed) | the same |
| 3, a new one | -- | 20.3 s (34 computed) | |

The photo is taken by making each KDA layer's call in two at the position. With
`K3_SNAP_TEST=3000` the CLI cuts the 5,862-token prompt's step the same way (138 KDA calls on
the card instead of 69): the logits are `g6k_cpu.f32`'s at every position, the prompt 117.3 s
with its logits written (`raw/root_6k_cut`). Release r3's `kimi_k3.exe` (0fb3905d, this build)
on the same prompt with no cut: `g6k_cpu.f32`'s logits again, 5,862 positions of 5,862
(`raw/r3_6k`, 120.6 s with its logits written).

## Several conversations, in files (2026-11-11)

`api_sess.py` against `coli serve` with the launcher's settings and 16 conversations kept in
files (`serve_sess.sh`; `K3_SESSIONS=16`, `K3_SESSION_DIR`, nothing kept in RAM): A is
`p15k.txt` as its system message and a question, B `sys2k.txt` (the first 9,000 characters of
it) and a question -- another conversation -- then A's next turn; then the server stopped
(killed after its idle write) and started again, and A's turn after that (`raw/sess1*`,
`raw/sess2*`).

| request | to the first token | what happened |
|---|---:|---|
| A1, 16,539 tokens | 231.5 s | the prompt (222 s in the runs above: the machine's drift) |
| B1, 2,956 tokens | 101.4 s | A written to its file first: 2.07 GB in 1.5 s |
| A2, 16,592 tokens | **26.6 s** | B written (1.15 GB, 3.7 s), A read back in 4.3 s, 16,558 reused, 34 computed |
| after a restart: A3, 16,637 tokens | **31.9 s** | A read back in 0.5 s, 16,602 reused; 24.9 s of the prompt's 30.9 the fresh start's expert reads |

Without the kept conversations A2 and A3 read their 16.6 thousand tokens again, as a new
conversation of that length does (219-221 s above). The expert cache is the size it is
without them: 82-83 slots a layer, 49 lent for A's rows, and B did not add to that count.
This build's `kimi_k3.exe` (f0d72290) on the 5,862-token prompt: `g6k_cpu.f32`'s logits at 5,862
positions of 5,862 (`raw/r4_6k`, 120.6 s).

## The not-done list, 2026-11-11

The same prompt (`p15k.txt`, 16,511 tokens, the launcher's step of 16,384) and 16 tokens of its
reply on release r4's source (`raw/r5_old`) and on this one (`raw/r5_new`), each with
`K3_LOGITS` and `K3_LOGITS_DECODE`: all 16,527 rows of logits the same bytes.

| | r4 | now |
|---|---:|---:|
| the dense cache read (33.9 GB) | 5.6 s | 4.6 s (read ahead by 12 threads) |
| the engine ready | 9.4 s | 7.5 s (6.9-7.5 over three starts) |
| the prompt, its logits written | 253.3 s | 244.0 s |
| a decode token's attention after it (the CPU's, 16,500 positions) | 271 ms | 253 ms |
| a decode token, the first 16 after the prompt | 2,076 ms | 1,676 ms (of it 964 -> 617 ms waiting for the drives: their drift) |

The shared experts on a thread of their own, on one build (`raw/r5_15k_par`, `raw/r5_15k_nopar`,
no logits written): 217.7 s with them, 220.7 s with `K3_SHARED_PAR=0`. The step's thread then
computes 93 s of routed experts for the CPU where it computed 55, beside the card; the card is
what a large step waits for. The 5,862-token prompt on this build: `g6k_cpu.f32`'s logits at
every position, 117.1 s (`raw/r5_6k`).

A cancelled prompt through the API (`raw/resume.log`, `api_resume.py`): the launcher's settings
but steps of 1,024, the 2,962 tokens of `sys2k.txt` and a question; `curl` sent it and gave up
after 150 s, in the second step; the engine kept the state after the first 988 tokens; the same
request again reused them and took 187 s to its first token where the whole prompt took 290.5 s
(a run before it whose client waited), the same reply.

Benches on the CPU (16 threads; `tests/test_k3_dsa bench`, `tests/bench_k3_mla`, the pick alone
with the engine's spinning team):

| a decode row, one layer | r4 | now |
|---|---:|---:|
| the indexer at 4,096 / 16,384 / 131,072 positions | 0.44 / 0.72 / 2.60 ms | 0.24 / 0.50 / 1.84 ms |
| its pick of 2,048 at 131,072 / 262,144 (one thread / the team) | 0.88 / 1.97 ms | 0.08 / 0.12 ms |
| the MLA core over 2,048 picked rows (96 heads) | 1.90 ms | 1.37-1.46 ms |
| of it softmax | 0.52 ms | 0.11 ms |

## A large step's rows beside the kernels (2026-11-12)

A prompt step on the card registers its big host buffers with it (`K3_PIN_STEP`; 18 buffers,
11.0 GB for a step of 16,384 rows, 0.6 s to register and 0.1 s to let go), and the DLL's calls
whose rows are registered move them on a copy stream beside their kernels (`pipe_rows` in
`backend_cuda_k3.cu`). The 16,511-token prompt, step 16,384, no logits written, r5's pair
(`kimi_k3.exe` 947d1767, `coli_cuda_k3.dll` 7ed12963) and this one's (`coli_cuda_k3.dll`
1a4faee4) in turn (`raw/ab_r5_1`, `raw/ab_hw_1`, `raw/ab_r5_2`, `raw/ab_hw_2`):

| | r5 | now |
|---|---:|---:|
| the prompt | 227.5 / 228.9 s | 205.5 / 205.2 s |
| attention: KDA / MLA | 43.6 / 26.4 s | 38.2 / 21.1 s |
| the KDA layers' calls on the card | 38.9 s | 35.6 s |
| the dense matrices the card does not hold (353 calls) | 24.2 s | 13.0 s |
| the dense matrices it holds (56 calls) | 8.5 s | 5.1 s |
| the latent's two matrices | 13.8 s | 6.7 s |
| the shared experts (beside the routed ones) | 44.0 s | 27.1 s |
| the routed experts, computed | 97.9 s | 95.6 s |
| their bytes copied / waiting for a slot | 67.6 / 5.0 s | 53.7 / 16.3 s |

The copies' queue (`qblock.cu`, built with `nvcc -O2 -arch=sm_89`): on this machine (WDDM, driver
621.14) a 50 MB copy of a stream that waits for nothing finishes 90-104 ms after it is queued when
another stream's copy queued before it waits for a 100 ms kernel, in either direction and from
pinned or pageable memory, and 2 ms when nothing waits; a kernel of another stream is not held up.
The pipe's first version queued each chunk's copies behind waits for its kernels: the shared
experts' pipe then held the routed experts' copies and the experts' rows a layer sends first
(`k3g_begin`: 14.0 s of `raw/pipe_15k_all`'s 211.6, 0.7 s now). Its two runs in the half hour
before midnight took 358.3 and 342.6 s (`raw/pipe_15k`, `raw/pipe_15k_b`), everything on the card
slower (the KDA layers 85.5 s), and the runs beside them were slow too (r5 230.7 s, the pipe's
build with `K3_PIN_STEP=0` 247.3 s), the adapter's memory 40-130 MB fuller at its peak than in
every run since; the same DLL with the same 11.0 GB registered took 211.6 s after midnight. Not
explained; not seen again. `monitor2.ps1` (`K3_MON=monitor2.ps1`)
adds the GPU memory WDDM gives the engine to the CSV: 22.3 GB dedicated and 35.6 GB shared (the
cache's 20 GiB of pinned slots, the experts' outputs and the step's registered buffers) at most.

The same bytes, on this build (`kimi_k3.exe` fc6b9154): the 5,862-token prompt has `g6k_cpu.f32`'s
logits at every position (`raw/r6_6k_logits`, 116.8 s: that prompt waits for the drives, 13.9 s of
its MoE); the 16,511-token prompt, its logits written, has `p15k_16k.f32`'s, the bytes every
release since r2 has had (`raw/r6_15k_logits`, 229.1 s where r5 took 244.0 with them written).
The `kimi_k3.exe` built from the committed source differs from fc6b9154 in its 4 bytes of PE
time stamp and checksum. `tests/test_k3_cuda_experts --pin` runs every call with
registered rows in several chunks, so a chunk reuses the buffers of the chunk two before it: the
CPU's bytes.

## The routed experts in tiles (2026-11-12)

`k3_gu_t` and `k3_dn_t` (`backend_cuda_k3.cu`; `K3_GPU_EXPERT_TILES`) take a routed expert's piece
of 32 rows or more: 32 rows and 64 outputs of gate and up (128 of down) a block, the weights decoded
once into shared memory a stage at a time, all 8 lanes of 4 rows x 2-4 outputs a thread, a dp4a's sum
made a float by adding it to 1.5 x 2^23's bits. `bench_ke.cu` on one random expert at K3.1's widths,
the old kernels against the tiles, every byte of gq, gs and y the same:

| rows | k3_gate_up + k3_down | k3_gu_t + k3_dn_t |
|---:|---:|---:|
| 8 | 0.061 ms (4.3 TMAC/s) | 0.180 ms (1.5) |
| 32 | 0.175 ms (6.0) | 0.168 ms (6.3) |
| 64 | 0.245 ms (8.6) | 0.154 ms (13.7) |
| 300 | 1.088 ms (9.1) | 0.554 ms (17.9) |
| 1,000 | 3.184 ms (10.4) | 1.577 ms (20.9) |

On the model, r6's `kimi_k3.exe` (fc6b9154) with r6's DLL (1a4faee4) and with this one (314741c4),
the 16,511-token prompt in turn (`raw/ke_ab_*`):

| | r6's DLL | the tiles |
|---|---:|---:|
| the prompt | 224.3 / 202.8 s | 192.3 / 188.7 s |
| the routed experts, computed | 104.8 / 91.7 s | 83.3 / 82.4 s |
| their bytes copied / waiting for a slot | 59.3 / 18.0, 56.2 / 11.6 s | 60.3 / 6.3, 60.5 / 5.7 s |
| the shared experts (beside them) | 30.9 / 25.8 s | 22.8 / 22.2 s |
| experts the CPU took | 6,381 / 5,678 | 5,056 / 4,952 |

(r6's first run is this series' slow one: its attention took 68 s against 55-61.) The same bytes:
the 16,511-token prompt's logits are `p15k_16k.f32`'s (`raw/ke_15k_logits`, 243.3 s with them
written, the attention 68 s there too) and the 5,862-token prompt's `g6k_cpu.f32`'s
(`raw/ke_6k_logits`, 117.5 s: it waits for the drives, 16.5 s of its MoE); `tests/test_k3_cuda_experts`
sends 5 of its 11 experts' pieces through the tiles, 0 of 5,526,528 outputs differ from the CPU's.

The phase now waits for the bus: the weights' copies took 54 s beside the old kernels and 60 s
beside the tiles, whose outputs come down into the pinned rows faster (0.33 TB of a prompt against
the weights' 1.34 TB up; on this machine the two directions together are slower than one).

## Pictures (2026-11-12)

Kimi K3.1's vision tower in the engine (`k3_vision.h`, `k3cu_vision`; docs/kimi_k3.md, "Kimi K3.1:
pictures"). The oracle is Moonshot's own `MoonViT3dPretrainedModel` and `PatchMergerMLPV2`
(`modeling_kimi_k3.py` of the Kimi-K3 release, the same vision_config; `tools/k3_vision_ref.py`)
in f32 on the card, TF32 off, the real weights. Two synthetic pictures (`vision/small.png` 640 x
400, `vision/big.png` 1568 x 830; `vision/shot2.png` 1280 x 720 for the agent):

| | 640 x 400 (30 x 46 patches, 345 rows) | 1568 x 830 (60 x 112, 1,680 rows) |
|---|---:|---:|
| `k3_vision.h` (CPU) against the oracle, rows | 1.03e-5 relative | 1.27e-5 |
| after the patches and positions / block 0 / the final norm | 5.6e-7 / 1.0e-6 / 8.7e-6 | |
| `k3cu_vision` (GPU) against the oracle | | 1.34e-5 (1.50e-5 from the CPU's) |
| the oracle in bf16 against itself in f32 | 4.34e-2 | |
| the CPU's tower | 2.4 s | 18.1 s (matrices 6.7, attention 11.1) |
| the GPU's, alone / inside the serve (20 GiB of dense matrices beside it) | | 0.76 / 0.76 s |

The norms' eps (the tower's `nn.RMSNorm(hidden)` have none): the bf16 oracle is 4.3% from the f32
one with FLT_EPSILON and 55% (the lowest row cosine 0.09) from one with bf16's 2^-7 -- the
activations' mean squares going into the norms are 0.0005-0.3, so the eps is not a detail.
FLT_EPSILON is what torch 2.15 uses in bf16 too, and what the model reads well with (below).

`tools/kimi_image.py` gives the processor's pixels byte for byte (`vision/pre_check.py`: PNG, JPEG,
RGBA, palette with and without transparency, grey, 6000 x 4000 at 16,590 rows, 5 x 3 at 1), and
the engine's tokens decode to `encoding_k3.py`'s segments with the pads expanded
(`vision/wire_check.py`: a user picture and a tool result's, 2,025 pads).

Through the API (worktree build, `-Sessions 0`, a fresh expert cache; `raw/vision_api_cpu.txt`,
`raw/vision_api_agent.txt`, `raw/vision_serve_*.txt`):

- the 640 x 400 picture, "describe it": the title "LookRS232 - COM3 115200 8N1", the terminal's
  seven lines word for word, the red circle, Connect and Send, "Baud: 115200" (391 prompt tokens,
  the prompt 80 s, all of it the drives');
- the screenshot from the user, title / baud rate / buttons: all three right (1,742 tokens,
  179 s with the answer);
- the screenshot as a tool result (Kimi Code's Computer Use shape): "Connect, Send, and Clear"
  (1,921 tokens, 132 s; the rows from the cache);
- an agent's two turns, GPU tower: after the first screenshot the model called the tool for the
  settings window; with the second, "No, the LookRS232 window shows a baud rate of 115200, while
  the serial port settings window shows 9600." The second turn restored the photo of the first
  prompt's end (1,930 of 3,236 tokens, the first picture in it) and the tower ran only for the new
  picture (0.65 s).

## A draft model (2026-11-17)

Speculative decoding for K3.1 Pro (`k3_draft.h`, `backend_cuda_k3_draft.cuh`; docs/kimi_k3.md,
"Kimi K3.1: a draft model"). The scripts are in `draft/`.

**Which one.** Four draft models for this target are on Hugging Face. By their own cards, at
width 8 and temperature 1.0:

| | first draft stands, SWE-bench Pro | accepted length, GSM8K / AIME 2026 | size |
|---|---:|---:|---:|
| Inferact/Kimi-K3.1-Pro-DSpark (2026-11-14) | 0.75 | 5.57 / 3.43 | 4.96 GB |
| Inferact/Kimi-K3.1-Pro-DSpark-preview | 0.67 | 4.60 / 2.97 | 5.37 GB |
| LithosAI/Kimi-K3.1-Pro-DSpark | 0.67 | 4.76 / 2.80 | 4.50 GB |
| RadixArk/Kimi-K3.1-Dflash2-Preview | -- | 5.29 / 3.32 | 5.57 GB |

Inferact's second checkpoint leads on every benchmark two of them share, and is the simplest
to run: plain grouped-query attention with a window of 2,048 positions, where the preview and
Dflash2 are MLA stacks. (gureshov/Kimi-K3.1-Pro-Coding-720E-DSpark is for a pruned target;
ZhuoBohang-AMD's is gated.)

**What it would guess on real work**, before any of the engine was written. Two recorded Kimi
Code sessions replayed through the engine as prompts (47,767 and 58,123 tokens; `build_req.py`),
with a temporary hook (`patch_hid.py`: `K3_HID_DUMP`) writing the hidden rows after ten layers,
every position's input id and the id the head ranks first; then the draft model in PyTorch over
those rows (`dspark2_eval.py`) at each of the 9,660 positions the model itself had written
(`spans.py`), a draft standing when it is the head's first id there:

| drafts 1..i all stand | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| the rows after layers 75, 79, 83, 87, 91 | 0.79 | 0.59 | 0.43 | 0.31 | 0.23 | 0.17 | 0.13 |
| the rows after layers 74, 78, 82, 86, 90 | 0.70 | 0.46 | 0.30 | 0.20 | 0.13 | 0.07 | 0.05 |
| the matrices and the head as int8 rows | 0.79 | 0.59 | 0.43 | 0.31 | 0.23 | 0.17 | 0.13 |
| ... as int4 in groups of 64 | 0.78 | 0.58 | 0.41 | 0.29 | 0.22 | 0.15 | 0.11 |
| a window of 1,024 / of 256 rows | 0.78 / 0.76 | 0.58 / 0.54 | 0.41 / 0.36 | | | | |

(the second, fourth and fifth rows on 1,500 of the positions; `acceptance_int8.txt` is the
third's whole output). So `aux_hidden_state_layer_ids` 76..92 count from 1, as vLLM captures
them. By what the model was writing: a tool call's arguments 0.83, 0.65, 0.50; a reply 0.83,
0.62, 0.43; reasoning 0.63, 0.38, 0.25. The confidence head says 0.95 or more at 57% of the
positions and 92% of those stand; the pick's own probability sorts better (at 0.98 or more, up
to 3 drafts: 92% of 4,623 stand).

**The same bytes.** `rowtest.sh`: a 51-token prompt in steps of 1, 2, 3 and 8 rows -- one
logits file. `drafttest.sh` (the CLI, greedy, `K3_LOGITS` with `K3_LOGITS_DECODE=1`, a fresh
process a run):

| | plain | `K3_DRAFT`, 0.98, at most 3 | `K3_DRAFT_P=0.3 K3_DRAFT_MAX=7` |
|---|---:|---:|---:|
| 51-token prompt, 48 tokens of reply | 62.6 s | 52.6 s; 24 steps, 24 of 24 drafts stood | |
| 3,759-token prompt, 120 tokens | 173.3 s | 153.4 s; 73 steps, 47 of 49 stood | 163.6 s; 25 steps, 95 of 143 stood |
| logits, every position | | the plain run's bytes | the plain run's bytes |

In the third column blocks ended with 0, 1, 2, 3, 4, 6 and 7 of their drafts standing: every
depth of what a block takes back. The card's draft model against PyTorch on the rows it was
given (`K3_DRAFT_DUMP`, `tools/k3_draft_ref.py`): 151 of 152, 503 of 504 and 167 of 168 picks
the same in the three runs, the differing one a tie at probability 0.28 each time, the
probabilities within 2.3e-3.

**What a draft costs here.** From the second row: a step of one row 1.445 s (this prompt, a
fresh process: 14 GB of experts read a token); with the draft model a step carried 1.67 rows
on average and took 2.10 s, so a draft row about 0.6 of a step, its routed experts read and
computed for it alone, plus 77 ms a step that is the draft model's: its pass (23-37 ms) and the
1.9 GiB of dense matrices it moved off the card (20.0 -> 18.1 GiB there). With those costs over
the replayed sessions (`econ3.py`), the gate's two numbers hardly matter between 0.95 and 0.99
and from 3 drafts up: x1.13 everywhere (0.98 and 3: x1.131; 0.8 and 7: x1.05; always one draft:
x1.09).

**Through the API** (`serve_ab.sh`: the launcher on port 18081, a request sent twice, the second
time answered from its prompt's photo so the time is the reply's; temperature 0):

| a recorded Kimi Code turn | without | with | |
|---|---:|---:|---:|
| 23,295 tokens, answered by a tool call, 91 tokens | 121.4 s | 102.9 s | +18% |
| 23,371 tokens (the user then asks for an account), 220 tokens of reasoning | 269.0 s | 247.7 s | +9% |
| their prompts, the first time | 288.8, 301.1 s | 294.0, 303.7 s | |

The replies are the same text in all eight responses. The start takes 2.1-2.7 s more (5 GB of
bf16 read and quantized on the card).

## Both drives whole (2026-11-17)

The user asked to keep all the weights on both drives, to say how much faster the drives read and
how much shorter the waits get, and to have the engine check at every start that `C:` holds them
all and copy what it lacks from `D:` (`c/k3_mirror.h`; docs/kimi_k3.md, "the weights on both
drives"). Scripts in `mirror/`.

**The copies.** `D:` got `C:`'s 96 files (934 GB, robocopy `/J /MT:4`, 159 s; `cmpcopy.c`: all
96 the same bytes, 1,003 GB compared in 177 s), and the engine's first start with `K3_FILL=1`
copied `D:`'s 92 `.p2` parts to `C:` itself: 558.6 GB in 276 s (2.0 GB/s, one file at a time,
`CopyFileEx` unbuffered). Each folder now holds all 188 files, 1.45 TB; `C:` has 0.94 TB free. A
start with nothing to copy checks the 188 headers in 0.2 s.

**A hang on the way.** The first run with the copies stopped in the middle of its prompt: no CPU,
no reads. gdb: two readers inside `exit()`, one in `k3_io_stop`'s `pthread_join`, the other waiting
for the C runtime's exit lock the first held. Every reader had opened its own C descriptors per
file and per copy, and with 188 files on two drives they went past msvcrt's 2,048 for the whole
process; an open failed, the read failed, and the reader called `exit()`. The readers now keep
Windows handles (`CreateFile`, `ReadFile` with an `OVERLAPPED` offset: no such limit), a reader's
read error prints and ends the process with `_exit` (stderr flushed first: msvcrt buffers it when
it is redirected), and the copies get no C descriptors on Windows.

**The same bytes** (`logits_check.sh`, a 3,759-token prompt and 32 tokens, every position's
logits): r11 as installed (the cut layout) = the new exe reading the copies (the hybrid below,
with 2 and with 3 readers a drive) = reading `C:` alone (`K3_MIRROR=0`) = the run whose start
copied the 558 GB. The replies too.

**How much faster** (`perf_cli.sh`: the same prompt and 128 tokens with the draft model, a fresh
process a run, the new exe in every column; the cut layout read through a folder of hard links to
`C:`'s 96 files, so both layouts sit side by side on the drives):

| | prompt | a decode token | waiting for reads | drives together: prompt, decode | runs |
|---|---:|---:|---:|---:|---|
| cut layout, 2 readers a drive (as before) | 107.3-107.5 s | 1,205-1,217 ms | 344-377 ms | 13.5, 11.4-11.5 GB/s | 4 |
| cut layout, 3 readers | 106.7 s | 1,244-1,249 ms | 398-410 ms | 13.6, 11.5 GB/s | 2 |
| both whole, every read cut 60/40, 2 readers | 111.0-111.1 s | 1,186-1,188 ms | 342-345 ms | 13.0, 11.7 GB/s | 2 |
| ... 3 readers | 110.4 s | 1,166-1,169 ms | 308-314 ms | 13.1, 12.2 GB/s | 2 |
| both whole, a prompt's reads whole by the shorter queue, 2 readers | 102.6-102.7 s | 1,168 ms | 325-335 ms | 14.1, 11.8 GB/s | 2 |
| **... 3 readers (the default now)** | **100.5 s** | **1,138-1,142 ms** | **284-293 ms** | **14.4, 12.3 GB/s** | 2 |

Against the cut layout: the prompt 6% sooner, a decode token 6%, its wait for reads 21% shorter,
the drives 7% faster together in both phases. The model the replicas were planned on (2026-11-13:
the last of N reads on idle drives 28-44% sooner) promised more for the decode than this: the
prefetcher keeps both drives busy 88% of a decode token already, so a read cut in two lands
sooner only by what balance gives back. Cutting the prompt's reads made it slower (two half-size
reads cost more than one when both drives are saturated: `C:` 8.5 -> 8.1 GB/s), hence the hybrid,
which also keeps `D:` busier than the cut layout's fixed 61/39 did (97% of the prompt instead of
88%). A third reader slows the cut layout (more read ahead, no more read) and speeds up the
copies'.

**Through the API** (`api_ab.sh`: the launcher on port 18081 with the draft model, the recorded
Kimi Code turn of the draft model's table sent twice, the second time answered from its prompt's
photo; cut, copies, copies, cut):

| a recorded turn of 23,295 tokens, answered by a tool call (91 tokens) | cut layout | both drives whole |
|---|---:|---:|
| the turn, prompt and reply | 406.5, 404.6 s | 404.4, 404.2 s |
| its reply alone | 103.9, 103.6 s | 101.2, 101.2 s |
| the drives together during the reply | 12.1-12.5 GB/s | 12.7-13.1 GB/s |

All eight replies are the same tool call. A prompt this long is the card's (299 s either way, 0.4
s of it waiting for reads, the drives busy 67-72%), and a reply at 23k positions spends more of a
token in attention, so the turn gains 2.5% of its reply and nothing of its prompt.

**Copies of some files only** -- what a release user's first start leaves on the download tool's
layout: `D:`'s parts copied to `C:`, `D:` without `C:`'s files. Cutting reads there would have put
85% of a decode token's bytes on `C:`, so with any expert file lacking its copy every read of a file
with one goes whole by queue, 2 readers a drive (`perf_cli.sh partial`: `K3_DIRS` a folder of hard
links to `D:`'s 92 parts). Against the cut layout, the same exe: the prompt 107.3 s (107.5), a
decode token 1,158 ms (1,210), its wait 313 ms (367) -- the queue sends some of the parts' reads to
`C:` when `D:` is the busier. Every file on both drives with the release's exe and the launcher's
defaults: 100.5 s, 1,150 ms, 275 ms. The release's exe's logits = r11's exe's (run from its backup,
reading every file from `C:` now: a prompt 174.5 s, which is what going back to r11 without
deleting `C:`'s copies of the parts would cost).

The first single runs of the day -- r11 waiting 750 ms a token, the copies' first runs 562-581 --
were on drives that had just taken 934 and 558 GB of writes and were still busy behind them; the
table's runs came an hour and more later, alternating.

## A long answer (2026-11-17)

One request through the API with release r12 as installed in `C:\colibri` (the draft model on,
every model file on both drives, the user's launcher: 24 threads), a fresh server on port 18081:
a short prompt (209 tokens) asking for a whole Python module, greedy, 2,048 tokens of code
(`long/`: `make_requests.py`, `long_answer.sh`, `stream_time.py` timing every streamed token,
`summarize.py`; the record is `long/long_answer.txt`).

| | |
|---|---:|
| the first token (the prompt, on a fresh server's empty expert cache) | 65.2 s |
| the 2,047 tokens after it | 2,095 s: **0.977 tokens/s**, 1,024 ms a token |
| by 256 tokens of the answer, tokens/s | 1.08, 1.09, 0.95, 0.95, 0.89, 0.94, 0.94, 1.02 |
| the draft model | 2,048 tokens in 687 steps (2.98 a step), 1,360 of 1,385 guesses stood |
| the drives during the answer | 13.5 GB/s together, `C:` busy 94% of the time, `D:` 97% |

A step carried three tokens and took 3.05 s: its four rows ask for the experts of four tokens,
which the drives bring at their pace, 13.8 GB a token -- even code the draft model guesses almost
always right. The run was to be the first of eight (code, prose, reasoning and a 16.5k-token
context, each with the draft model and without); the series was cut to this one.

## Not done

- **The draft model's context is not in the photos.** A conversation that comes back from a file,
  or a new one that starts after the system prompt's photo, drafts from the rows its own prompt
  step computes (the last 2,047) and from none before them; a turn that restores its own
  conversation's photo keeps what the ring still holds.
- **A large step's MoE waits for the bus.** 16,511 rows take 190 s where the drives need 105: the
  routed experts' weights go up (1.34 TB) while their outputs come down (0.33 TB), 60 s of copies
  in an 83 s phase. Their outputs summed on the card would need 3.8 GB of it at 16,384 rows (2-2.5
  GB are free); faster kernels gain little more here.
- The numbers of `docs/kimi_k3-4090.md` are Kimi K3's; its shards are no longer on this
  machine, so nothing here compares the two models on the same day.
