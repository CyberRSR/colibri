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

## Not done

- **A large step's MoE waits for the bus.** 16,511 rows take 190 s where the drives need 105: the
  routed experts' weights go up (1.34 TB) while their outputs come down (0.33 TB), 60 s of copies
  in an 83 s phase. Their outputs summed on the card would need 3.8 GB of it at 16,384 rows (2-2.5
  GB are free); faster kernels gain little more here.
- The numbers of `docs/kimi_k3-4090.md` are Kimi K3's; its shards are no longer on this
  machine, so nothing here compares the two models on the same day.
