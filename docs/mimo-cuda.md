# MiMo native dense CUDA offload

`MIMO_CUDA=1` moves MiMo's frequently used dense matrices to CUDA VRAM. It
supports Flash and Pro, including the released Pro-MOPD split-shard checkpoint.
FP8 e4m3 and BF16 stay in their original formats; activations stay float32.
There is no conversion of the checkpoint and no additional quantization.

The QKV matrices are placed first, then the language-model head, the dense MLP
and as many attention output projections as the budget permits. In particular,
a 24-GiB card does not have to fit complete layers. FP8 scales use the engine's
deinterleaved **per-row** layout, not the generic CUDA backend's block grid.
Pinned activation staging is reused between calls. The same kernel reduction
is used for decode and prefill so chunk sizes do not change the arithmetic.

The trunk's shapes (FP8 rows a multiple of 512 bytes, BF16 rows a multiple of 256
elements: every MiMo matrix) run on wide kernels: one warp per output row, each
lane loading 16 weight bytes at a time, e4m3 widened by the hardware conversion
(exact), and a tile of up to 8 activation rows sharing every weight it decodes,
so a prompt chunk streams a matrix once per 8 rows instead of once per row. A
decode step uses a one-row tile (more warps in flight). Each row's sums follow
the element index only, so a chunk of S rows gives the bytes of S one-row calls
(`tests/test_mimo_cuda.cu` checks it on both kernels). On the RTX 4090 a 27136 x
6144 FP8 qkv takes 0.22 ms per decode call instead of 0.63, and a 17-row prompt
chunk 1.7 ms instead of 9.2.

In a decode step routing, the streamed MXFP4 experts, attention/KV state and
vision remain on the CPU; a prompt's blocks run their experts, attention and
matrices on the GPU ([Prompts on the GPU](#prompts-on-the-gpu)).
CPU weight copies are retained by default. `MIMO_CUDA_RELEASE_HOST=1` releases
successfully uploaded host matrices so RAM can hold more experts. If CUDA
cannot compute a matrix, the engine reads it back through its original loader,
including QKV deinterleaving and the selected dense format. Recovered host
weights stay in RAM. Leave enough host-memory headroom for this recovery when
choosing an expert-cache size. CUDA and Vulkan placement are
mutually exclusive when CUDA successfully places a matrix. A missing DLL or
device leaves the existing CPU/Vulkan path available. Allocation failures leave
unplaced matrices on CPU; a compute failure disables further CUDA computation.
GPU and CPU reductions have different rounding: validate tokens/logits on the
intended workload, rather than assuming bit identity across backends.

## Build on Windows

Requirements: MinGW-w64 GCC with OpenMP and GNU make, CUDA Toolkit, Visual
Studio x64 C++ build tools. From the repository's `c` directory:

```powershell
.\build_mimo_cuda.bat
mingw32-make mimo CUDA_DLL=1 ARCH=native VK=0
.\build_mimo_cuda.bat test
python tools/make_mimo_tiny.py --output mimo_tiny --force --vision
$env:MIMO_CUDA = '1'
python tests/mimo_tiny_harness.py --binary ./mimo.exe --fixture ./mimo_tiny
python tests/test_mimo_prefix_serve.py
python tests/test_omp_spin.py
python tests/test_exp_f.py
$env:MIMO_CUDA_TEST = '1'
python tests/test_mimo_cuda_fallback.py
```

The DLL script discovers Visual Studio, uses `CUDA_PATH` or `nvcc` on PATH,
and defaults to `CUDA_ARCH=sm_89` (RTX 4090). Override it for other GPUs.
`coli_cuda_mimo.dll` is separate from the other engines' `coli_cuda.dll` and
links the CUDA runtime statically. Keep it beside `mimo.exe`. No change to
the installed release or system PATH is required. The default CPU build
still needs no CUDA SDK or runtime.

Linux builds use `make -C c mimo CUDA=1 CUDA_ARCH=sm_89` and
`make -C c cuda-mimo-test CUDA_ARCH=sm_89`. The native Windows RTX 4090 is the
hardware validation target; Linux is not covered by those local results.

## Run on the measured computer

```powershell
.\c\tools\start_mimo_cuda_4090.ps1
```

The example profile expects the model in `D:\AI\Models\MiMo-V2.6-Pro-MOPD`
and additional shards in `C:\AI\MiMoPro-Shards`; both paths are parameters.
It serves localhost port 18080. Existing launchers and model files are untouched.
`-DenseGB 21 -ReserveGB 1` places more o_proj (before the packing: five more, 115
matrices, 20.9 GiB, +2%: 1.81 to 1.85 tok/s), leaving about 1 GiB of the card free
while it runs; the default keeps 2 GiB for other programs.
The expert-cache default is 104 entries per MoE layer, with 16 CPU workers, two
expert readers per drive and the next layer's prefetch (`-IoPerDrive`, `-Prefetch`;
see [expert streaming](mimo.md#expert-streaming)). The profile enables host-copy release, freeing about 20 GiB of RAM
on this checkpoint. Its nominal RAM budget is 174 GiB; the cap is explicitly
bounded at 104 so recovery has headroom on the validated machine. Plain
`MIMO_CUDA=1` still retains host copies unless release is explicitly enabled.
The 192-GiB profile is not appropriate for smaller-RAM machines.
The launcher clears stale layer-truncation, trace and fault-injection settings
and sets the requested context explicitly: by default the checkpoint's full
1,048,576 positions (`-Context`), whose attention caches take memory only as a
conversation grows (102 KB a token; past about 300k tokens beside this cache the
expert pool gives memory back, see [`MIMO_KV_STEP`](mimo.md#environment)), and
replies of up to 65,536 tokens, reasoning included, when a request asks for no
fewer (`-MaxOutput`, the gateway's `--ngen`).

| Variable | Default | Meaning |
|---|---:|---|
| `MIMO_CUDA` | `COLI_CUDA`, else 0 | Enable this path; explicit 0 disables it |
| `MIMO_CUDA_DENSE_GB` | 20 | Maximum resident dense GiB; 0 disables placement |
| `MIMO_CUDA_RESERVE_GB` | 2 | Free VRAM left for scratch and other applications |
| `MIMO_CUDA_RELEASE_HOST` | 0 | Release uploaded host matrices; reload from checkpoint for CPU fallback |
| `MIMO_CUDA_BF16_PACK` | 1 | BF16 matrices (the head, the o_proj) go up packed without loss, 12 bits a weight -- the same results, see below; 0: plain bf16 |
| `MIMO_CUDA_OPROJ_INT8` | 0 | 1: every o_proj on the GPU as int8 per 128 columns -- **not exact**, see below |
| `MIMO_CUDA_OPROJ_SPREAD` | 1 | The o_proj that fit go up even layers first, so the CPU's are every other layer rather than the second half (the disks catch up during a slow attention: +1%); 0: in layer order |
| `COLI_GPU` | 0 | CUDA device ordinal |

The actual budget is the smaller of the requested budget and free VRAM minus
the reserve. Check `[CUDA] mimo placed` at startup and `[CUDA] mimo dense` at
exit: the latter reports actual matmul calls, not just device discovery.
The host-copy report counts released and restored matrices. For recovery tests,
`MIMO_CUDA_FAULT=N` switches to CPU at the Nth eligible matmul; it is never set
by the production profile or benchmark.
Large prefill scratch allocations can fall back to CPU without changing output
ownership. A 16K context allocation is not a performance claim at 16K tokens.

## Prompts on the GPU

A decode step needs 8 of a layer's 384 experts, so it is bound by reading them, and
the CPU computes them. A prompt is the opposite: a block of a few hundred rows calls
nearly every expert of every layer, and on the CPU (about 0.7 TFLOP/s for MXFP4) a
prompt went in at about 11 tokens a second -- a 20,000-token agent prompt took the
better part of an hour, most of it CPU arithmetic while the disks and the GPU waited.
Its full-attention layers grew with the square of the prompt on top. With `MIMO_CUDA`
a prompt block now runs on the GPU:

- **experts** (`MIMO_CUDA_MOE`, steps of 64 rows or more): an expert the cache holds is
  copied into one of 32 pinned staging buffers over the workers; one it does not hold is
  read from disk straight into one (`MIMO_CUDA_MOE_DIRECT`, `MIMO_CUDA_MOE_STAGES`), the
  cache taking only those it can without letting go of one the step uses. Each goes up
  into one of 12 device slots while the device computes the ones before, and the device
  computes gate, up, silu and down over the rows that chose it (f32 products of the exact
  e2m1 values, each 32-column group's sum times its scale, as `matmul_mxfp4` groups
  them). The down kernel stores its rows' outputs straight into the (row, choice) pairs
  the CPU path fills, in pinned host memory -- the card has one copy engine, and output
  copies queued behind their kernels held up every later expert's upload -- and each row
  sums its experts in its routing order as before, so a row's bytes depend neither on the
  rows beside it nor on when its experts arrived.
- **blocks of up to 32,768 rows** (`MIMO_CUDA_CHUNK`, bounded by the card's free
  memory at the first prompt: 4 x (hidden + experts) bytes a row beside 640 MB, and
  `MIMO_CUDA_SPARE_MB` more): a block reads each expert once for all its rows, so a
  25,000-token prompt reads the 531 GB of experts once, not 49 times as 512-row blocks
  did. Its attention goes in sub-blocks of 1024 rows (`MIMO_ATTN_ROWS`), every matrix
  call in pieces whose scratch stays under 192 MB (`MIMO_CUDA_SCRATCH_MB`), and the
  attention, the experts and the matrices take turns in one device arena: a WDDM card
  that runs out of memory pages it and crawls.
- **attention** (`MIMO_CUDA_ATTN`, blocks of 32 rows or more): a sub-block's normed rows
  go up once and stay on the device from the qkv product through RoPE (the host's
  tables), the attention and o_proj (`MIMO_CUDA_ATTN_DEV`); only its keys and values,
  for the caches, and o_proj's output come down. The attention is a flash kernel in f32:
  tiles of 128 queries (the heads that share a key head) by 128 keys at absolute
  positions, an online softmax with the sink. Earlier blocks' keys and values go up in
  chunks of `MIMO_CUDA_ATTN_MB` (256); a full layer keeps a block's on the device for its
  next sub-block.
- **matrices** of a prompt block (64 rows or more, `MIMO_CUDA_TILED_ROWS`): one tiled
  kernel for FP8 (the qkv, the dense MLP), bf16 (the routers), packed bf16 (the o_proj)
  and the experts' MXFP4, 128 x 128 tiles, 8 x 8 a thread, out of shared memory instead
  of the decode kernels' one warp a row. The scaled formats sum each scale group apart
  (FP8's 128 columns, MXFP4's 32) and add the sum times its scale, as the CPU does: one
  FMA chain with the scales folded into the weights was ten times less accurate, and
  the argmax left the CPU's at 7 of 658 positions. The o_proj the GPU does not hold go
  up for the block in their packed form (`MIMO_CUDA_PACKED_ROWS`).
- **the routers** of those MoE steps (`MIMO_CUDA_ROUTER`), from the rows already up for
  the experts.
- **the CPU's per-row work beside the device's** (`MIMO_PIPE`): while the device computes
  a sub-block's attention, the CPU readies the next one's rows (the step before's
  weighted sums, the first norm) and finishes the one before (its residual, the second
  norm), in each row's own order. The block's residual and normed rows sit in pinned
  memory (`MIMO_CUDA_PIN`).

Not the CPU's bytes: the same f32 quantities summed in other orders, as the CUDA
matmuls already were; a prompt block's rows get the same bytes whatever block or
sub-block they are in. A step smaller than these thresholds takes the decode path, and a
prompt has such steps: a block's last attention piece can be a few rows, and a prompt
resumed from a photo prefills only its tail. So a prompt's rows take the kernels of the
prompt's length (`MIMO_PROMPT_KERNELS`, default 1): every row of a prompt of 64 tokens or
more takes the prompt kernels wherever it is cut, and a turn resumed from a photo is the
bytes of the same prompt computed cold ([Turn photos](mimo.md#turn-photos-agent-clients-that-drop-the-reasoning));
the head still takes a prompt's last row alone, as before. For a short tail that can cost
time or save it: the GPU's experts move every expert the rows route to over PCIe, where
the CPU reads it in place, and the decode path takes the rows' attention on the CPU, which
grows with the context (35 rows after 4318 tokens: 11.6 s on the prompt kernels against
8.1 on the decode path; about 20 rows after 26,700: 7-10 s against 10-14). A live prefix's
tail (a client that sent the reasoning back) is never the cold prompt's bytes and keeps its
steps' kernels, as before. A decode step (one to three rows) keeps the CPU path
([Decode at long contexts](#decode-at-long-contexts)). Measured on
the RTX 4090 machine, a 4310-token prompt: 413.9 s with the CPU's prompts, 68.6 s in r2
([the first round](experiments/mimo-gpu-prompt-4090-2026-10-20/README.md)), 47.1 s now;
a 24,957-token prompt: about an hour, 244.8 s in r2, 92.2 s now
([the second](experiments/mimo-gpu-prompt2-4090-2026-10-20/README.md)). Against the CPU
path on a 658-token prompt: the same argmax at every position, max KL 8.6e-7, the same
greedy reply, and the same logits byte for byte in blocks of 256 or 330 rows, in
sub-blocks of 110 and with the keys sent up in chunks.

| Variable | Default | Meaning |
|---|---:|---|
| `MIMO_CUDA_MOE` | 1 | A prompt block's experts on the GPU; 0: the CPU's |
| `MIMO_CUDA_MOE_ROWS` | 64 | The fewest rows a MoE step needs to go to the GPU |
| `MIMO_CUDA_CHUNK` | 32768 | Prompt rows a block holds with the GPU's experts (lower when the card has less free memory: `MIMO_CUDA_SPARE_MB`, 0, keeps more for other programs); `MIMO_CHUNK` set wins |
| `MIMO_CUDA_ROUTER` | 1 | The routers of the GPU's MoE steps on the GPU too |
| `MIMO_CUDA_MOE_DIRECT` | 1 | Experts the cache does not hold read from disk straight into staging; 0: through the cache's slots |
| `MIMO_CUDA_MOE_STAGES` | 32 | The pinned staging buffers of the GPU's experts |
| `MIMO_CUDA_ATTN` | 1 | A prompt block's attention on the GPU; 0: the CPU's |
| `MIMO_CUDA_ATTN_DEV` | 1 | A sub-block's rows kept on the device from the qkv to o_proj; 0: they go down for RoPE and up again |
| `MIMO_CUDA_ATTN_ROWS` | 32 | The fewest rows an attention needs to go to the GPU |
| `MIMO_CUDA_ATTN_MB` | 256 | Earlier positions' keys and values sent up at once (more go in chunks) |
| `MIMO_ATTN_ROWS` | 1024 | A larger block's attention goes in sub-blocks of this many rows |
| `MIMO_PIPE` | 1 | The CPU's per-row work beside the device's sub-blocks; 0: one after the other |
| `MIMO_CUDA_PIN` | 1 | A prompt block's rows in pinned memory |
| `MIMO_CUDA_TILED_ROWS` | 64 | The fewest rows a matrix call needs for the tiled kernels |
| `MIMO_PROMPT_KERNELS` | 1 | A prompt's rows (from nothing or from an exact photo) choose their kernels by the prompt's length, not their step's (a smaller matrix call goes up padded to the tiled kernels' rows); 0: by the step's rows, as a live prefix's tail and a read-out always do |
| `MIMO_CUDA_PACKED_ROWS` | 64 | The fewest rows for an o_proj the GPU does not hold to go up for the call |
| `MIMO_CUDA_SCRATCH_MB` | 192 | The scratch of a matrix call's piece |
| `MIMO_CUDA_MOE_RCAP` | 1024 | Rows of an expert computed at once (tests set it low) |

## Decode at long contexts

A decode step's rows -- the next token and the MTP drafts it verifies, one to three --
keep the CPU path, and each full-attention layer reads its whole cache: on Pro 102 KB a
position over the 10 full layers, 2.6 GB a step at 25,000 positions, f32 in host RAM (a
device copy would not fit beside the 20-GiB dense budget). r3 took that attention as (row,
key/value head) tasks, 8 to 24 for 16 workers, each reading its values once for each of
its 16 query heads, the exps one x87 call at a time: about 400 ms of a token after a
24,957-token prompt, more as a conversation grows. Now a block of up to
`MIMO_ATTN_KEY_ROWS` (8) rows over `MIMO_ATTN_KEYS` (128) keys or more goes by keys
(`attend_keys`): the scores in chunks of keys over every worker, the softmax eight heads at
a time (`exp_f8`, the x87 `exp_f`'s bits for every float, 17 times faster), the values by
32 columns of a key head for all its heads and rows, each value row read once a step.
Every number is the one the loop made, summed in the loop's order, so the logits and the
replies are the same bytes (`tests/test_mimo_attention.c` compares the two). Measured on
the RTX 4090 machine, 128 greedy tokens after a 24,957-token prompt: 90.7 - 92.9 s with
r3's engine, 54.6 - 55.0 s now (1.37 - 1.40 -> 2.31 - 2.33 tok/s; the attention 388 - 404
-> 103 ms a token); after a 4310-token prompt 48.5 - 49.0 -> 45.5 - 45.8 s
([the measurement](experiments/mimo-long-decode-4090-2026-10-20/README.md)).

## BF16 packed without loss (`MIMO_CUDA_BF16_PACK`, on by default)

The head and the o_proj are BF16, and a trained matrix uses few of the exponents a
bf16 can hold: on Pro, a window of 15 consecutive exponents per row holds all but
0.018% of the weights. A BF16 matrix whose rows are a multiple of 256 therefore
goes up as two planes: each weight's sign/mantissa byte, and a 4-bit code, its
exponent minus the start of its row's window -- or 15, which says the exponent is
the next byte of the row's exception list. 12 bits a weight instead of 16. The
kernel rebuilds each weight's bf16 in registers (a byte add, two masks and a byte
permute per four weights; an exception costs a warp a prefix count) and sums
exactly as the bf16 kernel does, so the results are the same bytes:
`tests/test_mimo_cuda.cu` compares the two kernels on trained-like matrices, on
matrices whose weights are mostly exceptions and on degenerate rows (zeros of both
signs, one value, 15 and 16 exponents, subnormals), for 1 to 17 rows. Packing is
done once at start-up on the host, in parallel: the engine starts in 14.5 s instead
of 12.1.

In the 20-GiB budget the head and 51 o_proj fit instead of the head and 36 (52
packed matrices, 8.49 GiB for 11.31 GiB of bf16), so the CPU computes 19 o_proj a
token instead of 34. The CPU packs its own the same way (`MIMO_BF16_PACK`,
[mimo.md](mimo.md#environment)). Measured with the profile above, in one series:
1.791 tok/s for the engine before them, 1.824 with the CPU's packing alone, 1.918
with the GPU's alone, 1.936 with both (+8.1%; decode 2.032 to 2.226) -- every reply
the same bytes ([the packing experiment](experiments/mimo-pack-4090-2026-10-17/README.md),
which also measures the pooled expert cache of [mimo.md](mimo.md#environment): 1.990
tok/s with both, 1.809 over eight different prompts).

## The faster, inexact o_proj (`MIMO_CUDA_OPROJ_INT8=1`)

On a 24-GiB card 51 of Pro's 70 o_proj fit beside the qkv and the head when packed
(36 as plain bf16); the CPU computes the others every token. With
`MIMO_CUDA_OPROJ_INT8=1` each o_proj goes up as int8 with a float scale per (row,
128 columns), half its bf16 bytes, and all 70 fit (20.01 GiB in all: give the budget
`MIMO_CUDA_DENSE_GB=20.2`). The CPU computes none, and the RAM they leave is the
experts'. It is **not exact**: on Pro's own activations a projection is off by 0.5%
(median, 1.35% at most) against bf16; through 70 layers and the routers' top-8 cuts
that becomes, on a 451-token prompt read teacher-forced, the same argmax at 398 of
451 positions (88%), mean KL 0.13 (the exact CUDA path: 451/451, 7.6e-12). On eight
varied 64-token answers, three were the same bytes and the others parted after 22 to
145 characters into equivalent wording ("in the country" / "at the time"). The host
copies stay bf16, so a CPU that has to take over computes the released bytes.

Measured with the profile above before the packing: 1.71 to 2.08 tok/s for the
benchmark request (2.41 decode), 1.58 to 1.89 tok/s over eight different prompts;
in the second streaming round 2.168 against 1.814. Use it when speed matters more
than the bits.

## Reproduce the benchmark

```powershell
python c/tools/benchmark_mimo_cuda.py --engine c/mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --backend cuda --output c/build/mimo-cuda/my-cuda-run
```

Use `--backend cpu` with the same binary for a causal comparison, or
`--backend vulkan --engine C:/colibri/mimo.exe` for the installed release.
Run one arm at a time. Each arm starts a persistent engine, excludes one full
warmup request, then measures three identical 64-token greedy requests with
prefix reuse disabled. It saves raw replies, profile counters, GPU telemetry,
binary hashes and timings. Overall throughput includes prefill; decode speed
uses `(completion_tokens - 1) / (last_token_time - first_token_time)`.
Early EOS is rejected so a shorter answer cannot masquerade as an improvement.

For a full-model correctness comparison (numpy required):

```powershell
python c/tools/check_mimo_cuda.py --engine c/mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --output c/build/mimo-cuda/my-correctness-run
```

It compares all vocabulary logits at every prompt position, requires the same
argmax, bounds KL divergence, and compares the generated response bytes. Both
backends use the entire model and the same checkpoint, context and precision.
`--prompt-file` supplies another fully rendered chat prompt.

See the [RTX 4090 experiment](experiments/mimo-cuda-4090-2026-10-16/README.md)
for raw samples, actual GPU work, numerical checks, rejected tuning and the
limitations of the general Windows test suite.

The [streaming experiment](experiments/mimo-stream-4090-2026-10-16/README.md)
measures the same profile with the expert streaming and the wide kernels:
1.156 to 1.709 tok/s for a whole request, the CPU logits unchanged to the bit.
[Its second round](experiments/mimo-stream2-4090-2026-10-16/README.md) adds the
CPU's prompt kernels, 512-row prompt blocks and the spread o_proj: 1.159 to 1.814
tok/s for the benchmark, 1.061 to 1.679 over eight different prompts, a 2260-token
prompt 4.4x faster, every reply the release engine's bytes.
[The MTP experiment](experiments/mimo-mtp-4090-2026-10-17/README.md) adds the
drafts of the checkpoint's MTP layer, verified two rows a step: 2.206 tok/s for the
benchmark (1.999 without), 1.941 over the eight prompts sampled at temperature 0.7
(1.770), the greedy replies the same bytes.
[Its second round](experiments/mimo-mtp2-4090-2026-10-17/README.md) drafts with two
MTP layers and verifies the drafts they are sure of: 2.414 tok/s for the benchmark
against 2.258 with one draft a step in the same series, 2.071 and 1.977 over the eight
prompts greedy and sampled (2.002 and 1.888), every greedy reply unchanged.
[A third](experiments/mimo-mtp-tail-4090-2026-10-17/README.md) spreads the drafts'
softmax and the dense MLP's silu over the CPU workers: 2.485 against 2.450 tok/s for
the benchmark and 2.146 against 2.122 over the eight prompts greedy in one series, the
logits and every reply unchanged.
[A fourth](experiments/mimo-sample-4090-2026-10-17/README.md) sorts only the entries a
sampled draw's nucleus can hold: 2.122 against 2.036 tok/s over the eight prompts
sampled at temperature 0.7 in one series, the same draws and replies; greedy unchanged.
[A fifth](experiments/mimo-io-4090-2026-10-17/README.md) drops the prefetches a step did
not ask for while they are still queued and lets the layers that wait on the disk keep
their experts longer: 2.565 against 2.440 tok/s for the benchmark and 2.245 against
2.115 over the eight prompts greedy in one series, the logits and every reply unchanged.
[A sixth](experiments/mimo-omp-spin-4090-2026-10-17/README.md) runs the CPU's OpenMP
regions on a team that spins between them instead of MinGW libgomp's, which sleeps on
semaphores: the compute 7% shorter, most of it then waiting on the disk; 2.614 against
2.594 tok/s for the benchmark and 2.261 against 2.222 over the eight prompts greedy in
one series, the logits and every reply unchanged.
[A seventh](experiments/mimo-layout-4090-2026-10-17/README.md) finds why a change
elsewhere could make the CPU phases 3.5% slower: MinGW's `exp`, behind every `expf`,
switches the x87 control word twice a call, and at two of the 64 places the linker can put
it that costs 5.7 times the call. `exp_f.h` computes the same bits without the switch:
with the library at such a place 2.607 against 2.582 and 2.569 tok/s for the benchmark in
one series; where it sits now, no difference that shows (2.612 against 2.606 and 2.611,
2.251 against 2.244 and 2.251 over the eight prompts greedy); the logits and every reply
unchanged.
