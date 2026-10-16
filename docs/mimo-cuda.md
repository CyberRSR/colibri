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

Routing, streamed MXFP4 experts, attention/KV state and vision remain on CPU.
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
`-DenseGB 21 -ReserveGB 1` places five more o_proj (115 matrices, 20.9 GiB) and
measured +2% (1.81 to 1.85 tok/s), leaving about 1 GiB of the card free while it
runs; the default keeps 2 GiB for other programs.
The expert-cache default is 104 entries per MoE layer, with 16 CPU workers, two
expert readers per drive and the next layer's prefetch (`-IoPerDrive`, `-Prefetch`;
see [expert streaming](mimo.md#expert-streaming)). The profile enables host-copy release, freeing about 20 GiB of RAM
on this checkpoint. Its nominal RAM budget is 174 GiB; the cap is explicitly
bounded at 104 so recovery has headroom on the validated machine. Plain
`MIMO_CUDA=1` still retains host copies unless release is explicitly enabled.
The 192-GiB profile is not appropriate for smaller-RAM machines.
The launcher clears stale layer-truncation, trace and fault-injection settings
and sets the requested context explicitly.

| Variable | Default | Meaning |
|---|---:|---|
| `MIMO_CUDA` | `COLI_CUDA`, else 0 | Enable this path; explicit 0 disables it |
| `MIMO_CUDA_DENSE_GB` | 20 | Maximum resident dense GiB; 0 disables placement |
| `MIMO_CUDA_RESERVE_GB` | 2 | Free VRAM left for scratch and other applications |
| `MIMO_CUDA_RELEASE_HOST` | 0 | Release uploaded host matrices; reload from checkpoint for CPU fallback |
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

## The faster, inexact o_proj (`MIMO_CUDA_OPROJ_INT8=1`)

On a 24-GiB card 36 of Pro's 70 o_proj fit beside the qkv and the head; the CPU
computes the other 34 every token, 6.8 GB of bf16 from RAM, about 120 ms. With
`MIMO_CUDA_OPROJ_INT8=1` each o_proj goes up as int8 with a float scale per (row,
128 columns), half its bytes, and all 70 fit (20.01 GiB in all: give the budget
`MIMO_CUDA_DENSE_GB=20.2`). The CPU computes none, and the RAM they leave is the
experts'. It is **not exact**: on Pro's own activations a projection is off by 0.5%
(median, 1.35% at most) against bf16; through 70 layers and the routers' top-8 cuts
that becomes, on a 451-token prompt read teacher-forced, the same argmax at 398 of
451 positions (88%), mean KL 0.13 (the exact CUDA path: 451/451, 7.6e-12). On eight
varied 64-token answers, three were the same bytes and the others parted after 22 to
145 characters into equivalent wording ("in the country" / "at the time"). The host
copies stay bf16, so a CPU that has to take over computes the released bytes.

Measured with the profile above: 1.71 to 2.08 tok/s for the benchmark request (2.41
decode), 1.58 to 1.89 tok/s over eight different prompts. Use it when speed matters
more than the bits.

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
