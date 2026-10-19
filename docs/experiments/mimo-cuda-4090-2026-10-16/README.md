# MiMo-V2.6-Pro-MOPD on an RTX 4090

Measured on 2026-10-16, native Windows, using the complete 70-layer checkpoint.
The recommended launcher is `c/tools/start_mimo_cuda_4090.ps1`: CUDA dense budget
20 GiB, 2 GiB GPU reserve, uploaded host copies released, expert cache 104 per
MoE layer, 16 CPU workers, 16 readers, and a 16384-position context allocation.

## Results

The final confirmation series gives **7.7% more tokens/s for the entire request**
and **11.5% more decode tokens/s** than the installed Vulkan profile. The first
token is slower, so this is useful for continued generation rather than a
universal latency improvement.

| Final series | Overall tok/s | Decode tok/s | First token, s | Peak process RSS, GB |
|---|---:|---:|---:|---:|
| Installed Vulkan, chain32, cache88, 24 workers | 1.005 | 1.137 | 8.289 | 139.02 |
| CUDA, released host copies, cache104, **16 workers** | **1.083** | **1.268** | 9.426 | 151.55 |
| CUDA, released host copies, cache104, 24 workers | 1.056 | 1.228 | 9.255 | 151.56 |

The selected CUDA arm's raw overall samples are **1.0880, 1.0827, 1.0725**;
Vulkan's are **1.0020, 1.0055, 1.0089**. Means of the whole-device GPU utilization
samples, including warmup, are **8.42% versus 4.85%**. They also include desktop
and other processes; actual engine CUDA work is established by its own counters:
**28160 successful matmuls, 110 resident matrices, no compute/scratch fallback**
over each four-request run.

Earlier controlled measurements establish the contributions separately:

| Earlier series | Overall tok/s | Decode tok/s |
|---|---:|---:|
| Same CUDA-capable binary, CUDA disabled, cache88, 24 workers | 0.810 | 0.916 |
| CUDA enabled, retained host copies, cache88, 24 workers | 1.067 | 1.242 |
| CUDA enabled, retained host copies, cache88, 16 workers | 1.099 | 1.290 |
| CUDA, released host copies, cache88, 24 workers | 1.066 | 1.241 |
| CUDA, released host copies, cache104, 24 workers | 1.119 | 1.301 |
| Installed Vulkan in that earlier series | 1.059 | 1.199 |

The controlled CUDA on/off gain is **31.8%**. The cache-size change alone gives
**5.0%** in its controlled pair. Absolute speeds drifted between the earlier and
confirmation series, which is why the final comparison uses the latter rather
than combining the fastest CUDA result with a later Vulkan result. All arms,
including slower configurations, remain in this directory.

## What changed

The CUDA implementation places native FP8/BF16 matrices individually, prioritizing
all QKV projections, the language-model head and layer-0 MLP, then fitting as
many output projections as the budget allows. On this checkpoint 110 matrices
occupy 19.995 GiB. FP8 scales use MiMo's deinterleaved per-row layout. Activations
remain float32; neither the checkpoint nor its experts are requantized.

The original Windows MiMo binary uses Vulkan and has no MiMo CUDA calls. Its
32-layer chain places 67 dense matrices and also uses a small GPU expert tier.
CUDA dense offload alone was close to that Vulkan profile in overall throughput.
Releasing the CUDA matrices' host copies frees 19.995 GiB for the expert cache:
raising it from 88 to 104 reduces disk reads without raising the process peak
much above the retained-copy cap88 CUDA configuration.

CPU routing, native MXFP4 expert computation, attention/KV and vision are still
CPU operations. The GPU is not continuously saturated: expert reads and CPU
expert work dominate the remaining time. An unavailable CUDA device or failed
operation falls back to CPU. Released weights are restored from the original
checkpoint through their original loader, including QKV deinterleaving.
The injected failure tests exercise this restoration; a physical driver reset
or GPU removal was not performed.

## Measurement protocol and limits

One persistent engine per arm, one full warmup request excluded, then three
identical greedy requests of exactly 64 completion tokens. Prefix reuse is
disabled so each request processes the same 17-token prompt. The prompt is
"Explain briefly what binary search is." with the native ChatML non-thinking
cue. `samples.json` records each response's timing and SHA-256; `reply-*.txt`
contains the actual output. Early EOS is rejected. Every measured reply across
all arms is byte-identical.

Overall throughput includes prefill; decode throughput is `(tokens-1)` divided
by elapsed time from the first DATA frame to DONE. Loading and warmup are
excluded. This is a short-prompt decode benchmark, not a 16K-prompt or maximum-
context claim. No SSD or OS cache purge was performed. The expert cache remains
alive between requests. Arms run sequentially. The first set of runs and the
later confirmation pair were separated in time; the confirmation pair is the
primary comparison to the installed Vulkan executable.

`gpu.csv` samples NVIDIA telemetry every 500 ms, including warmup. GPU utilization
in the summary is the mean over that whole interval, not a kernel occupancy
measurement. A truncated final CSV row caused by stopping the monitor is
excluded and counted. Process RSS is the engine's peak working-set statistic in
decimal GB, not total system commit or GiB. System RAM headroom is also needed
for GPU fallback, other applications and larger prompts.

The controlled manifests isolate CUDA on/off on one binary, and cache88/cache104
on one released-host binary. The 16-worker experiment is retained, including its
time-separated regression for cache104. Repeating 24 workers in the final series
favored 16 workers, which the launcher now uses; the older negative result is
retained in `threads-history.json`. These are measurements
on one host and one fixed prompt, not a claim of a universal speedup.

## Correctness and checks

- GPU scalar-reference tests cover float32, BF16, per-row FP8 and int8, irregular
  dimensions, production hidden size 6144, bit-identical batch/chunk results,
  invalid inputs, and shutdown/reinitialization: passed.
- The independent vendor-derived MiMo tiny oracle passes greedy, teacher forcing,
  image cases, cache eviction, prompt-chunk invariance and corruption checks, both
  with retained and released CUDA host copies.
- The 14 MiMo serving/prefix tests pass with released copies, including snapshots,
  sliding-window restoration and image requests.
- The 6 opt-in fallback tests pass: missing Windows DLL, invalid device, zero
  budget, partial placement, released copies, and mid-step CPU recovery at the
  thirteenth eligible matmul in native and float32 modes.
- Full-model CPU/CUDA comparison, cap104: all 17 prompt positions have identical
  argmax; maximum absolute logit difference `5.340576171875e-05`, maximum KL
  `1.4426740834108535e-11`; the 32-token greedy reply is byte-identical. Both runs
  use all 70 layers and the same native checkpoint. Raw logit arrays remain in
  the local `c/build/mimo-cuda/release-correctness` directory, with hashes in the
  published result. They can be regenerated with `check_mimo_cuda.py`.

The general repository suite is **not fully green on this Windows environment**.
`make check` ran 17 C binaries; the unchanged DeepSeek-V4 framing test failed when
MSVCRT `tmpfile()` returned NULL. A separate general Python run reported 2416 tests,
8 failures, 10 errors, 319 skips. Its failure areas include an installed `tests`
package shadowing repository imports, non-Windows compiler assumptions, missing
optional `safetensors`, symlink privileges, `os.pread`, CRLF fixture hashes,
path separators and CPU-thread mocks. The corresponding test/implementation
paths are unchanged by this branch. These failures are not counted as passes;
this work's numerical and fallback gates are the targeted checks above.

The final dependency-free CPU build and its complete MiMo tiny oracle also pass;
the combined CUDA/Vulkan code path compiles in a syntax check. The new benchmark
evidence guard and registry agreement checks pass. The actual PowerShell
launcher was exercised through `/v1/models` and `/v1/chat/completions` on localhost
port 18089, including stale truncation/fault/context variables in its inherited
environment. The final 16-worker profile returned `Четыре` to the Russian
arithmetic prompt (25 prompt tokens, 4 completion tokens). The test server was
stopped afterward; `api-smoke-16t/result.json` records the response and binary hashes.

## Reproduce

Build using `c/build_mimo_cuda.bat` (sm_89, CUDA 13.1/MSVC 14.44) and
`mingw32-make -C c mimo CUDA_DLL=1 ARCH=native VK=0` (GCC 14.2, MSVCRT).
Source commits are recorded beside each arm. The native dense and released-host
engine binaries have different hashes; all measured copies and the installed
release were retained locally. The original `C:/colibri` installation and model
weights were not overwritten.

`result.json.commit` identifies the measurement-harness checkout. It is not a
claim that the installed prebuilt Vulkan executable was compiled from that
commit. Its SHA-256 identifies the baseline binary. The CUDA engine source is
unchanged between code commit `17ae9476` and the final profile/report commit;
the measured released-host engine SHA-256 is
`41c8b2aa61cb167506449c5b9eb0f0ce4989a7f21776e6c1cc79ba2a3403b70a`.

```powershell
python c/tools/benchmark_mimo_cuda.py --engine C:/colibri/mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --backend vulkan --cap 88 --threads 24 --read-threads 16 --context 16384 --tokens 64 --runs 3 --output c/build/mimo-cuda/new-vulkan
python c/tools/benchmark_mimo_cuda.py --engine c/mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --backend cuda --release-host --cap 104 --threads 16 --read-threads 16 --context 16384 --tokens 64 --runs 3 --output c/build/mimo-cuda/new-cuda
python c/tools/check_mimo_cuda.py --engine c/mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --release-host --cap 104 --output c/build/mimo-cuda/new-correctness
python c/experiment_manifest.py docs/experiments/mimo-cuda-4090-2026-10-16/cuda-vs-cpu.json docs/experiments/mimo-cuda-4090-2026-10-16/expert-cache.json docs/experiments/mimo-cuda-4090-2026-10-16/threads.json
```

See `summary.json` for hardware, model metadata hashes, full numeric values and
GPU telemetry summaries; each arm contains raw samples, replies, engine output
and executable hashes. `sha256.json` inventories the published evidence. Model
config/index hashes identify the measured layout; the 563-GB weight set itself
was not rehashed or redistributed.
