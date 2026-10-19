# MiMo-V2.6-Pro on an RTX 4090: streaming, round 2

Measured on 2026-10-16 on the machine of the
[first streaming experiment](../mimo-stream-4090-2026-10-16/README.md), native
Windows, the complete 70-layer checkpoint, the same launcher profile (CUDA dense
budget 20 GiB, host copies released, expert cache 104 per MoE layer, 16 CPU
workers, 16384-position context) and the same benchmark command. The engine is
commit `10313e4f`; the four commits since the first experiment (`3e8cf2ae`,
`4ba6a54e`, `2f044413`, `10313e4f`) are below.

## Results

The benchmark (three identical requests after a warmup, as before), run back to
back in one series:

| Arm | Overall tok/s | Decode tok/s | First token, s | Expert RAM hits |
|---|---:|---:|---:|---:|
| release engine (`66d529bd`) | 1.159 | 1.359 | 8.86 | 69.8% |
| **this engine, defaults** | **1.814** | **2.059** | **4.68** | 86.1% |
| this engine, `MIMO_CUDA_OPROJ_INT8=1` (not exact) | 2.168 | 2.527 | 4.58 | 86.1% |

**+56.5% overall and +51.5% decode** against the release engine; the first token
comes in half the time. Raw samples: release 1.1547, 1.1601, 1.1591; this engine
1.8092, 1.8141, 1.8148; int8 o_proj 2.1593, 2.1677, 2.1715 tok/s. **Every
measured reply of every arm is the same bytes** (SHA-256 `0305abcd...`, the reply
of both earlier experiments), the int8 arm's included on this prompt. The first
experiment's engine measured 1.709 in its own series.

**Eight different prompts** (`varied-prompts/`: English and Russian, 18 to 64
tokens each, one cold engine per arm, nothing repeated, so no cache has seen a
prompt before):

| Arm | Overall tok/s | Mean decode tok/s | Mean first token, s | Replies = release |
|---|---:|---:|---:|---:|
| release engine | 1.061 | 1.328 | 12.01 | -- |
| **this engine, defaults** | **1.679** | **2.018** | **6.39** | **8 / 8** |
| this engine, int8 o_proj | 1.976 | 2.456 | 6.25 | 3 / 8 |

**A 2260-token prompt** (`long-prompt/`, the CLI, 8 generated tokens): the
release engine prefills it in **903.9 s** with the profile's 64-row prompt blocks;
this engine in **205.4 s** with its new default of 512 rows (4.4x). A build
with the first experiment's kernels took 498.1 s with 64-row blocks
(`development.json`). Both replies are the same bytes.

Where a request's 35.2 s went with the defaults (from the engine's phase line,
the last measured request): the prompt 4.6 s (1.6 s waiting for the disks), then
63 generated tokens in 30.6 s, about 485 ms a token: 229 ms computing experts, 81
ms waiting for the disks, 151 ms of attention of which 121 ms are the 34 o_proj
the CPU computes (the 36 others are on the GPU), and the rest routing,
prediction and the head. With the int8 o_proj the attention takes 42 ms a token
and the disks wait 96 ms: the faster layers leave the readers less time.

## What changed

**The prompt's CPU kernels take several rows at once, with the same bits**
(`4ba6a54e`). A prompt block used to cost as many expert reads from RAM as it had
rows: the MXFP4 expert kernel, the bf16 matmuls (o_proj, router) and the
attention scores each processed one row at a time. Now the MXFP4 kernel decodes
each 32-weight group once for up to 8 rows, the bf16 matmul runs 4 rows x 2
outputs in register blocks over 64-row tiles, silu is fused into the up pass
(the gate and up blocks of a wave in one pass, then down), and the attention
scores are computed per key for all the query heads of a group with AVX2. Every
(row, output) sum keeps the one-row loop's order -- GCC compiles `s += q*k` as a
sequential add, which the vector code reproduces -- so a prompt's logits are the
bytes they were. 2260 tokens with 256-row blocks: 427 s before, 249 s after.

**Prompts in blocks of 512 rows by default** (`2f044413`). Each block streams
the union of its rows' experts once, so larger blocks read fewer: with the new
kernels 256 rows took 249 s, 1024 rows 229 s and 2048 rows 275 s (a block's union
no longer fits the cache), and with the 64-row tiles of the next build 512 rows
213 s and 1024 rows 208 s. 512 keeps the scratch modest.

**The CPU o_proj layers are spread through the model** (`10313e4f`,
`MIMO_CUDA_OPROJ_SPREAD`). On a 24-GiB card 36 of the 70 o_proj fit; they used to
be the first 36 layers', so the last 34 layers all had a slow attention. Even
layers' go up first now, so a slow attention comes every other layer and the
readers catch up there: 1.794 -> 1.813 tok/s in one build (`development.json`).
`-DenseGB 21 -ReserveGB 1` places five more and measured 1.851.

**An opt-in int8 o_proj on the GPU** (`3e8cf2ae`, `MIMO_CUDA_OPROJ_INT8=1`, with
`MIMO_CUDA_DENSE_GB=20.2`): every o_proj as int8 with a scale per (row, 128
columns), half the bytes, so all 70 fit and the CPU computes none. **Not exact**
(see [mimo-cuda.md](../../mimo-cuda.md)): 3 of the 8 varied replies are the same
bytes, the others part after a few dozen characters into equivalent wording.

**Prefetch sizes per prediction point** (`3e8cf2ae`, `MIMO_PREFETCH_ATTN`,
`MIMO_PREFETCH_MLP`): measured, the defaults (8 after the attention, none after
the MLP) stay the fastest.

## Correctness

- **The CPU logits on the real model are still unchanged to the bit**: the
  70-layer CPU run of `check_mimo_cuda.py` writes the logits file of both earlier
  experiments, SHA-256 `cebbf3a8...` (`correctness/result.json`).
- The CUDA path against the CPU path: 17/17 argmax, maximum absolute logit
  difference 5.3e-5, maximum KL 1.7e-11, the 32-token greedy reply the same bytes.
- Every exact reply above is the release engine's: the benchmark's 12 (`0305abcd...`),
  the 8 varied prompts', the long prompt's.
- The tiny oracle passes; its logits are byte-identical to the engine's before
  these changes with prompt blocks of 1, 7, 64 and 512 rows. The CUDA unit test
  and the 14 serve/prefix and 6 CUDA fallback tests pass.

## What did not help (`development.json`)

Reads cut into 4 MB pieces (1.483 against 1.792 with whole reads in the same
build; 8 MB pieces with 6 readers a drive 1.762), a second prediction after the
MLP (1.615 alone, 1.682 with 4 after the attention, against 1.710), a prefetch two
layers ahead (1.781 and 1.771 against 1.792), decayed use counts in the cache
(`MIMO_CACHE_DECAY=0.97`: 1.819 against 1.710 here, where the four requests are
one prompt, but no gain on varied prompts).

## Measurement protocol and limits

The benchmark as in the earlier experiments: one persistent engine per arm, one
excluded warmup, three identical greedy 64-token requests, prefix reuse off, the
17-token prompt "Explain briefly what binary search is."; arms run sequentially.
Repeating a prompt favors a cache, hence the varied prompts, which run cold: each
prompt is one request, the first one's time includes the engine's first reads of
everything, and the release engine gets no warmup either. The release binary is
the one both earlier experiments measured (engine SHA-256 `41c8b2aa...`). The long
prompt ran each engine once; the release engine was not measured with 512-row
blocks. One machine.

## Files

`summary.json` (every benchmark arm's medians, profile counters, the engine's
streaming and phase lines, hashes), the arms' raw samples, replies, GPU telemetry
and engine lines, `stream-vs-release.json` (one-variable manifest,
`c/experiment_manifest.py`), `varied-prompts/` (prompts, driver, per-prompt
results with replies, `summary.json`), `long-prompt/` (prompt, command, engine
lines, replies), `correctness/`, `development.json` (the arms and prompt runs
measured on the way, from working-tree states that were not committed),
`sha256.json`.

## Reproduce

```powershell
git checkout 10313e4f
c/build_mimo_cuda.bat
mingw32-make -C c mimo CUDA_DLL=1 ARCH=native VK=0
python c/tools/benchmark_mimo_cuda.py --engine c/mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --backend cuda --release-host --cap 104 --threads 16 --read-threads 16 --context 16384 --tokens 64 --runs 3 --output c/build/mimo-stream2/stream
python c/tools/benchmark_mimo_cuda.py ... --env MIMO_CUDA_OPROJ_INT8=1 --dense-gb 20.2 --output c/build/mimo-stream2/stream-int8
python docs/experiments/mimo-stream2-4090-2026-10-16/varied-prompts/drive_mimo.py --engine c/mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --cap 104 --tokens 64 --json varied.json docs/experiments/mimo-stream2-4090-2026-10-16/varied-prompts/prompts.txt
bash docs/experiments/mimo-stream2-4090-2026-10-16/long-prompt/run_prefill.sh c/mimo.exe stream MIMO_CHUNK=512
python c/tools/check_mimo_cuda.py --engine c/mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --release-host --cap 64 --threads 16 --output c/build/mimo-stream2/correctness
python c/experiment_manifest.py docs/experiments/mimo-stream2-4090-2026-10-16/stream-vs-release.json
```

`run_prefill.sh` reads `prompt.txt` beside it and writes its outputs there; the
measured runs used the same script from a scratch directory.
