# MiMo-V2.6-Pro on an RTX 4090: lossless BF16 packing and a pooled expert cache

Measured in the night of 16 to 17 October 2026 on the machine and with the profile
of the [second streaming round](../mimo-stream2-4090-2026-10-16/README.md) (CUDA
dense budget 20 GiB, host copies released, expert cache 104 per MoE layer, 16 CPU
workers, 16384-position context), the same benchmark command and the same eight
varied prompts. Two commits: `71f5a0c4` packs the BF16 matrices without loss,
`0369c3b1` pools the expert cache across the MoE layers by default.

## Results

Three series, each run back to back. Series A, the packing (commit `71f5a0c4`;
the engine before it is `10313e4f`):

| Arm | Overall tok/s | Decode tok/s | First token, s |
|---|---:|---:|---:|
| before the packing (`10313e4f`) | 1.791 | 2.032 | 4.73 |
| the CPU's packing only (`MIMO_CUDA_BF16_PACK=0`) | 1.824 | 2.087 | 4.90 |
| the GPU's packing only (`MIMO_BF16_PACK=0`) | 1.918 | 2.194 | 4.65 |
| **both (the defaults of `71f5a0c4`)** | **1.936** | **2.226** | 4.76 |

**+8.1% overall, +9.6% decode.** The GPU's packing is the most of it; the CPU's
adds 0.9% (decode 1.5%) once the GPU holds 51 o_proj, under the manifest's 1% bar
(`cpu-pack.json`: no-change), and 1.8% when it packs all 34 (the second row). The
GPU packing places 51 o_proj on the card instead of 36 (52 packed matrices with
the head: 8.49 GiB for 11.31 GiB of bf16);
the CPU then computes 19 o_proj a token, packed, 2.87 GB instead of 3.83 GB of
bf16 to read. Raw samples: 1.7867, 1.7932, 1.7914 before; 1.9284, 1.9367, 1.9357
after.

Series C, the pool (commit `0369c3b1`, its defaults against `MIMO_CACHE_GLOBAL=0`):

| Arm | Overall tok/s | Decode tok/s | First token, s |
|---|---:|---:|---:|
| a fixed cap per layer (`MIMO_CACHE_GLOBAL=0`) | 1.940 | 2.230 | 4.73 |
| **one pool (the default of `0369c3b1`)** | **1.990** | **2.321** | 5.01 |

**+2.6% overall, +4.1% decode**; the first token 0.3 s later. Raw samples: 1.9266,
1.9400, 1.9417 without the pool; 1.9854, 1.9898, 1.9929 with it. Against the
release engine of round 2 (1.159 tok/s, decode 1.359) the defaults of `0369c3b1`
are +72% overall and +71% decode on this benchmark.

Eight different prompts (`varied-prompts/`, one cold engine per run):

| Run | Overall tok/s | Mean decode tok/s | Mean first token, s | Replies = before |
|---|---:|---:|---:|---:|
| B: before the packing (`10313e4f`) | 1.651 | 1.979 | 6.44 | -- |
| B: the packing, the pool off | 1.771 | 2.168 | 6.55 | 8 / 8 |
| B: the packing and the pool | 1.797 | 2.239 | 6.89 | 8 / 8 |
| C: `0369c3b1`, the pool off | 1.778 | 2.174 | 6.49 | 8 / 8 |
| **C: `0369c3b1`, defaults** | **1.809** | **2.253** | 6.84 | **8 / 8** |

The packing is +7.3% over eight different prompts (series B) and the pool +1.5%
(B) to +1.7% (C) more; every reply is the bytes of the engine before them, which
are the release engine's (round 2: 1.061 tok/s over the same prompts). The pool
costs a new prompt 0.3 s of first token.

## What changed

**BF16 packed without loss** (`71f5a0c4`). A trained matrix's exponents crowd
into a few values: a window of 15 consecutive exponents per row holds all but
0.018% of the weights of Pro's o_proj and head (1,110,340 of 6.07 billion). Each
such matrix is stored as two planes, the sign/mantissa byte of every weight and a
4-bit code, its exponent minus the start of the row's window, or 15: the exponent
is the next byte of the row's exception list. 12 bits a weight instead of 16, the
same layout on both sides:

- the GPU kernel loads 8 sign/mantissa bytes and 4 code bytes a lane, rebuilds 8
  bf16 in registers (a byte add of the row's base, two masks and two byte permutes
  per four weights; a warp that meets an exception counts its lanes' exceptions
  with a prefix sum) and runs the bf16 kernel's FMAs in its order;
- the CPU kernel rebuilds each group of 8 into the words `bf16_decode8` loads and
  runs `matmul_bf16`'s loops (the decode step's and the 64-row tiles of a prompt
  block) in their order.

Both give the plain kernels' bytes. The budget counts a packed matrix's exact
size (`mimo_cuda_bytes`): a first version that took the drop of the card's free
memory as the size placed 42 to 46 o_proj instead of 51 -- the free figure WDDM
reports moved by more than the allocations.

**One expert pool** (`0369c3b1`). The MoE layers' caches share their 104 x 69
buffers: in a decode step a layer that misses takes the least recently used
buffer of any layer (not pinned, not loading, not a prefetch still waiting), and a
layer may grow to twice the cap. The layers whose routing spreads wider hold more,
and a decode step reads about 10% fewer experts on demand (2374 against 2651 a
request). A first version took buffers the same way in prompt blocks too, and a
new prompt's experts pushed out what the next layers held: its first token came
0.4 s later on the benchmark, 0.7 s on the varied prompts. A prompt block now
reuses the layer's own least recently used buffer once the layer holds the cap.

## What did not help

- Prefetches read in 8 or 4 MB pieces that make way for a demand between pieces
  (a reader reads a waiting demand itself and pauses while another reads one): the
  generation's stalls stayed where they were (5.62 s and 5.42 s a request against
  5.50 s, one build); not kept.
- Predicting prompt blocks too (`MIMO_PREFETCH_ROWS=64`): 1650 instead of 2885
  demand reads in the benchmark's prompt, but more reads in all and a later first
  token: 1.895 against 1.928, 1.748 against 1.771 on the varied prompts.

## Correctness

- **The 70-layer CPU logits are still the release engine's**, SHA-256
  `cebbf3a8...` (`correctness/`), with all 71 BF16 matrices packed (11.98 GB for
  15.97 GB): the CPU run packs every o_proj and the head.
- The CUDA path against the CPU path, for both commits: 17/17 argmax, maximum
  absolute logit difference 4.6e-5, KL at most 1.5e-11, the 32-token greedy reply
  the same bytes.
- `tests/test_mimo_cuda.cu` compares the packed kernel with the bf16 one, byte
  for byte, on trained-like matrices, matrices whose weights are mostly exceptions
  and degenerate rows (zeros of both signs, one value, 15 and 16 exponents,
  subnormals), 256 to 16384 columns, 1 to 17 rows, and checks the packed size.
- The tiny fixture's prompt logits with the packing are the release engine's and
  `10313e4f`'s bytes, native and f32 dense, prompt blocks of 1, 7, 64 and 512 rows.
  The tiny oracle, the 14 serve/prefix tests and the 6 CUDA fallback tests pass,
  with the pool on and off.

## Files

`summary.json`, the benchmark arms (raw samples, replies, GPU telemetry, engine
lines), one-variable manifests (`pack-vs-stream.json`, `gpu-pack.json`,
`cpu-pack.json`, `pool.json`; `c/experiment_manifest.py`), `varied-prompts/`
(prompts, driver, every run's per-prompt results and replies, `summary.json`),
`correctness/` (both commits), `sha256.json`. The arms named `hybrid*` ran a build
of the pool before it became the default (`MIMO_CACHE_GLOBAL` unset was 0 there).
Series A ran a build of the tree committed as `71f5a0c4`: a rebuild from the commit
differs from it in the four bytes of the PE link time and checksum, two comments
having moved in between.

## Reproduce

```powershell
git checkout 0369c3b1
c/build_mimo_cuda.bat
c/build_mimo_cuda.bat test
mingw32-make -C c mimo CUDA_DLL=1 ARCH=native VK=0
python c/tools/benchmark_mimo_cuda.py --engine c/mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --backend cuda --release-host --cap 104 --threads 16 --read-threads 16 --context 16384 --tokens 64 --runs 3 --output c/build/mimo-pack/pool
python c/tools/benchmark_mimo_cuda.py ... --env MIMO_CACHE_GLOBAL=0 --output c/build/mimo-pack/pool-off
python docs/experiments/mimo-pack-4090-2026-10-17/varied-prompts/drive_mimo.py --engine c/mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --cap 104 --tokens 64 --json varied.json docs/experiments/mimo-pack-4090-2026-10-17/varied-prompts/prompts.txt
python c/tools/check_mimo_cuda.py --engine c/mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --release-host --cap 64 --threads 16 --output c/build/mimo-pack/correctness
python c/experiment_manifest.py docs/experiments/mimo-pack-4090-2026-10-17/pack-vs-stream.json docs/experiments/mimo-pack-4090-2026-10-17/gpu-pack.json docs/experiments/mimo-pack-4090-2026-10-17/cpu-pack.json docs/experiments/mimo-pack-4090-2026-10-17/pool.json
```
