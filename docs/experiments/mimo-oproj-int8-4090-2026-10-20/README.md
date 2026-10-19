# MiMo-V2.6-Pro on an RTX 4090: every o_proj as int8 (not exact)

Measured on 20 October 2026 on the machine and profile of the
[MTP experiments](../mimo-mtp-4090-2026-10-17/README.md) (16 CPU workers, 104 experts a
layer in one pool, host copies released, the benchmark request of 64 greedy tokens
after a warmup), with the engine of `136b9f20` at a 16384-token context. The trial arm
sets `MIMO_CUDA_OPROJ_INT8=1` and `MIMO_CUDA_DENSE_GB=20.2`: all 70 o_proj go to the
GPU as int8 with a float scale per (row, 128 columns), 144 matrices in 19.58 GiB, and
the CPU computes none of them; the exact arm keeps the 51 packed bf16 o_proj the GPU
holds and the CPU's 19.

| Arm | Overall tok/s | Decode tok/s |
|---|---:|---:|
| int8 o_proj | 2.702 | 3.361 |
| exact (the default) | 2.538 | 3.133 |
| int8 o_proj again | 2.671 | 3.330 |

+5.8% overall and +6.8% decode, one ABA series, the same binary. The benchmark's 64
tokens happened to be the exact reply's bytes in every arm (SHA-256 `0305abcd...`),
but the int8 path is **not exact**: on Pro's own activations a projection is off by
0.5% (median, 1.35% at most) against bf16, and through 70 layers and the routers'
top-8 cuts a 451-token prompt read teacher-forced gave the same argmax at 398 of 451
positions (88%), mean KL 0.13 (the exact CUDA path: 451/451, 7.6e-12); on eight varied
64-token answers three were the same bytes and the others parted after 22 to 145
characters into equivalent wording ([docs/mimo-cuda.md](../../mimo-cuda.md)).

Files: each arm's `result.json` (medians, configuration, hashes), `samples.json` (every
request), `reply.txt` and the engine's `[CUDA]`/`[mimo]` lines.
