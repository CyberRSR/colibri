# MiMo-V2.6-Pro on an RTX 4090: prompts on the GPU, round 2

Measured on 20 October 2026 on the machine of the earlier MiMo experiments (Ryzen 9
7950X3D, 192 GB DDR5, RTX 4090, the model's shards on two NVMe volumes) with the RTX 4090
profile, as in [the first round](../mimo-gpu-prompt-4090-2026-10-20/README.md):
`long_prompt.py` (its copy here, unchanged) sends that round's two prompts
(`../mimo-gpu-prompt-4090-2026-10-20/prompts/`) one after the other over one engine's SERVE
pipe (thinking off, 16 greedy tokens, prefix reuse off) and times their first tokens. In
each series the two engines alternate: r2's (the code of release `mimo-pro-4090-r2`,
`b5b45f2c`), this round's, r2's, this round's.

## Results

| Prompt | Engine | Series 1 | Series 3 |
|---|---|---:|---:|
| 4310 tokens | r2 | 69.13 s, 69.14 s | 70.88 s, 69.32 s |
| 4310 tokens | **this round** | **47.26 s, 47.11 s** | **47.33 s, 47.07 s** |
| 24,957 tokens | r2 | 243.91 s, 244.71 s | 250.31 s, 247.71 s |
| 24,957 tokens | **this round** | **92.51 s, 92.18 s** | **92.14 s, 92.33 s** |

24,957 tokens: 2.64x faster in series 1, 2.70x in series 3; 4310 tokens: 1.47x and 1.48x.
Same arithmetic in f32 as before (the products of the checkpoint's exact weights, the
sums in the device's orders): see [Correctness](#correctness).

Series 3 ran the shipping engine (`mimo.exe` 8c0b21c0, `coli_cuda_mimo.dll` c1019362),
the card at 2.91 - 2.94 GHz throughout (`series3/gpu.csv`). Series 1 ran it before its
last change, which only touches decode steps ([Decode](#decode)). Series 2, between them,
ran the shipping engine while the machine was slower for some minutes: r2's engine took
315.1 s for the 24,957-token prompt once (its GPU matmuls 86.7 s there against 67.8 -
69.9 in its other runs), this round's 103.8 and 96.7 s (`summary.json` keeps it).

Where the 24,957-token prompt's time goes now (the prompt alone, run `k12` of
`development.json`, 93.9 s; the prompt path's code is the shipping engine's): the
attention's chain on the device 38.4 s (the rows up 1.7 s, qkv and RoPE 16.9, the attention
8.1, o_proj 10.0, the outputs down 1.7), the experts' loop 46.7 s (20.4 of it waiting for a
staging buffer while the device computes), the routers 4.6 s; the CPU's per-row loops (4.3
s) and the rows' weighted sums (8.9 s) run beside the device. In r2 the qkv took 57.8 s and
o_proj 23.7 s, their rows going down to the host and up again around the attention, the
attention's scores and values about 40 s, the experts 87 s. The 4310-token prompt is now
bound by its experts' reads: its experts' loop took 39.2 of its 47 s and read about 600 GB
from the drives, 15 GB/s, where `C:` alone gives about 10.5 GB/s for the two thirds of the
shards it holds and `D:` 6.5 ([mimo-io](../mimo-io-4090-2026-10-17/README.md)).

## What changed (`development.json`)

Each step on the 24,957-token prompt alone, one run each (the runs vary by a few
percent with the card's clocks):

| Step | First token |
|---|---:|
| r2 | 235.5 s |
| one tiled kernel for every prompt matrix (FP8, bf16, packed bf16, the experts' MXFP4): 128 x 128 tiles, 8 x 8 a thread, for r2's 64 x 64; a stream for each expert slot | 215.1 s |
| a sub-block's attention kept on the device: its rows go up once, the qkv, RoPE (the host's tables), the attention and o_proj run there, and only its keys and values (for the caches) and o_proj's output come down | 148.0 s |
| the experts' outputs stored by the down kernel straight into pinned host memory: the card has one copy engine and takes copies in the order they come, so each output copy queued behind its expert's kernels held up every later expert's upload | 125.7 s |
| flash attention: tiles of 128 queries (the heads that share a key head) by 128 keys at absolute positions, an online softmax with the sink | 123.0 s |
| experts the cache does not hold read from the disks straight into the pinned staging buffers, not into a cache slot and copied again | 117.9 s |
| the routers from the rows already on the device; the block's rows in pinned memory | 113.2 s |
| a full layer's keys and values kept on the device for the block's next sub-block | 111.8 s |
| the CPU's per-row loops (weighted sums, norms, residuals) beside the device's sub-blocks | 97.5 s |
| one device arena for the attention, the experts and the matrices; 12 expert slots | 97.9 s |
| the scaled formats' sums by scale group (the correctness fix below) | 93.9 s |

The kernels at MiMo's shapes, on the device alone (`kernels/bench-grouped-1.txt`: 1024
rows, an expert's 520, 25,024 keys; r2's kernels against this round's): the qkv 30.5 ->
36.5 TFLOP/s, o_proj 30.8 -> 35.7, an expert's gate and up 19.4 -> 23.4 and down 22.6 ->
33.7, a full layer's attention 18.8 -> 39.0, this round's within 2.8e-7 of a double
reference. The last step costs them a little, within the runs' spread: with the two
builds alternating the qkv runs at 36.5 and 34.8 TFLOP/s summed by group against 39.7 and
38.2 in one chain, the experts' gate and up at 23.4 against 25.6 and 26.7.

Dead ends (`development.json`): tiles of 128 x 256 (no faster: 39.7 against 40.1 TFLOP/s
for the qkv, one block an SM); four 64-output groups a block (no gain); the experts'
outputs copied down by the copy engine, polled, or by a narrow store kernel (1.8 - 2.5 ms
an expert against 1.65 for the down kernel's own stores, `c/tools/bench_mimo_moe.cu`); a
free expert slot chosen for each expert instead of the slots in turn (slower).

## Correctness

- Against the CPU path on the first round's 658-token prompt in 256-row blocks
  (`correctness/`, `check_mimo_cuda.py --chunk 256`, the shipping engine): the same argmax
  at all 658 positions, max |logit difference| 0.0084, max KL 8.6e-7, the 32-token greedy
  reply the same bytes. (r2: 0.0085 and 8.7e-7.)
- The same prompt's GPU run in blocks of 330 rows, in 330-row blocks of 110-row
  sub-blocks (the pipelined loop, a full layer's keys and values kept on the device between
  sub-blocks), and in 330-row blocks with earlier positions' keys sent up in chunks of 128
  (`MIMO_CUDA_ATTN_MB=1`): the logits of every position the check's GPU run's, byte for
  byte (`correctness/probes.json`, `correctness/probe.py`).
- The 4310-token prompt's reply is r2's in every run. The 24,957-token prompt's leaves
  r2's at its tenth token, a tie (`topk/`, greedy with `MIMO_MTP=0`, each step's top
  three): after "On an RTX 4090" r2's engine scores "," 33.979 and " (" 33.883, this
  round's " (" 34.116 and "," 34.007 -- 0.1 apart either way, where every other step's
  first two are 0.3 to 16 apart. The same quantities summed in other orders break the tie
  the other way.
- On the way the check failed (`k10`: the argmax the CPU's at 622 of 658 positions, max KL
  7.5). Rerunning its GPU side with each path turned off in turn
  (`development.json`, `correctness_bisect`) left the tiled matrices: with only those on
  the GPU, 651 of 658 (max KL 0.36 from position 206); with the decode kernels instead,
  658 (max KL 1.8e-8). The new kernel summed FP8's and MXFP4's products in one FMA chain
  with the scales folded into the weights; the CPU and r2's kernels sum each scale group
  (FP8's 128 columns, MXFP4's 32) apart and add the sum times its scale. One chain is ten
  times less accurate (2.5e-7 of the products' magnitude against 2.7e-8 at the qkv's
  shape, `kernels/`), and a long prompt's massive-activation channels felt it. Summed by
  group again: 658 of 658, max KL 8.6e-7 (`k11`), and the experts' down kernel gives r2's
  bytes.
- `tests/test_mimo_cuda.cu` now pins the tiled kernels' order: every output must be the
  bytes of its sums in order, FP8 by scale group, bf16 one chain. The folded kernel passed
  the old bound (1e-6 of the magnitude) and fails this (`kernels/unit-tests-folded.txt`);
  the shipping one passes all 56 checks (`kernels/unit-tests.txt`). The tiny fixture with
  every GPU prompt path forced on (pieces of 3 expert rows, 50 KB of matrix scratch, also
  in 16-row sub-blocks): all checks pass; its logits the same bytes in one chunk of keys
  and in chunks of 128, in blocks of 256 rows and of 7, in one attention piece and in
  pieces of 16 rows pipelined or not, the argmax the CPU's at 256 of 256.

## Decode

The decode benchmark (`benchmark/`, 64 tokens after a 17-token prompt, which the GPU
prompt path does not take; a warmup and three runs an engine), r2's engine and the
shipping one alternating: 2.607 and 2.619 tok/s against 2.621 and 2.617 (decode 3.22 -
3.24 both), every reply `0305abcd...` as in r2 and colibri 2.0.

Before its last change this round's engine measured 2.578 against r2's 2.623
(`benchmark/before-*`): every matrix call asked the driver twice whether the caller's
buffers were pinned, to send a prompt block's straight, and a decode step's calls paid it
too (their wall time over the benchmark's 14,208 calls: FP8 4.466 -> 4.573 s, packed bf16
2.137 -> 2.278 s). Only a prompt block's calls ask now; the check's GPU logits are the same
bytes as before the change.

## The release's build (`release-build/`)

Release `mimo-pro-4090-r3` was built from the merge into main (`c087ce36`), which also
carries main's own changes to other engines and `compat.h`: its `mimo.exe` (49f87449) and
`coli_cuda_mimo.dll` (6dc4f791) are other bytes than the measured ones. Run the same way,
they gave the measured build's CPU and GPU logits and replies on the correctness prompt
byte for byte (`check/same-bytes.json`), the two prompts' first tokens in 47.30 and 92.22
s, and 2.645 tok/s for the decode benchmark (decode 3.28, every reply `0305abcd...`).

## Files

`summary.json` (every run of the three series), `series1/` to `series3/` (each run's
`results.json` and `engine.txt`, the card's telemetry), `development.json` (the steps,
the dead ends, the decode fix and the bisect), `correctness/`, `topk/` (the 24,957-token
prompt's top three a step for both engines, and the scratch patch that printed them),
`kernels/` (`c/tools/bench_mimo_cuda.cu` with this round's backend and with the folded
kernel, two runs each, and the unit tests with both), `benchmark/`, `release-build/`,
`long_prompt.py`, `sha256.json`.

## Reproduce

```powershell
c/build_mimo_cuda.bat
mingw32-make -C c mimo CUDA_DLL=1 ARCH=native VK=0
python docs/experiments/mimo-gpu-prompt2-4090-2026-10-20/long_prompt.py --engine C:\colibri\dev\c\mimo.exe --prompt docs/experiments/mimo-gpu-prompt-4090-2026-10-20/prompts/p4k.txt --prompt docs/experiments/mimo-gpu-prompt-4090-2026-10-20/prompts/p25k.txt --tokens 16 --env MIMO_CHUNK= --out c/build/gpu-prompt2
python c/tools/check_mimo_cuda.py --engine C:\colibri\dev\c\mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --release-host --prompt-file docs/experiments/mimo-gpu-prompt-4090-2026-10-20/correctness/prompt.txt --tokens 32 --chunk 256 --output c/build/gpu-prompt2-check
python c/tools/benchmark_mimo_cuda.py --engine C:\colibri\dev\c\mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --backend cuda --release-host --cap 104 --threads 16 --read-threads 16 --context 16384 --tokens 64 --runs 3 --output c/build/gpu-prompt2-bench
c/build_mimo_cuda.bat test
```

The engine's path must be absolute (Windows does not start a relative path written with
forward slashes).
