# MiMo-V2.6-Pro on an RTX 4090: the MTP drafts' CPU tail over the workers

Measured on 17 October 2026 on the machine and with the profile of
[the two-draft experiment](../mimo-mtp2-4090-2026-10-17/README.md) (CUDA dense budget
20 GiB, host copies released, expert cache 104 per MoE layer in one pool, 16 CPU workers,
16384-position context, two MTP drafts a step cut at 0.6), the same benchmark command and
the same eight varied prompts. Commit `15446df8` spreads two loops of `c/mimo.c` over
the workers; the trunk's numbers do not move, and the drafts' probabilities kept their
bits in every draft checked:

- `mtp_run`'s draft probability, `1 / sum exp(logit - max)` over the 152576-entry
  vocabulary after each block's head: 152576 `expf` one after another, 4.4 ms a draft.
  Now in chunks of 512 over the workers, each chunk's sum in double in order and the
  chunks' sums added in order, so the result depends neither on the thread count nor on
  which worker took a chunk. The argmax before it stays one scalar pass (80 us).
- `dense_mlp`'s `silu(gate) * up`, an `expf` per element (the trunk's dense layer 0 and
  every MTP block call, `mtp_warm`'s prompt rows too): each element alone, the same
  `expf`.

Why no more than that: MinGW's libgomp waits at a barrier on a semaphore (its posix
configuration: `bar.o` calls `gomp_sem_wait`, `sem_wait`), so at 16 threads a barrier
costs about 75 us and a fork and join about 50 us, whatever `OMP_WAIT_POLICY` says
(`micro/barrier.txt`) -- a threaded argmax, two barriers more, was slower than the
scalar pass; and a vectorized `expf` would not give the runtime's bits (the trunk's must
not move). In `micro/variants.c` (16 threads, head-like logits, 1.4 ms idle before each
call as the GPU head leaves the workers; a dynamic schedule absorbs the slow fork; every
p and every silu bit the same as the serial loops'):

| Loop | Before | After |
|---|---:|---:|
| a draft's argmax and sum | 4.41 ms | 0.52 ms |
| silu, 1 row of 16384 | 0.51 ms | 0.12 ms |
| silu, 3 rows | 1.55 ms | 0.24 ms |
| silu, 50 rows | 26.4 ms | 2.0 ms |

## Results

One series, run back to back in ABA order: before (`07d76d68`'s tree, built the same
day), after (`15446df8`'s), before; the result files name `07d76d68` as the checkout
for both, the engines' SHA-256 tell them apart. The benchmark (three identical greedy
requests after a warmup; the last request's phases):

| Arm | Overall tok/s | Decode tok/s | Generation | Drafting |
|---|---:|---:|---:|---:|
| before | 2.447 (2.441-2.450) | 2.971 | 21.27 s | 1.15 s |
| **after** | **2.485** (2.479-2.489) | **3.027** | 20.78 s | 0.81 s |
| before | 2.453 (2.442-2.461) | 2.983 | 21.10 s | 1.10 s |

**+1.4% overall, +1.7% decode** against the mean of the two before arms; every request
of the after arm is faster than every request of both before arms. The drafting of a
request falls from 1.10-1.15 s to 0.81 s, a block call (two a step) about 15 ms instead
of 21. The same drafts in every arm (26 steps, 24 of 24 and 12 of 13 kept), every reply
the release engine's bytes (SHA-256 `0305abcd...`).

Eight different prompts, greedy (`drive_mimo.py`, one engine per run, 64 tokens each):

| Arm | Overall tok/s | Mean decode tok/s | Drafting | Drafts kept (first, second) |
|---|---:|---:|---:|---|
| before | 2.1213 | 2.7379 | 9.57 s | 151 of 162, 95 of 102 |
| **after** | **2.1462** | **2.7834** | 7.04 s | 151 of 162, 95 of 102 |
| before | 2.1217 | 2.7387 | 9.57 s | 151 of 162, 95 of 102 |

**+1.2% overall, +1.6% mean decode**, each of the eight prompts decoding faster in the
after arm than in either before arm; all eight replies the recorded ones (the MTP
experiment's, the release engine's) in every arm.

## Correctness

- The trunk's numbers: with the after binary the 70-layer CPU logits are still SHA-256
  `cebbf3a8...` and the CUDA logits `01479691...`, the 32-token greedy replies the same
  bytes (`correctness/result.json`, `c/tools/check_mimo_cuda.py`).
- The bits of p: an instrumented copy of the after engine (`check/make_check.py`; not
  committed) computed every draft's probability the old way beside the new one and
  compared the float bits, and the threaded silu against the serial loop, on the
  traces' workloads below: 815 greedy and 808 sampled drafts, p's bits the same in every
  one; in more than 2000 silu calls (over 127 M elements) no element differs.
- The traces (`traces/`; `MIMO_MTP=3 MIMO_MTP_MIN_P=0`, the eight prompts at 128
  tokens, as [the two-draft experiment's](../mimo-mtp2-4090-2026-10-17/traces) were
  written): greedy, all 273 steps of its `g.trace` again -- positions, drafts, kept,
  every printed probability -- so the cut at 0.4 or 0.6 decides every step the same
  (the closest product came 1.4e-3 from 0.6, far beyond a last bit). Sampled
  (temperature 0.7, top_p 0.9): the before and the after engine wrote the same trace,
  byte for byte (271 steps), and the same eight replies; neither is `t07.trace`, which
  an earlier build wrote (its lines have no verified count): from the second step (the
  same printed probabilities, 1 kept there against 3) the draws part. In the new
  sampled trace the closest product came 1.7e-4 from 0.6. `traces/compare.txt`.
- The tiny suites pass with the after binary, CPU and `MIMO_CUDA=1` harness output byte
  for byte the before binary's; 14 prefix/serve tests, 6 CUDA fallback tests
  (`MIMO_CUDA_TEST=1`), the sampled-draft test.

## Files

`summary.json` (every arm, raw samples, the last request's phases, the checks),
`series.sh` (the series as it ran), `benchmark/` (per arm `result.json` with the
samples, `engine.txt`), `varied-prompts/` (per run results, replies and engine lines),
`correctness/`, `traces/` (the after engine's traces, the check binary's lines,
`compare_traces.py` and `compare.txt`), `check/make_check.py`, `micro/` (`variants.c`,
`barrier.c` and what they printed), `sha256.json`.

## Reproduce

```powershell
mingw32-make -C c mimo CUDA_DLL=1 ARCH=native VK=0
python c/tools/benchmark_mimo_cuda.py --engine c/mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --backend cuda --release-host --cap 104 --threads 16 --read-threads 16 --context 16384 --tokens 64 --runs 3 --output c/build/mtp-tail/b-after
python docs/experiments/mimo-mtp-4090-2026-10-17/varied-prompts/drive_mimo.py --engine C:/path/to/c/mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --cap 104 --tokens 64 --env SEED=20261017 --json g-after.json docs/experiments/mimo-mtp-4090-2026-10-17/varied-prompts/prompts.txt
python docs/experiments/mimo-mtp-4090-2026-10-17/varied-prompts/drive_mimo.py --engine C:/path/to/c/mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --cap 104 --tokens 128 --env SEED=20261017 --env MIMO_MTP=3 --env MIMO_MTP_MIN_P=0 --env MIMO_MTP_TRACE=C:/path/to/g.trace --json trace-g.json docs/experiments/mimo-mtp-4090-2026-10-17/varied-prompts/prompts.txt
python c/tools/check_mimo_cuda.py --engine c/mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --output c/build/mtp-tail/correctness --cap 64 --threads 16 --release-host
```

The sampled traces add `--temperature 0.7 --top-p 0.9`. The before arm is the same
commands on a build of `07d76d68`.
