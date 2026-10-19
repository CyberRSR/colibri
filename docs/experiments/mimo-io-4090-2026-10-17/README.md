# MiMo-V2.6-Pro on an RTX 4090: reading the experts where the drives have time

Measured on 17 October 2026 on the machine and with the profile of
[the two-draft experiment](../mimo-mtp2-4090-2026-10-17/README.md) (CUDA dense budget
20 GiB, host copies released, expert cache 104 per MoE layer in one pool, 16 CPU workers,
16384-position context, two MTP drafts a step cut at 0.6), the same benchmark command and
the same eight varied prompts. The change is in `c/mimo.c`'s expert streaming: what is
read and when, never a number -- every expert is still computed from the same bytes, in
the same rows and order. Commit `54418f93`.

## Where a decode step waited

A copy of the engine logged every expert read (queued, started, finished, which drive,
used or not) and every MoE step's phases (`analysis/inst.patch`, `MIMO_IOLOG`; the scripts
beside it read the log). The last benchmark request with every change off (development
series 4, `analysis/where-it-waits.txt`): 21.2 s of generation in 27 steps, 2.9 s of it
waiting for experts, 296 GB read.

- **The first half of the layers carries the stall.** Layers 1-32 read 8.4-8.6 experts a
  step for a union of 16-18 (their routing repeats less from token to token) behind a
  1 ms GPU attention, and wait 2.8-3.0 ms a layer a step; the `C:` drive is busy 96-97% of
  that time. Layers 33-69 read 4.0-4.6 for a union of 14.5, every second one computes its
  o_proj on the CPU (2.4 ms attentions on average), and wait 0.4-0.5 ms: `C:` idles 27-30%
  there, `D:` 39-49% (`analysis/layers.txt`). Both drives idle through every MTP drafting,
  0.9 s a request. The eight varied prompts show the same split.
- **The disk is behind where it matters, and the prediction's misses ride along.** Of the
  request's 9343 prefetches 6464 were used -- 3606 of them still loading when their layer
  asked -- and 2879 not: 1162 read already, 723 being read, and **994 still queued** when
  their layer's routing came. A queued one was then read before the next layer's
  prefetches.
- **The drives are at their ceiling.** `analysis/drives.txt` (20 MiB unbuffered reads):
  `C:` gives 10.4 GB/s with 2 readers and no more with 3 to 6, `D:` 6.3 with 2 and 6.55
  with 3; together 16.7, independent. The engine's two readers a drive already take `C:`,
  the drive the steps wait for, to its ceiling.
- **The prediction cannot reach further.** Of the experts a step's union holds
  (`analysis/prediction.txt`): the next layer's router after this layer's attention (the
  prefetcher) predicts 79%; two layers ahead 71%; after this layer's MLP 83%, but with
  ~1 ms left before the demand; the chosen token's embedding, for the next step's layers
  during the MTP drafting, 25%. By rank, a row's first two choices come true 97% of the
  time, its eighth 50-60%.

So the step had two ways to go faster at the same bandwidth: not reading what nobody will
use, and reading the first half's experts less often -- the second half has the time.

## What changed

- **Stale prefetches leave the queue** (`io_cancel`, `MIMO_PREFETCH_CANCEL`, default 1).
  When a layer's routing comes, each predicted expert it did not choose whose read no
  reader has taken yet is taken out of the queue and its slot freed (a read already
  started finishes; its slot is the first to be reused, as before).
- **A prediction's reads go likeliest first.** A row's choice weighs top-k + 2 minus its
  rank instead of one vote, so a row's first choice comes before another row's last; one
  row keeps its rank order. What a step will not ask for is then mostly at the end of the
  queue, where the cancel finds it unread.
- **The layers that wait keep their experts longer** (`MIMO_CACHE_BONUS`, default 4 steps).
  Each MoE layer keeps a running average (1/16 a step) of how long its decode steps wait
  on the disk; the pool's least-recently-used choice sees its buffers as used that many
  steps later times its share of the stall over the average MoE layer's, at most three
  times. No layer range is written down: on this profile the first half earns the bonus.
- `MIMO_IO_DELAY_US` (tests): every read waits that long first, so the tiny fixture's
  queues fill as on a slow disk; the harness checks that prefetches are cancelled and no
  token moves.

## Results

One series, back to back in ABA order (`series.sh`): before (the build of `6efa0fe2`'s
tree, the previous rounds' engine), after (the build of `54418f93`'s), before;
`binaries.sha256`, each in its own folder beside a copy of `coli_cuda_mimo.dll`. First the
eight prompts greedy (`drive_mimo.py`, one cold engine per run, 64 tokens each,
`SEED=20261017`):

| Arm | Overall tok/s | Mean decode tok/s | Requests | Generation (stalled) | Read |
|---|---:|---:|---:|---:|---:|
| before | 2.1116 | 2.7310 | 220.69 s | 166.40 s (25.38) | 2468 GB |
| **after** | **2.2449** | **2.9471** | 207.58 s | 154.00 s (17.20) | 2315 GB |
| before | 2.1175 | 2.7346 | 220.07 s | 166.03 s (26.01) | 2468 GB |

**+6.2% overall, +7.8% mean decode** against the mean of the two before arms; each prompt
decodes faster in the after arm than in either before arm. The stalls fall by a third
(25.7 to 17.2 s), generation minus stalled by 3.7 s (141.0 and 140.0 to 136.8: the drives
write less into the RAM the experts are computed from), 6% fewer bytes read. Every reply
the bytes of the MTP experiment's (`mtp.json`, the release engine's) in all three arms,
the same drafts kept (151 of 162, 95 of 102).

Then the benchmark (three identical greedy requests after a warmup):

| Arm | Overall tok/s | Decode tok/s | Last request: generation (stalled), minus stalled | Read | Cancelled |
|---|---:|---:|---:|---:|---:|
| before | 2.4423 (2.4280-2.4429) | 2.9653 | 21.25 s (2.93), 18.32 s | 295.96 GB | -- |
| **after** | **2.5652 (2.5225-2.5692)** | **3.1553** | 19.94 s (1.81), 18.13 s | 278.37 GB | 633 |
| before | 2.4381 (2.4201-2.4431) | 2.9616 | 21.27 s (2.97), 18.30 s | 295.94 GB | -- |

**+5.1% overall, +6.5% decode**; every reply SHA-256 `0305abcd...` in every arm, the same
drafts (26 steps, 24 of 24 and 12 of 13 kept). The stall falls 2.95 to 1.81 s; the
compute does not move (generation minus stalled 18.32 and 18.30 against 18.13 s), so the
gain is the disk's.

## Correctness

- The 70-layer CPU logits are still the release engine's (SHA-256 `cebbf3a8...`) and the
  CUDA logits the earlier rounds' (`01479691...`), the after binary, `check_mimo_cuda.py
  --cap 64 --threads 16 --release-host` (`correctness/`); 17/17 argmax, the 32-token greedy
  replies the same bytes. Its CPU run cancelled 245 prefetches.
- Every benchmark reply of the series and of the 26 development arms `0305abcd...`; the
  eight greedy replies the recorded ones in all three series arms.
- The tiny suites: `mimo_tiny_harness.py` (CPU and `MIMO_CUDA=1`), with the new slow-read
  check (37 prefetches cancelled unread, the reference's tokens), the 14 prefix/serve
  tests, the 6 CUDA fallback tests (`MIMO_CUDA_TEST=1`), the 2 sampled-draft tests.

## On the way (`development.json`)

Development series of the benchmark, instrumented copies with the IOLOG on (series 3
without). Series 1-3 compared separate builds, which is how the layout effect below
showed; series 4 and 5 each ran one binary with every feature behind a switch, so their
arms differ in nothing but the switch:

- Series 4 (`analysis/v4.patch`), in order: everything off 2.436; cancel 2.510 (+3.0%);
  cancel and order 2.481; cancel, order and bonus (static, layers 1-32 four steps) 2.549;
  with the MTP-window prefetch added 2.545 / 2.546; cancel and bonus 2.544; off 2.440.
  The stall 2.93 -> 2.37 (cancel) -> 1.91 s (with the bonus); generation minus stalled
  18.2-18.3 s in every arm but `c-order` (18.55).
- **The MTP-window prefetch** (dropped): the chosen token's row through layer 0's attention
  before the drafts (its K/V row is written again by the step before anything reads it)
  and layers 1-2 predicted from it: `C:`'s idle time in the drafting fell 0.89 -> 0.64 s,
  the stall 0.05-0.11 s, and the extra attention and predictions cost about as much.
- Series 5 (`analysis/v5.patch`), the bonus: static 1-32 at 4 steps 2.550 at both ends of
  the series; 6, 8: 2.587, 2.582; adaptive 2, 4, 8: 2.533, 2.596, 2.596. Every one leaves
  the stall at 1.8-1.97 s: past four steps the reads move to the second half and its
  stall grows by what the first half's falls (8 steps: 1.06 + 0.87 s against 1.38 +
  0.52). The spread between arms is the machine's (generation minus stalled 17.8-18.2 s).
  The adaptive rule matches the static one without naming layers, so it is the default.
- **Replay** (`analysis/sim.py`, `analysis/sims.txt`): the logged routing and candidates
  through the pooled cache reproduce the engine's read counts exactly (11659 against
  11660 a request). A bonus for buffers used twice or more (segmented LRU) reads 0.6%
  less at best; prefetching only ranks 0-6 7% less, but as demand reads, slower in the
  steps that wait.
- **Code layout**: a build that differed from series 1's other binary by the ordering's few
  lines (in `mimo_prefetch`) ran its CPU phases ~3.5% slower in every arm, with
  the line's switch on or off -- the router's step 0.29 against 0.21 ms, attention +9%;
  `moe` byte for byte the same code 112 bytes further on. Hence one binary a series, and
  generation minus stalled beside every tok/s. The committed build shows no such loss
  (18.13 against 18.32 / 18.30 s).

## Measurement protocol and limits

As in the earlier experiments: one persistent engine per benchmark arm, a warmup, three
identical 64-token requests, prefix reuse off; the varied prompts one cold engine per run,
each prompt one request. One machine, one series; the machine drifts by a few percent
from series to series (generation minus stalled 17.8-18.9 s in comparable development
arms), so arms are compared within one series only. Throughout, a stale `timeout.exe 60
t_dyn.exe` of another session's test (started 13:13) spun about a third of a core in the
kernel; every arm ran with it. The development arms ran with the IOLOG on (an atomic
count and a 32-byte record an event), so their tok/s are not the series'. The bonus's
numbers are of this profile: on a card or a budget where the second half's attentions are
as short as the first's, the drives have no time there to take reads from the first, and
the bonus can only move the stall. Repeating one prompt favors the cache; the varied
prompts gain more than the benchmark.

## Files

`summary.json` (the series' arms: samples, the last request's phases and streaming
counters, the varied runs' totals), `benchmark/` and `varied-prompts/` (the arms' results
and engine lines), `correctness/` (`logits.sha256`: the two logits files),
`development.json` (every development arm), `series.sh`, `dev.sh`, `binaries.sha256`,
`analysis/` (the IOLOG patch and the development variants' patches against `6efa0fe2`'s
`c/mimo.c`, the scripts and their outputs), `sha256.json`.

## Reproduce

```bash
git checkout 54418f93
mingw32-make -C c mimo CUDA_DLL=1 ARCH=native VK=0
python c/tools/benchmark_mimo_cuda.py --engine c/mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --backend cuda --release-host --cap 104 --threads 16 --read-threads 16 --context 16384 --tokens 64 --runs 3 --output c/build/io/after
python c/tools/benchmark_mimo_cuda.py ... --env MIMO_PREFETCH_CANCEL=0 --env MIMO_CACHE_BONUS=0 --output c/build/io/off
python docs/experiments/mimo-mtp-4090-2026-10-17/varied-prompts/drive_mimo.py --engine C:/path/to/c/mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --cap 104 --tokens 64 --env SEED=20261017 --json greedy.json docs/experiments/mimo-mtp-4090-2026-10-17/varied-prompts/prompts.txt
git apply docs/experiments/mimo-io-4090-2026-10-17/analysis/inst.patch   # on 6efa0fe2: the IOLOG copy
```

`MIMO_PREFETCH_CANCEL=0 MIMO_CACHE_BONUS=0` turns two of the three off; the reads' order
has no switch (the before binary has the old one). `drive_mimo.py` needs the engine's
absolute path (it starts it as given).
