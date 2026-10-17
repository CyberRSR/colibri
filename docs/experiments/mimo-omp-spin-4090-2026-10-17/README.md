# MiMo-V2.6-Pro on an RTX 4090: OpenMP regions on a team that spins

Measured on 17 October 2026 on the machine and with the profile of
[the two-draft experiment](../mimo-mtp2-4090-2026-10-17/README.md) (CUDA dense budget
20 GiB, host copies released, expert cache 104 per MoE layer in one pool, 16 CPU workers,
16384-position context, two MTP drafts a step cut at 0.6), the same benchmark command and
the same eight varied prompts, on top of [the I/O round](../mimo-io-4090-2026-10-17/README.md).
Commit `ca05d0e4` links `c/omp_spin.c` into Windows builds: it changes which threads run
the engine's OpenMP regions and how they wait between them, never what they compute.

## What the regions cost

MinGW-w64's libgomp parks its team on POSIX semaphores between regions (its
`config/posix` barrier never spins, whatever `OMP_WAIT_POLICY` or `GOMP_SPINCOUNT` say).
`micro/` (`tests/omp_spin_check.c --time`, 16 threads, three repetitions):

| | libgomp's team | the spinning team |
|---|---:|---:|
| an empty region | 47.2-47.7 us | 1.0-1.6 us |
| two barriers in a region, more | 116.9-118.3 us | 1.5-2.0 us |
| a region met inside a task, a task | 0.60 us (9.6 us of a thread's time) | 0.08 us (1.3 us) |

A decode step opens about 730 top-level regions (attention, the router and the prefetch,
the two waves of `moe`, the CPU o_proj, the MTP layers' fp8 matmuls, silu, softmax) and
about 70,000 inside `moe_wave`'s tasks (`matmul_mxfp4`'s own `parallel for`, a team of one
each). A wrapped copy of the engine counted them in the first measurement of this cut, on
`ede829c7`: the top-level regions took about 50 ms a step beyond their longest thread's
work (the forks' wake-ups and the joins), the nested ones 0.65 s of thread time a step
(about 40 ms of the step at 16 threads), 1150 barriers about 4 ms: some 90 ms of a
~790 ms step.

## What changed

- `c/omp_spin.c`, linked with `--wrap` in Windows builds (the Makefile's `OMP_SPIN_WRAP`):
  `GOMP_parallel`, `GOMP_barrier`, the nonmonotonic dynamic loops' start, next and end
  (long and unsigned long long), `omp_get_thread_num`, `omp_get_num_threads` and
  `omp_in_parallel` go to a team whose threads spin `COLI_OMP_SPIN_US` (2000) between
  regions, then sleep on a condition variable. A region met inside one runs at once on
  the thread that meets it, a team of one as under libgomp, without building one. GCC's
  outlined functions are the same: a static loop takes the share libgomp would give it,
  a dynamic one takes chunks as libgomp hands them out. A region opened while another
  thread's runs on the team goes to libgomp. `COLI_OMP_SPIN=0`: libgomp's team throughout.
- `tests/test_omp_spin.py`: the driver spinning, sleeping after every region and on
  libgomp's team, with a second master; and the libgomp entry points that mimo.c (plain,
  CUDA, `VK=1`) and the Vulkan objects a `VK=1` build links call, all served.

## Results

One series, back to back in ABA order (`series.sh`): before (the Makefile build of
`d5acaa4d`, `dcd4b07e...`, the I/O round's after binary but for the PE timestamp and
checksum), after (the build with `omp_spin.c`, `347f0acd...`), before; `binaries.sha256`,
each in its own folder beside a copy of `coli_cuda_mimo.dll`. mimo.c's and the CUDA
loader's functions keep their addresses in both binaries (`omp_spin.o` is linked after
them; only libgomp and what follows it move). First the eight prompts greedy
(`drive_mimo.py`, one cold engine per run, 64 tokens each, `SEED=20261017`):

| Arm | Overall tok/s | Mean decode tok/s | Requests | Generation (stalled) | Minus stalled | Experts, attention |
|---|---:|---:|---:|---:|---:|---:|
| before | 2.2070 | 2.8946 | 211.15 s | 157.10 s (16.85) | 140.25 s | 96.87, 26.76 s |
| **after** | **2.2605** | **2.9950** | 206.15 s | 151.69 s (23.52) | 128.17 s | 89.37, 23.69 s |
| before | 2.2369 | 2.9427 | 208.33 s | 154.35 s (16.95) | 137.40 s | 95.37, 25.81 s |

**+1.7% overall, +2.6% mean decode** against the mean of the two before arms; each prompt
decodes faster in the after arm than in either before arm. The compute (generation minus
stalled) is 7.7% shorter, 10.7 s, and the stalls take 6.6 s of it back: the drives read
what they read before (2304 GB against 2319 and 2316). Every reply the bytes of the MTP
experiment's (`mtp.json`, the release engine's) in all three arms, the same drafts kept
(151 of 162, 95 of 102).

Then the benchmark (three identical greedy requests after a warmup):

| Arm | Overall tok/s | Decode tok/s | Last request: generation (stalled), minus stalled | Experts, attention |
|---|---:|---:|---:|---:|
| before | 2.5916 (2.5568-2.5956) | 3.1943 | 19.69 s (1.88), 17.81 s | 12.58, 3.25 s |
| **after** | **2.6142 (2.5509-2.6248)** | **3.2337** | 19.41 s (2.76), 16.65 s | 11.81, 2.99 s |
| before | 2.5964 (2.5413-2.5988) | 3.1993 | 19.65 s (1.85), 17.80 s | 12.57, 3.26 s |

**+0.8% overall, +1.2% decode**, against a spread of 0.2% between the before arms; every
reply SHA-256 `0305abcd...`, the same drafts (26 steps, 24 of 24 and 12 of 13 kept). The
measured requests compute 0.7, 1.1 and 1.2 s less (17.41, 16.67, 16.65 s against
17.77-18.18), and the last one waits 0.9 s of its 1.2 on the disk: 277 GB read against 279
and 278.

The same cut on `ede829c7`, before the I/O round (`earlier-series.json`, the same script
over that series): the benchmark 2.4487 / 2.4642 / 2.4657 (+0.3%, inside the spread), the
eight prompts 2.1310 / 2.1015 / 2.1160 (-1.0%), the compute 4.0% and 2.8% shorter and
the stalls about as much longer, or more. With the I/O round's shorter stalls, part of
the cut now reaches the clock: about a quarter of it in the benchmark, two fifths over
the eight prompts.

## Correctness

- The 70-layer CPU logits are still the release engine's (SHA-256 `cebbf3a8...`) and the
  CUDA logits the earlier rounds' (`01479691...`), the after binary on the spinning team,
  `check_mimo_cuda.py --cap 64 --threads 16 --release-host` (`correctness/`); 17/17
  argmax, the 32-token greedy replies the same bytes.
- Every benchmark reply of the series `0305abcd...`; the eight greedy replies the recorded
  ones in all three arms.
- With the after binary: `mimo_tiny_harness.py` (CPU and `MIMO_CUDA=1`), the 14
  prefix/serve tests, the 6 CUDA fallback tests (`MIMO_CUDA_TEST=1`), the 2 sampled-draft
  tests, and `test_omp_spin.py` (its check of the Vulkan objects skips `backend_vulkan.c`
  and `vk_chain.c` here, which need the Vulkan headers; mimo.c with `-DCOLI_VULKAN` and
  `vk_tier.c` pass).

## Measurement protocol and limits

As in the earlier experiments: one persistent engine per benchmark arm, a warmup, three
identical 64-token requests, prefix reuse off; the varied prompts one cold engine per run,
each prompt one request. One machine, one series; the machine drifts by a few percent
from series to series (this series' before arms against the I/O round's after arm, the
same code: +1.0% and +1.2% on the benchmark, -1.7% and -0.4% over the eight prompts), so arms are
compared within one series only. Throughout, a stale `timeout.exe 60 t_dyn.exe` of
another session's test spun about 0.4 of a core; every arm ran with it. The spinning team
keeps its 15 threads busy for up to 2 ms after each region: on a machine shared with other
work it takes CPU time libgomp's team would leave (`COLI_OMP_SPIN=0`, or a shorter
`COLI_OMP_SPIN_US`). Only Windows builds link the file; the other platforms' builds keep
their OpenMP runtime's team.

## Files

`summary.json` (the series' arms: samples, the requests' phases, the last request's
streaming counters, the varied runs' totals; `make_summary.py` over the scratch series),
`earlier-series.json` (the first measurement's series on `ede829c7`, the same script),
`benchmark/` and `varied-prompts/` (the arms' results and engine lines), `correctness/`
(`logits.sha256`: the two logits files), `micro/` (`omp_spin_check --time` on both teams),
`series.sh`, `binaries.sha256`, `sha256.json`.

## Reproduce

```bash
git checkout ca05d0e4
mingw32-make -C c mimo CUDA_DLL=1 ARCH=native VK=0
python c/tools/benchmark_mimo_cuda.py --engine c/mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --backend cuda --release-host --cap 104 --threads 16 --read-threads 16 --context 16384 --tokens 64 --runs 3 --output c/build/omp/after
python c/tools/benchmark_mimo_cuda.py ... --env COLI_OMP_SPIN=0 --output c/build/omp/libgomp
python docs/experiments/mimo-mtp-4090-2026-10-17/varied-prompts/drive_mimo.py --engine C:/path/to/c/mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --cap 104 --tokens 64 --env SEED=20261017 --json greedy.json docs/experiments/mimo-mtp-4090-2026-10-17/varied-prompts/prompts.txt
python c/tests/test_omp_spin.py   # with MinGW's gcc on PATH
```

`COLI_OMP_SPIN=0` gives libgomp's team in the same binary; the series compared the builds.
