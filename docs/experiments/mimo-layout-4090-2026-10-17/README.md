# MiMo-V2.6-Pro on an RTX 4090: the code-layout effect was MinGW's exp

Measured on 17 October 2026 on the machine and with the profile of
[the two-draft experiment](../mimo-mtp2-4090-2026-10-17/README.md) (CUDA dense budget
20 GiB, host copies released, expert cache 104 per MoE layer in one pool, 16 CPU workers,
16384-position context, two MTP drafts a step cut at 0.6), the same benchmark command and
the same eight varied prompts, on top of [the OpenMP spin round](../mimo-omp-spin-4090-2026-10-17/README.md)
(`a495a0c0`, code `ca05d0e4`). The change computes the same bits as before. What changes
is which code runs them.

## The effect

During [the I/O round](../mimo-io-4090-2026-10-17/README.md) a development build (`v3`)
that differed from another (`v2`) only by a few lines in `mimo_prefetch` ran every CPU
phase slower, with the lines' switch on or off. `moe` was the same code 112 bytes further
on. Its event logs (`analysis/phases.txt`, the three measured requests, main thread):

| Phase, a call | v2 | v3, switch off | v3, switch on |
|---|---:|---:|---:|
| after attention to the step's reads: norm, router, top-k | 0.209 ms | 0.289 ms | 0.291 ms |
| the next layer's prediction (router, top-k) | 0.299 ms | 0.377 ms | 0.384 ms |
| attention | 1.689 ms | 1.792 ms | 1.800 ms |
| a wave of experts | 2.088 ms | 2.178 ms | 2.231 ms |
| the two MTP drafts | 16.29 ms | 17.46 ms | 17.63 ms |
| the step's slots (no math) | 0.030 ms | 0.031 ms | 0.032 ms |

It reproduces today with the same two binaries (`analysis/runs-s1.txt`, `runs-s2.txt`): the
benchmark at 2.512, 2.535, 2.540 and 2.542 tok/s for v2, at 2.477, 2.479, 2.503 and 2.501
for v3, interleaved. Generation minus stalled is 18.14 s against 18.77 (+3.5%) in the
unsampled pair. The per-processor busy time and effective clock (`analysis/cpu.txt`, PDH)
are the same in both runs within a few points on every logical processor, so the threads
run on the same cores.

## Where the time goes

An external sampler (`analysis/winsample2.c`) suspends the engine's main thread about 1000
times a second and records its RIP, and with `winsample3.c` also RSP and the return
addresses above it. It does this for 72 s after the benchmark's warmup.
`analysis/samples.txt`: the profiles of the two binaries match function for function
but one. **`__expl_internal`** takes 1.5% of the main thread in v2 and 5.5% in v3. That is
libmingwex's x87 core of `exp`, behind every `expf` the engine computes, because MinGW's
(msvcrt) `math.h` makes `expf(x)` `(float)exp((double)x)`. In v3, more than half the samples
in it sit right after its two `FLDCW`. Its code switches the control word to chopping for
two `FRNDINT` and back. In both runs of each binary (`analysis/rsp.txt`):

| | samples in `__expl_internal` | right after the two `FLDCW` |
|---|---:|---:|
| v3, v3 | 3803, 3874 (5.6%, 5.7%) | 2170, 2194 |
| v2, v2 | 1086, 1030 (1.6%, 1.5%) | 64, 49 |

The stack is not the cause: RSP modulo 4 KiB at those samples takes the same values in
both binaries (the same frames, 0x558, 0x568, 0x5d8, 0x788, 0x4c8), and the data
`__expl_internal` reads has the same addresses. Only its code moved: it starts at
0x140072b40 in v2 and at 0x140072bc0 in v3, the same offset modulo 128.

`micro/` places the function's own bytes, from HEAD's build, at every 16-byte slot of a
4 KiB page and at every byte of the first 2 KiB, each copy in its own page (`gen_xexp.py`,
`xbench.c`, one pinned core). Each copy is checked against `exp()` first:

| | slots | median a call | slow copies |
|---|---:|---:|---|
| libmingwex's `__expl_internal` | 256 x 16 bytes | 19.29 ns | 0x3c0 and 0xbc0: 111.4 ns |
| | 2048 x 1 byte | 19.29 ns | 0x1c4, 0x3c0, 0x3c8, 0x5c4: 111-113 ns |
| the same with `FISTTP` + `FILD` (`exp_f.h`) | 256 x 16 bytes | 14.27 ns | none (max 15.08) |
| | 2048 x 1 byte | 14.29 ns | none (max 20.56) |

At those few addresses the two `FLDCW` (or the `FLD` between the `FRNDINT`s) meet a
512-byte boundary in a way this Zen 4 handles 5.7 times slower; most nearby placements
with the same instructions on the same boundary are not slow. The copies show the
mechanism, not its hardware cause. The library's object is 64-byte aligned, so in a build
`__expl_internal` lands in one of 64 slots of a page, and two of them (0x3c0, 0xbc0) are
slow: one build in 32. `analysis/addresses.txt` has where it landed in each build of the
last rounds. All 13 builds are fine but v3 (0xbc0) and this round's two built on purpose.

## What changed

- `c/exp_f.h`, `exp_f()`: libmingwex's x87 sequence inline, its two truncations by `FISTTP`
  + `FILD` (SSE3), which give the same integers without touching the control word (the
  sign of a truncated zero, the one difference, never reaches the result). The same limits
  as the library's `exp`: above it `HUGE_VAL`, below 0, a NaN back quieted. It is used only
  where `expf` is that x87 `exp`: MinGW-w64 x86-64 on msvcrt (not `_UCRT`), with SSE3.
  Elsewhere `exp_f` is `expf`.
- `c/mimo.c`: silu, the router's sigmoid, the attention softmax, `exp_sum` (the drafts'
  softmax) and `sample_rank` call `exp_f`. In the after binary no function of a decode
  step calls `exp`. What still calls it is the picture tower's softmax and the logprob
  read-outs, which sum `exp` in double over the vocabulary when a request asks for logprobs.
- `c/tests/exp_f_check.c` and `tests/test_exp_f.py`: `exp_f` against the library's `expf`
  on all 2^32 floats, built as the engines are (-O3, x86-64-v3, OpenMP), 6 s on 16
  threads. The main thread and the threads winpthreads starts (libgomp's, the spinning
  team's) run with the x87 control word 0x037f, as the check's do (both functions run the
  same x87 arithmetic under any other). `tests/test_mimo_sample_draft.py` compiles mimo.c's
  sampler beside `exp_f.h` now.

A silu value costs 17 ns against 32 (`micro/exp_f_check.txt`).

## Results

One series, back to back (`series.sh`). Before is HEAD's Makefile build (`347f0acd...`,
the OpenMP round's after binary), after is `36a4fdfe`'s with `exp_f` (`af1c7ca6...`), then
before again; `binaries.sha256`, each in its own folder beside a copy of
`coli_cuda_mimo.dll`. In both, `__expl_internal` sits at a fast address (0x100 and 0xac0).
First the eight prompts greedy (`drive_mimo.py`, one cold engine per run, 64 tokens each,
`SEED=20261017`):

| Arm | Overall tok/s | Mean decode tok/s | Requests | Generation (stalled) | Minus stalled | Experts, attention |
|---|---:|---:|---:|---:|---:|---:|
| before | 2.2444 | 2.9848 | 207.63 s | 152.10 s (21.14) | 130.96 s | 91.26, 23.75 s |
| after | 2.2513 | 2.9889 | 206.99 s | 152.15 s (21.50) | 130.65 s | 91.43, 23.96 s |
| before | 2.2508 | 2.9775 | 207.03 s | 152.53 s (22.62) | 129.91 s | 90.29, 24.36 s |

Then the benchmark (three identical greedy requests after a warmup):

| Arm | Overall tok/s | Decode tok/s | Last request: generation (stalled), minus stalled | Experts, attention |
|---|---:|---:|---:|---:|
| before | 2.6061 (2.5784-2.6097) | 3.2239 | 19.54 s (2.57), 16.97 s | 11.98, 3.09 s |
| after | 2.6121 (2.5735-2.6123) | 3.2320 | 19.47 s (2.65), 16.82 s | 11.93, 3.06 s |
| before | 2.6108 (2.5830-2.6180) | 3.2295 | 19.45 s (2.48), 16.97 s | 11.98, 3.09 s |

Where the library already sits at a fast address, `exp_f` changes nothing that shows:
+0.16% over the eight prompts and +0.14% for the benchmark, both inside the spread of the
two before arms (0.29%, 0.18%). The benchmark's last two requests compute 0.9% less
(16.82 s against 16.94-16.97), and the stall takes it back.

Then three benchmark arms with `__expl_internal` moved to a slow address by a pad object
linked after `omp_spin.o`: mimo.c's 177 functions keep their addresses, and the library
code moves. HEAD's code with a 704-byte pad (`__expl_internal` at 0x1400723c0; the data
sections move one whole page with the text) / the code with `exp_f` and a 256-byte pad
(0x140072bc0) / HEAD's code again:

| Arm | Overall tok/s | Decode tok/s | Last request: generation (stalled), minus stalled | Experts, attention |
|---|---:|---:|---:|---:|
| HEAD, slow exp | 2.5816 (2.5354-2.5874) | 3.1887 | 19.74 s (2.14), 17.60 s | 12.07, 3.27 s |
| **exp_f, slow exp** | **2.6071 (2.5827-2.6146)** | **3.2284** | 19.51 s (2.67), 16.84 s | 11.95, 3.07 s |
| HEAD, slow exp | 2.5687 (2.5475-2.5716) | 3.1692 | 19.84 s (2.24), 17.60 s | 12.05, 3.29 s |

With the library at the slow address, HEAD's code computes 3.7% longer than at its own
address (17.60 s against 16.97, measured requests 17.6-17.9 s against 16.9-17.1). That is
v3's loss again, now in HEAD's code: 1.3% of the benchmark's tok/s, the stall absorbing the
rest. With `exp_f` the same placement costs nothing: 16.84 s, as the after arm's 16.82;
**+1.24% overall and +1.56% decode** against the two arms around it, whose spread is 0.5%.
Every request computes less (17.19, 16.83 and 16.84 s against 17.6-17.9).

## Correctness

- `exp_f` gives the library's `expf` bits for every one of the 2^32 floats (`micro/exp_f_check.txt`,
  `tests/test_exp_f.py`).
- The 70-layer CPU logits are still the release engine's (SHA-256 `cebbf3a8...`) and the
  CUDA logits the earlier rounds' (`01479691...`), from the after binary with
  `check_mimo_cuda.py --cap 64 --threads 16 --release-host` (`correctness/`): 17/17 argmax,
  the 32-token greedy replies the same bytes.
- Every benchmark reply of all nine arms `0305abcd...`, the same drafts kept (26 steps,
  24 of 24 and 12 of 13). The eight greedy replies are the recorded ones in all three
  varied arms, with the same drafts (151 of 162, 95 of 102).
- With the after binary: `mimo_tiny_harness.py` (CPU and `MIMO_CUDA=1`), the 14
  prefix/serve tests, the 6 CUDA fallback tests (`MIMO_CUDA_TEST=1`), the 2 sampled-draft
  tests, `test_omp_spin.py` (its two Vulkan objects skipped for want of the headers, as
  before) and `test_exp_f.py`.

## Measurement protocol and limits

As in the earlier experiments: one persistent engine per benchmark arm, a warmup, three
identical 64-token requests, prefix reuse off. The varied prompts run one cold engine per
run, each prompt one request. One machine and one series, compared within the series. A
stale `timeout.exe 60 t_dyn.exe` of another session's test spun about 0.4 of a core
throughout, in every arm. The sampled runs (`analysis/runs-s*.txt`) suspend the main
thread about 1000 times a second. Their tok/s are within a percent of the unsampled
ones but are not the series'. The slow addresses are this machine's (a Ryzen 9 7950X3D,
Zen 4). Another core may have others, or none. `exp_f` has no `FLDCW` to place, and over
2304 placements of its sequence none was slow, but inline it sits wherever its callers
land: what the series shows is one such build. Builds with UCRT or another C library keep
their own `expf` and whatever layout effects they have.

## Files

`summary.json` (the series' arms: samples, the requests' phases, the last request's
streaming counters, the varied runs' totals; `make_summary.py` over the scratch series),
`benchmark/` and `varied-prompts/` (the arms' results and engine lines), `correctness/`
(`logits.sha256`: the two logits files), `micro/` (the placement sweeps and the all-floats
check), `analysis/` (the v2/v3 phases from the I/O round's logs, the sampled runs and their
summaries, the samplers, `cpumon.c`, the sweep's generator and driver, `addresses.txt`),
`series.sh`, `binaries.sha256`, `sha256.json`. The analysis scripts ran from
`c/build/layout`.

## Reproduce

```bash
git checkout 36a4fdfe
mingw32-make -C c mimo CUDA_DLL=1 ARCH=native VK=0
python c/tests/test_exp_f.py   # with MinGW's gcc on PATH
python c/tools/benchmark_mimo_cuda.py --engine c/mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --backend cuda --release-host --cap 104 --threads 16 --read-threads 16 --context 16384 --tokens 64 --runs 3 --output c/build/layout/after
# the placement sweep (from c/build/layout, a MinGW build of mimo in bin-before/):
python gen_xexp.py mingw bin-before/mimo.exe 256 16 x.S && gcc -O2 -o xbench.exe xbench.c x.S && ./xbench.exe 4 100000
python gen_xexp.py fisttp 256 16 x.S && gcc -O2 -o xbench.exe xbench.c x.S && ./xbench.exe 4 100000
```

To see the effect in HEAD's code, link a pad object after `omp_spin.o` so that `nm
mimo.exe` shows `__expl_internal` at 0x...3c0 or 0x...bc0 of a page (`.text`,
`.skip N`). Here N was 704 for `ca05d0e4`.
