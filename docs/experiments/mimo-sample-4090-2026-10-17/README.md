# MiMo-V2.6-Pro on an RTX 4090: sampled draws that sort only the nucleus

Measured on 17 October 2026 on the machine and with the profile of
[the two-draft experiment](../mimo-mtp2-4090-2026-10-17/README.md) (CUDA dense budget
20 GiB, host copies released, expert cache 104 per MoE layer in one pool, 16 CPU workers,
16384-position context, two MTP drafts a step cut at 0.6), the same benchmark command and
the same eight varied prompts. Commit `77c682eb` changes `sample()` and `sample_draft()`
in `c/mimo.c`, the draws at temperature > 0: a decode step draws once for every
verified draft it reaches and once more for the next token when they all stood.

- Before: per draw, `expf((logit - max) / T)` over the 152576 entries one after another
  (~4.6 ms with MinGW's x87 `exp`), the total added in order, `qsort` of all 152576
  (p, id) pairs (~11.5 ms), then top_p's walk. In the engine (`micro/`, a timed copy)
  **17.5 ms a `sample()`, 17.7 ms a `sample_draft()`**: over the eight prompts sampled,
  471 draws, 8.3 s of the 232 s.
- After: the `expf` over the workers in chunks of 512, each entry's the same `expf` (the
  workers' x87 control word is the master's, 0x37f; 4 M inputs, the same bits), the
  total still added in order, so the cut keeps its bits; then only the entries of p at
  least (total - cut) / vocab are sorted -- all the others together come short of
  total - cut, so the nucleus is among them; 12.7 on average on the engine's rows. They
  come out in the whole sort's order but among entries of equal p; the walks' sums do
  not see that order, only which entry a draw lands on, so where the draw's entry shares
  its p, or the draft shares the nucleus' last p with entries outside it, the whole
  vocabulary is sorted as before and the draw reads the same place. None of the
  engine's 471 rows needed it. **0.63 ms a `sample()`, 0.64 ms a `sample_draft()`** in
  the engine, the same draws.

## Results

One series, run back to back in ABA order: before (`60061c42...`, the Makefile build of
`ede829c7`'s tree, the previous rounds' binary), after (the Makefile build of the
change), before; `binaries.sha256`. Eight different prompts sampled at temperature 0.7,
top_p 0.9, `SEED=20261017` (`drive_mimo.py`, one engine per run, 64 tokens each):

| Arm | Overall tok/s | Mean decode tok/s | Requests | Generation (stalled) | Drafts kept (first, second) |
|---|---:|---:|---:|---:|---|
| before | 2.0261 | 2.5869 | 231.48 s | 176.59 s (26.55) | 150 of 166, 99 of 105 |
| **after** | **2.1220** | **2.7368** | 221.02 s | 166.59 s (27.02) | 150 of 166, 99 of 105 |
| before | 2.0456 | 2.6113 | 229.27 s | 174.80 s (27.02) | 150 of 166, 99 of 105 |

**+4.2% overall, +5.3% mean decode** against the mean of the two before arms; each of
the eight prompts decodes faster in the after arm than in either before arm. The
generation phase is 8-10 s shorter, the draws' 8.3 s and some; the stalls do not grow
(the time was the main thread's between steps, not taken from the disk). Every reply
the same bytes in all three arms, and the bytes of the two-draft experiment's sampled
run (`t07-mtp2-p06.json`, an earlier build), the same drafts kept.

The benchmark, greedy (three identical requests after a warmup), as a check -- greedy
decoding takes the argmax and never sorts:

| Arm | Overall tok/s | Decode tok/s | Last request: generation (stalled) |
|---|---:|---:|---:|
| before | 2.4689 (2.4682-2.4746) | 3.0041 | 21.00 s (3.08) |
| after | 2.4685 (2.4620-2.4702) | 3.0028 | 20.97 s (3.04) |
| before | 2.4713 (2.4674-2.4739) | 3.0025 | 20.98 s (3.03) |

-0.06%, inside the before arms' spread; every reply SHA-256 `0305abcd...`, the same
drafts (26 steps, 24 of 24 and 12 of 13 kept).

## Correctness

- Greedy: the benchmark replies above; the eight prompts greedy with the after binary
  (`varied-prompts/g-after*`, untimed, 2.1385 tok/s): all eight the recorded replies
  (the MTP experiment's, the release engine's), 151 of 162 and 95 of 102 drafts kept as
  recorded.
- Sampled: the replies above; before that, the timed copies (`micro/timed-*`) gave the
  same eight replies and the same drafts kept, prompt by prompt.
- The draws, row by row (`micro/bench-real.txt`): the 471 rows the timed before engine
  met (200 for `sample()`, 271 for `sample_draft()`, each of those with its draft and
  three more: the argmax, the second, any entry), 16 seeds each: 20544 draws, token,
  kept flag and the next `rand()` the old code's in every one;
  16.44 ms a draw against 0.74 ms (16 threads, the master alone 1 ms before each call,
  as the workers sleep between steps). On synthetic rows full of ties (logits
  continuous or on grids of 1/4 to 1/1024, ties at the top; `micro/bench-synthetic.txt`):
  4800 draws the old code's, 137 of the 240 rows sorting the whole vocabulary for some
  draw (60 of them at top_p 1, which always does).
- `c/tests/test_mimo_sample_draft.py` now also holds `sample()` and `sample_draft()` to
  the draws of the code before (kept in the test as the reference) on rows full of ties:
  34560 draws, 4915 of them sorting the whole vocabulary because the draw's entry shares
  its p, 3280 because the draft shares the nucleus' last; with those two checks removed,
  5312 draws differ. The distribution test passes as before.
- The tiny suites pass with the after binary (CPU and `MIMO_CUDA=1` harness output byte
  for byte the before binary's), 14 prefix/serve tests, 6 CUDA fallback tests
  (`MIMO_CUDA_TEST=1`).

## Files

`summary.json` (every arm, the timed engines' per-draw times, the micro-benchmark's
lines), `series.sh` (the series as it ran), `binaries.sha256`, `varied-prompts/` (per
run results, replies and engine lines), `benchmark/` (per arm `result.json` with the
samples, `engine.txt`), `micro/`: `make_variants.py` (the timed engines' sources:
`ede829c7`'s and the change's, each draw timed, `MIMO_SAMPLE_DUMP` writing its rows),
`build.sh`, `gen_bench.py` and `bench_driver.c` (the old code against the new on the
dumped rows or synthetic ones), what they printed, and the timed runs' lines and
replies. The rows themselves (287 MB) are not kept.

## Reproduce

```powershell
mingw32-make -C c mimo CUDA_DLL=1 ARCH=native VK=0
python docs/experiments/mimo-mtp-4090-2026-10-17/varied-prompts/drive_mimo.py --engine C:/path/to/c/mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --cap 104 --tokens 64 --temperature 0.7 --top-p 0.9 --env SEED=20261017 --json s-after.json docs/experiments/mimo-mtp-4090-2026-10-17/varied-prompts/prompts.txt
python c/tools/benchmark_mimo_cuda.py --engine c/mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --backend cuda --release-host --cap 104 --threads 16 --read-threads 16 --context 16384 --tokens 64 --runs 3 --output c/build/sample-draft/b-after
```

The before arm is the same commands on a build of `ede829c7`. The micro-benchmark: copy
`micro/` to `c/build/sample-draft/`, run `make_variants.py` and `build.sh before-timed`,
the varied command above on `before-timed/mimo.exe` with `--env
MIMO_SAMPLE_DUMP=C:/path/to/rows.bin`, then `gen_bench.py`, build `bench.c` with the
engine's flags (`-O3 -march=native -fopenmp -static`) and run `bench rows.bin 16 1.0`.
