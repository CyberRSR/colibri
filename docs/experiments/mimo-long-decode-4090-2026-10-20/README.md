# MiMo-V2.6-Pro on an RTX 4090: decode at long contexts

Measured on 20 October 2026 on the machine of the earlier MiMo experiments (Ryzen 9
7950X3D, 192 GB DDR5, RTX 4090, the model's shards on two NVMe volumes) with the RTX 4090
profile. `long_prompt.py` (its copy here, unchanged from [the GPU prompts' second
round](../mimo-gpu-prompt2-4090-2026-10-20/README.md)) sends the first round's two prompts
(`../mimo-gpu-prompt-4090-2026-10-20/prompts/`: 4310 and 24,957 tokens of this fork's
documents and a question about them) one after the other over one engine's SERVE pipe --
thinking off, greedy, prefix reuse off -- now for 128 tokens each; the engine's line for
each request gives the time its reply took and the attention's part of it. In each series
r3's engine (the code of release `mimo-pro-4090-r3`, `4b8bd1c8`: `mimo.exe` 8c0b21c0) and
this round's (`8008234f`: `mimo.exe` feb10604) alternate -- r3's, this round's, r3's, this
round's -- with the same CUDA DLL (c1019362).

## Results

The 128 tokens' generation (the first comes with the prompt):

| Prompt | Engine | Series 1 | Series 2 |
|---|---|---:|---:|
| 24,957 tokens | r3 | 92.85 s, 90.70 s | 91.00 s, 90.72 s |
| 24,957 tokens | **this round** | **55.01 s, 54.65 s** | **54.62 s, 54.62 s** |
| 4310 tokens | r3 | 49.03 s, 48.58 s | 48.52 s, 48.49 s |
| 4310 tokens | **this round** | **45.82 s, 45.56 s** | **45.52 s, 45.51 s** |

After the 24,957-token prompt a reply now comes at 2.309 - 2.325 tokens a second against
1.368 - 1.400 (127 tokens over the generation; 1.66 - 1.69x), its attention 13.1 s against
49.2 - 51.3 (103 against 388 - 404 ms a token). After 4310 tokens: 2.772 - 2.791 against
2.590 - 2.619 tok/s, the attention 57 against 84 - 88 ms a token -- about 47 of those 57 are
the attention's matrices (the qkv on the GPU, the o_proj the GPU does not hold on the CPU),
what it takes at a short context (2.93 - 2.94 s over the benchmark's 63 decode tokens). The
same replies in every run, the same MTP drafts kept (24,957 tokens: 62 steps, 64 drafts
kept; 4310: 54 steps, 73), the prompts' first tokens the same in both engines (47.1 - 48.1 s
and 92.1 - 93.9 s: their attention runs on the GPU).

Where this round's 54.6 s after 24,957 tokens go (series 2, `4-new`): the experts 26.7 s,
the attention 13.1 s, waiting for the disks 10.4 s, the rest (MTP drafting 2.3 s, the
routers, sampling, the dense layers) about 4.5 s. With r3 the attention took 49.2 of 90.7 s.
The card's clocks stayed at 2.85 - 2.94 GHz under load in both series (`series*/gpu.csv`).

The decode benchmark (`benchmark/`, 64 tokens after a 17-token prompt: too few keys for
`attend_keys`, so both engines run the loop by heads; a warmup and three runs an engine),
r3's engine and this round's alternating: 2.640 and 2.635 tok/s against 2.638 and 2.632
(decode 3.263 and 3.260 against 3.268 and 3.250), every reply `0305abcd...` as in r3 and
colibri 2.0.

## What changed

A decode step's rows -- the next token and the MTP drafts it verifies, one to three --
take their attention on the CPU: the full layers' keys and values are f32 in host RAM, 102 KB
a position over Pro's 10 full layers (2.6 GB at 25,000 positions), and a device copy would
not fit beside the 20-GiB dense budget. r3 split a full layer's attention into (row,
key/value head) tasks: 8 to 24 for the 16 workers, each reading its key head's keys and,
once for each of its 16 query heads, its values, the exps one x87 call at a time. At 24,957
tokens that was 376 ms of a token on the deployed r3 through the API (64 tokens in 44.4 s,
attention 24.0 s), and it grows with the context.

`attend_keys` (`c/mimo.c`) takes a block of up to 8 rows over a full layer's cache of 128
keys or more (`MIMO_ATTN_KEY_ROWS`, `MIMO_ATTN_KEYS`), and makes each number the loop made,
summed in the loop's order:

- **scores** by chunks of keys over every worker, all rows and heads at once; each score is
  its own sum, as before. Each chunk keeps its largest score a head, and the chunks' largest
  taken in key order is the loop's first maximum.
- **softmax** by (row, key head, eight heads): the exps eight at a time with `exp_f8`
  (`c/exp_f.h`: e^x in double, rounded to float from 2^-44 below and above; where the two
  roundings agree, so does the library's (float)(double) of its x87 result; the 731 floats
  of all 2^32 where they do not go to `exp_f` itself; 1.0 ns a value against 17), the
  denominator added in key order, then each weight e / den. The scores of a key head lie by
  groups of eight heads, so a task's 25,000 keys are one 800 KB stretch in its core's L2.
- **values** by (key head, 32 columns), every row and head, the keys in blocks of 64: each
  output the same fma chain over the keys from zero, each value row read once a call
  instead of once a query head.

Pro's full-layer shape on the 16 workers, ms a call (`kernels/bench-final.txt`, the median
of three; `tests/test_mimo_attention.c bench`, linked with the spinning OpenMP team as
`mimo.exe` is):

| Keys | Rows | r3's loop | `attend_keys` |
|---:|---:|---:|---:|
| 128 | 1 | 0.12 | 0.06 |
| 128 | 3 | 0.24 | 0.15 |
| 1024 | 1 | 1.23 | 0.12 |
| 1024 | 3 | 2.24 | 0.36 |
| 4320 | 1 | 4.79 | 0.45 |
| 4320 | 3 | 9.08 | 1.23 |
| 24,960 | 1 | 43.3 | 5.56 |
| 24,960 | 3 | 96.4 | 9.94 |

For three rows a call at 24,960 keys moves some 560 MB -- the keys 154, the values 102, the
scores' 38 about eight times over -- near what the memory gives in its 10 ms; one row's
scores read the keys at about 63 GB/s. The steps and the dead ends (other part widths and
blocks, the keys prefetched, the division moved into the values, streaming stores) are in
`development.json`.

## Correctness

- Every reply of both series is the same bytes in all eight runs of a prompt (4310 tokens:
  `23ac7428...`, 24,957: `76748fbc...`), with the same drafts kept.
- `check_mimo_cuda.py` with this round's engine (`correctness/`): on the 17-token
  benchmark prompt the CPU logits are `cebbf3a8...` and the CUDA ones `01479691...`, as in
  every round, the replies `63ed9984...`. On the 658-token correctness prompt in 256-row
  blocks: `874ee6e8...` and `03ce216b...`, the replies `f0aaa17f...` -- r3's own bytes
  (round 2's `correctness/result.json`), with every decode step's full layers through
  `attend_keys`; and with `MIMO_ATTN_KEYS=1 MIMO_ATTN_KEY_ROWS=256`, every CPU prompt
  block's full layers through it too, the same logits byte for byte (`check658-keys`).
- `tests/test_mimo_attention.c` (`kernels/test-mimo-attention.txt`): `attend_keys` against
  `attend_heads`, r3's loop, on 112 cases -- Pro's shape for 1 to 4 rows at 1 to 4101 keys,
  two, twelve and 24 heads a key head (spare lanes, three lane groups), sinks, tied scores,
  heads whose exps round to zero: every output the same bytes, built native, for
  x86-64-v3, and on Linux (gcc 11.4, where `exp_f` is glibc's `expf`, one a call).
- `tests/exp_f_check.c` (`kernels/exp_f_check.txt`): `exp_f8` gives `exp_f`'s bits for
  all 2^32 floats (`tests/test_exp_f.py` now checks it).
- The tiny fixture (`correctness/tiny-*.txt`): `mimo_tiny_harness.py` on Windows and Linux,
  with two new block-invariance variants and one greedy variant that force `attend_keys`
  on every full layer (`MIMO_ATTN_KEYS=1`, blocks of 1 and 3: the logits the bytes of
  `attend_heads`' blocks, the tokens the reference's with MTP drafts); r3's and this
  round's engine on a 200-token prompt in blocks of 64, 3 and 1, with and without the
  forcing: one logits file, one reply. The serve tests (14 prefix, 6 Brio, 2 System One),
  the 6 CUDA fallback tests, the 2 sampled-draft tests and `test_omp_spin.py` (mimo.c
  calls only what the spinning team serves) pass.

## Files

`summary.json` (every run: first token, generation, its stall, experts and attention, the
reply's SHA-256), `series1/`, `series2/` (each run's `results.json` and `engine.txt`, the
card's telemetry), `benchmark/`, `correctness/` (the three checks' reports, engine lines
and replies, the tiny fixture's runs), `kernels/` (the attention bench, the unit test and
`exp_f_check` outputs), `development.json`, `long_prompt.py`, `sha256.json`.

## Reproduce

```powershell
c/build_mimo_cuda.bat
mingw32-make -C c mimo tests/test_mimo_attention.exe CUDA_DLL=1 ARCH=native VK=0
c/tests/test_mimo_attention.exe
c/tests/test_mimo_attention.exe bench 24960 3 10
python -m unittest discover -s c/tests -p test_exp_f.py
python c/tests/mimo_tiny_harness.py --binary c/mimo.exe --fixture c/mimo_tiny
python docs/experiments/mimo-long-decode-4090-2026-10-20/long_prompt.py --engine C:\colibri\dev\c\mimo.exe --prompt docs/experiments/mimo-gpu-prompt-4090-2026-10-20/prompts/p4k.txt --prompt docs/experiments/mimo-gpu-prompt-4090-2026-10-20/prompts/p25k.txt --tokens 128 --env MIMO_CHUNK= --out c/build/long-decode
python c/tools/check_mimo_cuda.py --engine C:\colibri\dev\c\mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --release-host --cap 64 --threads 16 --output c/build/long-decode-check17
python c/tools/check_mimo_cuda.py --engine C:\colibri\dev\c\mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --release-host --prompt-file docs/experiments/mimo-gpu-prompt-4090-2026-10-20/correctness/prompt.txt --tokens 32 --chunk 256 --env MIMO_ATTN_KEYS=1 --env MIMO_ATTN_KEY_ROWS=256 --output c/build/long-decode-check658-keys
python c/tools/benchmark_mimo_cuda.py --engine C:\colibri\dev\c\mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --backend cuda --release-host --cap 104 --threads 16 --read-threads 16 --context 16384 --tokens 64 --runs 3 --output c/build/long-decode-bench
```

The engine's path must be absolute (Windows does not start a relative path written with
forward slashes). `MIMO_ATTN_KEYS=0` gives r3's attention in this round's engine.
