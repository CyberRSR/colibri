# MiMo-V2.6-Pro on an RTX 4090: MTP drafts verified in each decode step

Measured on 17 October 2026 on the machine and with the profile of the
[packing and pooling experiment](../mimo-pack-4090-2026-10-17/README.md) (CUDA dense
budget 20 GiB, host copies released, expert cache 104 per MoE layer in one pool, 16
CPU workers, 16384-position context), the same benchmark command and the same eight
varied prompts. Commit `ed62c9af` drafts the token after the next with the
checkpoint's first MTP layer and verifies it as a second row of each decode step
([mimo.md](../../mimo.md#speculative-decoding-mtp)); `MIMO_MTP=0` is the engine before.

## Results

One series, the same binary, run back to back. The benchmark (three identical greedy
requests after a warmup):

| Arm | Overall tok/s | Decode tok/s | First token, s | Drafts kept |
|---|---:|---:|---:|---:|
| one token a step (`MIMO_MTP=0`) | 1.999 | 2.329 | 4.97 | -- |
| **MTP drafts (the default)** | **2.206** | **2.617** | 4.97 | 30 of 34 |

**+10.4% overall, +12.4% decode**, the first token no later; every reply the same
bytes (SHA-256 `0305abcd...`, the release engine's). Raw samples: 2.0001, 1.9980,
1.9988 without drafts; 2.2133, 2.2057, 2.1984 with them. Against the release engine
of round 2 (1.159 tok/s, decode 1.359): +90% overall, +93% decode.

Where the time went (the phase lines of the last request, its 64 tokens): 34 steps
of two rows, 709 ms each, 1.88 tokens a step -- 377 ms a token against 423 ms
without drafts: the experts 220 ms a token instead of 229 (13.9 distinct experts a
layer for two tokens, not 16, but a dropped draft's are computed too), the disks
49 ms instead of 69, the attention 75 ms instead of 95 (every dense matrix read once
for both rows), the drafting 11 ms.

Eight different prompts (`varied-prompts/`, one cold engine per run):

| Run | Overall tok/s | Mean decode tok/s | Mean first token, s | Drafts kept |
|---|---:|---:|---:|---:|
| greedy, `MIMO_MTP=0` | 1.844 | 2.301 | 6.75 | -- |
| **greedy, MTP** | **1.982** | **2.520** | 6.82 | 213 of 250 |
| temperature 0.7, top_p 0.9, `MIMO_MTP=0` | 1.770 | 2.186 | 6.87 | -- |
| **temperature 0.7, top_p 0.9, MTP** | **1.941** | **2.451** | 6.75 | 214 of 249 |

Greedy: +7.5% overall, +9.5% decode, all eight replies the bytes of the engine
before the packing (round 3), which are the release engine's. Sampled -- the gateway's
default when a request names no temperature is 0.7, and top_p 0.9 -- +9.7% overall,
+12.1% decode; a draft stands with the probability the sampler gives it, so the
tokens have the sampler's distribution but are other draws than the plain run's.

A 2260-token prompt (`long-prompt/`; prompts are not drafted): 202.9 s, against
205.4 s in round 2; the decode of its 8 tokens is too short to say anything.

## Correctness

- The 70-layer CPU logits are still the release engine's (SHA-256 `cebbf3a8...`,
  `correctness/`); that run's 32-token greedy reply, drafted, is the CUDA run's.
- CUDA against CPU: 17/17 argmax, maximum absolute logit difference 4.6e-5, KL
  1.5e-11, the greedy replies the same bytes.
- `c/tests/test_mimo_sample_draft.py` cuts `sample()` and `sample_draft()` out of
  `mimo.c` and compares 400,000 draws of each, for drafts inside and outside top-p's
  nucleus at three temperatures: the frequencies agree within 0.003 and a draft
  stands at its probability.
- The tiny fixture has no MTP layer; its oracle, prompt-block logits (1, 7, 64 and
  512 rows, native and f32), the 14 serve/prefix and 6 CUDA fallback tests pass.

## On the way (`development.json`)

- The first build drafted greedy requests only and filled the MTP block's window
  before the first token: 1.991 to 2.221 tok/s on the benchmark, 1.816 to 1.981 over
  the varied prompts, in its own series.
- The MTP layer fed the trunk's hidden state before its final norm instead of after
  kept the same 30 of 34 drafts on the benchmark prompt; the switch was dropped.
- The long prompt on the way: the packed bf16 prompt kernel of `71f5a0c4` decoded a
  group of weights for every 4 rows (218.3 s); unpacking an output's row once a
  64-row tile, 211.1 s; a prompt block that grows a pooled layer only to the cap,
  206.7 s (207.7 without the pool).

## Measurement protocol and limits

As in the earlier experiments: one persistent engine per benchmark arm, a warmup,
three identical 64-token requests, prefix reuse off; the varied prompts one cold
engine per run, each prompt one request. One series, sequential. The series ran a
build of the tree committed as `ed62c9af` before its last two edits: a then-present
`MIMO_MTP_PRENORM` switch, off (the path the default runs), was dropped, and two
comments rewritten; a run of the committed build gives the benchmark prompt's reply
and its 30 of 34 kept drafts again. Acceptance depends on the text: 64% to 97% per
prompt here.

## Files

`summary.json`, the two benchmark arms (raw samples, replies, GPU telemetry, engine
lines), `mtp.json` (one-variable manifest, `c/experiment_manifest.py`),
`varied-prompts/` (prompts, the driver, per-prompt results and replies for the four
runs, `summary.json` with the drafts kept), `long-prompt/`, `correctness/`,
`development.json`, `sha256.json`.

## Reproduce

```powershell
git checkout ed62c9af
c/build_mimo_cuda.bat
mingw32-make -C c mimo CUDA_DLL=1 ARCH=native VK=0
python c/tools/benchmark_mimo_cuda.py --engine c/mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --backend cuda --release-host --cap 104 --threads 16 --read-threads 16 --context 16384 --tokens 64 --runs 3 --output c/build/mimo-mtp/mtp
python c/tools/benchmark_mimo_cuda.py ... --env MIMO_MTP=0 --output c/build/mimo-mtp/mtp-off
python docs/experiments/mimo-mtp-4090-2026-10-17/varied-prompts/drive_mimo.py --engine c/mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --cap 104 --tokens 64 --temperature 0.7 --top-p 0.9 --json sampled.json docs/experiments/mimo-mtp-4090-2026-10-17/varied-prompts/prompts.txt
python c/tests/test_mimo_sample_draft.py
python c/experiment_manifest.py docs/experiments/mimo-mtp-4090-2026-10-17/mtp.json
```
