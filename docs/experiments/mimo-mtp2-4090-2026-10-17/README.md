# MiMo-V2.6-Pro on an RTX 4090: two MTP drafts a step, verified while the layers are sure

Measured on 17 October 2026 on the machine and with the profile of [the MTP
experiment](../mimo-mtp-4090-2026-10-17/README.md) (CUDA dense budget 20 GiB, host copies
released, expert cache 104 per MoE layer in one pool, 16 CPU workers, 16384-position
context), the same benchmark command and the same eight varied prompts. Commit
`ed77c521` decodes the dense CPU weights of a step of a few rows once for all of them;
`16c234f5` drafts with the checkpoint's first two MTP layers and verifies the drafts the
layers are sure of ([mimo.md](../../mimo.md#speculative-decoding-mtp)). `MIMO_MTP=1
MIMO_MTP_MIN_P=0` is the MTP experiment's engine with those kernels.

## Results

One series (series 2), one binary, run back to back. The benchmark (three identical
greedy requests after a warmup; the last request's drafts):

| Arm | Overall tok/s | Decode tok/s | Drafts kept |
|---|---:|---:|---|
| the `ed62c9af` build (one draft a step) | 2.140 | 2.530 | 30 of 34 |
| one draft (`MIMO_MTP=1 MIMO_MTP_MIN_P=0`) | 2.258 | 2.699 | 29 of 33 |
| one draft, verified at 0.6 | 2.266 | 2.710 | 27 of 28 |
| two drafts, both verified every step (`MIMO_MTP_MIN_P=0`) | 2.384 | 2.885 | 22 of 23, 17 of 22 |
| two drafts at 0.4 | 2.381 | 2.881 | 23 of 23, 14 of 19 |
| **two drafts at 0.6 (the default)** | **2.414** | **2.932** | 24 of 24, 12 of 13 |
| three drafts at 0.6 | 2.301 | 2.763 | 21 of 22, 11 of 12, 4 of 4 |

**+6.9% overall, +8.6% decode** against one draft a step, every reply the release
engine's bytes (SHA-256 `0305abcd...`). Against the `ed62c9af` build in the same series,
+12.8%; against the release engine of round 2 (1.159 tok/s), +108%.

Eight different prompts (`varied-prompts/`, one cold engine per run, 64 tokens each):

| Run | Overall tok/s | Mean decode tok/s | Drafts kept (first, second) |
|---|---:|---:|---|
| greedy, one draft | 2.002 | 2.560 | 209 of 246 |
| greedy, two, both verified | 2.005 | 2.544 | 154 of 185, 119 of 151 |
| greedy, two at 0.4 | 2.054 | 2.642 | 157 of 182, 104 of 119 |
| **greedy, two at 0.6** | **2.071** | **2.672** | 151 of 162, 95 of 102 |
| temperature 0.7, top_p 0.9, one draft | 1.888 | 2.358 | 205 of 253 |
| sampled, two, both verified | 1.947 | 2.446 | 155 of 185, 120 of 152 |
| sampled, two at 0.4 | 1.941 | 2.470 | 154 of 188, 107 of 125 |
| **sampled, two at 0.6** | **1.977** | **2.515** | 150 of 166, 99 of 105 |

Greedy +3.4% overall, +4.4% decode, all eight replies the bytes of the MTP experiment's
(the release engine's) in every run; sampled +4.7%, +6.7%, the tokens the sampler's
distribution but other draws. Verifying both drafts every step is no faster than one
draft on these prompts greedy (2.005 against 2.002; prompt 3, whose drafts mostly fall,
1.67 decode against 2.23): the cut is what makes the second layer pay.

Where the time went (the last benchmark request's generation, 64 tokens): one draft,
23.37 s in 34 steps -- experts 14.51 s, attention 4.09 s, disk stalls 2.53 s, drafting
0.77 s; two drafts at 0.6, 21.49 s in 27 steps -- experts 13.27, attention 3.32, stalls
2.38, drafting 1.26. A verified draft's row brings its experts whether it stands or not
(a layer needs 8 for one token, 13.9 for two, 19.3 for three), about 220 ms a row on this
profile; both drafts verified every step spent 14.00 s on experts, the cut 13.27.

## Which hidden state the second layer reads

vLLM's MiMo-V2 MTP builds one layer (`num_mtp_layers = 1`) and repeats it, feeding it
its own output after its final norm; SGLang's multi-layer MTP, Xiaomi's day-0 code for
MiMo-V2-Flash, gives every layer the target's hidden state instead (non-chain), layer k
embedding the token k + 1 ahead. Measured on the benchmark prompt, three drafts, every
draft verified, 64 tokens (`development.json`, a dropped `MIMO_MTP_CHAIN` switch), the
same reply bytes in all three:

| Layer k reads | Second drafts kept | Third drafts kept | Tokens a step |
|---|---:|---:|---:|
| **the trunk's hidden state** | **14 of 19** | **13 of 14** | **3.30** |
| the layer before's residual | 14 of 23 | 3 of 14 | 2.60 |
| the layer before's output after its final norm | 11 of 26 | 1 of 11 | 2.41 |

## The cut

`traces/` holds every step's drafts and the layers' own probabilities for them
(`MIMO_MTP_TRACE`, three drafts verified every step, the eight prompts at 128 tokens,
greedy and sampled), and `mtp_policy.py`, which replays them through a cost per verified
row taken from the benchmark. A first draft its layer gave 0.8 or more stood 172 times in
172 greedy, 0.99 or more 100 of 100 sampled; one under 0.5 about three times in ten.
Simulated, verifying while the probabilities multiply to 0.5-0.6 beats verifying every
draft by about 8% with two layers, greedy and sampled (`g-policy.txt`, `t07-policy.txt`;
the steps taken as samples, so an estimate); the series above measured 0.4 and 0.6 and
found 0.6 the faster on all three workloads, by less than simulated: +1.3% on the
benchmark, +3.3% greedy and +1.5% sampled over verifying both drafts.

## Correctness

- The 70-layer CPU logits are still the release engine's (SHA-256 `cebbf3a8...`, the
  committed build, `correctness/`), the CUDA logits the MTP experiment's
  (`01479691...`), and both 32-token greedy replies, drafted, the same bytes.
- 200 tokens past the window (217 positions, the benchmark prompt, the by-hand path):
  `MIMO_MTP=0`, `1`, `2` and `3` and the committed defaults give the same bytes
  (`development.json`).
- The tiny fixture now carries three MTP layers (`make_mimo_tiny.py --mtp 3`, the
  Makefile's fixture): every tiny check runs drafted, and the harness checks one to three
  drafts a step against Xiaomi's tokens with the layers' own drafts (nearly all dropped,
  in windows of 8 they keep wrapping), the cut, and the reference's tokens kept and
  missed (`MIMO_MTP_ORACLE`). It catches the ring bug below; the 14 serve/prefix (at one,
  two and three drafts), 6 CUDA fallback, Brio and SystemOne tests pass on it.

## On the way (`development.json`)

- Series 1, the first build (every draft verified, before the turn-end cap): the
  benchmark 2.259 (the `ed62c9af` build), 2.310 (one draft), 2.390 (two), 2.253 (three).
  Its MTP=1 against the `ed62c9af` build is the kernels alone: +2.2%, decode 2.691 ->
  2.762, the request's CPU o_proj 2.88 -> 2.37 s.
- The window ring: a sliding window's ring held exactly 128 positions, so the rows of a
  step whose drafts were dropped overwrote positions the next step still read once a
  conversation passed the window ("window ring lost position" on the first 128-token
  reply). With one draft the position it overwrites is the one leaving the window; with
  N the ring now keeps N - 1 more.
- The context: `serve_budget` lets a turn fill the context, which a step with two drafts
  then overflowed; a step now drafts no more tokens than can still go out.

## Measurement protocol and limits

As in the earlier experiments: one persistent engine per benchmark arm, a warmup, three
identical 64-token requests, prefix reuse off; the varied prompts one cold engine per
run, each prompt one request, SEED=20261017. Series 2 is sequential, in the order of the
tables; it ran a build of `16c234f5`'s tree before its last edits (the two defaults, two
log lines), every arm naming `MIMO_MTP` and `MIMO_MTP_MIN_P`. The machine drifted
between series: the `ed62c9af` build ran 2.259 in series 1 and 2.140 in series 2, the
CPU's share slower, the disks faster, so arms are compared within a series only. The
committed build then ran with its defaults (`confirmation/`): the benchmark 2.405, 2.262
with one draft, 2.409 again; the eight prompts greedy 2.069, every reply unchanged.

## Files

`summary.json` (every benchmark arm, raw samples, the last request's phases and drafts),
`benchmark/` (series 2), `confirmation/`, `series1/`, `varied-prompts/` (per-run results,
replies and engine lines, `summary.json`), `traces/`, `correctness/`, `development.json`,
`mtp2.json` and `min-p.json` (one-variable manifests, `c/experiment_manifest.py`),
`sha256.json`.

## Reproduce

```powershell
git checkout 16c234f5
c/build_mimo_cuda.bat
mingw32-make -C c mimo CUDA_DLL=1 ARCH=native VK=0
python c/tools/benchmark_mimo_cuda.py --engine c/mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --backend cuda --release-host --cap 104 --threads 16 --read-threads 16 --context 16384 --tokens 64 --runs 3 --output c/build/mtp2/default
python c/tools/benchmark_mimo_cuda.py ... --env MIMO_MTP=1 --env MIMO_MTP_MIN_P=0 --output c/build/mtp2/one-draft
python docs/experiments/mimo-mtp-4090-2026-10-17/varied-prompts/drive_mimo.py --engine C:/path/to/c/mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --cap 104 --tokens 64 --env SEED=20261017 --json greedy.json docs/experiments/mimo-mtp-4090-2026-10-17/varied-prompts/prompts.txt
python docs/experiments/mimo-mtp2-4090-2026-10-17/traces/mtp_policy.py docs/experiments/mimo-mtp2-4090-2026-10-17/traces/g.trace
python c/experiment_manifest.py docs/experiments/mimo-mtp2-4090-2026-10-17/mtp2.json docs/experiments/mimo-mtp2-4090-2026-10-17/min-p.json
```

`drive_mimo.py` needs the engine's absolute path (it starts it as given).
