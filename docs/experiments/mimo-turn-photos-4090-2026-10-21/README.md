# MiMo-V2.6-Pro on an RTX 4090: agent turns that send the reply back without its reasoning

Measured on 21 October 2026 on the machine of the earlier MiMo experiments (Ryzen 9
7950X3D, 192 GB DDR5, RTX 4090, the model's shards on two NVMe volumes) with the RTX 4090
profile. An OpenAI-compatible agent client resends the whole conversation every turn, and
sends the assistant's turns back with their content and tool calls but without
`reasoning_content`. The template renders such a turn as `<think></think>`, so with thinking
on the new prompt leaves the state the last turn built right after its `<think>` cue: the
live prefix (`kv_prefix_reuse`, all or nothing) never matched, and r4 computed the whole
conversation again on every turn.

`agent_loop.py` drives the gateway (`api_run.py`: this checkout's `coli serve` with the
environment of `START-MIMO`, the engine given by `COLI_ENGINE`) the way such a client does,
thinking on, greedy, streaming: turn 1 is the GPU prompt rounds' 24,957-token prompt
(`../mimo-gpu-prompt-4090-2026-10-20/prompts/p25k.txt`) behind a system prompt and a
`read_file` tool; turn 2 answers a `read_file` call with 6,000 characters of
`docs/mimo-cuda.md`; turns 3 and 4 are one-line user follow-ups. Each reply is cut at 64
tokens (the time of a turn is its prompt's and a short reply's). `drop` sends the assistant
turns back as most clients do; `keep` also sends `reasoning_content`, which keeps the live
prefix. Each series runs r4's engine (`base`: the branch's code before this round,
`9b043a09`, built here: `mimo.exe` 109fcd19) and this round's (`new`: `0accfa07`, `mimo.exe`
6e75a032),
with the same CUDA DLL (c1019362); series 2 in the opposite order.

## Results

Time to the first streamed token of each turn, seconds:

| Client | Engine | Turn 1 (25,054 tokens) | Turn 2 (tool result, +1647) | Turn 3 (+23) | Turn 4 (+17) |
|---|---|---:|---:|---:|---:|
| drop | r4 | 94.0, 96.3 | 97.5, 100.2 | 97.2, 100.3 | 97.7, 99.9 |
| drop | **this round** | 94.3, 96.9 | **34.5, 34.4** | **10.3, 10.3** | **7.2, 7.1** |
| keep | r4 | 93.8, 94.2 | 34.5, 34.3 | 13.8, 14.8 | 10.1, 10.8 |
| keep | this round | 96.2, 94.8 | 34.5, 34.4 | 15.1, 14.7 | 11.0, 10.9 |

(series 1, series 2; the tails are the drop client's, the keep client's being the same
messages after its live prefix: +1648, +24 and +18 tokens.)

With r4 a client that drops the reasoning waited 97-100 s for every turn at this length
(the whole conversation, `[PREFIX] no reuse`); now each turn resumes from the photo of the
turn before (`[PIN] resumed from the photo of 25054 tokens, 1647 to prefill`, then of 26701
with 23 to prefill, of 26724 with 17) and takes what a client that keeps the reasoning
takes, or less. The keep client's turns are r4's: in series 2, where the two keep runs came
last, 14.74 and 10.85 s against 14.76 and 10.79 (series 1's gap is the machine's drift over
the series; its new keep run came last).

**Every turn returned the same reasoning, content and tool calls in both engines** --
r4 computing the whole conversation, this round resuming from a photo -- and in both series
(`summary.json`: one hash per turn, `44c37556...` for turn 1, `f23d2b1b`, `a0cfa9b2`,
`514b6fd9` for the drop client's turns 2-4, `5f0f0653`, `ecef578e`, `9c476085` for the keep
client's).

Short tails are faster resumed from a photo than on the live prefix here: their rows run on
the GPU's prompt kernels (below), where the live prefix's few rows take their attention over
26,800 keys on the CPU (turn 3: attention 0.55 s against 7.1-8.3 s, experts on the GPU 9.5 s
against 3.3-3.4 s and 3.1-3.3 s stalled). At 4318 tokens it is the other way round (`runs/tail-*`, a
35-row tail resumed from a photo, two runs each): 11.66 and 11.61 s on the prompt kernels,
8.05 and 8.07 s with `MIMO_PROMPT_KERNELS=0` (the step-by-step choice, not the cold prompt's
bytes): the GPU's experts move every expert those rows route to over PCIe.

## What changed

- **Turn photos** (`c/mimo.c`, "TURN PHOTOS"): every prompt that generates also takes a
  photo of its end, as `SUBMIT pin=1` does for System One -- the 60 windowed layers' rings
  (about 80 MB on Pro), the token ids (the full layers' rows stay where they are), the
  logits -- into a pool of its own (`MIMO_TURN_PHOTOS`, 2): System One's photos stay as many
  as `COLI_PIN_SLOTS` says, and a read-out (`logprobs=k`) neither takes one nor resumes from
  one. A prompt that the live prefix does not cover resumes from the deepest photo, of
  either pool, that is a prefix of it and whose rows the state still holds. Nothing changed
  in the gateway: the engine takes the photo, for every endpoint and client.
- **The MTP layers**: their windows hold rows of the reply the next turn threw away. A
  photo put back empties them, as `model_reset` does, and `mtp_warm` fills them again before
  the first draft; the photo also holds the trunk's last hidden rows, so that `mtp_warm`
  fills them from the positions it fills them from after the prompt computed cold, and the
  drafts are the same (with fewer than 128 new tokens they would otherwise see only those).
- **A prompt's kernels** (`MIMO_PROMPT_KERNELS`, 1): on the GPU a step's kernels depend on
  its rows: 64 or more take the prompt kernels (tiled matrices, the GPU's experts and
  routers; 32 for the attention), fewer the decode path. A tail of a few dozen rows would
  therefore not be the cold prompt's bytes (on the tiny fixture: logits 1.7e-4 apart), and
  neither would a block's last attention piece of fewer than 64 rows (a prompt's blocks go
  in pieces of 1024). Now the rows of a prompt that starts from nothing or from an exact
  photo choose by the prompt's length; a smaller matrix call goes up padded to 64 rows,
  whose tiled kernel computes each row by itself, and the head still takes the last row
  alone. A live prefix's tail keeps its steps' own kernels, as before: its reply came from
  decode steps, so no choice makes it the cold prompt.
- **Exact photos only**: a turn photo is taken of a prompt whose rows are the bytes a
  longer prompt computed cold gives them -- not one that went on from the live prefix, and
  on the GPU not one shorter than 64 tokens (`COLI_PREFIX_LOG=1` says `[PIN] no turn
  photo`). With `MIMO_PROMPT_KERNELS=0` every turn takes one.
- `MIMO_PROMPT_LOGITS=<file>` appends each served prompt's last logits, what the checks
  below compare.

The second stage the task allowed for -- keeping a conversation's full-layer rows (2.55 GB
at 25,000 tokens) when a side request with another system prompt (a title, a summary)
rewrites them from position 0 -- is not done: these runs have no side request, and whether
the agents in use send them is not measured yet.

## Correctness

- Through the gateway: every turn of the eight runs above, as said, the same bytes in r4
  and this round, which in drop mode means a turn resumed from a photo against the same
  conversation computed whole.
- Over the SERVE pipe (`turns.py`, `plan-exact.json`, `runs/exact`): a 4318-token
  conversation, then the turn after it -- a short tail (35 tokens) and a long one (1830) --
  resumed from the photo, then the same prompt computed cold after an unrelated prompt (whose
  prefill releases the photo): the logits after the prompt the same bytes
  (`792181ec...`, `0fda4a84...`), the 32-token replies the same bytes; 11.6 s against 39.1 s,
  34.1 s against 43.2 s. The same with the round's first build (`runs/dev-exact-a90970df`).
- `check_mimo_cuda.py` with this round's engine (`correctness/`): on the 17-token
  benchmark prompt the CPU logits `cebbf3a8...` and the CUDA ones `01479691...`, the
  replies `63ed9984...`; on the 658-token correctness prompt in 256-row blocks
  `874ee6e8...` and `03ce216b...`, the replies `f0aaa17f...` -- r4's bytes, as in every
  round. (A prompt's kernels change nothing there: their blocks take the prompt kernels
  whole. Where a block of more than 1024 rows ends in an attention piece of fewer than 64,
  that piece's rows now take the prompt kernels where r4 gave them the decode path: the
  same quantities in another order. The 25,054-token turn 1 above ends in one of 478 and
  is r4's bytes.)
- The decode benchmark (`benchmark/`, 64 tokens after a 17-token prompt, prefix reuse off
  as always, so no photo; a warmup and three runs an engine, r4's and this round's
  alternating): 2.667 and 2.667 tok/s against 2.671 and 2.669 (decode 3.321 and 3.321
  against 3.322 and 3.319), every reply `0305abcd...`.
- The tiny fixture (`c/tests/test_mimo_prefix_serve.py`, `correctness/tiny-*.txt`):
  the 14 prefix and System One tests as before, and turn photos against a cold engine --
  an agent loop of five turns that drops the reasoning, each turn resumed from the photo of
  the one before with its logits, reply and every MTP step (drafts made, verified and kept,
  their probabilities) the cold prompt's; one, two and three drafts verified whatever their
  probability, with tails longer and shorter than the 8-token window (the hidden rows'
  test: it fails without them); System One's one-slot pool kept, a read-out never resumed
  from a turn photo, a repeated turn resumed from the one before, a picture, the switches.
  With `MIMO_CUDA_TEST=1` the same loop on the GPU, also in 64-row attention pieces, the
  cold prompt's bytes; with `MIMO_PROMPT_KERNELS=0` a resumed tail is not; a live prefix's
  tail keeps its own kernels. The vendor-oracle harness (23 checks, MTP included), the
  Brio and System One gateway tests, the CUDA fallback tests and the sampled-draft tests
  pass.

## The release's build (`release-build/`)

Release `mimo-pro-4090-r5` was built from the merge into main (`f375866f`): its `mimo.exe`
(4b098911) and `coli_cuda_mimo.dll` (16d2f342) are other bytes than the measured ones (main
carries other engines' changes; the DLL is never the same bytes). Run the same way
(`release-build/run_release.sh`), they gave the measured build's CPU and CUDA logits and
replies on both correctness prompts byte for byte, the resumed turns' logits and replies
(`792181ec...`, `0fda4a84...`; 11.6 against 39.1 s and 33.9 against 43.1 s), the agent
loop dropping the reasoning in 94.4, 34.6, 10.3 and 7.1 s with every turn's hash the
measured one's (`same-bytes.json`), and the decode benchmark at 2.676 and 2.664 tok/s, the
measured build between them at 2.667, every reply `0305abcd...`. `make test-c` passes on the
merge (without `CUDA_DLL=1`, in its own worktree).

## Files

`summary.json` and `summary.txt` (every turn of both series: times, prompt tokens, how its
prompt started, the hash of what it returned), `series1/`, `series2/` (each run's
`turns.json`, the gateway's log with the engine's lines, the card's telemetry), `runs/`
(the SERVE-pipe runs: `results.json`, the engine's log, `compare.json`, `logits.sha256`),
`correctness/`, `benchmark/`, `release-build/`, the scripts (`agent_loop.py`, `api_run.py`,
`run_api.sh`, `summarize.py`, `turns.py`, `make_plans.py`, `compare.py`, `run_tail.sh`,
`run_checks.sh`) and the plans, `sha256.json`.

## Reproduce

```sh
c/build_mimo_cuda.bat
mingw32-make -C c mimo CUDA_DLL=1 ARCH=native VK=0
python c/tests/test_mimo_prefix_serve.py
MIMO_CUDA_TEST=1 python c/tests/test_mimo_prefix_serve.py MimoTurnPhotoCudaTest
python c/tests/mimo_tiny_harness.py --binary c/mimo.exe --fixture c/mimo_tiny
cd docs/experiments/mimo-turn-photos-4090-2026-10-21
python make_plans.py
python turns.py --engine /c/colibri/dev/c/mimo.exe --plan plan-exact.json --out runs/exact && python compare.py runs/exact
sh run_tail.sh
sh run_api.sh <r4 mimo.exe> /c/colibri/dev/c/mimo.exe series1 forward
sh run_api.sh <r4 mimo.exe> /c/colibri/dev/c/mimo.exe series2 reverse
python summarize.py series1 series2
sh run_checks.sh <r4 mimo.exe> /c/colibri/dev/c/mimo.exe
```

Engine paths are absolute (MSYS paths from Git Bash are converted for Windows). r4's engine
here was built from `9b043a09` in a worktree; builds of the same code differ only in the PE
header's link time and checksum (this round's 6e75a032 and b0f6da69: 4 bytes).
