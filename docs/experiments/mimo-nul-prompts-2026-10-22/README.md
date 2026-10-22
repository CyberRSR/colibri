# MiMo-V2.6-Pro: prompts holding NUL bytes

Checked on 22 October 2026 on the machine of the earlier MiMo experiments (Ryzen 9
7950X3D, 192 GB DDR5, RTX 4090, the model's shards on two NVMe volumes).

## What was wrong

An agent that reads a binary file puts its bytes, NULs and all, into a tool turn. The
gateway refused every prompt holding a NUL with 400 "NUL bytes are not supported in
prompts." (`Engine.generate` in `c/openai_server.py`), whatever engine served it, and the
agent's turn ended there: a coding agent's request of 11 messages, 38,974 prompt tokens,
whose last two tool turns held an executable's strings (168 NULs), was answered 400.

MiMo's engine never needed that: both serve loops tokenize a `SUBMIT` payload by its byte
count (`mimo_serve_start` -> `tok_encode(text, len)`), the SERVE reader takes the payload
by its length, and the byte-level tokenizer maps byte 0 to its own token like any other.

## What changed (`97917a18`)

- `c/mimo.c`: the handshake says `CAPS vision=<0|1> nul=1`.
- `c/openai_server.py`: `Engine.nul_prompts` is the engine's `nul=1`; `generate()`
  refuses a prompt holding NUL only to an engine that did not say it. Every other engine
  keeps the 400 rather than risk a prompt cut at its first NUL.
- `docs/serve_protocol.md`, `docs/mimo.md`: the key and what it means.

## Checks

- **The tokenizer against the reference** (`nul_cases.py`, `tok_nul_check.c`,
  `cases.txt`, `tokenizer-check.txt`): HF `tokenizers` 0.23.3 on MiMo-V2.6-Pro's
  `tokenizer.json` (NFC normalizer, byte-level BPE) against the engine's `tok_encode`,
  fed by byte count as the serve path does: NUL alone and in runs, before and after
  spaces, letters, digits, contractions, CR/LF and tabs, the other C0 controls and DEL,
  noncharacters and lone continuation characters, next to `<|im_start|>`/`<|im_end|>`,
  inside Cyrillic and CJK words: 24 of 24 the same ids, every one decoding back to its
  bytes. The two tool turns of the refused request (not published here: a third party's
  binary) gave the same ids as well, 26 of 26 in all.
- **The refused request, replayed** through the gateway with the fix and r5's other
  files: 200, the reply reading the strings out of the tool turns; the engine's account
  `[PREFIX] no reuse, 0 of 38974 prompt tokens`, then `[PIN] turn photo of 38974 tokens`.
- **Tests**: `tests/test_openai_server.py` (a NUL goes out byte for byte to an engine
  that says `nul=1`, and to one that says nothing or only `vision`, nothing is written
  and the answer is 400 on `messages`; 425 passed), `tests/test_mimo_prefix_serve.py`
  (the tiny fixture's engine says `nul=1`, and with one token a byte every NUL is a
  position of its own: the read-out covers all of them and a reply follows), with the
  Brio and System One serve tests: 55 passed, the 9 CUDA ones skipped.

## The release's build (`release-build/`)

Release `mimo-pro-4090-r6` was built from the merge into main (`798c0c9b`): `mimo.exe`
4be4fbcd, and r5's `coli_cuda_mimo.dll` (16d2f342; `backend_cuda_mimo.cu` has not changed
since `4b8bd1c8`). `__expl_internal` at ...5c0, off the slow page slots. On the 17-token
correctness prompt (`check_mimo_cuda.py --release-host --cap 64 --threads 16`) its CPU and
CUDA logits are r5's bytes (`cebbf3a8...`, `01479691...`) and so are both replies. The
decode benchmark (64 tokens after that prompt, a warmup and three runs): 2.657, 2.662 and
2.663 tok/s (decode 3.302 - 3.307), against 2.667 for r5's measured build; every reply
`0305abcd...`.

## Commands

```sh
python nul_cases.py D:/AI/Models/MiMo-V2.6-Pro-MOPD/tokenizer.json cases.txt   # needs `tokenizers`
gcc -O2 -I c docs/experiments/mimo-nul-prompts-2026-10-22/tok_nul_check.c -o tok_nul_check
./tok_nul_check D:/AI/Models/MiMo-V2.6-Pro-MOPD/tokenizer.json docs/experiments/mimo-nul-prompts-2026-10-22/cases.txt
python c/tools/check_mimo_cuda.py --engine c/mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --release-host --cap 64 --threads 16 --output release-build/check17
python c/tools/benchmark_mimo_cuda.py --engine c/mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --backend cuda --release-host --cap 104 --threads 16 --read-threads 16 --context 16384 --tokens 64 --runs 3 --output release-build/benchmark
```
