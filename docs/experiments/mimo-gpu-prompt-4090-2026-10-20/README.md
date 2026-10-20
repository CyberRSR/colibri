# MiMo-V2.6-Pro on an RTX 4090: prompts on the GPU

Measured on 20 October 2026 on the machine of the earlier MiMo experiments (Ryzen 9
7950X3D, 192 GB DDR5, RTX 4090, the model's shards on two NVMe volumes) with the RTX 4090
profile (CUDA dense budget 20 GiB, host copies released, 104 experts a layer in one pool,
16 CPU workers, two readers a drive). `long_prompt.py` sends each prompt in `prompts/` as
one request over the engine's SERVE pipe (thinking off, 16 greedy tokens, prefix reuse
off) and times its first token. The engine's own line splits the prompt's time.

## Results

| Prompt | Engine | First token | Experts | Attention |
|---|---|---:|---:|---:|
| 4310 tokens | before: attention and experts on the CPU | 413.9 s | 251.3 s | 123.8 s |
| 4310 tokens | the attention on the GPU only | 360.4 s | 251.5 s | 70.1 s |
| 4310 tokens | **the defaults** (experts, attention, matrices, routers on the GPU) | **68.6 s** | 37.1 s | 20.0 s |
| 24,957 tokens | **the defaults** | **244.8 s** | 87.4 s | 136.9 s |

4310 tokens: 6.0x. The 24,957-token prompt was not run on the CPU (about an hour by the
4310-token run's parts: its experts scale with the prompt, its full-attention scores with
the square of it); the user's agents had reported about 40 minutes for prompts of that
size. On the GPU it went in as one block (`[mimo] prompt blocks on the GPU: up to 32768
rows`), its 531 GB of experts read once.

Where the 24,957-token prompt's 244.8 s go: the experts 87.4 s (the GPU's MXFP4 kernels at
about 12 TFLOPS: 1.04 PFLOP), the qkv 57.8 s and o_proj 23.7 s (their rows go down to the
host and back up between the matrices and the attention), the attention's scores and values
about 40 s, the routers 6.4 s.

The decode benchmark of `benchmark/` (64 tokens after a 17-token prompt, which the GPU
prompt path does not take): 2.668 tok/s, decode 3.32, every reply `0305abcd...` as before.

## Correctness

- Against the CPU path on a 658-token prompt in 256-row blocks (`correctness/`,
  `check_mimo_cuda.py --chunk 256`): the same argmax at all 658 positions, max |logit
  difference| 0.0085, max KL 8.7e-7, the 32-token greedy reply the same bytes. (Before the
  routers went to the GPU: 0.0041 and 2.0e-7.)
- The same tool on the 17-token benchmark prompt, whose blocks stay on the decode path:
  CPU logits SHA-256 `cebbf3a8...` and CUDA `01479691...`, as before.
- The 4310-token prompt's reply was the CPU engine's until the routers went to the GPU;
  then it reads "On an RTX 4090, the CUDA path speeds up Mi" instead of "... moves MiMo":
  the same quantities summed in another order choose another expert where two are as good.
- `tests/test_mimo_cuda.cu`: the tiled FP8, bf16 and packed bf16 kernels against a double
  reference (most weights exceptions included) and a row's bytes across blocks; the tiny
  harness with every GPU prompt path forced on, pieces of 3 expert rows and 50 KB of matrix
  scratch.

## On the way (`development.json`)

- The first GPU experts went up in a fixed order and deadlocked when the layer's slots were
  pinned by experts later in that order; now each goes as its bytes arrive, its outputs into
  the (row, choice) pairs moe_cpu fills, and each row sums them in its routing order.
- In 512-row blocks the GPU experts gained nothing (362.9 s): each block reads every expert
  again (3.1 TB for 4310 tokens). 4096-row blocks: 135.1 s; one block: 96.0 s.
- Tiled kernels for the prompt matrices: qkv 24.1 -> 9.4 s on the 4310-token prompt.
- A 12,500-row block's dense MLP wanted 1.6 GB of scratch and the card paged its memory
  (5 GB "shared"): matrix calls now go in pieces of 192 MB, an expert's rows in pieces of
  1024, and the block's size comes from the card's free memory.

## Files

`summary.json` (the table), each run's `results.json` and `engine.txt`, `prompts/`,
`long_prompt.py`, `correctness/`, `benchmark/`, `development.json`, `sha256.json`.

## Reproduce

```powershell
c/build_mimo_cuda.bat
mingw32-make -C c mimo CUDA_DLL=1 ARCH=native VK=0
python docs/experiments/mimo-gpu-prompt-4090-2026-10-20/long_prompt.py --engine c/mimo.exe --prompt docs/experiments/mimo-gpu-prompt-4090-2026-10-20/prompts/p4k.txt --prompt docs/experiments/mimo-gpu-prompt-4090-2026-10-20/prompts/p25k.txt --tokens 16 --env MIMO_CHUNK= --out c/build/gpu-prompt
python c/tools/check_mimo_cuda.py --engine c/mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --release-host --prompt-file docs/experiments/mimo-gpu-prompt-4090-2026-10-20/correctness/prompt.txt --tokens 32 --chunk 256 --output c/build/gpu-prompt-check
```
