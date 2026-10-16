# MiMo-V2.6-Pro: expert streaming on an RTX 4090

Measured on 2026-10-16 on the machine of the
[CUDA dense experiment](../mimo-cuda-4090-2026-10-16/README.md), native Windows,
the complete 70-layer checkpoint, the same launcher profile (CUDA dense budget
20 GiB, host copies released, expert cache 104 per MoE layer, 16 CPU workers,
16384-position context) and the same benchmark command.

## Results

Three arms run back to back: the measured release engine of the CUDA experiment,
then the engine of commit `845474fc` with its defaults, and the same binary with
the prefetcher off.

| Arm | Overall tok/s | Decode tok/s | First token, s | Expert RAM hits |
|---|---:|---:|---:|---:|
| release engine (`66d529bd`) | 1.156 | 1.356 | 8.89 | 69.8% |
| streaming, prefetch off | 1.587 | 1.784 | 5.00 | 69.8% |
| **streaming, defaults** | **1.709** | **1.942** | **5.00** | 86.1% |

**+47.9% overall and +43.3% decode** against the release engine; the first token
comes 44% sooner. The prefetcher alone is +7.7% overall over the same binary
without it. Raw samples: release 1.1524, 1.1561, 1.1559; streaming 1.7065, 1.7152,
1.7092; no prefetch 1.5825, 1.5871, 1.5880 tok/s. **Every measured reply of every
arm is the same bytes** (SHA-256 `0305abcd...`, the reply of every arm of the CUDA
experiment). The hit rate with the prefetcher counts the experts it had read or
was reading when the step asked for them.

Where a request's 37.4 s went with the defaults (the release engine's 55.3 s in
brackets), from the engine's own counters: experts waiting for the disk 5.6 s
(22.0), expert computation 20.2 s (16.7), attention 9.9 s (15.1) of which the qkv
matmuls 1.2 s and o_proj 7.5 s. The expert computation is longer than before
because the disks now write into RAM while the experts are computed: the 7950X3D's
DDR5 gives about 65 GB/s, measured, and 17 GB/s of reads take their share of it.

## What changed

**Reads overlap computation.** Before, a MoE step read its missing experts in
parallel, waited for all of them, then computed all eight. Now a pool of readers
runs beside the computation, and a step computes the experts the cache holds at
once and then each wave of arrivals (`moe_cpu`).

**Readers per device, demands first.** The release's shards sit on two devices:
`C:` (one PCIe 5.0 NVMe, 88 shards, about 10.5 GB/s for 19 MB unbuffered reads)
and `D:` (three PCIe 4.0 NVMe striped behind the chipset, 42 shards, about 6.5 GB/s:
the chipset link, not the drives). Each device gets its own queues and two readers;
a reader takes demands before prefetches, so a demand waits behind at most two
reads of its own device. One queue of 16 readers made every read share the
bandwidth with every other, demands with prefetches, and was slower than the
release (1.076 tok/s). On Windows each reader opens its own unbuffered handle per
shard: reads through one synchronous handle are serialized by its file object.

**The next layer's experts are read ahead.** After a layer's attention, the next
MoE layer's router is applied to the residual (normed by that layer's
post-attention norm), and the experts it ranks top-8 are queued as prefetches.
77.3% of the experts the decode steps chose were predicted; a prediction that does
not come true costs a read and a slot that is the first to be reused. Predicting
again after the MLP raised that to 85.0% but read more and was slower (1.475
against 1.490 in its pair); so were two extra experts per prediction (1.450) and
24 workers (1.460). See `development.json`.

**The trunk's CUDA matmuls run on wide kernels.** The FP8 qkv took 0.63 ms per
decode call and 9.2 ms per 17-row prompt chunk (each row re-read the matrix);
it takes 0.22 and 1.7 ms. Measured with the engine's own `mimo_cuda_matmul`, copies
included: o_proj and the head's prompt chunks are 4-5x faster too.

**The attention's host part.** RoPE made its cosines and sines once per head (136
heads a row); it makes them once per row. The sliding window's ring was copied,
1.3 MB a layer and a token, into a scratch buffer; it is read in place. The routers
keep their released bf16 bytes instead of an f32 copy: the same scores from half
the reads. Together they took the attention's non-matmul time from about 4.1 s to
1.1 s a request.

## Correctness

- **The CPU logits on the real model are unchanged to the bit.** The 70-layer
  CPU run of `check_mimo_cuda.py` gives the logits file of the CUDA experiment's
  two correctness runs, SHA-256 `cebbf3a8...`, with another cache size (64 against
  104 and 88) and another thread count (16 against 24). The streaming changes
  when bytes arrive and which experts stay cached, never which numbers are computed.
- The CUDA path against the CPU path, full model (`correctness/`): 17/17 argmax,
  maximum absolute logit difference 6.5e-5 (5.3e-5 with the narrow kernels),
  maximum KL 1.6e-11, the 32-token greedy reply byte-identical.
- The tiny oracle (Xiaomi's modeling code) passes with the prefetcher off and on,
  with no reader thread, with CUDA, and with `MIMO_CACHE_DECAY`; its logits are
  byte-identical to the engine's before these changes, native and f32 dense, with
  prompt blocks of 7 across the 8-row window.
- The 14 serve/prefix tests and the 6 CUDA fallback tests pass. The CUDA unit test
  covers both wide kernels at shapes from 512 to 6144 columns and chunks of 3 to
  17 rows: chunked results are byte-identical to one-row calls.

## Measurement protocol and limits

As in the CUDA experiment: one persistent engine per arm, one excluded warmup,
three identical greedy 64-token requests, prefix reuse off, the 17-token prompt
"Explain briefly what binary search is." Arms ran sequentially. The release arm's
binary is the one the CUDA experiment measured (engine SHA-256 `41c8b2aa...`); its
first series of the day gave 1.146 tok/s, this series 1.156.

Repeating one prompt favors a cache: the routing of request 2 is the routing of
request 1. The prefetcher's predictions come from the same routing, so they do
not profit from the repetition the same way, but the hit rates here are those of
a repeated prompt, not of a varied workload. This is one machine; the per-device
queues pay most where the shards span devices of different speeds.

## Files

`summary.json` (every arm's medians, profile counters, the engine's streaming and
phase lines, hashes), the arms' raw samples, replies, GPU telemetry and engine
lines, `stream-vs-release.json` and `prefetch.json` (one-variable manifests,
`c/experiment_manifest.py`), `development.json` (the arms measured on the way, from
working-tree states that were not committed), `correctness/`, `sha256.json`.

## Reproduce

```powershell
git checkout 845474fc
c/build_mimo_cuda.bat
mingw32-make -C c mimo CUDA_DLL=1 ARCH=native VK=0
python c/tools/benchmark_mimo_cuda.py --engine c/mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --backend cuda --release-host --cap 104 --threads 16 --read-threads 16 --context 16384 --tokens 64 --runs 3 --output c/build/mimo-stream/stream
python c/tools/benchmark_mimo_cuda.py ... --env MIMO_PREFETCH=0 --output c/build/mimo-stream/stream-noprefetch
python c/tools/check_mimo_cuda.py --engine c/mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --release-host --cap 64 --threads 16 --output c/build/mimo-stream/correctness
python c/experiment_manifest.py docs/experiments/mimo-stream-4090-2026-10-16/stream-vs-release.json docs/experiments/mimo-stream-4090-2026-10-16/prefetch.json
```

`--read-threads` is still recorded in the config; the engine no longer reads it
(`MIMO_IO_PER_DRIVE`).
