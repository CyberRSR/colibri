#!/bin/bash
# rowtest.sh m...: the engine over a short prompt in steps of m rows (K3_CHUNK=m), every
# position's logits written (K3_LOGITS); then each run's file against the first's, byte for byte.
SP="$(cd "$(dirname "$0")" && pwd)"
ENG="${K3_ENGINE_DIR:-$(cd "$SP/../../../../c" && pwd)}"   # the folder with kimi_k3.exe and its DLLs
SP="$SP/runs"; mkdir -p "$SP/rowtest"
printf 'Write a Python function that returns the n-th Fibonacci number, then explain in two sentences why the iterative form is faster than the recursive one.' > "$SP/rowtest/prompt.txt"
export K3_DIRS='D:\AI\Models\Kimi-K3.1-Pro' K3_CUDA=0 K3_GPU_EXPERTS=1 K3_READAHEAD_GB=20 K3_CUDA_DENSE=1 \
       K3_CUDA_DENSE_GB=20 K3_CUDA_RESERVE_GB=2 K3_CUDA_RELEASE_HOST=1 K3_EXPERT_GB=1000 RAM_GB=174 \
       K3_MAXT=131072 K3_DIRECT=1 K3_PIPE=1 K3_IO_PER_DRIVE=2 K3_PREFETCH=1 OMP_NUM_THREADS=16 \
       COLI_NO_OMP_TUNE=1 AUTOPIN=0 USAGE_SAVE=0 COLI_VULKAN=0 COLI_CUDA=0 COLI_GPU=0 K3_STAT_EVERY=8
cd "$ENG" || exit 1
first=""
for m in "$@"; do
    K3_CHUNK=$m K3_LOGITS="$(cygpath -w "$SP/rowtest/logits_m$m.bin")" ./kimi_k3.exe "C:/AI/Kimi-K3.1-Pro" --chat \
        --prompt-file "$SP/rowtest/prompt.txt" --ngen 0 > "$SP/rowtest/m$m.out" 2> "$SP/rowtest/m$m.err"
    printf "m=%s rc=%s: " "$m" "$?"; grep -h -E 'prompt: [0-9]+ tokens|prefill done' "$SP/rowtest/m$m.err" | cut -c1-120 | tr '\n' ' '
    ls -la "$SP/rowtest/logits_m$m.bin" | awk '{printf "%s bytes", $5}'
    if [ -z "$first" ]; then first=$m; echo
    else
        if cmp -s "$SP/rowtest/logits_m$first.bin" "$SP/rowtest/logits_m$m.bin"; then echo "  = m=$first, byte for byte"
        else echo "  DIFFERS from m=$first: first differing byte $(cmp "$SP/rowtest/logits_m$first.bin" "$SP/rowtest/logits_m$m.bin" | awk '{print $5}')"; fi
    fi
done
echo rowtest done
