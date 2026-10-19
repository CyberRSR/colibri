#!/bin/bash
# The ABA series: the eight varied prompts greedy (one cold engine each arm), then the benchmark,
# each before / after / before, binaries in their own folders beside a copy of the CUDA DLL.
# Sequential: nothing else heavy runs meanwhile.
cd /c/colibri/dev
T=c/build/io-prefetch
S=$T/series
mkdir -p $S
V=docs/experiments/mimo-mtp-4090-2026-10-17/varied-prompts
MODEL=D:/AI/Models/MiMo-V2.6-Pro-MOPD
SHARDS=C:/AI/MiMoPro-Shards
sha256sum $T/bin-before/mimo.exe $T/bin-after/mimo.exe $T/bin-before/coli_cuda_mimo.dll $T/bin-after/coli_cuda_mimo.dll > $S/binaries.sha256
nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv > $S/gpu-before.csv

bench() {   # name binary-dir
  echo "=== $(date +%H:%M:%S) benchmark $1 ($2)"
  python c/tools/benchmark_mimo_cuda.py --engine $T/$2/mimo.exe --model $MODEL --shards $SHARDS --backend cuda --release-host \
    --cap 104 --threads 16 --read-threads 16 --context 16384 --tokens 64 --runs 3 --output "$S/$1" > "$S/$1.stdout" 2>&1
  echo "    exit $? $(date +%H:%M:%S)"; tail -2 "$S/$1.stdout"
}
varied() {  # name binary-dir
  echo "=== $(date +%H:%M:%S) varied $1 ($2)"
  python $V/drive_mimo.py --engine C:/colibri/dev/$T/$2/mimo.exe --model $MODEL --shards $SHARDS --cap 104 --tokens 64 \
    --env SEED=20261017 --json "$S/$1.json" --log "$S/$1-engine.txt" $V/prompts.txt > "$S/$1.txt" 2>&1
  echo "    exit $? $(date +%H:%M:%S): $(tail -1 "$S/$1.txt")"
}

varied g-before-1 bin-before
varied g-after bin-after
varied g-before-2 bin-before
bench b-before-1 bin-before
bench b-after bin-after
bench b-before-2 bin-before
echo "SERIES DONE $(date +%H:%M:%S)"
