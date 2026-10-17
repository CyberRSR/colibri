#!/bin/bash
# The ABA series of exp_f on HEAD a495a0c0 (code ca05d0e4): the eight varied prompts greedy (one
# cold engine each arm), then the benchmark, each before / after / before; then the benchmark
# with MinGW's __expl_internal moved to a slow address (a pad object after omp_spin.o): HEAD's
# code / exp_f's / HEAD's. Binaries in their own folders beside a copy of the CUDA DLL.
# Sequential: nothing else heavy runs meanwhile.
cd /c/colibri/dev
T=c/build/layout
S=$T/series
mkdir -p $S
V=docs/experiments/mimo-mtp-4090-2026-10-17/varied-prompts
MODEL=D:/AI/Models/MiMo-V2.6-Pro-MOPD
SHARDS=C:/AI/MiMoPro-Shards
sha256sum $T/bin-*/mimo.exe $T/bin-*/coli_cuda_mimo.dll > $S/binaries.sha256
nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv > $S/gpu-before.csv

bench() {   # name binary-dir
  echo "=== $(date +%H:%M:%S) benchmark $1 ($2)"
  python c/tools/benchmark_mimo_cuda.py --engine $T/$2/mimo.exe --model $MODEL --shards $SHARDS --backend cuda --release-host \
    --cap 104 --threads 16 --read-threads 16 --context 16384 --tokens 64 --runs 3 --output "$S/$1" > "$S/$1.stdout" 2>&1
  echo "    exit $? $(date +%H:%M:%S)"; tail -2 "$S/$1.stdout"
  grep -E "generation" "$S/$1/engine.log" | tail -1 | sed 's/^/    /'
}
varied() {  # name binary-dir
  echo "=== $(date +%H:%M:%S) varied $1 ($2)"
  python $V/drive_mimo.py --engine C:/colibri/dev/$T/$2/mimo.exe --model $MODEL --shards $SHARDS --cap 104 --tokens 64 \
    --env SEED=20261017 --json "$S/$1.json" --log "$S/$1-engine.txt" $V/prompts.txt > "$S/$1.txt" 2>&1
  echo "    exit $? $(date +%H:%M:%S): $(tail -1 "$S/$1.txt")"
}

if tasklist | grep -qi "^mimo.exe"; then echo "another mimo.exe runs; stop"; exit 1; fi
varied g-before-1 bin-before
varied g-after bin-after
varied g-before-2 bin-before
bench b-before-1 bin-before
bench b-after bin-after
bench b-before-2 bin-before
echo "=== $(date +%H:%M:%S) TIMED SERIES DONE"
bench b-bad-1 bin-bad
bench b-after-bad bin-after-bad
bench b-bad-2 bin-bad
echo "SERIES DONE $(date +%H:%M:%S)"
