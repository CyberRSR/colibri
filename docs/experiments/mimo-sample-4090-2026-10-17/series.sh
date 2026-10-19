#!/bin/bash
# The sampler series: timed ABA arms first (eight varied prompts sampled at temperature 0.7,
# top_p 0.9, then the benchmark, greedy), then the untimed greedy check of the after binary
# over the eight prompts. Sequential: nothing else heavy runs meanwhile.
cd /c/colibri/dev
T=c/build/sample-draft
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
varied() {  # name binary-dir tokens temperature top_p [env...]
  name=$1; bin=$2; tokens=$3; temp=$4; topp=$5; shift 5
  args=(--env SEED=20261017)
  for e in "$@"; do args+=(--env "$e"); done
  echo "=== $(date +%H:%M:%S) varied $name ($bin) tokens $tokens T=$temp top_p=$topp ${args[*]}"
  python $V/drive_mimo.py --engine C:/colibri/dev/$T/$bin/mimo.exe --model $MODEL --shards $SHARDS --cap 104 --tokens $tokens \
    --temperature "$temp" --top-p "$topp" "${args[@]}" --json "$S/$name.json" --log "$S/$name-engine.txt" $V/prompts.txt > "$S/$name.txt" 2>&1
  echo "    exit $? $(date +%H:%M:%S): $(tail -1 "$S/$name.txt")"
}

varied s-before-1 bin-before 64 0.7 0.9
varied s-after bin-after 64 0.7 0.9
varied s-before-2 bin-before 64 0.7 0.9
bench b-before-1 bin-before
bench b-after bin-after
bench b-before-2 bin-before
echo "=== $(date +%H:%M:%S) TIMED SERIES DONE"

varied g-after bin-after 64 0 1
echo "SERIES DONE $(date +%H:%M:%S)"
