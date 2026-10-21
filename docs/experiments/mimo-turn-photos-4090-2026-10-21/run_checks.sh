#!/bin/sh
# check_mimo_cuda on the two correctness prompts with this round's engine, then the decode
# benchmark, r4's engine and this round's alternating. usage: run_checks.sh BASE_ENGINE NEW_ENGINE
cd "$(dirname "$0")"
base=$1; new=$2
R=../../..
P=../mimo-gpu-prompt-4090-2026-10-20/correctness/prompt.txt
python $R/c/tools/check_mimo_cuda.py --engine "$new" --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --release-host --cap 64 --threads 16 --output correctness/check17 || exit 1
python $R/c/tools/check_mimo_cuda.py --engine "$new" --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --release-host --prompt-file $P --tokens 32 --chunk 256 --output correctness/check658 || exit 1
for arm in 1-base:$base 2-new:$new 3-base:$base 4-new:$new; do
  python $R/c/tools/benchmark_mimo_cuda.py --engine "${arm#*:}" --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --backend cuda --release-host --cap 104 --threads 16 --read-threads 16 --context 16384 --tokens 64 --runs 3 --output "benchmark/${arm%%:*}" || exit 1
done
