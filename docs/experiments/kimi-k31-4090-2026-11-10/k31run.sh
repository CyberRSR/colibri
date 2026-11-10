#!/bin/bash
# k31run.sh NAME NGEN [VAR=value ...] -- one real-model run of kimi_k3 on Kimi K3.1 Pro with the
# 4090 launcher's settings: stdout/stderr and a 2 s monitor CSV under runs/NAME.*; extra
# VAR=value override the profile. K3_EXE picks the engine (default: stage/kimi_k3.exe),
# K3_RUN_PROMPT the chat prompt, K3_RUN_PROMPT_FILE a file holding it.
SP="${K31_SP:-$(cd "$(dirname "$0")" && pwd)}"   # the folder with stage/ (the engine and its DLLs), runs/, ref/ and the prompts
name=$1; ngen=$2; shift 2
export K3_DIRS="D:\\AI\\Models\\Kimi-K3.1-Pro" K3_CUDA=0 K3_GPU_EXPERTS=1 K3_READAHEAD_GB=20 K3_CUDA_DENSE=1 \
       K3_CUDA_DENSE_GB=20 K3_CUDA_RESERVE_GB=2 K3_CUDA_RELEASE_HOST=1 K3_EXPERT_GB=1000 RAM_GB=174 \
       K3_MAXT=131072 K3_CHUNK=8192 K3_DIRECT=1 K3_PIPE=1 K3_IO_PER_DRIVE=2 K3_PREFETCH=1 OMP_NUM_THREADS=16 \
       COLI_NO_OMP_TUNE=1 AUTOPIN=0 USAGE_SAVE=0 COLI_VULKAN=0 COLI_CUDA=0 COLI_GPU=0 K3_STAT_EVERY=8
for kv in "$@"; do export "$kv"; done
prompt=${K3_RUN_PROMPT:-"What is the capital of France? Answer in one short sentence."}
exe=${K3_EXE:-$SP/stage/kimi_k3.exe}
cd "$(dirname "$exe")" || exit 1
rm -f "$SP/runs/$name.stop"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$SP/monitor.ps1" -Out "$SP/runs/$name.csv" -StopFile "$SP/runs/$name.stop" -Seconds 2 &
t0=$(date +%s%N)
if [ -n "$K3_RUN_PROMPT_FILE" ]; then
    "$exe" "C:/AI/Kimi-K3.1-Pro" --chat --prompt-file "$K3_RUN_PROMPT_FILE" --ngen "$ngen" > "$SP/runs/$name.out" 2> "$SP/runs/$name.err"
else
    "$exe" "C:/AI/Kimi-K3.1-Pro" --chat "$prompt" --ngen "$ngen" > "$SP/runs/$name.out" 2> "$SP/runs/$name.err"
fi
rc=$?
echo "rc=$rc wall_ms=$(( ($(date +%s%N) - t0) / 1000000 ))" >> "$SP/runs/$name.err"
touch "$SP/runs/$name.stop"; wait
echo "$name done rc=$rc"
