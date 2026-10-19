#!/usr/bin/env bash
# run_prefill.sh <engine> <tag> [KEY=VALUE ...]: CLI run of long_prompt.txt, 8 generated tokens
SP="$(cd "$(dirname "$0")" && pwd)"   # the measured run used a scratch directory of its session
ENG="$1"; TAG="$2"; shift 2
env -i PATH="$PATH" SYSTEMROOT="$SYSTEMROOT" WINDIR="$WINDIR" TEMP="$TEMP" TMP="$TMP" \
  MIMO_DIRS=C:/AI/MiMoPro-Shards MIMO_DENSE_BITS=0 MIMO_IDOT=0 MIMO_DIRECT=1 MIMO_CHUNK=64 \
  OMP_NUM_THREADS=16 COLI_NO_OMP_TUNE=1 COLI_TEMP=0 MIMO_CTX=16384 COLI_VULKAN=0 \
  MIMO_CUDA=1 MIMO_CUDA_DENSE_GB=20 MIMO_CUDA_RESERVE_GB=2 MIMO_CUDA_RELEASE_HOST=1 SERVE=0 \
  "$@" \
  "$ENG" D:/AI/Models/MiMo-V2.6-Pro-MOPD --cap 104 --ngen 8 --prompt "$(cat $SP/prompt.txt)" > "$SP/pf-$TAG.txt" 2> "$SP/pf-$TAG.log"
echo "$TAG exit $?: $(grep 'prefill' $SP/pf-$TAG.log | head -1)"
grep "streaming" "$SP/pf-$TAG.log" | tail -1
