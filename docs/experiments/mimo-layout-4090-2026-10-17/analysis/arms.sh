#!/bin/bash
# Benchmark arms (the mimo-mtp2 profile), each optionally sampled after the warmup.
# usage: arms.sh SERIES "name exe-path SAMPLE_SECS INTERVAL_US MAXTHREADS SKIP [ENV=V ...]" ...
# SAMPLE_SECS 0: no sampling (cpumon still runs). Sequential; nothing else heavy meanwhile.
cd /c/colibri/dev
L=c/build/layout
S=$L/runs/$1; shift
mkdir -p $S
for arm in "$@"; do
  set -- $arm
  name=$1; exe=$2; secs=$3; us=$4; maxth=$5; skip=$6; shift 6
  args=()
  for e in "$@"; do args+=(--env "$e"); done
  if tasklist | grep -qi "^mimo.exe"; then echo "another mimo.exe runs; stop"; exit 1; fi
  echo "=== $(date +%H:%M:%S) $name ($exe) sample=$secs/$us/$maxth/$skip $*"
  python c/tools/benchmark_mimo_cuda.py --engine $exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards \
    --backend cuda --release-host --cap 104 --threads 16 --read-threads 16 --context 16384 --tokens 64 --runs 3 \
    "${args[@]}" --output "$S/$name" > "$S/$name.stdout" 2>&1 &
  bpid=$!
  until grep -q "^run 0:" "$S/$name.stdout" 2>/dev/null || ! kill -0 $bpid 2>/dev/null; do sleep 0.5; done
  pid=$(powershell -NoProfile -Command "(Get-Process mimo).Id")
  echo "    engine pid $pid at $(date +%H:%M:%S)"
  $L/cpumon.exe 74 500 "$S/$name.cpu" &
  cpid=$!
  if [ "$secs" != "0" ]; then $L/${SAMPLER:-winsample2}.exe $pid $secs $us "$S/$name.samples" $maxth $skip 2>&1 | sed 's/^/    /'; fi
  wait $bpid; rc=$?
  wait $cpid
  echo "    exit $rc $(date +%H:%M:%S): $(tail -1 "$S/$name.stdout")"
  grep -E "^run [1-3]" "$S/$name.stdout" | sed 's/^/    /'
  grep -E "streaming|generation" "$S/$name/engine.log" | tail -2 | sed 's/^/    /'
done
echo "DONE $(date +%H:%M:%S)"
