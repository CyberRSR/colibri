#!/bin/bash
# Development arms: each a benchmark run (the mimo-mtp2 profile) of a scratch engine with its IOLOG.
# usage: dev.sh SERIES "name bindir [ENV=V ...]" ...   (sequential; nothing else heavy meanwhile)
cd /c/colibri/dev
T=c/build/io-prefetch
S=$T/dev/$1; shift
mkdir -p $S
for arm in "$@"; do
  set -- $arm
  name=$1; bin=$2; shift 2
  args=()
  for e in "$@"; do args+=(--env "$e"); done
  echo "=== $(date +%H:%M:%S) $name ($bin) $*"
  python c/tools/benchmark_mimo_cuda.py --engine $T/$bin/mimo.exe --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards \
    --backend cuda --release-host --cap 104 --threads 16 --read-threads 16 --context 16384 --tokens 64 --runs 3 \
    --env MIMO_IOLOG=C:/colibri/dev/$S/$name.iolog "${args[@]}" --output "$S/$name" > "$S/$name.stdout" 2>&1
  echo "    exit $? $(date +%H:%M:%S): $(tail -1 "$S/$name.stdout")"
  grep -E "streaming|generation" "$S/$name/engine.log" | tail -2 | sed 's/^/    /'
done
echo "DONE $(date +%H:%M:%S)"
