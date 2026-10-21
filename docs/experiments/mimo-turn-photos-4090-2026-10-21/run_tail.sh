#!/bin/sh
# the short tail's time, MIMO_PROMPT_KERNELS 1 (default) and 0 alternating
cd "$(dirname "$0")"
for i in 1 2; do
  python turns.py --engine /c/colibri/dev/c/mimo.exe --plan plan-tail.json --out runs/tail-$i-kernels1
  python turns.py --engine /c/colibri/dev/c/mimo.exe --plan plan-tail.json --out runs/tail-$i-kernels0 --env MIMO_PROMPT_KERNELS=0
done
