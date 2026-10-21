#!/bin/sh
# The agent loop at 25k through the gateway: r4's engine ("base", the branch's code before this
# round, built from 9b043a09) and this round's ("new"), drop and keep, in two series of opposite
# order. usage: run_api.sh BASE_ENGINE NEW_ENGINE SERIES_DIR ORDER   (ORDER: forward | reverse)
cd "$(dirname "$0")"
base=$1; new=$2; dir=$3; order=$4
if [ "$order" = forward ]; then runs="base:drop new:drop base:keep new:keep"; else runs="new:drop base:drop new:keep base:keep"; fi
i=0
for r in $runs; do
  i=$((i + 1)); eng=${r%%:*}; mode=${r##*:}
  if [ "$eng" = base ]; then e=$base; else e=$new; fi
  echo "== $i $eng $mode"
  python api_run.py --engine "$e" --mode "$mode" --out "$dir/$i-$eng-$mode" || exit 1
done
