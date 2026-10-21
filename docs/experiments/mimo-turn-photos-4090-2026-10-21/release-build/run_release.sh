#!/bin/sh
# The release's own build (the merge into main) run as the measured one was: the turns resumed
# from photos against the same prompts cold, the agent loop dropping the reasoning, and the
# decode benchmark alternating with the measured build. usage: run_release.sh RELEASE_ENGINE MEASURED_ENGINE
cd "$(dirname "$0")/.."
rel=$1; measured=$2
python turns.py --engine "$rel" --plan plan-exact.json --out release-build/exact && python compare.py release-build/exact || exit 1
python api_run.py --engine "$rel" --mode drop --out release-build/api-drop || exit 1
for arm in 1-release:$rel 2-measured:$measured 3-release:$rel; do
  python ../../../c/tools/benchmark_mimo_cuda.py --engine "${arm#*:}" --model D:/AI/Models/MiMo-V2.6-Pro-MOPD --shards C:/AI/MiMoPro-Shards --backend cuda --release-host --cap 104 --threads 16 --read-threads 16 --context 16384 --tokens 64 --runs 3 --output "release-build/benchmark/${arm%%:*}" || exit 1
done
