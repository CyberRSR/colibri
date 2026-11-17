#!/bin/bash
# perf_cli.sh <prompt file> <config>...: a prompt and NGEN (128) tokens of reply with the draft model on (as the
# launcher runs it), a fresh process each, from K3_ENGINE_DIR (default: the repo's c/; the measured runs used the
# new kimi_k3.exe beside C:\colibri's DLLs). Each run's lines are appended to runs/perf_cli.log.
#   cut      each expert from one drive, as tools/k3_download.py laid the model out: the model folder is
#            C:\AI\Kimi-K3.1-Pro-cutview (hard links to C:'s 96 files of the cut layout), D:'s .p2 parts via K3_DIRS
#   cut3     the same, 3 readers a drive
#   mirror   every file on both drives, every expert's read cut 60/40 over C: and D: (2 readers a drive)
#   mirror3  the same, 3 readers a drive
#   hyb      every file on both drives: a decode step's reads cut, a prompt step's whole by the shorter queue
#   hyb3     the same, 3 readers a drive (the default with copies)
#   c_only   every file from C: alone
#   full     every file on both drives, the launcher's defaults (readers the engine's choice)
#   partial  every file on C:, D:'s .p2 parts only on D: (a folder of hard links): a release user's first start
# The cut view: mkdir C:\AI\Kimi-K3.1-Pro-cutview, then New-Item -ItemType HardLink for every file of
# C:\AI\Kimi-K3.1-Pro but the *.p2.safetensors (no bytes copied).
SP="$(cd "$(dirname "$0")" && pwd)"
ENG="${K3_ENGINE_DIR:-$(cd "$SP/../../../../c" && pwd)}"
P=$1; shift
NGEN=${NGEN:-128}
export PYTHONIOENCODING=utf-8
export K3_DIRS='D:\AI\Models\Kimi-K3.1-Pro' K3_CUDA=0 K3_GPU_EXPERTS=1 K3_READAHEAD_GB=20 K3_CUDA_DENSE=1 \
       K3_CUDA_DENSE_GB=20 K3_CUDA_RESERVE_GB=2 K3_CUDA_RELEASE_HOST=1 K3_EXPERT_GB=1000 RAM_GB=174 \
       K3_MAXT=131072 K3_DIRECT=1 K3_PIPE=1 K3_PREFETCH=1 OMP_NUM_THREADS=16 \
       COLI_NO_OMP_TUNE=1 AUTOPIN=0 USAGE_SAVE=0 COLI_VULKAN=0 COLI_CUDA=0 COLI_GPU=0 K3_STAT_EVERY=32 \
       K3_CHUNK=16384 K3_DENSE_CACHE=1 K3_DRAFT='C:\AI\Kimi-K3.1-Pro-DSpark' K3_DRAFT_P=0.98 K3_DRAFT_MAX=3
unset K3_DRAFT_DUMP K3_DRAFT_ORACLE K3_LOGITS K3_LOGITS_DECODE K3_MIRROR K3_FILL K3_MIRROR_SPLIT K3_MIRROR_ANY K3_MIRROR_ROWS K3_IO_PER_DRIVE
mkdir -p "$SP/runs"
n=$(ls "$SP"/runs/*.err 2>/dev/null | wc -l)
for cfg in "$@"; do
    n=$((n + 1))
    case $cfg in
        cut)     model='C:/AI/Kimi-K3.1-Pro-cutview'; extra=(K3_MIRROR=0 K3_IO_PER_DRIVE=2) ;;
        cut3)    model='C:/AI/Kimi-K3.1-Pro-cutview'; extra=(K3_MIRROR=0 K3_IO_PER_DRIVE=3) ;;
        mirror)  model='C:/AI/Kimi-K3.1-Pro'; extra=(K3_MIRROR=1 K3_MIRROR_SPLIT=0.6 K3_MIRROR_ROWS=-1 K3_IO_PER_DRIVE=2) ;;
        mirror3) model='C:/AI/Kimi-K3.1-Pro'; extra=(K3_MIRROR=1 K3_MIRROR_SPLIT=0.6 K3_MIRROR_ROWS=-1 K3_IO_PER_DRIVE=3) ;;
        hyb)     model='C:/AI/Kimi-K3.1-Pro'; extra=(K3_MIRROR=1 K3_MIRROR_SPLIT=0.6 K3_MIRROR_ROWS=16 K3_IO_PER_DRIVE=2) ;;
        hyb3)    model='C:/AI/Kimi-K3.1-Pro'; extra=(K3_MIRROR=1 K3_MIRROR_SPLIT=0.6 K3_MIRROR_ROWS=16 K3_IO_PER_DRIVE=3) ;;
        c_only)  model='C:/AI/Kimi-K3.1-Pro'; extra=(K3_MIRROR=0 K3_IO_PER_DRIVE=2) ;;
        full)    model='C:/AI/Kimi-K3.1-Pro'; extra=(K3_MIRROR=1) ;;   # the launcher's defaults
        partial) model='C:/AI/Kimi-K3.1-Pro'; extra=(K3_MIRROR=1 'K3_DIRS=D:\AI\Models\Kimi-K3.1-Pro-p2view') ;;   # D: holds its .p2 parts only (hard links)
        *) echo "unknown $cfg"; continue ;;
    esac
    out="$SP/runs/$(printf %02d $n)_$cfg"
    t0=$(date +%s%N)
    (cd "$ENG" && env "${extra[@]}" ./kimi_k3.exe "$model" --chat --prompt-file "$P" --ngen $NGEN > "$out.out" 2> "$out.err")
    rc=$?
    s=$(( ($(date +%s%N) - t0) / 1000000 ))
    {
        echo "== $(printf %02d $n) $cfg rc=$rc, $((s / 1000)).$(( (s % 1000) / 100 )) s in all ($(date +%H:%M))"
        grep -h -E 'K3-MIRROR\]|expert readers|init done|prefill done|the prompt: attention|\] reads \(|\] decode [0-9]+ tokens|a decode token|K3-DRAFT\] run|prefetch \(run\)' "$out.err" | cut -c1-330 | sed 's/^/   /'
    } | tee -a "$SP/runs/perf_cli.log"
done
