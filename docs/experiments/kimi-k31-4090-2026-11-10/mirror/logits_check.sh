#!/bin/bash
# logits_check.sh <prompt file> <tag>...: a prompt and 32 tokens of reply, greedy, the draft model off, every
# position's logits written (K3_LOGITS, K3_LOGITS_DECODE=1); every tag's file and reply are compared with the first
# tag's. KEEP=1 keeps a tag's logits from an earlier run.
#   r11     C:\colibri's kimi_k3.exe as installed then (the cut layout: C:'s parts + D:'s .p2 parts)
#   fill    the new exe with K3_FILL=1: copies to C: what it lacks of D:'s files first, then the run (the copies read)
#   nomir   the new exe, K3_MIRROR=0: every file from C:
#   mirror  the new exe, K3_IO_PER_DRIVE=2: the copies read (a prompt's reads whole by queue, a decode's cut)
#   final   the new exe with the launcher's readers (3 a drive with copies)
SP="$(cd "$(dirname "$0")" && pwd)"
ENG="${K3_ENGINE_DIR:-$(cd "$SP/../../../../c" && pwd)}"   # the new exe and its DLLs
P=$1; shift
R="$SP/runs"; mkdir -p "$R"
export PYTHONIOENCODING=utf-8
bad=0
chk(){ if [ "$1" != 0 ]; then echo "   FAILED (exit $1)"; bad=1; fi; }
export K3_DIRS='D:\AI\Models\Kimi-K3.1-Pro' K3_CUDA=0 K3_GPU_EXPERTS=1 K3_READAHEAD_GB=20 K3_CUDA_DENSE=1 \
       K3_CUDA_DENSE_GB=20 K3_CUDA_RESERVE_GB=2 K3_CUDA_RELEASE_HOST=1 K3_EXPERT_GB=1000 RAM_GB=174 \
       K3_MAXT=131072 K3_DIRECT=1 K3_PIPE=1 K3_IO_PER_DRIVE=2 K3_PREFETCH=1 OMP_NUM_THREADS=16 \
       COLI_NO_OMP_TUNE=1 AUTOPIN=0 USAGE_SAVE=0 COLI_VULKAN=0 COLI_CUDA=0 COLI_GPU=0 K3_STAT_EVERY=16 \
       K3_LOGITS_DECODE=1 K3_CHUNK=16384 K3_DENSE_CACHE=1
unset K3_DRAFT K3_DRAFT_DUMP K3_DRAFT_ORACLE K3_DRAFT_P K3_DRAFT_MAX K3_MIRROR K3_FILL K3_MIRROR_SPLIT K3_MIRROR_ANY K3_MIRROR_ROWS
run(){
    tag=$1; dir=$2; shift 2
    (cd "$dir" && env "$@" K3_LOGITS="$(cygpath -w "$R/logits_$tag.bin")" ./kimi_k3.exe "C:/AI/Kimi-K3.1-Pro" --chat \
        --prompt-file "$P" --ngen 32 > "$R/$tag.out" 2> "$R/$tag.err")
    rc=$?
    echo "$tag rc=$rc"; chk $rc
    grep -h -E 'K3-FILL\]|K3-MIRROR\]|expert readers|prefill done|\] decode [0-9]+ tokens|a decode token|\] reads \(' "$R/$tag.err" | cut -c1-300 | sed 's/^/   /'
    ls -la "$R/logits_$tag.bin" | awk '{printf "   logits %s bytes\n", $5}'
}
for tag in "$@"; do
    if [ -n "$KEEP" ] && [ -f "$R/logits_$tag.bin" ]; then echo "$tag: its logits from before"; continue; fi
    case $tag in
        r11)    run r11 /c/colibri ;;
        r11s)   run r11s "${K3_R11_DIR:?the folder of r11's exe beside its DLLs}" ;;
        fill)   run fill "$ENG" K3_FILL=1 ;;
        nomir)  run nomir "$ENG" K3_MIRROR=0 ;;
        mirror) run mirror "$ENG" ;;
        final)  run final "$ENG" -u K3_IO_PER_DRIVE ;;
    esac
done
ref=""
for tag in "$@"; do
    [ -f "$R/logits_$tag.bin" ] || continue
    if [ -z "$ref" ]; then ref=$tag; continue; fi
    if cmp -s "$R/logits_$ref.bin" "$R/logits_$tag.bin"; then echo "LOGITS $ref = $tag: the same bytes"
    else echo "LOGITS $ref != $tag: $(cmp "$R/logits_$ref.bin" "$R/logits_$tag.bin" | head -1)"; bad=1; fi
    if cmp -s <(sed 's/TUNE decode:.*//' "$R/$ref.out") <(sed 's/TUNE decode:.*//' "$R/$tag.out"); then echo "REPLY $ref = $tag"
    else echo "REPLY $ref != $tag"; bad=1; fi
done
echo "logits check done: $([ $bad = 0 ] && echo ALL OK || echo FAILURES)"
