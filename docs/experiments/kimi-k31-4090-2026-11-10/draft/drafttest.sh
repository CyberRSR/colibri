#!/bin/bash
# drafttest.sh <name> <prompt file> <ngen> [K3_CHUNK] [extra env...]: the engine on a prompt,
# greedy, twice -- without the draft model and with it -- every position's logits written (the
# prompt's and the reply's: K3_LOGITS, K3_LOGITS_DECODE=1). The two files and the two replies must be
# the same bytes; the second run also writes K3_DRAFT_DUMP for dspark2_check.py.
SP="$(cd "$(dirname "$0")" && pwd)"
ENG="${K3_ENGINE_DIR:-$(cd "$SP/../../../../c" && pwd)}"   # the folder with kimi_k3.exe and its DLLs
SP="$SP/runs"
name=$1; prompt=$2; ngen=$3; chunk=${4:-16384}; shift 4 2>/dev/null
R="$SP/dt_$name"; mkdir -p "$R"
export K3_DIRS='D:\AI\Models\Kimi-K3.1-Pro' K3_CUDA=0 K3_GPU_EXPERTS=1 K3_READAHEAD_GB=20 K3_CUDA_DENSE=1 \
       K3_CUDA_DENSE_GB=20 K3_CUDA_RESERVE_GB=2 K3_CUDA_RELEASE_HOST=1 K3_EXPERT_GB=1000 RAM_GB=174 \
       K3_MAXT=131072 K3_DIRECT=1 K3_PIPE=1 K3_IO_PER_DRIVE=2 K3_PREFETCH=1 OMP_NUM_THREADS=16 \
       COLI_NO_OMP_TUNE=1 AUTOPIN=0 USAGE_SAVE=0 COLI_VULKAN=0 COLI_CUDA=0 COLI_GPU=0 K3_STAT_EVERY=16 \
       K3_LOGITS_DECODE=1 K3_CHUNK=$chunk
for kv in "$@"; do export "$kv"; done
cd "$ENG" || exit 1
run(){   # tag, then the run's own env
    tag=$1; shift
    env "$@" K3_LOGITS="$(cygpath -w "$R/logits_$tag.bin")" ./kimi_k3.exe "C:/AI/Kimi-K3.1-Pro" --chat \
        --prompt-file "$prompt" --ngen "$ngen" > "$R/$tag.out" 2> "$R/$tag.err"
    printf "%s rc=%s: " "$tag" "$?"
    grep -h -E 'prefill done|\] decode [0-9]+ tokens|K3-DRAFT\] (run|on|off|C:)|a decode token' "$R/$tag.err" | cut -c1-330
    ls -la "$R/logits_$tag.bin" | awk '{printf "   logits %s bytes\n", $5}'
}
[ -n "$SKIP_PLAIN" ] || run plain K3_DRAFT=
dtag=${DTAG:-draft}
run $dtag K3_DRAFT="${K3_DRAFT_DIR:-C:\\AI\\Kimi-K3.1-Pro-DSpark}" K3_DRAFT_DUMP="$(cygpath -w "$R/dump_$dtag")" K3_DRAFT_LOG=1
if cmp -s "$R/logits_plain.bin" "$R/logits_$dtag.bin"; then echo "LOGITS: the same bytes with the draft model and without"
else echo "LOGITS DIFFER: $(cmp "$R/logits_plain.bin" "$R/logits_$dtag.bin" | head -1)"; fi
if cmp -s <(sed 's/TUNE decode:.*//' "$R/plain.out") <(sed 's/TUNE decode:.*//' "$R/$dtag.out"); then echo "REPLY: the same bytes"; else echo "REPLY DIFFERS"; fi
grep "drafts stood" "$R/$dtag.err" | awk '{print $4" of "$6}' | sort | uniq -c | tr '
' ';'; echo
echo drafttest done
