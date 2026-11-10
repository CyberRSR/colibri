#!/bin/bash
# s1.sh -- decode: the lending build (stage/) against the load-fixed build before it (stage_loadfix/),
# alternated twice; then a ~15k-token prompt in steps of 8,192 and in one step of 16,384, logits compared.
SP="${K31_SP:-$(cd "$(dirname "$0")" && pwd)}"   # the folder with stage/ (the engine and its DLLs), runs/, ref/ and the prompts
show(){ printf "%s: " "$1"; grep -h -E 'init done|prompt:|prefill done|the prompt:|decode [0-9]+ tokens|a decode token|RAM\]|rc=' "$SP/runs/$1.err" | cut -c1-330; }
for i in 1 2; do
    K3_EXE="$SP/stage/kimi_k3.exe" bash "$SP/k31run.sh" dec_new$i 60 > /dev/null; show dec_new$i
    K3_EXE="$SP/stage_loadfix/kimi_k3.exe" bash "$SP/k31run.sh" dec_old$i 60 > /dev/null; show dec_old$i
done
K3_RUN_PROMPT_FILE="$SP/p15k.txt" bash "$SP/k31run.sh" p15k_8k 0 K3_CHUNK=8192 K3_LOGITS="$SP/ref/p15k_8k.f32" > /dev/null; show p15k_8k
K3_RUN_PROMPT_FILE="$SP/p15k.txt" bash "$SP/k31run.sh" p15k_16k 0 K3_CHUNK=16384 K3_LOGITS="$SP/ref/p15k_16k.f32" > /dev/null; show p15k_16k
if cmp -s "$SP/ref/p15k_8k.f32" "$SP/ref/p15k_16k.f32"; then echo "p15k: one step of 16,384 = steps of 8,192, every position's logits ($(stat -c %s "$SP/ref/p15k_8k.f32") bytes)"
else echo "p15k: LOGITS DIFFER ($(cmp "$SP/ref/p15k_8k.f32" "$SP/ref/p15k_16k.f32" 2>&1 | head -1))"; fi
echo series done
