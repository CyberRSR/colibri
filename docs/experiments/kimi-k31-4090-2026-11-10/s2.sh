#!/bin/bash
# s2.sh -- the build with k3cu_mla_sel, K3_DENSE_CACHE and equal steps (stage/) on Kimi K3.1 Pro:
#   1. the 5,862-token prompt's logits against the CPU alone (ref/g6k_cpu.f32)
#   2. the 16,511-token prompt in ONE step (K3_CHUNK=16384): logits against steps of 8,192 with the masked kernel
#   3. K3_DENSE_CACHE: no file / the start that writes it / the start that reads it: the same logits, the init times
#   4. timing, no logits: 16,511 tokens new (one step, lists) against the build before it at K3_CHUNK=8192 (3 steps, masks)
SP="${K31_SP:-$(cd "$(dirname "$0")" && pwd)}"   # the folder with stage/ (the engine and its DLLs), runs/, ref/ and the prompts
show(){ printf "%s: " "$1"; grep -h -E 'init done|dense cache|prompt:|prefill done|the prompt:|decode [0-9]+ tokens|a decode token|RAM\]|gated MLA.s attention|rc=' "$SP/runs/$1.err" | cut -c1-360; }
same(){ if [ -s "$SP/ref/$1" ] && cmp -s "$SP/ref/$1" "$SP/ref/$2"; then echo "$3: THE SAME BYTES ($(stat -c %s "$SP/ref/$1"))"; else echo "$3: DIFFER ($(cmp "$SP/ref/$1" "$SP/ref/$2" 2>&1 | head -1))"; fi; }
K3_RUN_PROMPT="$(cat $SP/long6k_prompt.txt)" bash "$SP/k31run.sh" s2_6k 0 K3_LOGITS="$SP/ref/s2_6k.f32" > /dev/null; show s2_6k
same s2_6k.f32 g6k_cpu.f32 "5,862 tokens, lists on the GPU vs the CPU alone"
K3_RUN_PROMPT_FILE="$SP/p15k.txt" bash "$SP/k31run.sh" s2_15k 0 K3_CHUNK=16384 K3_LOGITS="$SP/ref/s2_15k.f32" > /dev/null; show s2_15k
same s2_15k.f32 p15k_8k.f32 "16,511 tokens, one step over lists vs steps of 8,192 with masks"
rm -f "$SP/ref/s2_15k.f32"
bash "$SP/k31run.sh" s2_dcA 20 K3_LOGITS="$SP/ref/s2_dcA.f32" > /dev/null; show s2_dcA
bash "$SP/k31run.sh" s2_dcB 20 K3_DENSE_CACHE=1 K3_LOGITS="$SP/ref/s2_dcB.f32" > /dev/null; show s2_dcB
bash "$SP/k31run.sh" s2_dcC 20 K3_DENSE_CACHE=1 K3_LOGITS="$SP/ref/s2_dcC.f32" > /dev/null; show s2_dcC
same s2_dcB.f32 s2_dcA.f32 "the start that writes the dense cache vs none"
same s2_dcC.f32 s2_dcA.f32 "the start that reads the dense cache vs none"
ls -la "C:/AI/Kimi-K3.1-Pro/k3_dense_cache.bin"
K3_RUN_PROMPT_FILE="$SP/p15k.txt" bash "$SP/k31run.sh" s2_t15k_new 0 K3_CHUNK=16384 K3_DENSE_CACHE=1 > /dev/null; show s2_t15k_new
K3_EXE="$SP/stage_lend/kimi_k3.exe" K3_RUN_PROMPT_FILE="$SP/p15k.txt" bash "$SP/k31run.sh" s2_t15k_old 0 K3_CHUNK=8192 > /dev/null; show s2_t15k_old
echo series done
