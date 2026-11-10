#!/bin/bash
# s4.sh -- the release candidate (stage/) on Kimi K3.1 Pro, the launcher's settings (AUTOPIN=0, K3_DENSE_CACHE=1, K3_CHUNK=16384):
#  A1 5,862 tokens: logits against the CPU alone, with K3_CHECK; twice more without
#  A2 16,511 tokens in one step: logits against steps of 8,192 with the masked kernel (the build before)
#  A3 the same under RAM_GB=110 (the cache lends down to its floor), 20 tokens after it: the same logits
#  B5 decode, 60 tokens: the dense cache read against quantizing the checkpoint, alternated twice
#  B6 16,511 tokens, B7 5,862 tokens: the prompts' times without logits; B8 5,862 tokens then 40 of decode
SP="${K31_SP:-$(cd "$(dirname "$0")" && pwd)}"   # the folder with stage/ (the engine and its DLLs), runs/, ref/ and the prompts
R="K3_CHUNK=16384 K3_DENSE_CACHE=1"
show(){ printf "%s: " "$1"; grep -h -E 'init done|dense cache|prompt:|prefill done|the prompt:|decode [0-9]+ tokens|a decode token|RAM\]|K3-CHECK\] (the|on)|gated MLA.s attention|rc=' "$SP/runs/$1.err" | cut -c1-360; }
same(){ if [ -s "$SP/ref/$1" ] && cmp -s "$SP/ref/$1" "$SP/ref/$2"; then echo "$3: THE SAME BYTES ($(stat -c %s "$SP/ref/$1"))"; else echo "$3: DIFFER ($(cmp "$SP/ref/$1" "$SP/ref/$2" 2>&1 | head -1))"; fi; }
P6K="$(cat $SP/long6k_prompt.txt)"
K3_RUN_PROMPT="$P6K" bash "$SP/k31run.sh" f_6k_ck 0 $R K3_CHECK=1 K3_CHECK_DUMP="$SP/ref/f_6k.ck" K3_LOGITS="$SP/ref/f_6k.f32" > /dev/null; show f_6k_ck
same f_6k.f32 g6k_cpu.f32 "A1 5,862 tokens with K3_CHECK vs the CPU alone"
for i in 2 3; do
    K3_RUN_PROMPT="$P6K" bash "$SP/k31run.sh" f_6k_$i 0 $R K3_LOGITS="$SP/ref/f_6k.f32" > /dev/null; show f_6k_$i
    same f_6k.f32 g6k_cpu.f32 "A1 5,862 tokens, run $i vs the CPU alone"
done
rm -f "$SP/ref/f_6k.f32"
K3_RUN_PROMPT_FILE="$SP/p15k.txt" bash "$SP/k31run.sh" f_15k 0 $R K3_LOGITS="$SP/ref/f_15k.f32" > /dev/null; show f_15k
same f_15k.f32 p15k_8k.f32 "A2 16,511 tokens, one step over lists vs steps of 8,192 with masks"
rm -f "$SP/ref/f_15k.f32"
K3_RUN_PROMPT_FILE="$SP/p15k.txt" bash "$SP/k31run.sh" f_15k_ram110 20 $R RAM_GB=110 K3_LOGITS="$SP/ref/f_15k_r.f32" > /dev/null; show f_15k_ram110
python - "$SP/ref/f_15k_r.f32" "$SP/ref/p15k_8k.f32" <<'EOF'
import sys, os
a, b = sys.argv[1], sys.argv[2]
n = os.path.getsize(b)
ok = os.path.getsize(a) >= n
with open(a, 'rb') as fa, open(b, 'rb') as fb:
    left = n
    while ok and left:
        k = min(left, 64 << 20); ok = fa.read(k) == fb.read(k); left -= k
print("A3 RAM_GB=110, the prompt's 16,511 positions vs steps of 8,192 unpressed:", "THE SAME BYTES" if ok else "DIFFER", f"({n} bytes; {(os.path.getsize(a) - n) // (163840 * 4)} decode rows after them)")
EOF
python - "$SP/runs/f_15k_ram110.csv" <<'EOF'
import sys
ws = [float(l.split(';')[8]) for l in open(sys.argv[1]).read().splitlines()[1:] if l.count(';') >= 9]
print(f"A3 RAM_GB=110: the engine's working set at most {max(ws):.1f} GB")
EOF
rm -f "$SP/ref/f_15k_r.f32"
for i in 1 2; do
    bash "$SP/k31run.sh" f_dec_dc$i 60 $R > /dev/null; show f_dec_dc$i
    bash "$SP/k31run.sh" f_dec_nodc$i 60 K3_CHUNK=16384 K3_DENSE_CACHE=0 > /dev/null; show f_dec_nodc$i
done
K3_RUN_PROMPT_FILE="$SP/p15k.txt" bash "$SP/k31run.sh" f_t15k 0 $R > /dev/null; show f_t15k
K3_RUN_PROMPT="$P6K" bash "$SP/k31run.sh" f_t6k 40 $R > /dev/null; show f_t6k
echo series done
