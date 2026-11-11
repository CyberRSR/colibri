#!/bin/bash
# serve_root.sh NAME N [VAR=value ...] -- the gateway with the launcher's settings (the engine's own log kept),
# N new conversations sharing a 16.5k-token system message (api_root.py), then the gateway stopped.
SP="${K31_SP:-$(cd "$(dirname "$0")" && pwd)}"
name=$1; n=$2; shift 2
cd "$SP/serve_stage" || exit 1
export K3_DIRS='D:\AI\Models\Kimi-K3.1-Pro' K3_CUDA=0 K3_GPU_EXPERTS=1 K3_READAHEAD_GB=20 K3_CUDA_DENSE=1 K3_CUDA_DENSE_GB=20 \
       K3_CUDA_RESERVE_GB=2 K3_CUDA_RELEASE_HOST=1 K3_EXPERT_GB=1000 K3_MAXT=131072 K3_CHUNK=16384 K3_DENSE_CACHE=1 AUTOPIN=0 \
       K3_DIRECT=1 K3_PIPE=1 K3_IO_PER_DRIVE=2 K3_PREFETCH=1 COLI_K3_CKPT=5 OMP_NUM_THREADS=16 COLI_NO_OMP_TUNE=1 COLI_VULKAN=0 \
       COLI_CUDA=0 COLI_GPU=0 K3_PREFIX_LOG=1
for kv in "$@"; do export "$kv"; done
python coli serve --model 'C:\AI\Kimi-K3.1-Pro' --ram 174 --ctx 131072 --ngen 4096 --host 127.0.0.1 --port 18090 \
    --model-id kimi-k3.1-pro --kv-slots 1 --queue-timeout 7200 --max-queue 8 > "$SP/runs/$name.log" 2>&1 &
python "$SP/api_root.py" 18090 "$SP/runs/${name}_session.json" "$SP/p15k.txt" "$n" 2>&1 | tee "$SP/runs/${name}_session.out"
PID=$(netstat -ano 2>/dev/null | grep "127.0.0.1:18090 .*LISTENING" | awk '{print $NF}' | head -1)
[ -n "$PID" ] && taskkill //PID "$PID" //T //F > /dev/null 2>&1
until ! tasklist 2>/dev/null | grep -qi kimi_k3; do sleep 2; done
grep -h -E "CKPT\]|PREFIX\]|prompt of|photographed" "$SP/runs/$name.log" | cut -c1-260
echo "$name done"
