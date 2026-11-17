#!/bin/bash
# api_ab.sh <config> <request.json>...: the launcher of K3_ENGINE_DIR (default the repo's c/) on port 18081,
# -Sessions 0, the draft model on; each request sent twice (the second time its prompt's photo is restored, so
# the time is the reply's alone); then the engine and the gateway stopped by their PIDs (nothing else).
#   cut     -ModelDir C:\AI\Kimi-K3.1-Pro-cutview -NoMirror: each expert from one drive (C:'s parts, D:'s .p2)
#   mirror  -ModelDir C:\AI\Kimi-K3.1-Pro: every file on both drives (the launcher's defaults)
# Requests: recorded Kimi Code turns (draft/build_req.py), not in the repository.
SP="$(cd "$(dirname "$0")" && pwd)"
W="${K3_ENGINE_DIR:-$(cd "$SP/../../../../c" && pwd)}"
cfg=$1; shift
case $cfg in
    cut)    largs='-ModelDir C:\AI\Kimi-K3.1-Pro-cutview -NoMirror -NoFill' ;;
    mirror) largs='-ModelDir C:\AI\Kimi-K3.1-Pro -NoFill' ;;
    *) echo "unknown $cfg"; exit 1 ;;
esac
n=$(ls -d "$SP"/runs/api_* 2>/dev/null | wc -l); n=$((n + 1))
O="$SP/runs/api_$(printf %02d $n)_$cfg"; mkdir -p "$O"
L="$O/serve.log"
printf '@echo off\r\ncd /d %s\r\npowershell -NoProfile -ExecutionPolicy Bypass -File tools\\start_kimi_k31_pro_4090.ps1 -Port 18081 -Sessions 0 -Draft C:\\AI\\Kimi-K3.1-Pro-DSpark %s > "%%~dp0serve.log" 2>&1\r\n' \
    "$(cygpath -w "$W")" "$largs" > "$O/start.cmd"
unset K3_DRAFT K3_DRAFT_P K3_DRAFT_MAX K3_DRAFT_DUMP K3_DRAFT_ORACLE K3_LOGITS K3_HID_DUMP K3_HID_LAYERS K3_MIRROR K3_FILL
cmd //c "$(cygpath -w "$O/start.cmd")" &
up=0
for i in $(seq 1 150); do
    curl -s -m 2 http://127.0.0.1:18081/v1/models > /dev/null 2>&1 && { up=1; break; }
    sleep 2
done
pids(){
    powershell.exe -NoProfile -Command "\$g = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*serve*--port 18081*' } | Select-Object -First 1; \$e = if (\$g) { Get-CimInstance Win32_Process -Filter \"Name='kimi_k3.exe' AND ParentProcessId=\$(\$g.ProcessId)\" | Select-Object -First 1 }; '' + \$g.ProcessId + ' ' + \$e.ProcessId" | tr -d '\r'
}
stop(){
    set -- $(pids)
    [ -n "$2" ] && powershell.exe -NoProfile -Command "Stop-Process -Id $2 -Force" && sleep 3
    [ -n "$1" ] && powershell.exe -NoProfile -Command "Stop-Process -Id $1 -Force"
    echo "stopped: gateway ${1:-none}, engine ${2:-none}"
}
if [ $up = 0 ]; then echo "FAILED: the server did not come up"; tail -20 "$L"; stop; exit 1; fi
echo "== $(printf %02d $n) $cfg: server up after $((i * 2)) s ($(date +%H:%M))" | tee -a "$SP/runs/api_ab.log"
k=0
for r in "$@"; do
    for pass in 1 2; do
        k=$((k + 1))
        t0=$(date +%s%N)
        code=$(curl -s -m 7200 -o "$O/$k.resp.json" -w '%{http_code}' -H 'Content-Type: application/json' --data-binary "@$r" http://127.0.0.1:18081/v1/chat/completions)
        ms=$(( ($(date +%s%N) - t0) / 1000000 ))
        usage=$(python -I -c "import json,sys; r=json.load(open(sys.argv[1],encoding='utf-8')); print(r.get('usage'))" "$O/$k.resp.json" 2>/dev/null)
        echo "   $(basename "$r") pass $pass: HTTP $code in $((ms / 1000)).$(( (ms % 1000) / 100 )) s; $usage" | tee -a "$SP/runs/api_ab.log"
    done
done
sleep 2
grep -h -E "\] reads \(|K3-DRAFT\] turn|prompt of [0-9]+ tokens|K3-MIRROR\]|expert readers|prefetch \(turn\)" "$L" | cut -c1-330 | sed 's/^/   /' | tee -a "$SP/runs/api_ab.log"
stop
