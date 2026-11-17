#!/bin/bash
# long_answer.sh <config> <request.json>...: for each request a fresh server -- the install's START-KIMI-K31-PRO.cmd
# (K3_INSTALL, default C:\colibri: release r12 and the user's launcher, 24 threads), on port 18081 with -Sessions 0 --
# the request streamed by stream_time.py, the engine's lines of the turn appended to runs/long_answer.log, then
# the engine and the gateway stopped by their PIDs (nothing else). Requests: make_requests.py.
#   draft    the launcher's defaults: the draft model on, every model file on both drives
#   nodraft  the same with -NoDraft
SP="$(cd "$(dirname "$0")" && pwd)"
R="${RUNS:-$SP/runs}"   # where the runs go
I="$(cygpath -w "${K3_INSTALL:-/c/colibri}")"
cfg=$1; shift
case $cfg in
    draft)   largs='' ;;
    nodraft) largs='-NoDraft' ;;
    *) echo "unknown $cfg"; exit 1 ;;
esac
unset K3_DRAFT K3_DRAFT_P K3_DRAFT_MAX K3_DRAFT_DUMP K3_DRAFT_ORACLE K3_LOGITS K3_LOGITS_DECODE K3_HID_DUMP K3_HID_LAYERS \
      K3_MIRROR K3_FILL K3_MIRROR_SPLIT K3_MIRROR_ROWS K3_MIRROR_ANY K3_IO_PER_DRIVE K3_DIRS
mkdir -p "$R"
pids(){
    powershell.exe -NoProfile -Command "\$g = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*serve*--port 18081*' } | Select-Object -First 1; \$e = if (\$g) { Get-CimInstance Win32_Process -Filter \"Name='kimi_k3.exe' AND ParentProcessId=\$(\$g.ProcessId)\" | Select-Object -First 1 }; '' + \$g.ProcessId + ' ' + \$e.ProcessId" | tr -d '\r'
}
stop(){
    set -- $(pids)
    [ -n "$2" ] && powershell.exe -NoProfile -Command "Stop-Process -Id $2 -Force" && sleep 3
    [ -n "$1" ] && powershell.exe -NoProfile -Command "Stop-Process -Id $1 -Force"
    echo "   stopped: gateway ${1:-none}, engine ${2:-none}"
}
free_gb(){
    powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB)" | tr -d '\r'
}
for r in "$@"; do
    # the RAM of the run before back first: the engine sizes its expert cache by what is free
    for i in $(seq 1 60); do [ "$(free_gb)" -ge 165 ] && break; sleep 2; done
    n=$(ls -d "$R"/[0-9][0-9]_* 2>/dev/null | wc -l); n=$((n + 1))
    O="$R/$(printf %02d $n)_${cfg}_$(basename "$r" .json)"; mkdir -p "$O"
    printf '@echo off\r\ncd /d %s\r\ncall %s\\START-KIMI-K31-PRO.cmd -Port 18081 -Sessions 0 %s > "%%~dp0serve.log" 2>&1\r\n' \
        "$I" "$I" "$largs" > "$O/start.cmd"
    f0=$(free_gb)
    cmd //c "$(cygpath -w "$O/start.cmd")" &
    up=0
    for i in $(seq 1 150); do
        curl -s -m 2 http://127.0.0.1:18081/v1/models > /dev/null 2>&1 && { up=1; break; }
        sleep 2
    done
    if [ $up = 0 ]; then echo "FAILED: the server did not come up"; tail -20 "$O/serve.log"; stop; exit 1; fi
    echo "== $(basename "$O"): server up after $((i * 2)) s, $f0 GB free before it ($(date '+%Y-%m-%d %H:%M'))" | tee -a "$R/long_answer.log"
    python -I "$SP/stream_time.py" "$r" "$O/reply" | sed 's/^/   /' | tee -a "$R/long_answer.log"
    sleep 2
    grep -h -E "^Draft model:|init done|expert readers|prompt of [0-9]+ tokens|prefetch \(turn\)|\] reads \(|K3-DRAFT\] turn" "$O/serve.log" \
        | cut -c1-330 | sed 's/^/   /' | tee -a "$R/long_answer.log"
    stop | tee -a "$R/long_answer.log"
    sleep 5
done
