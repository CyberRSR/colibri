#!/bin/bash
# serve_ab.sh <tag> <launcher args...> -- <request.json> [more requests...]
# The engine behind its gateway on port 18081 with the launcher's arguments, each request
# sent in order (a name ending in @2 is sent twice: the second time its prompt's photo is restored and
# the time is the reply's alone), the responses and times kept in sab_<tag>/, then both processes
# stopped by their PIDs.
D="$(cd "$(dirname "$0")" && pwd)"
W="${K3_ENGINE_DIR:-$(cd "$D/../../../../c" && pwd)}"   # the folder with kimi_k3.exe, coli and tools/
tag=$1; shift
largs=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do largs+=("$1"); shift; done
shift
R="$D/runs/sab_$tag"; mkdir -p "$R"; rm -f "$R"/*.resp.json "$R/serve.log" "$R/times.txt"
cat > "$R/start.cmd" <<EOF
@echo off
cd /d $(cygpath -w "$W")
powershell -NoProfile -ExecutionPolicy Bypass -File tools\\start_kimi_k31_pro_4090.ps1 -Port 18081 -Sessions 0 ${largs[*]} > "%~dp0serve.log" 2>&1
EOF
unset K3_HID_DUMP K3_HID_LAYERS K3_LOGITS K3_DRAFT_DUMP
cd "$W" || exit 1
cmd //c "$(cygpath -w "$R/start.cmd")" &
for i in $(seq 1 150); do
    curl -s -m 2 http://127.0.0.1:18081/v1/models > /dev/null 2>&1 && break
    sleep 2
done
curl -s -m 2 http://127.0.0.1:18081/v1/models > /dev/null 2>&1 || { echo "FAILED: the server did not come up"; tail -5 "$R/serve.log"; }
echo "[$tag] server up after $((i * 2)) s"
n=0
for spec in "$@"; do
    r=${spec%@2}; times=1; [ "$spec" != "$r" ] && times=2
    for k in $(seq 1 $times); do
        n=$((n + 1))
        t0=$(date +%s%N)
        code=$(curl -s -m 7200 -o "$R/$n.resp.json" -w '%{http_code}' -H 'Content-Type: application/json' --data-binary "@$D/$r" http://127.0.0.1:18081/v1/chat/completions)
        ms=$(( ($(date +%s%N) - t0) / 1000000 ))
        echo "$n $r pass $k: HTTP $code in $ms ms" | tee -a "$R/times.txt"
    done
done
powershell.exe -NoProfile -Command "\$e = Get-Process kimi_k3 -ErrorAction SilentlyContinue; \$g = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*serve*--port 18081*' }; foreach (\$p in \$e) { Stop-Process -Id \$p.Id -Force }; Start-Sleep 3; foreach (\$p in \$g) { Stop-Process -Id \$p.ProcessId -Force }; 'stopped: gateway ' + (\$g | Measure-Object).Count + ', engine ' + (\$e | Measure-Object).Count"
grep -h -E "K3-DRAFT\] (turn|on|off|C:)|\[K3-CUDA\] dense:|prompt of [0-9]+ tokens" "$R/serve.log" | cut -c1-240
echo "[$tag] done"
