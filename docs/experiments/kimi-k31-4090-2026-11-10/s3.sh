#!/bin/bash
# s3.sh -- the history's pins (AUTOPIN, a serve's default) against none, in decode: the same prompt, 60 tokens,
# alternated twice; the dense cache read at each start.
SP="${K31_SP:-$(cd "$(dirname "$0")" && pwd)}"   # the folder with stage/ (the engine and its DLLs), runs/, ref/ and the prompts
show(){ printf "%s: " "$1"; grep -h -E 'init done|dense cache|PIN\]|prefill done|decode [0-9]+ tokens|a decode token|rc=' "$SP/runs/$1.err" | cut -c1-330; }
for i in 1 2; do
    bash "$SP/k31run.sh" pin_on$i 60 AUTOPIN=1 K3_DENSE_CACHE=1 > /dev/null; show pin_on$i
    bash "$SP/k31run.sh" pin_off$i 60 AUTOPIN=0 K3_DENSE_CACHE=1 > /dev/null; show pin_off$i
done
echo series done
