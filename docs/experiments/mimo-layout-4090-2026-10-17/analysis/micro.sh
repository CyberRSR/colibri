#!/bin/bash
# The placement sweeps: libmingwex's __expl_internal (its bytes from HEAD's build) and exp_f.h's
# sequence, every 16-byte slot of a page and every byte of the first 2 KiB, pinned to CPU 4.
cd /c/colibri/dev/c/build/layout
export PATH="/c/Users/Cyber/AppData/Local/Nuitka/Nuitka/Cache/downloads/gcc/x86_64/14.2.0posix-19.1.1-12.0.0-msvcrt-r2/mingw64/bin:$PATH"
mkdir -p micro
for kind in mingw fisttp; do
  for g in "256 16" "2048 1"; do
    set -- $g
    if [ $kind = mingw ]; then python gen_xexp.py mingw bin-before/mimo.exe $1 $2 micro/x.S; else python gen_xexp.py fisttp $1 $2 micro/x.S; fi
    gcc -O2 -o micro/xbench.exe xbench.c micro/x.S && micro/xbench.exe 4 100000 > micro/$kind-$1x$2.txt
    echo "$kind $1x$2: $(grep -c SLOW micro/$kind-$1x$2.txt) slow; $(awk 'NR>1{print $4}' micro/$kind-$1x$2.txt | sort -n | awk '{a[NR]=$1} END{print "min",a[1],"median",a[int(NR/2)],"max",a[NR]}')"
  done
done
gcc -O3 -march=x86-64-v3 -fopenmp -Wall ../../tests/exp_f_check.c -o micro/exp_f_check.exe -static -lm && micro/exp_f_check.exe > micro/exp_f_check.txt
cat micro/exp_f_check.txt
rm -f micro/x.S micro/xbench.exe micro/exp_f_check.exe
