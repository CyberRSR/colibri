#!/bin/bash
# Build the scratch engines (make_variants.py's sources) with mimo's Makefile command line
# (mingw32-make -C c mimo CUDA_DLL=1 ARCH=native VK=0), each beside a copy of the CUDA DLL.
set -e
export PATH="/c/Users/Cyber/AppData/Local/Nuitka/Nuitka/Cache/downloads/gcc/x86_64/14.2.0posix-19.1.1-12.0.0-msvcrt-r2/mingw64/bin:$PATH"
cd /c/colibri/dev/c
for v in "$@"; do
  gcc -D_FILE_OFFSET_BITS=64 -O3 -march=native -fopenmp -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation \
    -Wno-unused-function -DCOLI_CUDA -DCOLI_MIMO_CUDA -I. build/sample-draft/$v/mimo.c backend_cuda_mimo_loader.o \
    -o build/sample-draft/$v/mimo.exe -lm -fopenmp -static -lpsapi
  cp coli_cuda_mimo.dll build/sample-draft/$v/
  sha256sum build/sample-draft/$v/mimo.exe
done
