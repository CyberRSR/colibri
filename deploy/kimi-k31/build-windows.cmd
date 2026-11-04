@echo off
rem Build Kimi K3.1 Pro's engine (kimi_k3.exe, coli_cuda_k3.dll) and put it into the install: build-windows.ps1
rem Arguments go to the script, e.g.  build-windows.cmd -Zip   or   build-windows.cmd -CudaArch sm_120 -SkipTests
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build-windows.ps1" %*
if errorlevel 1 pause
