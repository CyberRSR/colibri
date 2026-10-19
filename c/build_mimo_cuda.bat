@echo off
rem Build the native MiMo dense CUDA DLL; no system PATH or installed release changes.
rem CUDA_ARCH defaults to RTX 4090 (sm_89); override for other GPUs.
setlocal
if "%CUDA_ARCH%"=="" set "CUDA_ARCH=sm_89"
where cl.exe >nul 2>&1
if not errorlevel 1 goto compiler_ready
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (echo Visual Studio C++ tools not found.>&2 & exit /b 1)
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "MIMO_VS=%%i"
if "%MIMO_VS%"=="" (echo Visual Studio C++ tools not found.>&2 & exit /b 1)
call "%MIMO_VS%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
:compiler_ready
set "MIMO_NVCC=nvcc"
if not "%CUDA_PATH%"=="" set "MIMO_NVCC=%CUDA_PATH%\bin\nvcc.exe"
pushd "%~dp0"
if "%~1"=="test" goto tests
"%MIMO_NVCC%" -O3 -std=c++17 -arch=%CUDA_ARCH% -ftz=false -Xcompiler=-W3 -shared --cudart=static -DMIMO_CUDA_BUILDING_DLL backend_cuda_mimo.cu -o coli_cuda_mimo.dll
set "RC=%ERRORLEVEL%"
goto done
:tests
"%MIMO_NVCC%" -O3 -std=c++17 -arch=%CUDA_ARCH% -ftz=false -Xcompiler=-W3 --cudart=static tests\test_mimo_cuda.cu backend_cuda_mimo.cu -o tests\test_mimo_cuda.exe
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" goto done
tests\test_mimo_cuda.exe
set "RC=%ERRORLEVEL%"
:done
popd
exit /b %RC%
