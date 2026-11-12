# Build Kimi K3.1 Pro's engine on Windows 11 and put it into the install beside this script.
#
#   kimi_k3.exe       MinGW-w64 GCC:  mingw32-make kimi_k3.exe CUDA_DLL=1 ARCH=native VK=0
#   coli_cuda_k3.dll  CUDA toolkit + Visual Studio 2022's x64 C++ tools (c\build_k3_cuda.bat)
#   coli_cuda.dll     unchanged by K3.1: the release's is kept (-CudaDll rebuilds it, c\build_cuda.bat)
#
# Then (unless -SkipTests) the checks: the indexer's unit test, the tiny K3 and K3.1 models
# against their references (numpy needed) and, on the GPU, coli_cuda_k3.dll
# against the CPU's arithmetic bit for bit, K3.1's masked attention included
# (tests\test_k3_cuda_experts.exe). Then the binaries go into the install, their SHA-256 into
# BUILD-INFO-K31.txt, and with -Zip the install into a zip beside this script.
#
# Needs on PATH: gcc and mingw32-make (MinGW-w64 built against msvcrt -- winlibs
# x86_64-...-mingw-w64msvcrt-...: a UCRT gcc, MSYS2's ucrt64 included, gives the engine a
# C library whose expf and tanhf are not the card's, and it then keeps a prompt step's experts
# on the CPU; the script says so and stops unless -AllowUcrt), python 3.10+,
# git (for the commit in BUILD-INFO-K31.txt; optional). The CUDA toolkit (12.x or 13.x; CUDA_PATH
# is set by its installer) and Visual Studio 2022 or its Build Tools with "Desktop development
# with C++" are found by c\build_k3_cuda.bat itself. Run from any directory:
#   powershell -ExecutionPolicy Bypass -File deploy\kimi-k31\build-windows.ps1 [-Zip]
param(
    # The colibri checkout (this script's ..\..)
    [string]$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,
    # The install the binaries go into
    [string]$Bundle = (Join-Path $PSScriptRoot 'colibri-kimi-k31-pro-4090-r9-windows-x86_64'),
    # The GPU's architecture: sm_89 = RTX 40xx, sm_120 = RTX 50xx, sm_86 = RTX 30xx
    [string]$CudaArch = 'sm_89',
    # Rebuild coli_cuda.dll too
    [switch]$CudaDll,
    [switch]$SkipTests,
    # Build with a UCRT gcc anyway: the engine then keeps a prompt step's experts, its gated
    # MLA and KDA layers and its index picks on the CPU (see the C runtime check below)
    [switch]$AllowUcrt,
    # Leave the install as it is (build and test only)
    [switch]$NoCopy,
    [switch]$Zip
)
$ErrorActionPreference = 'Stop'

function Find-Tool([string[]]$names, [string]$hint) {
    foreach ($n in $names) {
        $c = Get-Command $n -ErrorAction SilentlyContinue
        if ($c) { return $c.Source }
    }
    throw "$($names -join ' / ') not found on PATH: $hint"
}
function Run([string]$what, [scriptblock]$block) {
    Write-Host "== $what" -ForegroundColor Cyan
    & $block
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE)" }
}
# The C runtime gcc links against, from the macros it predefines. A UCRT one gives the engine
# its own expf and tanhf, which are not the bits coli_cuda_k3.dll's tables hold, so the engine
# refuses to hand it anything: the routed experts, the gated MLA and KDA layers and the index
# picks all stay on the CPU. Nothing else about the build changes.
function Get-GccCrt([string]$gcc) {
    $probe = Join-Path ([IO.Path]::GetTempPath()) ('k3crt' + [IO.Path]::GetRandomFileName() + '.c')
    Set-Content -LiteralPath $probe -Value '' -Encoding ASCII
    try {
        $defines = & $gcc -dM -E $probe 2>$null
        if ($LASTEXITCODE -ne 0) { return 'unknown' }
        if (@($defines | Select-String -Pattern '^#define _UCRT(\s|$)' -Quiet)) { return 'ucrt' }
        return 'msvcrt'
    } finally { Remove-Item -LiteralPath $probe -Force -ErrorAction SilentlyContinue }
}

$gcc = Find-Tool @('gcc') 'install MinGW-w64 GCC (an msvcrt one: winlibs-x86_64-posix-seh-gcc-<v>-mingw-w64msvcrt-<v>.zip; MSYS2 ucrt64 works but keeps the prompt on the CPU, see below)'
$make = Find-Tool @('mingw32-make', 'make') 'install mingw32-make with MinGW-w64 GCC'
$python = Find-Tool @('python', 'py') 'install Python 3 (python.org)'
$c = Join-Path $Repo 'c'
if (!(Test-Path -LiteralPath (Join-Path $c 'kimi_k3.c'))) { throw "no c\kimi_k3.c under $Repo (pass -Repo)" }
if (!(Select-String -LiteralPath (Join-Path $c 'kimi_k3.c') -Pattern 'dsa_config' -SimpleMatch -Quiet)) {
    throw "$Repo is not a checkout with Kimi K3.1's engine (no dsa_config in c\kimi_k3.c): check out branch claude/kimi-k31-dsa"
}
$crt = Get-GccCrt $gcc
Write-Host "gcc: $gcc ($crt)"
Write-Host "make: $make"
Write-Host "python: $python"
Write-Host "CUDA arch: $CudaArch$(if ($env:CUDA_PATH) { ", CUDA_PATH $env:CUDA_PATH" })"
if ($crt -eq 'ucrt') {
    $msg = "this gcc targets the UCRT: kimi_k3.exe will keep a prompt step's routed experts, its gated " +
           'MLA and KDA layers and its index picks on the CPU, which costs minutes per 8192-row step ' +
           '(the engine says "experts: off" when it starts). Build with an msvcrt MinGW-w64 -- winlibs ' +
           'x86_64-...-mingw-w64msvcrt-... -- or pass -AllowUcrt to accept that engine.'
    if (!$AllowUcrt) { throw $msg }
    Write-Warning $msg
}

Push-Location $c
try {
    $env:CUDA_ARCH = $CudaArch
    Run 'coli_cuda_k3.dll (build_k3_cuda.bat: nvcc + MSVC)' { cmd /c build_k3_cuda.bat }
    if ($CudaDll) { Run 'coli_cuda.dll (build_cuda.bat)' { cmd /c build_cuda.bat } }
    else {   # the engine and the GPU test load coli_cuda.dll from c\: the release's
        foreach ($f in @('coli_cuda.dll', 'cudart64_12.dll')) {
            $src = Join-Path $Bundle $f
            if (!(Test-Path -LiteralPath (Join-Path $c $f)) -and (Test-Path -LiteralPath $src)) { Copy-Item -LiteralPath $src -Destination $c }
        }
        if (!(Test-Path -LiteralPath (Join-Path $c 'coli_cuda.dll'))) { throw "no coli_cuda.dll in ${c} or ${Bundle}: pass -CudaDll to build it" }
    }
    $flags = @('CUDA_DLL=1', 'ARCH=native', 'VK=0')
    Run 'kimi_k3.exe (MinGW-w64)' { & $make kimi_k3.exe @flags }
    if (!$SkipTests) {
        $hasNumpy = $true
        & $python -c "import numpy" 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "numpy is missing: python -m pip install numpy (the tiny models' generator and K3.1's reference need it); skipping those checks"
            $hasNumpy = $false
        }
        Run 'tests\test_k3_dsa.exe (build)' { & $make tests/test_k3_dsa.exe @flags }
        Run 'the indexer''s scores, pick and LayerNorm' { & .\tests\test_k3_dsa.exe }
        if ($hasNumpy) {
            Run 'the tiny fixtures' {
                & $python tools/make_kimi_k3_tiny.py --output ./kimi_k3_tiny --force
                if ($LASTEXITCODE -eq 0) { & $python tools/make_kimi_k3_tiny.py --output ./kimi_k31_tiny --force --dsa }
            }
            Run 'the tiny K3 model against Moonshot''s reference' {
                & $python tests/test_kimi_k3_tiny.py --binary ./kimi_k3.exe --fixture ./kimi_k3_tiny }
            Run 'the tiny K3.1 model against the numpy DSA reference' {
                & $python tests/test_kimi_k31_dsa.py --binary ./kimi_k3.exe --fixture ./kimi_k31_tiny --k3-fixture ./kimi_k3_tiny }
            Run 'the expert cache lending its slots (the caches grown, a large step): the same logits' {
                & $python tests/kimi_k3_lend_harness.py --binary ./kimi_k3.exe --fixture ./kimi_k31_tiny }
            Run 'the dense cache written, read, stale and damaged: the same logits' {
                & $python tests/kimi_k3_dense_cache_harness.py --binary ./kimi_k3.exe --fixture ./kimi_k31_tiny }
            Run 'the photos: divergent turns, and a request sent again' {
                & $python tests/test_kimi_k3_ckpt.py --binary ./kimi_k3.exe --fixture ./kimi_k31_tiny }
            Run 'the photo of the system prompt''s end: a new conversation starts after it' {
                & $python tests/kimi_k3_root_harness.py --binary ./kimi_k3.exe --fixture ./kimi_k31_tiny }
            Run 'conversations kept aside, in RAM and in files, across a restart; the caches'' count; the pins'' floor' {
                & $python tests/kimi_k3_sessions_harness.py --binary ./kimi_k3.exe --fixture ./kimi_k31_tiny }
        }
        Run 'tests\test_k3_cuda_experts.exe (build)' { & $make tests/test_k3_cuda_experts.exe @flags }
        Run 'coli_cuda_k3.dll against the CPU, K3.1''s masked attention' { & .\tests\test_k3_cuda_experts.exe --dsa }
        Run 'coli_cuda_k3.dll against the CPU, everything else' { & .\tests\test_k3_cuda_experts.exe }
    }
} finally {
    Pop-Location
}

if ($NoCopy) { Write-Host 'Built (not copied: -NoCopy).'; exit 0 }
if (!(Test-Path -LiteralPath $Bundle -PathType Container)) { throw "no install at $Bundle (pass -Bundle)" }
$files = @('kimi_k3.exe', 'coli_cuda_k3.dll')
if ($CudaDll) { $files += 'coli_cuda.dll' }
foreach ($f in $files) { Copy-Item -LiteralPath (Join-Path $c $f) -Destination (Join-Path $Bundle $f) -Force }
$commit = ''
if (Get-Command git -ErrorAction SilentlyContinue) { $commit = (& git -C $Repo rev-parse HEAD 2>$null) }
# ARCH=native is this board's instruction set: an AVX-512 build refuses to run on a CPU
# without it, and nothing says so but the line below.
$cpu = ((Get-ItemProperty -LiteralPath 'HKLM:\HARDWARE\DESCRIPTION\System\CentralProcessor\0' -Name ProcessorNameString -ErrorAction SilentlyContinue).ProcessorNameString -replace "`0", '').Trim()
$lines = @("Kimi K3.1 Pro's engine, built $(Get-Date -Format 'yyyy-MM-dd HH:mm') on $env:COMPUTERNAME",
           "commit: $(if ($commit) { $commit } else { 'unknown' })",
           "kimi_k3.exe: mingw32-make -C c kimi_k3 CUDA_DLL=1 ARCH=native VK=0 on the $crt C runtime ($((& $gcc --version)[0]))",
           $(if ($cpu) { "ARCH=native is $cpu: rebuild with ARCH=znver4 / x86-64-v3 before moving the binaries to another box" } else { 'ARCH=native is this machine''s CPU: rebuild before moving the binaries to another box' }),
           "coli_cuda_k3.dll: c\build_k3_cuda.bat, CUDA_ARCH=$CudaArch",
           $(if ($CudaDll) { "coli_cuda.dll: c\build_cuda.bat" } else { "coli_cuda.dll: release kimi-k3-4090-r1's (unchanged by K3.1)" }),
           $(if ($SkipTests) { 'checks: skipped (-SkipTests)' } else { 'checks: test_k3_dsa, test_kimi_k3_tiny, test_kimi_k31_dsa, tests\test_k3_cuda_experts.exe (--dsa and all) passed' }),
           '')
foreach ($f in @('kimi_k3.exe', 'coli_cuda_k3.dll', 'coli_cuda.dll')) {
    $p = Join-Path $Bundle $f
    if (Test-Path -LiteralPath $p) { $lines += "$((Get-FileHash -Algorithm SHA256 -LiteralPath $p).Hash.ToLower())  $f" }
}
Set-Content -LiteralPath (Join-Path $Bundle 'BUILD-INFO-K31.txt') -Value $lines -Encoding ASCII
Write-Host "Copied $($files -join ', ') into $Bundle; BUILD-INFO-K31.txt written" -ForegroundColor Green

if ($Zip) {
    $zipPath = Join-Path $PSScriptRoot ((Split-Path -Leaf $Bundle) + '.zip')
    if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath }
    # bsdtar, not Compress-Archive: PowerShell 5.1 writes its entry names with a backslash,
    # and outside Windows that unpacks as files named "web\index.html" instead of a tree.
    $tar = Join-Path $env:SystemRoot 'System32\tar.exe'
    if (!(Test-Path -LiteralPath $tar)) { throw "no $tar to pack the install with" }
    & $tar -a -cf $zipPath -C (Split-Path -Parent $Bundle) (Split-Path -Leaf $Bundle)
    if ($LASTEXITCODE -ne 0) { throw "packing the install failed (exit $LASTEXITCODE)" }
    Write-Host "Zip: $zipPath ($([math]::Round((Get-Item $zipPath).Length / 1MB, 1)) MB)" -ForegroundColor Green
}
