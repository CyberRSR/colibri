# Native MiMo CUDA profile for RTX 4090 / 192 GiB RAM / split model shards.
param(
    [string]$ModelDir = 'D:\AI\Models\MiMo-V2.6-Pro-MOPD',
    [string]$ShardDir = 'C:\AI\MiMoPro-Shards',
    [ValidateRange(8,104)][int]$Cap = 104,
    [ValidateRange(1,32)][int]$Threads = 16,
    [ValidateRange(1,8)][int]$IoPerDrive = 2,
    [ValidateRange(0,2)][int]$Prefetch = 1,
    [ValidateRange(1,21)][double]$DenseGB = 20,
    # VRAM left free for scratch and other programs; -DenseGB 21 -ReserveGB 1 puts
    # five more o_proj on the GPU (+2% measured) and leaves about 1 GiB free
    [ValidateRange(0.5,8)][double]$ReserveGB = 2,
    [ValidateRange(512,32768)][int]$Context = 16384,
    [ValidateRange(1,65535)][int]$Port = 18080,
    # Faster and NOT exact: every o_proj on the GPU as int8 (docs/mimo-cuda.md)
    [switch]$OprojInt8
)
$ErrorActionPreference = 'Stop'
$EngineDir = Split-Path -Parent $PSScriptRoot
foreach ($file in @('mimo.exe', 'coli_cuda_mimo.dll', 'coli')) {
    if (!(Test-Path -LiteralPath (Join-Path $EngineDir $file) -PathType Leaf)) {
        throw "Missing $file in $EngineDir. See docs/mimo-cuda.md for the build commands."
    }
}
if (!(Test-Path -LiteralPath (Join-Path $ModelDir 'config.json') -PathType Leaf)) {
    throw "Missing model config in $ModelDir"
}
if ($ShardDir -and !(Test-Path -LiteralPath $ShardDir -PathType Container)) {
    throw "Missing shard directory: $ShardDir"
}
$env:MIMO_DIRS = $ShardDir
$env:MIMO_CUDA = '1'
if ($OprojInt8 -and -not $PSBoundParameters.ContainsKey('DenseGB')) { $DenseGB = 20.2 }
$env:MIMO_CUDA_DENSE_GB = [string]$DenseGB
$env:MIMO_CUDA_OPROJ_INT8 = $(if ($OprojInt8) { '1' } else { '0' })
$env:MIMO_CUDA_RESERVE_GB = [string]$ReserveGB
$env:MIMO_CUDA_RELEASE_HOST = '1'
$env:MIMO_CHUNK = '512'
$env:MIMO_CTX = [string]$Context
$env:MIMO_DENSE_BITS = '0'
$env:MIMO_IDOT = '0'
$env:MIMO_DIRECT = '1'
$env:MIMO_IO_PER_DRIVE = [string]$IoPerDrive
$env:MIMO_PREFETCH = [string]$Prefetch
$env:OMP_NUM_THREADS = [string]$Threads
$env:COLI_NO_OMP_TUNE = '1'
$env:COLI_VULKAN = '0'
$env:COLI_GPU = '0'
foreach ($key in @('MIMO_CUDA_FAULT', 'MIMO_LAYERS', 'MIMO_TRACE', 'MIMO_LOGITS', 'MIMO_ROUTE_TRACE',
                   'MIMO_IO_THREADS', 'MIMO_READ_THREADS')) {
    Remove-Item -LiteralPath "Env:$key" -ErrorAction SilentlyContinue
}
Write-Host "MiMo native CUDA: dense budget $DenseGB GiB, cache $Cap experts/layer, CPU $Threads, readers $IoPerDrive per drive, prefetch $Prefetch$(if ($OprojInt8) { ', o_proj int8 (not exact)' })"
Write-Host "API: http://127.0.0.1:$Port/v1"
& python (Join-Path $EngineDir 'coli') serve --model $ModelDir --cap $Cap --ram 174 --ctx $Context `
    --host 127.0.0.1 --port $Port --model-id 'mimo-v2.6-pro-mopd' --kv-slots 1 `
    --queue-timeout 7200 --max-queue 8
exit $LASTEXITCODE
