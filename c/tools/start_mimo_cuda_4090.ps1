# Native MiMo CUDA profile for RTX 4090 / 192 GiB RAM / split model shards.
param(
    [string]$ModelDir = 'D:\AI\Models\MiMo-V2.6-Pro-MOPD',
    [string]$ShardDir = 'C:\AI\MiMoPro-Shards',
    [ValidateRange(8,100)][int]$Cap = 88,
    [ValidateRange(1,32)][int]$Threads = 24,
    [ValidateRange(1,32)][int]$ReadThreads = 16,
    [ValidateRange(1,21)][double]$DenseGB = 20,
    [ValidateRange(512,32768)][int]$Context = 16384,
    [ValidateRange(1,65535)][int]$Port = 18080
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
$env:MIMO_CUDA_DENSE_GB = [string]$DenseGB
$env:MIMO_CUDA_RESERVE_GB = '2'
$env:MIMO_DENSE_BITS = '0'
$env:MIMO_IDOT = '0'
$env:MIMO_DIRECT = '1'
$env:MIMO_READ_THREADS = [string]$ReadThreads
$env:OMP_NUM_THREADS = [string]$Threads
$env:COLI_NO_OMP_TUNE = '1'
$env:COLI_VULKAN = '0'
Write-Host "MiMo native CUDA: dense budget $DenseGB GiB, cache $Cap experts/layer, CPU $Threads, readers $ReadThreads"
Write-Host "API: http://127.0.0.1:$Port/v1"
& python (Join-Path $EngineDir 'coli') serve --model $ModelDir --cap $Cap --ram 160 --ctx $Context `
    --host 127.0.0.1 --port $Port --model-id 'mimo-v2.6-pro-mopd' --kv-slots 1 `
    --queue-timeout 7200 --max-queue 8
exit $LASTEXITCODE
