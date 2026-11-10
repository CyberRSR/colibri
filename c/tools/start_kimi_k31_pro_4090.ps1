# Kimi K3.1 Pro on the RTX 4090 / 192 GiB RAM / two-NVMe machine (docs/kimi_k3.md, Kimi K3.1).
# K3.1 Pro is Kimi K3 post-trained, with a DSA lightning indexer on each of its 24 MLA layers
# (dsa_config): past 2,048 positions a query row attends the 2,048 rows its layer's indexer
# picks. kimi_k3.exe needs to be a build that reads dsa_config (checked below); the rest is K3's
# profile. The checkpoint as tools/download_kimi_k31_pro_4090.ps1 lays it out: the model folder on C:
# and every MoE layer's other part on D:, read by two readers per drive. The largest
# dense matrices go to the GPU (K3_CUDA_DENSE), their host copies released for the
# expert cache. A decode step's routed experts run on the CPU, which reads them in place;
# a prompt step's go to the GPU too as they are read, with its other dense matrices
# (coli_cuda_k3.dll, the CPU's bytes), unless -CpuExperts.
param(
    # The engine's model folder (config, tokenizer, 61% of every layer's experts)
    [string]$ModelDir = 'C:\AI\Kimi-K3.1-Pro',
    # The other drive's folders, ';'-separated (K3_DIRS); "" for a model in one folder
    [string]$ShardDir = 'D:\AI\Models\Kimi-K3.1-Pro',
    # The model name the API answers to
    [string]$ModelId = 'kimi-k3.1-pro',
    # The whole process's RAM ceiling, GB: the expert cache gets what the rest leaves
    [ValidateRange(16,190)][double]$RamGB = 174,
    [ValidateRange(1,32)][int]$Threads = 16,
    [ValidateRange(1,8)][int]$IoPerDrive = 2,
    # Read the next MoE layer's predicted experts ahead (decode)
    [ValidateRange(0,1)][int]$Prefetch = 1,
    # Dense matrices on the GPU, GiB, and the VRAM left free beside them; -DenseGB 21.5
    # -ReserveGB 1 puts 35 more on the GPU (0.78 -> 0.81 tok/s measured) and leaves about
    # 0.6 GiB free: past the free memory Windows pages the GPU's to RAM and every call slows
    [ValidateRange(0,23)][double]$DenseGB = 20,
    [ValidateRange(0.5,8)][double]$ReserveGB = 2,
    # Context, tokens: the MLA caches and K3.1's index keys take 67 KB a token (8.8 GB at 128k).
    # The caches are allocated as the conversation reaches them, so an idle session costs the
    # expert cache 4k positions' worth, not this whole ceiling; a conversation that does run to
    # -Context takes its 8.8 GB out of the expert cache as it goes (K3_KV_GROW=0: all of it up
    # front). RamGB stays the ceiling either way: what the caches grow into is what the experts
    # give back, so a long conversation does not overrun the process' budget.
    [ValidateRange(512,1048576)][int]$Context = 131072,
    # The most tokens a reply may have when a request does not ask for fewer (reasoning included)
    [ValidateRange(256,1048576)][int]$MaxOutput = 32768,
    # Prompt tokens a prefill step takes at once. A step reads each expert it needs once for all
    # its rows, and a few hundred rows already need most of them (1.4 TB at 4,096 rows), so a
    # prompt costs about one pass over the drives per step: the fewer steps, the better. The
    # steps of a prompt are of one size and none is short: a prompt up to an eighth longer than
    # this is one step (18,432 rows), a longer one equal steps. Past 512 rows a step borrows 1.4
    # MB of RAM a row from the expert cache while it runs and gives it back at its end
    # (K3_LEND), so the size of a step costs decode nothing.
    [ValidateRange(1,16384)][int]$Chunk = 16384,
    # A long prompt's read-ahead pool, GB: the next MoE layer's experts read while this one
    # computes (2,637 tokens 170 -> 137 s at 20); its RAM comes out of the expert cache
    # (decode 0.83 -> 0.81 tok/s), 0 gives decode all of it
    [ValidateRange(0,64)][double]$ReadAheadGB = 20,
    # Photos of the recurrent state at every prompt's and reply's end (COLI_K3_CKPT, about
    # 434 MB each): a turn whose client sends the history back without its reasoning, or
    # edits a tool result, resumes from the deepest photo its prompt still matches and
    # prefills only the rest, instead of the whole conversation again; 0 = off
    [ValidateRange(0,8)][int]$Photos = 4,
    [ValidateRange(1,65535)][int]$Port = 18080,
    # The quantized dense matrices kept in a file (K3_DENSE_CACHE): the first start writes it
    # (33 GB beside the model, or at this path), the next ones read it instead of reading and
    # quantizing the checkpoint's 115 GB of BF16. The bytes are the same; the file is checked
    # against the checkpoint and written again when either changed. -NoDenseCache: off
    [string]$DenseCache = '',
    [switch]$NoDenseCache,
    # Half of every layer's expert cache seeded at the start from the experts the history used
    # most (AUTOPIN) and never evicted. Off here: K3's router spreads its use evenly, so the
    # history's favourites are found little more often than any others (5.7% of a prompt's
    # experts against 4.7% at random) while the half of the cache they hold is taken from the
    # one that follows the conversation -- decode 0.78 against 0.85 tok/s measured, and 7 s of
    # reads (68 GB) at every start. -Pins turns it on
    [switch]$Pins,
    # No GPU: the dense matrices stay on the CPU (a CPU build, or to compare)
    [switch]$Cpu,
    # A prompt step's routed experts and other dense matrices on the CPU alone (K3_GPU_EXPERTS=0)
    [switch]$CpuExperts,
    # Start even when coli_cuda_k3.dll cannot do K3.1's indexer: the picks then run on the CPU,
    # which is correct to the bit but keeps a long prompt's drives and card idle for minutes
    [switch]$AllowCpuIndexer
)
$ErrorActionPreference = 'Stop'
$EngineDir = Split-Path -Parent $PSScriptRoot
$need = @('kimi_k3.exe', 'coli')
if (!$Cpu) { $need += 'coli_cuda.dll' }
foreach ($file in $need) {
    if (!(Test-Path -LiteralPath (Join-Path $EngineDir $file) -PathType Leaf)) {
        throw "Missing $file in $EngineDir. See docs/kimi_k3.md (Kimi K3.1) for the build commands (or pass -Cpu)."
    }
}
# A kimi_k3.exe older than K3.1 ignores dsa_config and runs every MLA layer dense: the same up
# to 2,048 positions, another model's attention past them
$engineText = [System.IO.File]::ReadAllText((Join-Path $EngineDir 'kimi_k3.exe'), [System.Text.Encoding]::ASCII)
if (!$engineText.Contains('dsa_config')) {
    throw "kimi_k3.exe in $EngineDir predates Kimi K3.1 (it does not read dsa_config): build it from this branch (build-windows.ps1)"
}
Remove-Variable engineText
if (!$Cpu -and !$CpuExperts -and !(Test-Path -LiteralPath (Join-Path $EngineDir 'coli_cuda_k3.dll') -PathType Leaf)) {
    Write-Warning "No coli_cuda_k3.dll in $EngineDir (c\build_k3_cuda.bat): a prompt's experts stay on the CPU"
}
# K3.1's prompt picks need the card's indexer: a coli_cuda_k3.dll older than dsa_config binds
# without a word (k3cu_dsa_mask is an optional export) and the engine then scores every MLA layer
# on the CPU, which is the same to the bit but holds a long prompt for minutes with the drives and
# the card idle. The engine says so once, in a line nobody reads; say it where the user looks.
# -Cpu and -CpuExperts already chose the CPU's arithmetic, so there the banner says it and stops.
$indexerOnGpu = $false
if (!$Cpu -and !$CpuExperts) {
    $dll = Join-Path $EngineDir 'coli_cuda_k3.dll'
    if (Test-Path -LiteralPath $dll -PathType Leaf) {
        $dllText = [System.IO.File]::ReadAllText($dll, [System.Text.Encoding]::ASCII)
        $indexerOnGpu = $dllText.Contains('k3cu_dsa_mask')
        Remove-Variable dllText
    }
    if (!$indexerOnGpu -and !$AllowCpuIndexer) {
        throw "coli_cuda_k3.dll in $EngineDir $(if (Test-Path -LiteralPath $dll) { 'predates Kimi K3.1 (no k3cu_dsa_mask)' } else { 'is missing' }): a prompt's index scores would run on the CPU, minutes for a long prompt with the card idle. Build both binaries with deploy\kimi-k31\build-windows.ps1, or pass -AllowCpuIndexer to run them anyway."
    }
}
foreach ($file in @('config.json', 'tokenizer.json')) {
    if (!(Test-Path -LiteralPath (Join-Path $ModelDir $file) -PathType Leaf)) {
        throw "Missing $file in $ModelDir (tools/download_kimi_k31_pro_4090.ps1 fetches the model and writes tokenizer.json)"
    }
}
$shardDirs = @($ShardDir -split ';' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
foreach ($dir in $shardDirs) {
    if (!(Test-Path -LiteralPath $dir -PathType Container)) { throw "Missing shard directory: $dir" }
}
$layout = Join-Path $ModelDir 'k3_layout.json'
if (Test-Path -LiteralPath $layout -PathType Leaf) {
    $state = Get-Content -LiteralPath $layout -Raw | ConvertFrom-Json
    if (!$state.complete) {
        throw "$ModelDir is not completely downloaded yet (k3_layout.json); rerun tools/download_kimi_k31_pro_4090.ps1"
    }
}
$env:K3_DIRS = $shardDirs -join ';'
$env:K3_CUDA = '0'                     # no VRAM tier: a decode step's experts on the CPU, read in place
$env:K3_GPU_EXPERTS = $(if ($Cpu -or $CpuExperts) { '0' } else { '1' })
$env:K3_READAHEAD_GB = $(if ($Cpu -or $CpuExperts) { '0' } else { [string]$ReadAheadGB })
$env:K3_CUDA_DENSE = $(if ($Cpu -or $DenseGB -le 0) { '0' } else { '1' })
$env:K3_CUDA_DENSE_GB = [string]$DenseGB
$env:K3_CUDA_RESERVE_GB = [string]$ReserveGB
$env:K3_CUDA_RELEASE_HOST = '1'
$env:K3_EXPERT_GB = '1000'             # the cache takes what RamGB leaves (the engine clamps it)
$env:K3_MAXT = [string]$Context
$env:K3_CHUNK = [string]$Chunk
$env:K3_DENSE_CACHE = $(if ($NoDenseCache) { '0' } elseif ($DenseCache) { $DenseCache } else { '1' })
$env:AUTOPIN = $(if ($Pins) { '1' } else { '0' })
$env:K3_DIRECT = '1'
$env:K3_PIPE = '1'
$env:K3_IO_PER_DRIVE = [string]$IoPerDrive
$env:K3_PREFETCH = [string]$Prefetch
$env:COLI_K3_CKPT = [string]$Photos
$env:OMP_NUM_THREADS = [string]$Threads
$env:COLI_NO_OMP_TUNE = '1'
$env:COLI_VULKAN = '0'
$env:COLI_CUDA = '0'                   # the launcher's own GPU tier is for other engines
$env:COLI_GPU = '0'
foreach ($key in @('K3_LAYERS', 'K3_TRACE', 'K3_LOGITS', 'K3_X0', 'K3_MMAP', 'K3_IO_THREADS', 'K3_LOAD_THREADS',
                   'K3_IO_DELAY_US', 'K3_VK', 'COLI_GPU_FAIL_AFTER', 'K3_VAL_LOGITS', 'K3_VALIDATE_LAYER',
                   'COLI_K3_CKPT_DIR')) {
    Remove-Item -LiteralPath "Env:$key" -ErrorAction SilentlyContinue
}
Write-Host "Kimi K3.1 Pro: $ModelDir$(if ($shardDirs.Count) { '; experts also from ' + ($shardDirs -join '; ') })"
Write-Host "RAM $RamGB GB, CPU $Threads, readers $IoPerDrive per drive, prefetch $Prefetch, $(if ($env:K3_CUDA_DENSE -eq '1') { "dense matrices on the GPU up to $DenseGB GiB" } else { 'dense matrices on the CPU' })$(if ($env:K3_GPU_EXPERTS -eq '1') { ', a prompt step''s experts on the GPU too' })"
Write-Host "K3.1's index scores and picks: $(if ($indexerOnGpu) { 'on the GPU (k3cu_dsa_mask)' } elseif ($Cpu -or $CpuExperts) { 'on the CPU, as asked' } else { 'ON THE CPU: a long prompt waits here, see coli_cuda_k3.dll' })"
Write-Host "Context $Context tokens, replies up to $MaxOutput tokens, prompt steps of $Chunk (read-ahead $($env:K3_READAHEAD_GB) GB), $Photos turn photos"
Write-Host "Dense matrices' quantization: $(if ($NoDenseCache) { 'from the checkpoint at every start (-NoDenseCache)' } else { 'kept in ' + $(if ($DenseCache) { $DenseCache } else { Join-Path $ModelDir 'k3_dense_cache.bin' }) + ' (33 GB, written by the first start)' })"
Write-Host "API: http://127.0.0.1:$Port/v1, model $ModelId"
& python (Join-Path $EngineDir 'coli') serve --model $ModelDir --ram $RamGB --ctx $Context `
    --ngen $MaxOutput --host 127.0.0.1 --port $Port --model-id $ModelId --kv-slots 1 `
    --queue-timeout 7200 --max-queue 8
exit $LASTEXITCODE
