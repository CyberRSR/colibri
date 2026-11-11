param([string]$Out, [string]$StopFile, [int]$Seconds = 2)
# monitor.ps1's columns plus the GPU memory WDDM gives kimi_k3 (GPUProcessMemory: dedicated = VRAM,
# shared = system memory the GPU maps, pinned/registered included) and the adapter's totals, in MB.
$inv = [System.Globalization.CultureInfo]::InvariantCulture
"time;c_mbs;d_mbs;c_busy;d_busy;cpu;gpu;gpu_mem_mb;ws_gb;proc_cpu_s;p_ded_mb;p_shr_mb;p_commit_mb;a_ded_mb;a_shr_mb" | Out-File -FilePath $Out -Encoding ascii
while (-not (Test-Path $StopFile)) {
    Start-Sleep -Seconds $Seconds
    try {
        $disks = Get-CimInstance -ClassName Win32_PerfFormattedData_PerfDisk_LogicalDisk -ErrorAction Stop
        $cd = $disks | Where-Object { $_.Name -eq 'C:' }
        $dd = $disks | Where-Object { $_.Name -eq 'D:' }
        $cpu = (Get-CimInstance -ClassName Win32_PerfFormattedData_PerfOS_Processor -Filter "Name='_Total'").PercentProcessorTime
        $g = (& nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits) -split ','
        $p = Get-Process kimi_k3 -ErrorAction SilentlyContinue | Select-Object -First 1
        $ws = if ($p) { $p.WorkingSet64 / 1GB } else { 0 }
        $pc = if ($p) { $p.CPU } else { 0 }
        $pd = 0; $ps = 0; $pcm = 0
        if ($p) {
            $pm = Get-CimInstance -ClassName Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory |
                  Where-Object { $_.Name -like ("pid_" + $p.Id + "_*") }
            foreach ($i in $pm) { $pd += $i.DedicatedUsage; $ps += $i.SharedUsage; $pcm += $i.TotalCommitted }
        }
        $am = Get-CimInstance -ClassName Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory |
              Sort-Object -Property DedicatedUsage -Descending | Select-Object -First 1
        [string]::Format($inv, "{0};{1:F0};{2:F0};{3:F0};{4:F0};{5:F0};{6};{7};{8:F1};{9:F0};{10:F0};{11:F0};{12:F0};{13:F0};{14:F0}",
            (Get-Date -Format 'HH:mm:ss'), $cd.DiskReadBytesPersec / 1MB, $dd.DiskReadBytesPersec / 1MB,
            100 - $cd.PercentIdleTime, 100 - $dd.PercentIdleTime, $cpu, $g[0].Trim(), $g[1].Trim(), $ws, $pc,
            $pd / 1MB, $ps / 1MB, $pcm / 1MB, $am.DedicatedUsage / 1MB, $am.SharedUsage / 1MB) | Out-File -FilePath $Out -Append -Encoding ascii
    } catch { }
}
