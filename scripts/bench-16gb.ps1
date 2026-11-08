# Portable GPU/CPU benchmark for a 16 GB NVIDIA card and an AMD or Intel CPU.
#
#   .\bench-16gb.ps1                 # default: 3 iterations, includes 6.186e9
#   .\bench-16gb.ps1 -SkipLarge      # quick: only 1e6..1e9
#   .\bench-16gb.ps1 -Iterations 5 -LargeIterations 2
#
# The script never forces a GPU engine that the CLI preflight rejects.  A 16 GB
# card is enough for all four engines at 1e6/1e9, for the block engines at
# 6.186e9, and for block-28 at 61.865e9.  If a point does not fit the free VRAM
# on the target machine, the CLI skips that engine and the skip is recorded in
# summary.csv.  CPU/SIMD is forced at 1e8 because the CLI otherwise skips it as
# a deliberately slow case.
#
# Results are written to bench-results\<timestamp>\:
#   environment.txt, summary.csv, summary.md, report.md, and the raw
#   JSON/stderr per run.  report.md is the combined CPU/SIMD/GPU table.

[CmdletBinding()]
param(
    [int]$Iterations = 3,
    [int]$LargeIterations = 3,
    [switch]$SkipCpu,
    [switch]$SkipGpu,
    [switch]$SkipLarge,
    [switch]$ForceGpu,
    [string]$OutputDir
)

$ErrorActionPreference = 'Stop'
# Keep benchmark output in English-style invariant formatting even on a
# non-English Windows installation.
$invariantCulture = [Globalization.CultureInfo]::InvariantCulture
[Threading.Thread]::CurrentThread.CurrentCulture = $invariantCulture
[Threading.Thread]::CurrentThread.CurrentUICulture = $invariantCulture
$scriptDir = $PSScriptRoot
$script:failed = $false
$script:results = @()

function Find-RequiredFile {
    param(
        [Parameter(Mandatory)][string]$Label,
        [Parameter(Mandatory)][string[]]$Candidates
    )
    foreach ($candidate in $Candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    throw "$Label not found. Looked for:`n  $($Candidates -join "`n  ")"
}

$exe = Find-RequiredFile -Label 'fib-simd-lab.exe' -Candidates @(
    (Join-Path $scriptDir 'fib-simd-lab.exe'),
    (Join-Path $scriptDir 'dist\fib-simd-lab.exe'),
    (Join-Path (Split-Path -Parent $scriptDir) 'dist\fib-simd-lab.exe')
)

$engineNames = @{
    resident = 'resident27_fib.exe'
    block27 = 'block27_fib.exe'
    block28 = 'block28_fib.exe'
    block29 = 'block29_fib.exe'
}
$engineDirs = @{
    resident = 'resident-27'
    block27 = 'block-27'
    block28 = 'block-28'
    block29 = 'block-29'
}

$enginePaths = @{}
foreach ($key in $engineNames.Keys) {
    $name = $engineNames[$key]
    $dir = $engineDirs[$key]
    $parent = Split-Path -Parent $scriptDir
    $candidates = @(
        (Join-Path $scriptDir "engines\$name"),
        (Join-Path $scriptDir "cuda-ntt\$dir\build\$name"),
        (Join-Path $parent "cuda-ntt\$dir\build\$name")
    )
    $enginePaths[$key] = Find-RequiredFile -Label $name -Candidates $candidates
}

$gpuName = ''
$gpuTotalMiB = 0
$gpuFreeMiB = 0
$gpuDriver = ''
$gpuCuda = ''
$smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if ($smi) {
    $line = & $smi.Source --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader,nounits 2>$null |
        Select-Object -First 1
    if ($line) {
        $parts = $line -split ',\s*'
        if ($parts.Count -ge 4) {
            $gpuName = $parts[0]
            [void][int]::TryParse($parts[1], [ref]$gpuTotalMiB)
            [void][int]::TryParse($parts[2], [ref]$gpuFreeMiB)
            $gpuDriver = $parts[3]
        }
    }
    foreach ($headerLine in (& $smi.Source 2>$null | Select-Object -First 8)) {
        if ($headerLine -match 'CUDA (?:UMD )?Version:\s*([0-9]+\.[0-9]+)') {
            $gpuCuda = $Matches[1]
            break
        }
    }
}

if (-not $smi -and -not $SkipGpu) {
    Write-Host 'WARNING: nvidia-smi was not found; GPU points will be skipped.' -ForegroundColor Yellow
    $SkipGpu = $true
}

# The bundled engines are CUDA 13.0 binaries. A driver that reports CUDA 12.x
# cannot initialise the runtime even though nvidia-smi itself works.
if (-not $SkipGpu -and $gpuCuda -and -not $ForceGpu) {
    $cudaMajor = [int]($gpuCuda.Split('.')[0])
    if ($cudaMajor -lt 13) {
        Write-Host "WARNING: NVIDIA driver reports CUDA $gpuCuda, but the bundled engines require CUDA 13.0." -ForegroundColor Yellow
        Write-Host '         Update the NVIDIA driver (580 or newer for CUDA 13) or use a CUDA 12.x build.' -ForegroundColor Yellow
        Write-Host '         GPU points are skipped; CPU/SIMD points still run. Use -ForceGpu to try anyway.' -ForegroundColor Yellow
        $SkipGpu = $true
    }
}

$includeLarge = (-not $SkipLarge) -and (-not $SkipGpu)
if ($includeLarge -and $gpuTotalMiB -lt 15800) {
    Write-Host "WARNING: GPU reports $gpuTotalMiB MiB total; this profile expects a 16 GB card. Large points are skipped." -ForegroundColor Yellow
    $includeLarge = $false
}

if (-not $OutputDir) {
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $OutputDir = Join-Path $scriptDir "bench-results\$stamp"
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path

function Convert-NsToMs([object]$Value) {
    if ($null -eq $Value) { return '' }
    return [math]::Round(([double]$Value) / 1000000.0, 3)
}

function Convert-Ms([object]$Value) {
    if ($null -eq $Value) { return '' }
    return [math]::Round([double]$Value, 3)
}

function Invoke-Point {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$Backend,
        [Parameter(Mandatory)][uint64]$N,
        [int]$Runs = 1,
        [string]$Hash = ''
    )
    $jsonPath = Join-Path $OutputDir "$Name.json"
    $errPath = Join-Path $OutputDir "$Name.err.txt"
    $cliArgs = @(
        '--backend', $Backend,
        '--method', 'doubling',
        '--iterations', [string]$Runs,
        '--no-print', '--no-banner', '--no-hwinfo', '--no-color', '--no-progress',
        '--json'
    )
    if ($Hash) { $cliArgs += @('--gpu-sha256', $Hash) }
    if ($Backend -eq 'cpu') { $cliArgs += '--force-simd' }
    if ($Backend -like 'gpu*') {
        $cliArgs += @(
            '--gpu-resident-bin', $enginePaths.resident,
            '--gpu-block27-bin', $enginePaths.block27,
            '--gpu-block28-bin', $enginePaths.block28,
            '--gpu-block29-bin', $enginePaths.block29
        )
    }
    $cliArgs += [string]$N

    Write-Host "== $Name : $Backend n=$N iterations=$Runs =="
    $argLine = ($cliArgs | ForEach-Object {
        if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
    }) -join ' '
    $watch = [Diagnostics.Stopwatch]::StartNew()
    # Start-Process avoids Windows PowerShell 5.1 turning the CLI's informational
    # stderr lines into terminating NativeCommandError records.
    $process = Start-Process -FilePath $exe -ArgumentList $argLine -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $jsonPath -RedirectStandardError $errPath
    $code = $process.ExitCode
    $watch.Stop()

    $data = $null
    if ((Test-Path -LiteralPath $jsonPath) -and ((Get-Item -LiteralPath $jsonPath).Length -gt 0)) {
        try {
            $data = Get-Content -LiteralPath $jsonPath -Raw -Encoding UTF8 | ConvertFrom-Json
        } catch {
            Write-Host "WARNING: could not parse JSON from $Name" -ForegroundColor Yellow
        }
    }

    if ($data -and $data.measurements) {
        $kernel = [string]$data.kernel
        $skipped = ''
        if ($data.gpu_preflight -and $data.gpu_preflight.skipped) {
            $skipped = (@($data.gpu_preflight.skipped) -join ';')
        }
        foreach ($m in @($data.measurements)) {
            $engineLabel = [string]$m.engine
            if (-not $engineLabel) {
                if ([string]$m.backend -eq 'cpu') { $engineLabel = 'SIMD' }
                else { $engineLabel = [string]$m.backend }
            }
            $script:results += [pscustomobject]@{
                point = $Name
                n = [string]$N
                backend = [string]$m.backend
                engine = $engineLabel
                kernel = $kernel
                median_ms = Convert-NsToMs $m.median_ns
                stream_ms = Convert-Ms $m.gpu_stream_ms
                call_ms = Convert-Ms $m.call_wall_ms
                total_ms = Convert-Ms $m.total_wall_ms
                peak_vram_mib = Convert-Ms $m.peak_vram_mib
                host_peak_mib = Convert-Ms $m.host_peak_mib
                exact = [string]$m.exact
                sha256 = [string]$m.output_sha256
                skipped = $skipped
                exit = $code
                wall_s = [math]::Round($watch.Elapsed.TotalSeconds, 3)
            }
        }
        if ($code -ne 0) {
            Write-Host "WARNING: $Name exited with code $code (see $errPath)" -ForegroundColor Yellow
            $script:failed = $true
        }
    } elseif ($data -and $data.gpu_preflight -and $data.gpu_preflight.skipped) {
        $skipped = (@($data.gpu_preflight.skipped) -join ';')
        $script:results += [pscustomobject]@{
            point = $Name
            n = [string]$N
            backend = $Backend
            engine = ''
            kernel = [string]$data.kernel
            median_ms = ''
            stream_ms = ''
            call_ms = ''
            total_ms = ''
            peak_vram_mib = ''
            host_peak_mib = ''
            exact = 'skipped'
            sha256 = ''
            skipped = $skipped
            exit = $code
            wall_s = [math]::Round($watch.Elapsed.TotalSeconds, 3)
        }
        Write-Host "  skipped by preflight: $skipped"
        if ($code -ne 0) {
            Write-Host "WARNING: $Name exited with code $code (see $errPath)" -ForegroundColor Yellow
            $script:failed = $true
        }
    } else {
        $errorText = ''
        if (Test-Path -LiteralPath $errPath) {
            $errorText = (Get-Content -LiteralPath $errPath -Raw -Encoding UTF8).Trim()
        }
        if (-not $errorText) { $errorText = "exit code $code" }
        $script:results += [pscustomobject]@{
            point = $Name
            n = [string]$N
            backend = $Backend
            engine = ''
            kernel = ''
            median_ms = ''
            stream_ms = ''
            call_ms = ''
            total_ms = ''
            peak_vram_mib = ''
            host_peak_mib = ''
            exact = 'error'
            sha256 = ''
            skipped = ''
            exit = $code
            wall_s = [math]::Round($watch.Elapsed.TotalSeconds, 3)
        }
        Write-Host "WARNING: $Name produced no measurements: $errorText" -ForegroundColor Yellow
        $script:failed = $true
    }
}

$hash1e6 = '9AB5876C72650208A510CFBB65BBF859A87B773CC4EBEFD45D4E052CA7455BD6'
$hash1e9 = 'D57068140E4974AB4FE1F8847DA629A798C8DAC02AEF517E8C0711AFAF884B14'
$hash6186 = 'E2AC2BC743649BC0D295C29A08941DBBB00AB97B3205B3DDF2315B5D81A87750'
$hash61865 = '4F5D93B4BBD63C16C6FB68BA13DDEEECCE65714526E580F30AC8A2ADE605E580'

if (-not $SkipCpu) {
    foreach ($n in @(1000000, 10000000, 100000000)) {
        $suffix = switch ($n) {
            1000000 { '1e6' }
            10000000 { '1e7' }
            100000000 { '1e8' }
        }
        Invoke-Point -Name "simd-$suffix" -Backend 'cpu' -N $n -Runs $Iterations
        Invoke-Point -Name "numbigint-$suffix" -Backend 'num-bigint' -N $n -Runs $Iterations
        Invoke-Point -Name "gmp-$suffix" -Backend 'gmp' -N $n -Runs $Iterations
    }
}

if (-not $SkipGpu) {
    Invoke-Point -Name 'gpu-1e6' -Backend 'gpu-all' -N 1000000 -Runs $Iterations -Hash $hash1e6
    Invoke-Point -Name 'gpu-1e7' -Backend 'gpu-all' -N 10000000 -Runs $Iterations
    Invoke-Point -Name 'gpu-1e8' -Backend 'gpu-all' -N 100000000 -Runs $Iterations
    Invoke-Point -Name 'gpu-1e9' -Backend 'gpu-all' -N 1000000000 -Runs $Iterations -Hash $hash1e9
    if ($includeLarge) {
        Invoke-Point -Name 'gpu-6.186e9-resident' -Backend 'gpu' -N 6186556317 -Runs $LargeIterations -Hash $hash6186
        Invoke-Point -Name 'gpu-6.186e9-blocks' -Backend 'gpu-block' -N 6186556317 -Runs $LargeIterations -Hash $hash6186
        Invoke-Point -Name 'gpu-61.865e9-block28' -Backend 'gpu-block28' -N 61865563170 -Runs $LargeIterations -Hash $hash61865
    }
}

$envLines = @(
    "date=$(Get-Date -Format o)",
    "cpu=$((Get-CimInstance Win32_Processor | Select-Object -First 1).Name)",
    "logical_processors=$env:NUMBER_OF_PROCESSORS",
    "gpu=$gpuName",
    "gpu_driver=$gpuDriver",
    "gpu_cuda=$gpuCuda",
    "gpu_total_mib=$gpuTotalMiB",
    "gpu_free_mib_at_start=$gpuFreeMiB",
    "os=$([Environment]::OSVersion.VersionString)",
    "powershell=$($PSVersionTable.PSVersion)",
    "exe_sha256=$((Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash)"
)
foreach ($key in @('resident', 'block27', 'block28', 'block29')) {
    $envLines += "$($key)_sha256=$((Get-FileHash -LiteralPath $enginePaths[$key] -Algorithm SHA256).Hash)"
}
$envLines | Set-Content -LiteralPath (Join-Path $OutputDir 'environment.txt') -Encoding UTF8

$csvPath = Join-Path $OutputDir 'summary.csv'
$script:results | Export-Csv -LiteralPath $csvPath -NoTypeInformation -Encoding UTF8

$md = @()
$md += '# Portable benchmark results'
$md += ''
$md += "- CPU: $((Get-CimInstance Win32_Processor | Select-Object -First 1).Name)"
$md += "- GPU: $gpuName, driver $gpuDriver, CUDA $gpuCuda ($gpuTotalMiB MiB total, $gpuFreeMiB MiB free at start)"
$md += "- Date: $(Get-Date -Format o)"
$md += ''
$md += '| point | backend | engine | kernel | median ms | stream ms | call ms | total ms | peak VRAM MiB | host peak MiB | exact | skipped |'
$md += '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|'
foreach ($row in $script:results) {
    $md += "| $($row.point) | $($row.backend) | $($row.engine) | $($row.kernel) | $($row.median_ms) | $($row.stream_ms) | $($row.call_ms) | $($row.total_ms) | $($row.peak_vram_mib) | $($row.host_peak_mib) | $($row.exact) | $($row.skipped) |"
}
$md | Set-Content -LiteralPath (Join-Path $OutputDir 'summary.md') -Encoding UTF8

# ---------------------------------------------------------------------------
# Combined CPU/SIMD/GPU report
# ---------------------------------------------------------------------------
$rows = @($script:results | Where-Object { $_.median_ms -ne '' })
$order = @($rows | ForEach-Object { $_.n } | Sort-Object { [uint64]$_ } -Unique)
$byN = @{}
foreach ($row in $rows) {
    if (-not $byN.ContainsKey($row.n)) { $byN[$row.n] = @{} }
    $byN[$row.n][$row.engine] = $row
}

function Get-Cell($map, [string]$key) {
    if ($map.ContainsKey($key) -and $map[$key].median_ms -ne '') {
        return $map[$key].median_ms
    }
    return ''
}

function Get-Best($map, [string[]]$keys) {
    $values = @()
    foreach ($key in $keys) {
        if ($map.ContainsKey($key) -and $map[$key].median_ms -ne '') {
            $values += [double]$map[$key].median_ms
        }
    }
    if ($values.Count -eq 0) { return '' }
    return [math]::Round(($values | Measure-Object -Minimum).Minimum, 3)
}

function Format-Cell($value) {
    if ($value -eq '' -or $null -eq $value) { return '-' }
    return [string]$value
}

function Format-N([string]$value) {
    return [string]::Format(
        [Globalization.CultureInfo]::InvariantCulture,
        '{0:N0}',
        [uint64]$value
    )
}

$report = @()
$report += '# Combined CPU / SIMD / GPU benchmark'
$report += ''
$report += "- CPU: $((Get-CimInstance Win32_Processor | Select-Object -First 1).Name)"
$report += "- GPU: $gpuName, driver $gpuDriver, CUDA $gpuCuda ($gpuTotalMiB MiB total, $gpuFreeMiB MiB free at start)"
$report += "- Date: $(Get-Date -Format o)"
$report += ''
$report += 'All times are milliseconds, lower is better. `GPU speedup` is the best CPU'
$report += 'median divided by the best GPU median at the same `n`; values above 1 mean'
$report += 'the GPU is faster. A dash means that backend was not run or was skipped by'
$report += 'the CLI preflight (for example, an engine that does not fit in 16 GB VRAM).'
$report += ''
$report += '| n | CPU/SIMD | num-bigint | GMP | GPU resident | block-27 | block-28 | block-29 | best CPU | best GPU | GPU speedup |'
$report += '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|'
foreach ($n in $order) {
    $map = $byN[$n]
    $simd = Get-Cell $map 'SIMD'
    $native = Get-Cell $map 'num-bigint'
    $gmp = Get-Cell $map 'gmp'
    $resident = Get-Cell $map 'resident'
    $b27 = Get-Cell $map 'block-27'
    $b28 = Get-Cell $map 'block-28'
    $b29 = Get-Cell $map 'block-29'
    $bestCpu = Get-Best $map @('SIMD', 'num-bigint', 'gmp')
    $bestGpu = Get-Best $map @('resident', 'block-27', 'block-28', 'block-29')
    $speedup = ''
    if ($bestCpu -ne '' -and $bestGpu -ne '' -and [double]$bestGpu -gt 0) {
        $speedup = [math]::Round([double]$bestCpu / [double]$bestGpu, 2)
    }
    $displayN = Format-N $n
    $report += "| $displayN | $(Format-Cell $simd) | $(Format-Cell $native) | $(Format-Cell $gmp) | $(Format-Cell $resident) | $(Format-Cell $b27) | $(Format-Cell $b28) | $(Format-Cell $b29) | $(Format-Cell $bestCpu) | $(Format-Cell $bestGpu) | $(Format-Cell $speedup) |"
}
$report += ''
$report += '## GPU details'
$report += ''
$report += '| n | engine | stream ms | call ms | total ms | peak VRAM MiB | host peak MiB | exact | SHA-256 |'
$report += '|---:|---|---:|---:|---:|---:|---:|---|---|'
foreach ($row in ($rows | Where-Object { $_.engine -in @('resident', 'block-27', 'block-28', 'block-29') } | Sort-Object { [uint64]$_.n }, engine)) {
    $displayN = Format-N $row.n
    $sha = if ($row.sha256) { $row.sha256 } else { '-' }
    $report += "| $displayN | $($row.engine) | $(Format-Cell $row.stream_ms) | $(Format-Cell $row.call_ms) | $(Format-Cell $row.total_ms) | $(Format-Cell $row.peak_vram_mib) | $(Format-Cell $row.host_peak_mib) | $($row.exact) | $sha |"
}
$report += ''
$report += 'CPU/SIMD uses runtime dispatch; the kernel column shows the selected kernel.'
$report += 'On the Ryzen 9 5900HX target this is AVX2 (Zen 3 has no AVX-512), so the'
$report += 'portable binary does not require AVX-512. CPU 1e9 is intentionally outside'
$report += 'this profile: the scalar/SIMD path is much slower than the big-integer'
$report += 'libraries there, so 1e8 is the CPU ceiling and 1e9+ is measured on the GPU only.'
$report | Set-Content -LiteralPath (Join-Path $OutputDir 'report.md') -Encoding UTF8

Write-Host ''
Write-Host "Results:  $OutputDir"
Write-Host "CSV:      $csvPath"
Write-Host "Markdown: $(Join-Path $OutputDir 'summary.md')"
Write-Host "Report:   $(Join-Path $OutputDir 'report.md')"
Write-Host ''
foreach ($row in $script:results) {
    if ($row.median_ms -ne '') {
        Write-Host ("{0,-24} {1,-11} {2,-10} {3,10} ms  {4}" -f `
            $row.point, $row.backend, $row.engine, $row.median_ms, $row.kernel)
    } elseif ($row.exact -eq 'skipped') {
        Write-Host ("{0,-24} {1,-11} {2,-10} {3}" -f `
            $row.point, $row.backend, $row.engine, "skipped: $($row.skipped)")
    }
}

if ($script:failed) {
    Write-Host 'One or more benchmark points failed; see the JSON/stderr files.' -ForegroundColor Red
    exit 1
}

Write-Host 'Benchmark completed.' -ForegroundColor Green
exit 0
