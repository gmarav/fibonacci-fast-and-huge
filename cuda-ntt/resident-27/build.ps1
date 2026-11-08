# Build the GPU-resident fast-doubling engine.
#
# Host compiler for nvcc: MSVC (VS 18 BuildTools vcvars64.bat), target
# sm_120 (RTX 50 series).  MinGW g++ is used only for the GMP oracle and is
# never the nvcc host compiler.
#
# Produces, under cuda-ntt/resident-27/build:
#   resident27_fib.exe   host driver (CLI, timing, validation)
#   oracle_fib_gmp.exe   GMP mpz_fib_ui oracle (MinGW + static libgmp)
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\build.ps1
#   powershell -ExecutionPolicy Bypass -File .\build.ps1 -Validate
#   powershell -ExecutionPolicy Bypass -File .\build.ps1 -Bench -Sizes 1M,8M,32M -Runs 5
#   powershell -ExecutionPolicy Bypass -File .\build.ps1 -Bench -HostLoop -Sizes 1M
#   powershell -ExecutionPolicy Bypass -File .\build.ps1 -Clean

[CmdletBinding()]
param(
    [switch]$Clean,
    [switch]$Validate,
    [switch]$Bench,
    [string]$Sizes = "1M,8M,32M",
    [int]$Runs = 3,
    [switch]$HostLoop,
    [switch]$Csv,
    [switch]$NoGraph,
    [string]$Oracle = ""
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$build = Join-Path $root "build"
Push-Location $root

try {
    if ($Clean) {
        if (Test-Path $build) {
            Remove-Item -Recurse -Force $build
            Write-Host "cleaned $build"
        }
        if (-not ($Validate -or $Bench)) { return }
    }
    if (-not (Test-Path $build)) {
        New-Item -ItemType Directory -Path $build -Force | Out-Null
    }

    $vcvars = 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
    if (-not (Test-Path $vcvars)) {
        throw "MSVC Build Tools vcvars64.bat not found: $vcvars"
    }
    # CUDA 13.0 does not officially list VS 18 yet; the flag is required.
    $nvcc = "nvcc -O3 -std=c++17 -arch=sm_120 -Xcompiler /O2 -Xcompiler /EHsc -allow-unsupported-compiler"

    Write-Host "host environment: $vcvars"
    Write-Host "building resident27_fib.exe"
    cmd /c "`"$vcvars`" >nul && $nvcc resident27.cu resident27_fib.cpp -o build\resident27_fib.exe"
    if ($LASTEXITCODE -ne 0) { throw "nvcc exe build failed" }

    # --- GMP oracle (MinGW, static libgmp) -----------------------------------
    $gmpCandidates = @()
    if ($env:GMP_ROOT) { $gmpCandidates += $env:GMP_ROOT }
    foreach ($cand in (Get-ChildItem (Join-Path $root '..\..\build') -Recurse -Filter 'libgmp.a' -ErrorAction SilentlyContinue)) {
        $gmpCandidates += (Split-Path (Split-Path $cand.FullName -Parent) -Parent)
    }
    $gmpRoot = $null
    foreach ($cand in $gmpCandidates) {
        if ((Test-Path (Join-Path $cand 'include\gmp.h')) -and
            (Test-Path (Join-Path $cand 'lib\libgmp.a'))) {
            $gmpRoot = $cand
            break
        }
    }
    $oracleExe = Join-Path $build 'oracle_fib_gmp.exe'
    if ($null -eq $gmpRoot) {
        Write-Warning "static libgmp.a not found; the GMP oracle will not be built"
    } else {
        Write-Host "building oracle_fib_gmp.exe with $gmpRoot"
        & g++ -O2 -std=c++17 "oracle_fib_gmp.cpp" "-I$gmpRoot\include" `
            "$gmpRoot\lib\libgmp.a" -o $oracleExe
        if ($LASTEXITCODE -ne 0) { throw "g++ oracle build failed" }
    }

    $exe = Join-Path $build 'resident27_fib.exe'
    if ([string]::IsNullOrWhiteSpace($Oracle) -and (Test-Path $oracleExe)) {
        $Oracle = $oracleExe
    }

    if ($NoGraph) { $env:RESIDENT27_NO_GRAPH = "1" }

    if ($Validate) {
        $cliArgs = @('validate')
        if (-not [string]::IsNullOrWhiteSpace($Oracle)) { $cliArgs += @('--oracle', $Oracle) }
        & $exe @cliArgs
        if ($LASTEXITCODE -ne 0) { throw "validation failed" }
    }

    if ($Bench) {
        $cliArgs = @('bench', '--sizes', $Sizes, '--runs', "$Runs")
        if (-not [string]::IsNullOrWhiteSpace($Oracle)) { $cliArgs += @('--oracle', $Oracle) }
        if ($HostLoop) { $cliArgs += '--hostloop' }
        if ($Csv) { $cliArgs += '--csv' }
        & $exe @cliArgs
        if ($LASTEXITCODE -ne 0) { throw "benchmark failed" }
    }

    Write-Host ""
    Write-Host "artifacts under $build"
    Write-Host "  $exe --help"
} finally {
    Pop-Location
}
