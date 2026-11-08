# Build and run the CPU-only reference tests for the CUDA engines.
#
#   .\scripts\host-tests.ps1
#   .\scripts\host-tests.ps1 -KeepGoing
#
# These tests need only MinGW g++: no CUDA runtime, no GPU, no GMP.  They cover
# the host NTT/CRT references, segment normalisation, planner, arithmetic
# self-test, and the block-schedule drivers.  The GPU engines themselves are
# exercised by scripts/engine-smoke.ps1 on a machine with a CUDA device.

[CmdletBinding()]
param(
    [switch]$KeepGoing
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$cudaNtt = Join-Path $root 'cuda-ntt'
$buildDir = Join-Path $root 'build\host-tests'
$script:failed = $false

function Find-Gxx {
    $onPath = Get-Command g++.exe -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    $candidates = @()
    if ($env:MSYS2_ROOT) { $candidates += (Join-Path $env:MSYS2_ROOT 'mingw64\bin\g++.exe') }
    $candidates += @(
        'C:\msys64\mingw64\bin\g++.exe',
        (Join-Path $root '.tools\msys64\mingw64\bin\g++.exe')
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    throw 'g++ not found; install MSYS2/MinGW or put g++ on PATH'
}

function Invoke-Test {
    param(
        [Parameter(Mandatory)][string]$Exe,
        [string[]]$Arguments = @()
    )
    $display = Split-Path -Leaf $Exe
    if ($Arguments.Count -gt 0) { $display += ' ' + ($Arguments -join ' ') }
    Write-Host "  $display"
    $out = & $Exe @Arguments 2>&1
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        Write-Host "FAILED: $display (exit $code)" -ForegroundColor Red
        $out | ForEach-Object { Write-Host $_ }
        $script:failed = $true
        if (-not $KeepGoing) { throw "$display failed" }
    } else {
        $out | Select-Object -Last 2 | ForEach-Object { Write-Host "    $_" }
    }
}

$gxx = Find-Gxx
$suites = [ordered]@{
    'block-28' = @('driver_ref', 'ntt_ref', 'plan_test', 'segment_test', 'twiddle_test')
    'block-29' = @('driver_ref', 'ntt_ref', 'plan_test', 'segment_test', 'twiddle_test')
    'resident-27' = @('arith_selftest')
}

foreach ($engine in $suites.Keys) {
    $target = Join-Path $buildDir $engine
    New-Item -ItemType Directory -Force -Path $target | Out-Null
    Write-Host "== building $engine references =="
    foreach ($test in $suites[$engine]) {
        $source = Join-Path (Join-Path $cudaNtt $engine) ($test + '.cpp')
        $exe = Join-Path $target ($test + '.exe')
        & $gxx -O2 -std=c++17 -Wall -Wextra $source -o $exe
        if ($LASTEXITCODE -ne 0) { throw "g++ failed for $source" }
    }
}

New-Item -ItemType Directory -Force -Path (Join-Path $buildDir 'out'),
    (Join-Path $buildDir 'out28'), (Join-Path $buildDir 'out29') | Out-Null

Push-Location $buildDir
try {
    Write-Host '== running block-28 references =='
    Invoke-Test (Join-Path $buildDir 'block-28\driver_ref.exe')
    Invoke-Test (Join-Path $buildDir 'block-28\ntt_ref.exe') @('mont', 'out28')
    Invoke-Test (Join-Path $buildDir 'block-28\ntt_ref.exe') @('ntt', '10', 'out28')
    Invoke-Test (Join-Path $buildDir 'block-28\ntt_ref.exe') @('block', 'out28')
    Invoke-Test (Join-Path $buildDir 'block-28\plan_test.exe')
    Invoke-Test (Join-Path $buildDir 'block-28\segment_test.exe')
    Invoke-Test (Join-Path $buildDir 'block-28\twiddle_test.exe')

    Write-Host '== running block-29 references =='
    Invoke-Test (Join-Path $buildDir 'block-29\driver_ref.exe')
    Invoke-Test (Join-Path $buildDir 'block-29\ntt_ref.exe') @('mont', 'out29')
    Invoke-Test (Join-Path $buildDir 'block-29\ntt_ref.exe') @('ntt', '10', 'out29')
    Invoke-Test (Join-Path $buildDir 'block-29\ntt_ref.exe') @('block', 'out29')
    Invoke-Test (Join-Path $buildDir 'block-29\plan_test.exe')
    Invoke-Test (Join-Path $buildDir 'block-29\segment_test.exe')
    Invoke-Test (Join-Path $buildDir 'block-29\twiddle_test.exe')

    Write-Host '== running resident-27 references =='
    Invoke-Test (Join-Path $buildDir 'resident-27\arith_selftest.exe')
} finally {
    Pop-Location
}

if ($script:failed) {
    Write-Host ''
    Write-Host 'Host reference tests failed.' -ForegroundColor Red
    exit 1
}

Write-Host ''
Write-Host 'All host reference tests passed.'
