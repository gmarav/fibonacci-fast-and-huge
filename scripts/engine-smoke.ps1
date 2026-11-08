# Self-hosted GPU smoke test for the exact Fibonacci suite.
#
# Prerequisites: build the CUDA engines and the Rust CLI with .\build.ps1, and
# run this on the GPU machine (it is not suitable for hosted CPU runners).
#
#   pwsh -NoProfile -File .\scripts\engine-smoke.ps1
#
# Verifies that every expected executable exists, then runs all four engines
# once at n=1,000,000 with a fixed SHA-256 check.

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$required = @(
    'dist\fib-simd-lab.exe',
    'cuda-ntt\resident-27\build\resident27_fib.exe',
    'cuda-ntt\block-27\build\block27_fib.exe',
    'cuda-ntt\block-28\build\block28_fib.exe',
    'cuda-ntt\block-29\build\block29_fib.exe'
)

$missing = @()
foreach ($relative in $required) {
    if (-not (Test-Path -LiteralPath (Join-Path $root $relative) -PathType Leaf)) {
        $missing += $relative
    }
}
if ($missing.Count -gt 0) {
    $list = $missing -join [Environment]::NewLine
    throw "Missing build artifacts:$([Environment]::NewLine)  $list$([Environment]::NewLine)Run .\build.ps1 on this machine first."
}

$expectedSha256 = '9AB5876C72650208A510CFBB65BBF859A87B773CC4EBEFD45D4E052CA7455BD6'
$smokeArgs = @(
    '--backend', 'gpu-all',
    '--method', 'doubling',
    '--iterations', '1',
    '--no-print',
    '--no-banner',
    '--no-hwinfo',
    '--no-color',
    '--no-progress',
    '--gpu-sha256', $expectedSha256,
    '1000000'
)

Write-Host 'Running GPU smoke test: gpu-all, n=1000000, iterations=1'
& (Join-Path $root 'dist\fib-simd-lab.exe') @smokeArgs
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    throw "GPU smoke test failed: run.ps1 exited with code $exitCode. Check the CUDA engines, GPU driver, and available GPU memory."
}

Write-Host "GPU smoke test passed (sha256 $expectedSha256)."
