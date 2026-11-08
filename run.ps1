$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
Set-Location $root

$exe = Join-Path $root 'dist\fib-simd-lab.exe'
if (-not (Test-Path $exe)) {
    $fallback = Join-Path $root 'build\rust\release\fib-simd-lab.exe'
    if (Test-Path $fallback) { $exe = $fallback }
}

if (-not (Test-Path $exe)) {
    throw "fib-simd-lab.exe not found. Build it with .\build.ps1 first."
}

& $exe @args
exit $LASTEXITCODE
