@echo off
setlocal
rem Prefer PowerShell 7 (pwsh); fall back to Windows PowerShell 5.1.
set "PS=pwsh.exe"
where pwsh.exe >nul 2>nul
if not errorlevel 1 goto run
if exist "%ProgramFiles%\PowerShell\7\pwsh.exe" set "PS=%ProgramFiles%\PowerShell\7\pwsh.exe"
if exist "%ProgramFiles%\PowerShell\7\pwsh.exe" goto run
set "PS=powershell.exe"
:run
"%PS%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0bench.ps1" %*
exit /b %ERRORLEVEL%
