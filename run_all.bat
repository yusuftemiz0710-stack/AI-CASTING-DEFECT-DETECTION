@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0run_all.ps1" %*
endlocal
