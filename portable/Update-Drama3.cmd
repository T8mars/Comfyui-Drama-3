@echo off
setlocal
cd /d "%~dp0"
(
"%~dp0runtime\python\python.exe" -X utf8 "%~dp0tools\update_portable.py"
pause
exit /b
)
