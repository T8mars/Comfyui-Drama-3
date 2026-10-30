@echo off
setlocal
cd /d "%~dp0"
(
"%~dp0runtime\python\python.exe" -X utf8 "%~dp0tools\update_portable.py" --automatic
if errorlevel 75 exit /b 0
if errorlevel 1 (
  pause
  exit /b 1
)
"%~dp0runtime\python\python.exe" -X utf8 "%~dp0app.py" %*
if errorlevel 1 pause
exit /b
)
