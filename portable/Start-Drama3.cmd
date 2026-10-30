@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0Start-Drama3.exe" (
  "%~dp0Start-Drama3.exe" %*
  exit /b
)
(
"%~dp0runtime\python\python.exe" -X utf8 "%~dp0tools\update_portable.py" --automatic
if errorlevel 75 exit /b 0
if errorlevel 1 (
  pause
  exit /b 1
)
"%~dp0runtime\python\python.exe" -X utf8 -m drama3_t8.console_launcher "%~dp0app.py" %*
if errorlevel 1 pause
exit /b
)
