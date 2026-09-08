@echo off
setlocal
cd /d "%~dp0"
call "%~dp0demo_env.cmd"
if errorlevel 1 exit /b 1

if "%~1"=="" (
  "%PYTHON_EXE%" demo_fast.py test.png
  echo.
  pause
) else (
  "%PYTHON_EXE%" demo_fast.py %*
)
