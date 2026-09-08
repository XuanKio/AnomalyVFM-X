@echo off
setlocal
cd /d "%~dp0"
call "%~dp0demo_env.cmd"
if errorlevel 1 exit /b 1
"%PYTHON_EXE%" web_demo.py
echo.
pause
