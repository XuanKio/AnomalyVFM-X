@echo off
rem Shared, portable runtime configuration for demo.cmd and web_demo.cmd.
set "ANOMALYVFM_RUNTIME=%LOCALAPPDATA%\AnomalyVFM-X"
set "VENV_DIR=%ANOMALYVFM_RUNTIME%\venv"

rem Resolve a junction to its real target. Some Python packages are not found
rem reliably when the virtual environment is launched through a reparse point.
for /f "usebackq delims=" %%I in (`powershell.exe -NoProfile -Command "$item=Get-Item -LiteralPath ($env:ANOMALYVFM_RUNTIME + '\venv') -ErrorAction SilentlyContinue; if ($item.LinkType -and $item.Target) { $item.Target | Select-Object -First 1 }"`) do set "VENV_DIR=%%I"

set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "HF_HOME=%ANOMALYVFM_RUNTIME%\huggingface"
set "TORCH_HOME=%ANOMALYVFM_RUNTIME%\torch"

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Demo environment is not installed.
  echo Run setup_demo.cmd first.
  exit /b 1
)

"%PYTHON_EXE%" -c "import torch" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] PyTorch is missing from the demo environment:
  echo         %PYTHON_EXE%
  echo Run setup_demo.cmd to repair the environment.
  exit /b 1
)

set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
