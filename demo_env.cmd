@echo off
rem Shared, portable runtime configuration for demo.cmd and web_demo.cmd.
set "ANOMALYVFM_RUNTIME=%LOCALAPPDATA%\AnomalyVFM-X"
set "PYTHON_EXE=%ANOMALYVFM_RUNTIME%\venv\Scripts\python.exe"
set "HF_HOME=%ANOMALYVFM_RUNTIME%\huggingface"
set "TORCH_HOME=%ANOMALYVFM_RUNTIME%\torch"

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Demo environment is not installed.
  echo Run setup_demo.cmd first.
  exit /b 1
)

set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
