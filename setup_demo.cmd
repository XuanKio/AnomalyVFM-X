@echo off
setlocal
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
  echo [ERROR] uv is not installed or not available in PATH.
  echo Install uv from https://docs.astral.sh/uv/getting-started/installation/
  echo Then close and reopen the terminal and run setup_demo.cmd again.
  exit /b 1
)

set "ANOMALYVFM_RUNTIME=%LOCALAPPDATA%\AnomalyVFM-X"
set "VENV_DIR=%ANOMALYVFM_RUNTIME%\venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "HF_HOME=%ANOMALYVFM_RUNTIME%\huggingface"
set "TORCH_HOME=%ANOMALYVFM_RUNTIME%\torch"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

if not exist "%PYTHON_EXE%" (
  echo [1/5] Creating Python 3.10 environment...
  uv venv "%VENV_DIR%" --python 3.10 --python-preference managed
  if errorlevel 1 exit /b 1
) else (
  echo [1/5] Python environment already exists.
)

where nvidia-smi >nul 2>nul
if errorlevel 1 (
  set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
  echo [2/5] No NVIDIA GPU detected; installing CPU PyTorch...
) else (
  set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
  echo [2/5] NVIDIA GPU detected; installing CUDA 12.8 PyTorch...
)

uv pip install --python "%PYTHON_EXE%" torch==2.10.0 torchvision==0.25.0 --index-url "%TORCH_INDEX%"
if errorlevel 1 exit /b 1

echo [3/5] Installing the minimal inference dependencies...
uv pip install --python "%PYTHON_EXE%" transformers==4.56.1 huggingface_hub==0.36.2 safetensors==0.6.2 hf_xet==1.6.0
if errorlevel 1 exit /b 1

echo [4/5] Downloading only the required model files...
"%PYTHON_EXE%" -c "from huggingface_hub import snapshot_download; snapshot_download('openai/clip-vit-large-patch14-336', allow_patterns=['config.json','pytorch_model.bin']); snapshot_download('MaticFuc/anomalyvfm_clip', allow_patterns=['config.json','model.safetensors'])"
if errorlevel 1 exit /b 1

echo [5/5] Running the offline smoke test...
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
"%PYTHON_EXE%" demo_fast.py test.png
if errorlevel 1 exit /b 1

echo.
echo Setup completed successfully.
echo Start the browser demo with: web_demo.cmd
endlocal
