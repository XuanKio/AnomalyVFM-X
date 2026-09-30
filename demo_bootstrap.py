"""Standard-library launcher for the installed Windows demo environment."""

import os
from pathlib import Path
import subprocess
import sys


def select_demo_python(candidates: list[Path], env: dict[str, str]) -> Path:
    """Choose by a real import check, preserving the executable's original path."""
    probe = (
        "import sys; "
        "assert sys.version_info[:2] == (3, 10), 'Demo requires Python 3.10'; "
        "import cgi, torch, torchvision, PIL, transformers, huggingface_hub, safetensors; "
        "assert torch.__version__.split('+')[0] == '2.10.0', 'Unexpected torch version'; "
        "assert torchvision.__version__.split('+')[0] == '0.25.0', 'Unexpected torchvision version'"
    )
    failures = []
    seen = set()
    for candidate in candidates:
        key = os.path.normcase(os.path.abspath(candidate))
        if key in seen:
            continue
        seen.add(key)
        if not candidate.is_file():
            failures.append(f"{candidate}: not installed")
            continue
        try:
            check = subprocess.run(
                [str(candidate), "-E", "-s", "-c", probe], env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace", timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            failures.append(f"{candidate}: {exc}")
            continue
        if check.returncode == 0:
            return candidate
        details = check.stderr.strip().splitlines()
        failures.append(f"{candidate}: {details[-1] if details else 'import check failed'}")
    raise SystemExit(
        "No working demo Python found. Run setup_demo.cmd to repair it.\n"
        + "\n".join(failures)
    )


def launch_web_runtime(script: str) -> None:
    """Restart in the demo Python before importing cgi, torch or Hugging Face."""
    if os.name != "nt":
        return
    runtime = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "AnomalyVFM-X"
    if (os.environ.get("ANOMALYVFM_VERIFIED_PYTHON") == sys.executable
            and sys.flags.ignore_environment and sys.flags.no_user_site):
        print(f"Demo Python: {sys.executable}", flush=True)
        return

    # Do not inherit Python or legacy model-cache overrides from a VS Code terminal.
    env = os.environ.copy()
    for key in (
        "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
        "PYTORCH_TRANSFORMERS_CACHE", "PYTORCH_PRETRAINED_BERT_CACHE", "TORCH_HOME",
    ):
        env.pop(key, None)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    print("Checking demo Python...", flush=True)
    python = select_demo_python([
        Path(sys.executable),
        Path.home() / ".venvs/anomalyvfm-demo/Scripts/python.exe",
        runtime / "venv/Scripts/python.exe",
    ], env)
    # Do not resolve junctions here: use the exact path that passed the probe.
    env["ANOMALYVFM_VERIFIED_PYTHON"] = str(python)
    child = subprocess.Popen(
        [str(python), "-E", "-s", str(Path(script).resolve()), *sys.argv[1:]],
        cwd=str(Path(script).resolve().parent), env=env,
    )
    try:
        code = child.wait()
    except KeyboardInterrupt:
        # Windows sends Ctrl+C to both processes sharing this console.
        try:
            code = child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.terminate()
            code = child.wait()
    raise SystemExit(code)
