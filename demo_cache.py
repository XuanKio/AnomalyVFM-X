"""Find a complete, already-downloaded demo cache without network access."""

import json
import os
from pathlib import Path


REQUIRED_FILES = {
    "MaticFuc/anomalyvfm_clip": ("config.json", "model.safetensors"),
    "openai/clip-vit-large-patch14-336": ("config.json", "pytorch_model.bin"),
}


def select_model_cache(candidates: list[Path]) -> Path:
    failures = []
    for cache in candidates:
        try:
            for repo, filenames in REQUIRED_FILES.items():
                root = cache / ("models--" + repo.replace("/", "--"))
                revision = (root / "refs/main").read_text(encoding="utf-8").strip()
                if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
                    raise ValueError(f"Invalid cached revision for {repo}")
                snapshot = root / "snapshots" / revision
                for name in filenames:
                    path = snapshot / name
                    if not path.is_file() or path.stat().st_size == 0:
                        raise FileNotFoundError(f"Missing {repo}/{name}")
                json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            failures.append(f"{cache}: {exc}")
            continue
        # Preserve the proven path instead of traversing a Windows junction.
        return cache
    raise RuntimeError(
        "No complete local demo model cache found. Run setup_demo.cmd.\n"
        + "\n".join(failures)
    )


def configure_demo_cache() -> Path:
    runtime = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "AnomalyVFM-X"
    cache = select_model_cache([
        Path.home() / ".cache/huggingface-anomalyvfm/hub",
        runtime / "huggingface/hub",
        Path.home() / ".cache/huggingface/hub",
    ])
    # Transformers also supports legacy cache aliases; don't let them send the
    # backbone to a different directory than the adapter checkpoint.
    for key in ("TRANSFORMERS_CACHE", "PYTORCH_TRANSFORMERS_CACHE", "PYTORCH_PRETRAINED_BERT_CACHE"):
        os.environ.pop(key, None)
    os.environ["HF_HOME"] = str(cache.parent)
    os.environ["HF_HUB_CACHE"] = str(cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache)
    os.environ["TORCH_HOME"] = str(runtime / "torch")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    print(f"Demo model cache: {cache}", flush=True)
    return cache
