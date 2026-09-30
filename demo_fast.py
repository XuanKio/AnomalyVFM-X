"""Lightweight single-image demo for AnomalyVFM.

This keeps the original pretrained AnomalyVFM model but uses a smaller input
resolution by default so that inference fits comfortably on a 4 GB laptop GPU.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

from demo_cache import configure_demo_cache

# Select and validate both checkpoints before importing Hugging Face.
MODEL_CACHE_DIR = configure_demo_cache()

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download

from hf_model import AnomalyVFM, AnomalyVFMConfig


MODEL_ID = "MaticFuc/anomalyvfm_clip"
warnings.filterwarnings(
    "ignore", message=r"The transform `ToTensor\(\)` is deprecated.*", category=UserWarning
)


def load_demo_model() -> AnomalyVFM:
    """Read the cached checkpoint config explicitly before constructing the model."""
    try:
        config_path = Path(hf_hub_download(
            MODEL_ID, "config.json", cache_dir=MODEL_CACHE_DIR, local_files_only=True,
        ))
        weights_path = Path(hf_hub_download(
            MODEL_ID, "model.safetensors", cache_dir=MODEL_CACHE_DIR, local_files_only=True,
        ))
    except OSError as exc:
        raise RuntimeError(
            f"Demo checkpoint is incomplete in {MODEL_CACHE_DIR}. Run setup_demo.cmd."
        ) from exc
    config = AnomalyVFMConfig(**json.loads(config_path.read_text(encoding="utf-8")))
    print(f"Demo checkpoint: {weights_path.parent}", flush=True)
    return AnomalyVFM.from_pretrained(
        weights_path.parent,
        config=config,
        cache_dir=MODEL_CACHE_DIR,
        local_files_only=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a lightweight AnomalyVFM demo and save visual results."
    )
    parser.add_argument(
        "images",
        nargs="*",
        type=Path,
        default=[Path("test.png")],
        help="One or more input images (default: test.png).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=336,
        choices=(336, 672),
        help="336 is fast/low-memory; 672 is the original CLIP evaluation size.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs"), help="Result directory."
    )
    return parser.parse_args()


def configure_resolution(model: AnomalyVFM, size: int) -> None:
    """Update the spatial bookkeeping used by the original model wrapper."""
    patch_size = model.model.patch_size
    if size % patch_size:
        raise ValueError(f"Input size must be divisible by patch size {patch_size}.")
    model.config.image_size = size
    model.model.H = size
    model.feat_size = size // patch_size


def colorize(mask: np.ndarray) -> np.ndarray:
    """Convert a 0..1 mask to a compact blue-yellow-red heatmap."""
    x = np.clip(mask, 0.0, 1.0)
    red = np.clip(2.0 * x, 0.0, 1.0)
    green = np.clip(2.0 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    blue = np.clip(2.0 * (1.0 - x), 0.0, 1.0)
    return (np.stack((red, green, blue), axis=-1) * 255).astype(np.uint8)


def build_visualization(
    image: Image.Image, mask: torch.Tensor, score: float
) -> Image.Image:
    original = image.convert("RGB")
    width, height = original.size
    mask = F.interpolate(
        mask.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
    )[0, 0]
    values = mask.detach().float().cpu().numpy()

    # Per-image scaling makes the localization easy to see in a classroom demo.
    lo, hi = float(values.min()), float(values.max())
    normalized = (values - lo) / max(hi - lo, 1e-6)
    heatmap = Image.fromarray(colorize(normalized), mode="RGB")
    overlay = Image.blend(original, heatmap, alpha=0.38)

    title_height = 42
    canvas = Image.new("RGB", (width * 3, height + title_height), "white")
    canvas.paste(original, (0, title_height))
    canvas.paste(heatmap, (width, title_height))
    canvas.paste(overlay, (width * 2, title_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 12), "Input", fill="black")
    draw.text((width + 10, 12), "Anomaly heatmap", fill="black")
    draw.text(
        (width * 2 + 10, 12), f"Overlay | anomaly score: {score:.4f}", fill="black"
    )
    return canvas


def save_visualization(
    image: Image.Image, mask: torch.Tensor, score: float, output_path: Path
) -> None:
    build_visualization(image, mask, score).save(output_path, optimize=True)


def main() -> int:
    args = parse_args()
    missing = [str(path) for path in args.images if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input image(s): " + ", ".join(missing))

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    print(f"Loading {MODEL_ID} on {device}...")
    load_started = time.perf_counter()
    model = load_demo_model()
    configure_resolution(model, args.size)

    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device)
    model.eval()
    print(f"Model ready in {time.perf_counter() - load_started:.1f}s")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    transform = model.model.get_img_transform()

    for image_path in args.images:
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            score, mask = model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        score_value = float(score.squeeze().cpu())
        output_path = args.output_dir / f"{image_path.stem}_result.png"
        save_visualization(image, mask[0], score_value, output_path)

        memory = ""
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated() / 1024**3
            memory = f", peak VRAM {peak:.2f} GB"
        print(
            f"{image_path}: score={score_value:.4f}, inference={elapsed:.2f}s"
            f"{memory} -> {output_path}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
