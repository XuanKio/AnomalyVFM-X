"""Opt-in, sequential overlapping-tile inference for the local demo.

Maps remain uncalibrated anomaly scores. Tiling changes the spatial evidence
seen by the same model; it does not train the model or alter the image score.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps


def configure_resolution(model, size: int) -> None:
    """Reuse the demo configuration without loading HF at module import time."""
    from demo_fast import configure_resolution as configure_demo_resolution

    configure_demo_resolution(model, size)


def _positive_int(value, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _fraction(value, name: str, *, include_one: bool) -> None:
    try:
        valid = not isinstance(value, bool) and math.isfinite(value)
        valid = valid and 0 <= value and (value <= 1 if include_one else value < 1)
    except (TypeError, ValueError):
        valid = False
    if not valid:
        interval = "[0, 1]" if include_one else "[0, 1)"
        raise ValueError(f"{name} must be finite and in {interval}.")


def axis_starts(length: int, tile_size: int, overlap: float = 0.25) -> list[int]:
    """Cover an axis, anchoring the final tile to its far edge without padding."""
    _positive_int(length, "length")
    _positive_int(tile_size, "tile_size")
    _fraction(overlap, "overlap", include_one=False)
    if length <= tile_size:
        return [0]
    stride = max(1, int(tile_size * (1 - overlap)))
    last = length - tile_size
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def tile_boxes(width: int, height: int, tile_size: int = 672,
               overlap: float = 0.25) -> list[tuple[int, int, int, int]]:
    """Return row-major PIL crop boxes in EXIF-corrected source coordinates."""
    xs = axis_starts(width, tile_size, overlap)
    ys = axis_starts(height, tile_size, overlap)
    return [(x, y, min(x + tile_size, width), min(y + tile_size, height))
            for y in ys for x in xs]


def feather_weights(width: int, height: int) -> np.ndarray:
    """Positive triangular pixel-center weights, including all image edges."""
    _positive_int(width, "width")
    _positive_int(height, "height")

    def axis_weights(length):
        positions = np.arange(length, dtype=np.float32) + 0.5
        return np.minimum(positions, length - positions) / (length / 2)

    return np.outer(axis_weights(height), axis_weights(width)).astype(np.float32)


def _predict_map(model, transform, image: Image.Image, device: torch.device,
                 smoothing_kernel: int) -> tuple[float, np.ndarray]:
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        score, mask = model(tensor)
        if not isinstance(mask, torch.Tensor) or mask.ndim != 4 or mask.shape[:2] != (1, 1):
            raise ValueError("Model mask must have shape [1, 1, height, width].")
        if mask.shape[2] == 0 or mask.shape[3] == 0:
            raise ValueError("Model mask dimensions must be nonempty.")
        mask = F.avg_pool2d(mask.float(), smoothing_kernel, stride=1,
                            padding=smoothing_kernel // 2)
        # Match the web pipeline: mean on decoder grid, then CPU interpolation.
        mask_cpu = mask.detach().float().cpu()
        values = F.interpolate(mask_cpu, size=(image.height, image.width),
                               mode="bilinear", align_corners=False)[0, 0].numpy()
    if not np.isfinite(values).all():
        raise ValueError("Model produced a non-finite anomaly map.")
    if not isinstance(score, torch.Tensor) or score.numel() != 1:
        raise ValueError("Model image score must contain exactly one value.")
    score_value = float(score.detach().float().cpu().item())
    if not math.isfinite(score_value):
        raise ValueError("Model produced a non-finite image score.")
    return score_value, np.clip(values, 0, 1).astype(np.float32, copy=False)


def predict_comparison(model, image: Image.Image, device: torch.device,
                       input_size: int = 672, tile_size: int = 672,
                       overlap: float = 0.25, smoothing_kernel: int = 5,
                       global_weight: float = 0.25,
                       progress: Callable[[int, int], None] | None = None) -> dict:
    """Compare full-image inference with full-image + overlapping tile fusion.

    The caller owns model loading/device/dtype. This function sets eval mode
    and resolution, and is not safe for concurrent use of the same model.
    Output HxW float32 maps use the EXIF-corrected image coordinates.
    Progress receives (completed_tiles, total_tiles), first after baseline.
    Timings include preprocessing, inference, transfers and map reconstruction;
    callback time is excluded. CUDA peaks include the already-loaded model.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image.")
    _positive_int(input_size, "input_size")
    if input_size not in (336, 672):
        raise ValueError("input_size must be 336 or 672 for the demo checkpoint.")
    _positive_int(smoothing_kernel, "smoothing_kernel")
    if smoothing_kernel % 2 == 0:
        raise ValueError("smoothing_kernel must be odd.")
    _fraction(global_weight, "global_weight", include_one=True)
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None.")
    device = torch.device(device)
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    boxes = tile_boxes(width, height, tile_size, overlap)
    configure_resolution(model, input_size)
    model.eval()
    transform = model.model.get_img_transform()

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    synchronize()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    global_score, baseline = _predict_map(model, transform, image, device, smoothing_kernel)
    synchronize()
    baseline_seconds = time.perf_counter() - started
    if progress is not None:
        progress(0, len(boxes))

    started = time.perf_counter()
    weighted_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    callback_seconds = 0.0
    for index, (left, top, right, bottom) in enumerate(boxes, start=1):
        crop = image.crop((left, top, right, bottom))
        _, values = _predict_map(model, transform, crop, device, smoothing_kernel)
        weights = feather_weights(crop.width, crop.height)
        weighted_sum[top:bottom, left:right] += values * weights
        weight_sum[top:bottom, left:right] += weights
        if progress is not None:
            callback_started = time.perf_counter()
            progress(index, len(boxes))
            callback_seconds += time.perf_counter() - callback_started
    # Every source pixel has a positive contribution, including singleton axes.
    if not np.all(weight_sum > 0):
        raise RuntimeError("Tile stitching left uncovered pixels.")
    tiled = weighted_sum / weight_sum
    fused = (global_weight * baseline + (1 - global_weight) * tiled).astype(np.float32)
    synchronize()
    tiles_seconds = time.perf_counter() - started - callback_seconds
    metadata = {
        "original_size": [width, height],
        "input_size": input_size,
        "tile_size": tile_size,
        "overlap": overlap,
        "smoothing_kernel": smoothing_kernel,
        "global_weight": global_weight,
        "tile_boxes": [list(box) for box in boxes],
        "tile_count": len(boxes),
        "baseline_seconds": baseline_seconds,
        "tiles_seconds": tiles_seconds,
        "proposed_seconds": baseline_seconds + tiles_seconds,
        "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda" else None,
        "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3
        if device.type == "cuda" else None,
    }
    return {"baseline": baseline, "tiled": tiled, "fused": fused,
            "global_score": global_score, "metadata": metadata}
