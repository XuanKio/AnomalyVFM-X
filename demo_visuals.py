"""Model-independent, fixed-scale views of an anomaly score map.

Scores are model outputs on a 0..1 scale, not calibrated probabilities. The
binary view is a thresholded prediction, not a ground-truth annotation.
"""

from __future__ import annotations

import math
from numbers import Real

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _colorize(values: np.ndarray) -> np.ndarray:
    """Interpolate a fixed blue-cyan-yellow-red palette over the whole 0..1 range."""
    positions = (0.0, 0.125, 0.375, 0.625, 0.875, 1.0)
    colors = np.array(
        [(0, 0, 128), (0, 0, 255), (0, 255, 255),
         (255, 255, 0), (255, 0, 0), (128, 0, 0)],
        dtype=np.float64,
    )
    channels = [np.interp(values, positions, colors[:, i]) for i in range(3)]
    return np.rint(np.stack(channels, axis=-1)).astype(np.uint8)


def _label_font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _composite(panels: dict[str, Image.Image], threshold: float) -> Image.Image:
    width, height = panels["input"].size
    font = _label_font(max(18, min(28, width // 24)))
    labels = (
        ("input", "Input"),
        ("heatmap", "Anomaly heatmap (0-1)"),
        ("overlay", "Overlay"),
        ("mask", f"Predicted mask (t={threshold})"),
    )
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bounds = [measure.textbbox((0, 0), label, font=font) for _, label in labels]
    # Tiny inputs still get readable labels; image pixels themselves are not resized.
    column_width = max(width, max(box[2] - box[0] for box in bounds) + 24)
    title_height = max(box[3] - box[1] for box in bounds) + 24
    canvas = Image.new("RGB", (column_width * 4, height + title_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, ((key, label), box) in enumerate(zip(labels, bounds)):
        left = index * column_width
        label_width = box[2] - box[0]
        draw.text(
            (left + (column_width - label_width) // 2 - box[0], 12 - box[1]),
            label,
            font=font,
            fill="black",
        )
        canvas.paste(panels[key], (left + (column_width - width) // 2, title_height))
    return canvas


def render_prediction(
    image: Image.Image, values: np.ndarray, threshold: float = 0.5
) -> dict[str, Image.Image]:
    """Render an aligned score map without per-image normalization or cleanup.

    ``values`` must be a finite, real H-by-W map in [0, 1], matching ``image``.
    Pixels with raw score >= ``threshold`` are white in the predicted mask.
    Input, heatmap and overlay are RGB; mask is L. These four views retain the
    original image dimensions. Composite adds titles above horizontal columns.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL image.")
    width, height = image.size
    if width < 1 or height < 1:
        raise ValueError("image must have nonzero width and height.")
    if not isinstance(threshold, Real) or isinstance(threshold, (bool, np.bool_)):
        raise TypeError("threshold must be a real number in [0, 1].")
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and in [0, 1].")

    values = np.asarray(values)
    if values.shape != (height, width):
        raise ValueError(f"values must have shape {(height, width)}, got {values.shape}.")
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise TypeError("values must contain real numbers.")
    if not np.isfinite(values).all() or (values < 0.0).any() or (values > 1.0).any():
        raise ValueError("values must be finite and in [0, 1].")

    original = image.convert("RGB")
    heatmap = Image.fromarray(_colorize(values))
    panels = {
        "input": original,
        "heatmap": heatmap,
        "overlay": Image.blend(original, heatmap, alpha=0.45),
        "mask": Image.fromarray((values >= threshold).astype(np.uint8) * 255),
    }
    panels["composite"] = _composite(panels, threshold)
    return panels
