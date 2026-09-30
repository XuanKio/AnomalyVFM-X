"""Measure whether fine-defect evidence exists before mask smoothing.

Synthetic reference-derived masks are weak labels. This analysis does not
measure real-world accuracy or tune a threshold on the inspected image.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from demo_fast import configure_resolution, load_demo_model
from demo_visuals import render_prediction
from evaluate_visa_slice import PixelAccumulator


def stats(scores: np.ndarray, mask: np.ndarray | None) -> dict:
    y, x = np.unravel_index(int(scores.argmax()), scores.shape)
    result = {"max_score": float(scores.max()), "maximum_xy": [int(x), int(y)],
              "mask_pixels_at_0_5": int((scores >= .5).sum())}
    if mask is not None:
        positive = scores[mask]
        outside = scores[~mask]
        metrics = PixelAccumulator()
        metrics.add(scores, mask, abnormal=True)
        count = int(mask.sum())
        top = np.argpartition(scores.ravel(), -count)[-count:]
        result.update({
            "weak_label_pixels": count,
            "weak_label_score_max": float(positive.max()),
            "weak_label_score_mean": float(positive.mean()),
            "outside_weak_label_score_max": float(outside.max()),
            "weak_label_pixel_auroc_1001_bins": metrics.summary()["pixel_auroc_1001_bins"],
            "weak_label_recall_with_oracle_top_k": float(mask.ravel()[top].sum() / count),
            "maximum_inside_weak_label": bool(mask[y, x]),
        })
    return result


def run(source: Path, head_path: Path, output: Path, size: int = 672) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_demo_model()
    configure_resolution(model, size)
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device)
    model.eval()
    transform = model.model.get_img_transform()
    state = torch.load(head_path, map_location="cpu", weights_only=True)
    if state["size"] != size:
        raise ValueError("Experimental head has a different resolution")
    heads = {
        "baseline": (model.decoder.final.weight.detach().float(),
                     model.decoder.final.bias.detach().float()),
        "hardneg": (state["weight"].to(device), state["bias"].to(device)),
    }
    grid = size // model.model.patch_size
    rows = []
    with torch.inference_mode():
        for name in ("car_panel", "cable", "pcb", "metal_plate"):
            for kind in ("good", "bad"):
                with Image.open(source / name / f"{kind}.png") as opened:
                    image = opened.convert("RGB")
                mask = None
                if kind == "bad":
                    with Image.open(source / name / "refined_mask.png") as opened:
                        mask = np.asarray(opened.convert("L")) >= 128
                tensor = transform(image).unsqueeze(0).to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    _, tokens = model.model(tensor)
                    feature = tokens.permute(0, 2, 1).reshape(1, -1, grid, grid)
                    feature = model.decoder.bot(feature)
                    for block in model.decoder.blocks:
                        feature = block(feature)
                feature = feature.float()
                for variant, (weight, bias) in heads.items():
                    raw = F.conv2d(feature, weight, bias).sigmoid()
                    for kernel in (1, 3, 5):
                        smoothed = F.avg_pool2d(raw, kernel, stride=1, padding=kernel // 2)
                        full = F.interpolate(smoothed, size=(image.height, image.width),
                                             mode="bilinear", align_corners=False)
                        scores = full[0, 0].float().cpu().numpy()
                        row = {"image": f"{name}_{kind}", "variant": variant,
                               "smoothing_kernel": kernel, **stats(scores, mask)}
                        rows.append(row)
                        if kind == "bad" and kernel in (1, 5):
                            render_prediction(image, scores, .5)["overlay"].save(
                                output / f"{name}_{variant}_k{kernel}.png")
                print(f"evaluated {name}_{kind}", flush=True)
    report = {"source": str(source.resolve()), "head": str(head_path.resolve()),
              "size": size, "labels": "reference-derived weak synthetic masks; metal incomplete",
              "meaning": "Oracle top-K measures ranking at known weak-mask area, not deployable threshold",
              "rows": rows}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, choices=(336, 672), default=672)
    args = parser.parse_args()
    run(args.source, args.head, args.output, args.size)


if __name__ == "__main__":
    main()
