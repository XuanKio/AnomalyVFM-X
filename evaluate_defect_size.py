"""Measure real defect-region coverage by size, separating ranking from calibration.

The binary threshold for each class/variant uses only the first 50 healthy
VisA images in the existing evaluation report. Remaining normals and all
abnormal images are evaluation-only. Top-1% coverage is a ranking diagnostic,
not a deployable prediction rule.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from demo_fast import configure_resolution, load_demo_model
from evaluate_visa_slice import sha256


def size_bin(area: int) -> str:
    if area < 100:
        return "under_100"
    if area < 1000:
        return "100_to_999"
    return "1000_plus"


def threshold_from_normals(values: list[float], allowed_alarms: int = 2) -> float:
    if not 0 <= allowed_alarms < len(values):
        raise ValueError("Alarm budget must be smaller than calibration set")
    ranked = sorted(values, reverse=True)
    return float(np.nextafter(ranked[allowed_alarms], np.inf))


def region_coverage(scores: np.ndarray, region: np.ndarray,
                    threshold: float, top_one_percent: float) -> dict:
    inside = scores[region]
    return {"pixels": int(inside.size), "max_score": float(inside.max()),
            "coverage_at_calibrated_threshold": float(np.mean(inside >= threshold)),
            "coverage_in_image_top_1pct": float(np.mean(inside >= top_one_percent))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True,
                        help="Existing full evaluate_visa_slice report.json")
    parser.add_argument("--head", type=Path, required=True,
                        help="Experimental hard-negative head used in that report")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new empty output directory")
    manifest_path = args.dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    previous = json.loads(args.evaluation.read_text(encoding="utf-8"))
    if (previous["dataset_manifest_sha256"] != sha256(manifest_path)
            or previous["head_sha256"].get("hardneg") != sha256(args.head)
            or previous["image_size"] != 672 or previous["smoothing_kernel"] != 5
            or previous["images_evaluated"] != len(manifest["samples"])):
        raise ValueError("Evaluation report, manifest, or head differs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_demo_model()
    configure_resolution(model, 672)
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device)
    model.eval()
    transform = model.model.get_img_transform()
    state = torch.load(args.head, map_location="cpu", weights_only=True)
    if state["size"] != 672 or state["weight"].shape != model.decoder.final.weight.shape:
        raise ValueError("Head is incompatible with 672 decoder")
    heads = {"baseline": (model.decoder.final.weight.detach().float(),
                           model.decoder.final.bias.detach().float()),
             "hardneg": (state["weight"].to(device), state["bias"].to(device))}
    thresholds = {}
    heldout_healthy = {}
    for category in manifest["categories"]:
        for variant in heads:
            healthy = sorted((row for row in previous["per_image"]
                              if row["class"] == category and row["variant"] == variant
                              and row["label"] == "normal"), key=lambda row: row["image"])
            if len(healthy) != 100:
                raise ValueError(f"Expected 100 healthy images in {category}/{variant}")
            threshold = threshold_from_normals([row["max_score"] for row in healthy[:50]])
            thresholds[(category, variant)] = threshold
            heldout_healthy[(category, variant)] = {
                "images": len(healthy[50:]),
                "false_alarms": sum(row["max_score"] >= threshold for row in healthy[50:]),
                "calibration_images": 50, "calibration_allowed_alarms": 2,
                "threshold": threshold}
    args.output.mkdir(parents=True, exist_ok=True)
    regions = []
    skipped_tiny_components = 0
    started = time.perf_counter()
    abnormal = [row for row in manifest["samples"] if row["label"] != "normal"]
    grid = 672 // model.model.patch_size
    with torch.inference_mode():
        for number, row in enumerate(abnormal, 1):
            with Image.open(args.dataset / row["image"]) as opened:
                image = opened.convert("RGB")
            with Image.open(args.dataset / row["mask"]) as opened:
                truth = np.asarray(opened.convert("L")) > 0
            count, components, stats, _ = cv2.connectedComponentsWithStats(
                truth.astype(np.uint8), connectivity=8)
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
                coarse = F.avg_pool2d(F.conv2d(feature, weight, bias).sigmoid(),
                                      5, stride=1, padding=2)
                full = F.interpolate(coarse, size=truth.shape, mode="bilinear",
                                     align_corners=False)[0, 0].cpu().numpy()
                threshold = thresholds[(row["class"], variant)]
                top = float(np.quantile(full, .99))
                for index in range(1, count):
                    area = int(stats[index, cv2.CC_STAT_AREA])
                    if area < 8:
                        if variant == "baseline":
                            skipped_tiny_components += 1
                        continue
                    component = components == index
                    metrics = region_coverage(full, component, threshold, top)
                    regions.append({"class": row["class"], "image": row["image"],
                                    "variant": variant, "component": index,
                                    "area_bin": size_bin(area), **metrics})
            if number % 25 == 0 or number == len(abnormal):
                print(f"evaluated {number}/{len(abnormal)} abnormal images", flush=True)
    by_size = {}
    for category in manifest["categories"]:
        by_size[category] = {}
        for variant in heads:
            by_size[category][variant] = {}
            for bucket in ("under_100", "100_to_999", "1000_plus"):
                subset = [item for item in regions if item["class"] == category
                          and item["variant"] == variant and item["area_bin"] == bucket]
                by_size[category][variant][bucket] = {
                    "regions": len(subset),
                    "hit_10pct_at_calibrated_threshold": sum(
                        item["coverage_at_calibrated_threshold"] >= .1 for item in subset),
                    "hit_10pct_in_image_top_1pct": sum(
                        item["coverage_in_image_top_1pct"] >= .1 for item in subset),
                    "mean_coverage_at_calibrated_threshold": float(np.mean(
                        [item["coverage_at_calibrated_threshold"] for item in subset]))
                    if subset else None,
                }
    report = {"dataset_manifest_sha256": sha256(manifest_path),
              "prior_evaluation_sha256": sha256(args.evaluation),
              "head_sha256": sha256(args.head),
              "selection": "first 50 sorted normals/class calibrate; last 50 test; all bad test",
              "region_min_pixels": 8, "size_bin_pixels": [100, 1000],
              "hit_definition": "at least 10% of GT component pixels selected",
              "top_1pct_note": "per-image ranking diagnostic, not deployment threshold",
              "heldout_healthy": {category: {variant: heldout_healthy[(category, variant)]
                                               for variant in heads}
                                  for category in manifest["categories"]},
              "by_size": by_size, "skipped_components_under_8": skipped_tiny_components,
              "regions": regions, "seconds": time.perf_counter() - started}
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"heldout_healthy": report["heldout_healthy"],
                      "by_size": by_size, "seconds": report["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
