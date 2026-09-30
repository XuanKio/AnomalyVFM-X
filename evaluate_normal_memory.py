"""Evaluate a normal patch memory against the same CLIP decoder on real VisA.

This is an optional few-normal-reference method inspired by PatchCore. It is
not zero-shot and does not claim to reproduce PatchCore's feature extractor.
Ten normal images per category build the bank; none enter evaluation.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from analyze_visa_ranking import image_ap, image_auc
from demo_fast import configure_resolution, load_demo_model
from evaluate_visa_slice import PixelAccumulator, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-per-label", type=int, default=20,
                        help="Evaluation samples per class/label; 0 means all held-out images")
    parser.add_argument("--bank-images", type=int, default=10)
    parser.add_argument("--patch-stride", type=int, default=2)
    args = parser.parse_args()
    if args.limit_per_label < 0 or args.bank_images < 1 or args.patch_stride < 1:
        parser.error("Limits and strides must be positive (limit may be 0)")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_demo_model()
    configure_resolution(model, 672)
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device)
    model.eval()
    transform = model.model.get_img_transform()
    grid = 672 // model.model.patch_size
    accumulators = {(category, variant): PixelAccumulator()
                    for category in manifest["categories"] for variant in ("baseline", "normal_memory")}
    image_rows = []
    started = time.perf_counter()

    @torch.inference_mode()
    def extract(row):
        with Image.open(args.dataset / row["image"]) as opened:
            image = opened.convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            _, tokens = model.model(tensor)
        return image, tokens.float()

    for category in manifest["categories"]:
        normal = [row for row in manifest["samples"]
                  if row["class"] == category and row["label"] == "normal"]
        abnormal = [row for row in manifest["samples"]
                    if row["class"] == category and row["label"] != "normal"]
        if len(normal) <= args.bank_images or not abnormal:
            raise ValueError(f"Not enough VisA samples in {category}")
        bank_rows = normal[:args.bank_images]
        bank = []
        for row in bank_rows:
            _, tokens = extract(row)
            patches = F.normalize(tokens[0], dim=-1).reshape(grid, grid, -1)
            bank.append(patches[::args.patch_stride, ::args.patch_stride].reshape(-1, patches.shape[-1]))
        bank = torch.cat(bank).T.contiguous()
        evaluation = normal[args.bank_images:]
        if args.limit_per_label:
            evaluation = evaluation[:args.limit_per_label]
            abnormal = abnormal[:args.limit_per_label]
        evaluation += abnormal
        for number, row in enumerate(evaluation, 1):
            image, tokens = extract(row)
            with torch.inference_mode():
                feature = tokens.permute(0, 2, 1).reshape(1, -1, grid, grid)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    logits, _ = model.decoder(feature.to(dtype=next(model.decoder.parameters()).dtype))
                baseline = F.avg_pool2d(logits.sigmoid().float(), 5, stride=1, padding=2)
                patches = F.normalize(tokens[0], dim=-1)
                distances = []
                for chunk in patches.split(512):
                    similarities = chunk @ bank
                    distances.append(1 - similarities.max(dim=1).values)
                memory = torch.cat(distances).reshape(1, 1, grid, grid).clamp(0, 1)
                maps = {"baseline": baseline, "normal_memory": memory}
                if row["label"] == "normal":
                    truth = np.zeros((image.height, image.width), bool)
                else:
                    with Image.open(args.dataset / row["mask"]) as opened:
                        truth = np.asarray(opened.convert("L")) > 0
                for variant, coarse in maps.items():
                    full = F.interpolate(coarse, size=(image.height, image.width),
                                         mode="bilinear", align_corners=False)
                    scores = full[0, 0].float().cpu().numpy()
                    accumulators[(category, variant)].add(scores, truth, row["label"] != "normal")
                    image_rows.append({"class": category, "variant": variant,
                                       "image": row["image"], "label": row["label"],
                                       "max_score": float(scores.max())})
            if number % 20 == 0 or number == len(evaluation):
                print(f"{category}: evaluated {number}/{len(evaluation)}", flush=True)
        del bank
    summary = {}
    for category in manifest["categories"]:
        summary[category] = {}
        for variant in ("baseline", "normal_memory"):
            rows = [row for row in image_rows if row["class"] == category and row["variant"] == variant]
            positive = np.array([row["max_score"] for row in rows if row["label"] != "normal"])
            negative = np.array([row["max_score"] for row in rows if row["label"] == "normal"])
            metric = accumulators[(category, variant)].summary()
            summary[category][variant] = {
                "pixel_auroc_1001_bins": metric["pixel_auroc_1001_bins"],
                "pixel_ap_1001_bins": metric["pixel_ap_1001_bins"],
                "image_auroc_exact": image_auc(positive, negative),
                "image_ap_exact": image_ap(positive, negative),
                "healthy_images": len(negative), "abnormal_images": len(positive),
            }
    report = {"dataset_manifest_sha256": sha256(manifest_path),
              "dataset_source": manifest["source_url"], "bank_images_per_class": args.bank_images,
              "bank_patch_stride": args.patch_stride,
              "bank_selection": "first sorted normal images in manifest; excluded from evaluation",
              "limit_per_label": args.limit_per_label, "image_size": 672,
              "device": str(device), "seconds": time.perf_counter() - started,
              "metric_note": "Pixel AUROC/AP approximate with 1001 score bins; no threshold tuned",
              "by_class": summary, "per_image": image_rows}
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"seconds": report["seconds"], "by_class": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
