"""Compare local CLIP checkpoint and experimental mask heads on real VisA GT.

The slice is strictly evaluation-only. All variants share image input,
resolution, decoder features, 5x5 smoothing and a fixed 0.50 mask threshold.
Pixel AUROC/AP use 1001 fixed score bins to avoid retaining full-size maps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from demo_fast import configure_resolution, load_demo_model
from demo_visuals import render_prediction


BINS = 1001


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


class PixelAccumulator:
    def __init__(self) -> None:
        self.positive = np.zeros(BINS, np.int64)
        self.negative = np.zeros(BINS, np.int64)
        self.tp = self.fp = self.fn = 0
        self.normal_images = self.abnormal_images = 0
        self.false_alarm_images = self.detected_abnormal_images = 0

    def add(self, scores: np.ndarray, truth: np.ndarray, abnormal: bool) -> dict:
        if scores.shape != truth.shape or not np.isfinite(scores).all():
            raise ValueError("Scores and truth must be aligned and finite")
        index = np.clip((scores * (BINS - 1)).astype(np.int32), 0, BINS - 1)
        self.positive += np.bincount(index[truth], minlength=BINS)
        self.negative += np.bincount(index[~truth], minlength=BINS)
        predicted = scores >= 0.5
        tp = int((predicted & truth).sum())
        fp = int((predicted & ~truth).sum())
        fn = int((~predicted & truth).sum())
        self.tp += tp
        self.fp += fp
        self.fn += fn
        if abnormal:
            self.abnormal_images += 1
            self.detected_abnormal_images += int(predicted.any())
        else:
            self.normal_images += 1
            self.false_alarm_images += int(predicted.any())
        return {"gt_pixels": int(truth.sum()), "predicted_pixels": int(predicted.sum()),
                "tp": tp, "fp": fp, "fn": fn, "max_score": float(scores.max())}

    def summary(self) -> dict:
        positives, negatives = int(self.positive.sum()), int(self.negative.sum())
        true_pos = np.r_[0, np.cumsum(self.positive[::-1])].astype(np.float64)
        false_pos = np.r_[0, np.cumsum(self.negative[::-1])].astype(np.float64)
        recall = true_pos / max(positives, 1)
        fpr = false_pos / max(negatives, 1)
        precision_curve = true_pos / np.maximum(true_pos + false_pos, 1)
        auroc = float(np.sum(np.diff(fpr) * (recall[1:] + recall[:-1]) / 2))
        ap = float(np.sum(np.diff(recall) * precision_curve[1:]))
        precision = self.tp / max(self.tp + self.fp, 1)
        pixel_recall = self.tp / max(self.tp + self.fn, 1)
        return {"pixel_auroc_1001_bins": auroc, "pixel_ap_1001_bins": ap,
                "pixel_precision_at_0_5": precision, "pixel_recall_at_0_5": pixel_recall,
                "pixel_f1_at_0_5": 2 * precision * pixel_recall / max(precision + pixel_recall, 1e-12),
                "healthy_image_false_alarm_rate_at_0_5": self.false_alarm_images / max(self.normal_images, 1),
                "anomalous_image_any_mask_rate_at_0_5": self.detected_abnormal_images / max(self.abnormal_images, 1),
                "healthy_images": self.normal_images, "anomalous_images": self.abnormal_images,
                "gt_defect_pixels": positives, "tp": self.tp, "fp": self.fp, "fn": self.fn}


def parse_heads(values: list[str]) -> dict[str, Path]:
    heads = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--head format is name=path")
        name, path = value.split("=", 1)
        if not name.isidentifier() or name == "baseline" or name in heads:
            raise ValueError("Head names must be unique identifiers other than baseline")
        heads[name] = Path(path)
    return heads


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head", action="append", default=[], help="name=experimental_head.pt")
    parser.add_argument("--size", type=int, choices=(336, 672), default=672)
    parser.add_argument("--limit-per-label", type=int, default=0,
                        help="Smoke limit per class/label; 0 evaluates the whole slice")
    args = parser.parse_args()
    if args.limit_per_label < 0:
        parser.error("--limit-per-label cannot be negative")
    head_paths = parse_heads(args.head)
    dataset_manifest_path = args.dataset / "manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    samples = dataset_manifest["samples"]
    if args.limit_per_label:
        chosen, counts = [], {}
        for row in samples:
            key = (row["class"], row["label"] == "normal")
            if counts.get(key, 0) < args.limit_per_label:
                chosen.append(row)
                counts[key] = counts.get(key, 0) + 1
        samples = chosen
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_demo_model()
    configure_resolution(model, args.size)
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device)
    model.eval()
    transform = model.model.get_img_transform()
    grid = args.size // model.model.patch_size
    heads = {"baseline": (model.decoder.final.weight.detach().float().clone(),
                          model.decoder.final.bias.detach().float().clone())}
    for name, path in head_paths.items():
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["size"] != args.size or state["weight"].shape != heads["baseline"][0].shape:
            raise ValueError(f"Experimental head is incompatible: {name}")
        heads[name] = (state["weight"].to(device), state["bias"].to(device))
    accumulators = {(category, name): PixelAccumulator()
                    for category in dataset_manifest["categories"] for name in heads}
    per_image = []
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    previewed = set()
    with torch.inference_mode():
        for number, row in enumerate(samples, 1):
            path = args.dataset / row["image"]
            with Image.open(path) as source:
                image = source.convert("RGB")
            abnormal = row["label"] != "normal"
            if abnormal:
                with Image.open(args.dataset / row["mask"]) as source:
                    truth = np.asarray(source.convert("L")) > 0
            else:
                truth = np.zeros((image.height, image.width), dtype=bool)
            tensor = transform(image).unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                summary, tokens = model.model(tensor)
                feature = tokens.permute(0, 2, 1).reshape(1, -1, grid, grid)
                feature = model.decoder.bot(feature)
                for block in model.decoder.blocks:
                    feature = block(feature)
                image_score = float(model.predictor(summary).sigmoid().float().cpu().item())
            feature = feature.float()
            for name, (weight, bias) in heads.items():
                logits = F.conv2d(feature, weight, bias)
                scores = F.avg_pool2d(logits.sigmoid(), 5, stride=1, padding=2)
                full = F.interpolate(scores, size=(image.height, image.width),
                                     mode="bilinear", align_corners=False)[0, 0].float().cpu().numpy()
                data = accumulators[(row["class"], name)].add(full, truth, abnormal)
                per_image.append({"class": row["class"], "label": row["label"],
                                  "image": row["image"], "variant": name,
                                  "image_score": image_score, **data})
                preview_key = (row["class"], abnormal, name)
                if preview_key not in previewed:
                    folder = args.output / "previews" / row["class"]
                    folder.mkdir(parents=True, exist_ok=True)
                    label = "bad" if abnormal else "good"
                    render_prediction(image, full, .5)["overlay"].save(folder / f"{label}_{name}.png")
                    previewed.add(preview_key)
            if number % 25 == 0 or number == len(samples):
                print(f"evaluated {number}/{len(samples)}", flush=True)
    by_class = {category: {name: accumulators[(category, name)].summary() for name in heads}
                for category in dataset_manifest["categories"]}
    report = {"dataset_source": dataset_manifest["source_url"],
              "dataset_manifest_sha256": sha256(dataset_manifest_path),
              "dataset_prefix_sha256": dataset_manifest["local_prefix_sha256"],
              "head_sha256": {name: sha256(path) for name, path in head_paths.items()},
              "image_size": args.size, "smoothing_kernel": 5, "threshold": .5,
              "pixel_metric_note": "Approximate AUROC/AP from 1001 fixed bins, full-size official masks; image detection uses any positive mask pixel.",
              "images_evaluated": len(samples), "seconds": time.perf_counter() - started,
              "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
              "torch_version": torch.__version__, "by_class": by_class, "per_image": per_image}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"images_evaluated": len(samples), "seconds": report["seconds"],
                      "by_class": by_class}, indent=2), flush=True)


if __name__ == "__main__":
    main()
