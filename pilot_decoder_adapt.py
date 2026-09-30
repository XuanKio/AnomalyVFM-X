"""Small decoder-head adaptation experiment; never replaces the released model.

The frozen CLIP and decoder body produce cached features. Only the final 1x1
mask head is fitted on reviewed weak synthetic labels. A held-out category and
one real image are compared at the same, fixed inference threshold.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from demo_fast import configure_resolution, load_demo_model
from demo_visuals import render_prediction
from mask_ops import preserve_defects_at_decoder_scale
from pilot_quality_gate import write_gate


def resize_map(values: torch.Tensor, image: Image.Image) -> np.ndarray:
    return F.interpolate(values[None, None], size=(image.height, image.width),
                         mode="bilinear", align_corners=False)[0, 0].numpy().clip(0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scooter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, choices=(336, 672), default=672)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--hard-negative-fraction", type=float, default=0.0)
    parser.add_argument("--hard-negative-weight", type=float, default=0.5)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if not 0 <= args.hard_negative_fraction <= 1 or args.hard_negative_weight < 0:
        parser.error("hard-negative fraction must be in [0,1] and weight nonnegative")
    torch.manual_seed(args.seed)
    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    train_names = [r["name"] for r in manifest["pairs"] if r["accepted_for_pilot_training"]]
    rejected_names = [r["name"] for r in manifest["pairs"] if not r["accepted_for_pilot_training"]]
    if not train_names or not rejected_names:
        raise ValueError("Need both reviewed training pairs and a held-out category")
    paths = {}
    for name in train_names:
        paths[f"{name}_good"] = args.dataset / "train/ok" / f"{name}.png"
        paths[f"{name}_bad"] = args.dataset / "train/bad" / f"{name}.png"
    for name in rejected_names:
        paths[f"{name}_good"] = args.source / name / "good.png"
        paths[f"{name}_bad"] = args.source / name / "bad.png"
    paths["scooter_real"] = args.scooter
    paths["hazelnut_good"] = Path("demo_images/hazelnut_normal.png")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_demo_model()
    configure_resolution(model, args.size)
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device)
    model.eval()
    transform = model.model.get_img_transform()
    features: dict[str, torch.Tensor] = {}
    images: dict[str, Image.Image] = {}
    grid = args.size // model.model.patch_size
    with torch.inference_mode():
        for key, path in paths.items():
            image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            images[key] = image
            tensor = transform(image).unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                _, tokens = model.model(tensor)
                x = tokens.permute(0, 2, 1).reshape(1, -1, grid, grid)
                x = model.decoder.bot(x)
                for block in model.decoder.blocks:
                    x = block(x)
            features[key] = x[0].to(device="cpu", dtype=torch.float16)
            print(f"Cached {key}: {tuple(x.shape)}", flush=True)
    old_weight = model.decoder.final.weight.detach().float().cpu().clone()
    old_bias = model.decoder.final.bias.detach().float().cpu().clone()
    del model, tensor, tokens, x
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    weight = torch.nn.Parameter(old_weight.to(device).clone())
    bias = torch.nn.Parameter(old_bias.to(device).clone())
    original_weight, original_bias = old_weight.to(device), old_bias.to(device)
    labels = {}
    for name in train_names:
        with Image.open(args.dataset / "ground_truth/bad" / f"{name}.png") as image:
            mask = torch.from_numpy(np.array(image.convert("L"), copy=True)).float()[None, None] / 255
        mask = preserve_defects_at_decoder_scale(mask[0], grid * 2)
        labels[f"{name}_bad"] = (mask >= 0.5).float()[None].to(device)
        labels[f"{name}_good"] = torch.zeros_like(labels[f"{name}_bad"])
        print(f"Label {name}: {int(labels[f'{name}_bad'].sum())} grid pixels", flush=True)
    optimizer = torch.optim.AdamW([weight, bias], lr=0.002, weight_decay=0)
    for step in range(args.steps):
        optimizer.zero_grad()
        losses = []
        hard_losses = []
        for name in train_names:
            for kind in ("good", "bad"):
                key = f"{name}_{kind}"
                logits = F.conv2d(features[key].to(device=device, dtype=torch.float32)[None], weight, bias)
                target = labels[key]
                positives = target.sum()
                positive_weight = (target.numel() / positives.clamp(min=1) * 0.25).clamp(max=200)
                losses.append(F.binary_cross_entropy_with_logits(
                    logits, target, pos_weight=positive_weight if kind == "bad" else None
                ))
                if kind == "good" and args.hard_negative_fraction > 0:
                    # Mine the highest-scoring healthy cells across the entire
                    # image, without an object-specific ROI or test-image labels.
                    per_cell = F.softplus(logits.flatten())
                    k = max(1, round(args.hard_negative_fraction * per_cell.numel()))
                    hard_losses.append(per_cell.topk(k).values.mean())
        loss = torch.stack(losses).mean()
        if hard_losses:
            loss = loss + args.hard_negative_weight * torch.stack(hard_losses).mean()
        loss = loss + 0.01 * (weight - original_weight).square().mean()
        loss.backward()
        optimizer.step()
        if step in {0, args.steps // 2, args.steps - 1}:
            print(f"step {step + 1}/{args.steps} loss={float(loss.detach()):.5f}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"weight": weight.detach().float().cpu(), "bias": bias.detach().float().cpu(),
                "base_checkpoint": "MaticFuc/anomalyvfm_clip", "size": args.size,
                "train_names": train_names, "steps": args.steps, "seed": args.seed,
                "hard_negative_fraction": args.hard_negative_fraction,
                "hard_negative_weight": args.hard_negative_weight},
               args.output / "experimental_head.pt")
    rows = []
    html_rows = []
    with torch.inference_mode():
        for key in paths:
            feature = features[key].to(device=device, dtype=torch.float32)[None]
            views = []
            for variant, w, b in (("baseline", original_weight, original_bias),
                                   ("adapted", weight, bias)):
                logits = F.conv2d(feature, w, b)
                scores = F.avg_pool2d(logits.sigmoid().float(), 5, stride=1, padding=2)[0, 0].cpu()
                values = resize_map(scores, images[key])
                rendered = render_prediction(images[key], values, 0.5)
                prefix = f"{key}_{variant}"
                rendered["overlay"].save(args.output / f"{prefix}_overlay.png")
                rendered["mask"].save(args.output / f"{prefix}_mask.png")
                np.save(args.output / f"{prefix}_scores.npy", values)
                views.append(prefix)
                row = {"image": key, "variant": variant, "max_score": float(values.max()),
                       "mask_pixels_at_0_5": int((values >= 0.5).sum()),
                       "mean_score": float(values.mean())}
                if key.endswith("_bad") and key[:-4] in train_names:
                    with Image.open(args.dataset / "ground_truth/bad" / f"{key[:-4]}.png") as target_image:
                        weak_mask = np.asarray(target_image.convert("L")) >= 128
                    predicted = values >= 0.5
                    overlap = int((predicted & weak_mask).sum())
                    row["weak_label_recall"] = overlap / max(1, int(weak_mask.sum()))
                    row["weak_label_precision"] = overlap / max(1, int(predicted.sum()))
                rows.append(row)
            html_rows.append("<tr><td>" + key + "</td>" + "".join(
                f'<td><img src="{prefix}_overlay.png"><br><img src="{prefix}_mask.png"></td>'
                for prefix in views) + "</tr>")
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=False).stdout.strip()
    manifest_hash = hashlib.sha256((args.dataset / "manifest.json").read_bytes()).hexdigest()
    report = {"method": "frozen_CLIP_and_decoder_body_train_final_mask_head_only",
              "not_promoted_to_web_demo": True, "weak_train_pairs": train_names,
              "heldout_rejected_pairs": rejected_names,
              "steps": args.steps, "size": args.size, "seed": args.seed,
              "hard_negative_fraction": args.hard_negative_fraction,
              "hard_negative_weight": args.hard_negative_weight,
              "threshold": 0.5, "git_head": git_head, "dataset_manifest_sha256": manifest_hash,
              "torch_version": torch.__version__,
              "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
              "samples": rows}
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "comparison.html").write_text(
        '<!doctype html><html lang="vi"><meta charset="utf-8"><title>Pilot decoder comparison</title>'
        '<style>body{font:16px Segoe UI;background:#101621;color:#eee}table{border-collapse:collapse}'
        'td,th{border:1px solid #789;padding:8px;vertical-align:top}img{width:min(36vw,600px)}'
        '</style><h1>Thử nghiệm đầu định vị trên nhãn tổng hợp yếu</h1>'
        '<p>Cùng ngưỡng 0,50; ảnh kim loại và xe thật không có nhãn chuẩn đầy đủ. '
        'Bản thử nghiệm chưa được đưa vào web demo.</p><table><tr><th>Ảnh</th><th>Gốc</th>'
        '<th>Đầu định vị thử nghiệm</th></tr>' + "".join(html_rows) + '</table></html>',
        encoding="utf-8")
    write_gate(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
