"""Transductive zero-shot fusion: AnomalyVFM decoder + mutual patch scoring.

Idea (adapted from MuSc, ICLR 2024): in a batch of unlabeled images of one
product, a normal patch finds a near-identical patch in almost every other
image, while a defect patch does not. The score of a patch is the mean of its
smallest nearest-neighbour distances to the other images, so a few defective
images in the batch cannot pull the score down. No normal labels, no training,
no threshold. The map is fused with the released decoder map after both are
standardized with statistics of the same unlabeled batch.

All fusion rules are fixed before evaluation; no parameter is tuned on masks.
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

LAYERS = (6, 12, 18, 24)
RADII = (1, 3, 5)  # MuSc default r_list
PROJECTED_DIM = 256
VARIANTS = ("baseline", "mutual", "fusion")


def aggregate(tokens: torch.Tensor, grid: int, radius: int) -> torch.Tensor:
    """Average each patch with its r x r neighbourhood, then L2-normalize."""
    if radius > 1:
        spatial = tokens.T.reshape(1, -1, grid, grid).float()
        spatial = F.avg_pool2d(spatial, radius, stride=1, padding=radius // 2,
                               count_include_pad=False)
        tokens = spatial.reshape(tokens.shape[1], -1).T
    return F.normalize(tokens.float(), dim=-1).half()


def interval_mean(distances: torch.Tensor, fraction: float) -> torch.Tensor:
    """Mean of the smallest `fraction` of per-image distances (dim 1)."""
    keep = max(1, int(round(distances.shape[1] * fraction)))
    return distances.topk(keep, dim=1, largest=False).values.mean(dim=1)


@torch.inference_mode()
def mutual_scores(features: dict, grid: int, fraction: float, chunk: int) -> torch.Tensor:
    """features[layer]: (N, P, D) fp16 on device. Returns (N, grid, grid)."""
    count = next(iter(features.values())).shape[0]
    total = torch.zeros(count, grid * grid, device=next(iter(features.values())).device)
    for layer, tokens in features.items():
        for radius in RADII:
            tick = time.perf_counter()
            bank = torch.stack([aggregate(tokens[i], grid, radius) for i in range(count)])
            for query in range(count):
                nearest = []
                others = [j for j in range(count) if j != query]
                for start in range(0, len(others), chunk):
                    block = bank[others[start:start + chunk]]  # (k, P, D)
                    sim = torch.einsum("pd,kqd->kpq", bank[query], block)
                    nearest.append(1 - sim.amax(dim=2).float())  # (k, P)
                distances = torch.cat(nearest).T  # (P, N-1)
                total[query] += interval_mean(distances, fraction)
            del bank
            print(f"  mutual scoring layer {layer} r={radius} done "
                  f"({time.perf_counter() - tick:.0f}s)", flush=True)
    return (total / (len(features) * len(RADII))).reshape(count, grid, grid)


def standardize(maps: torch.Tensor) -> torch.Tensor:
    """Batch-level robust z-score: same transform for every image of a class."""
    flat = maps.flatten().float()
    sample = flat[torch.randperm(flat.numel(), generator=torch.Generator().manual_seed(0))[:2_000_000]]
    median = sample.median()
    spread = (sample - median).abs().median().clamp_min(1e-6)
    return (maps - median) / spread


def to_unit(maps: torch.Tensor) -> torch.Tensor:
    """Monotonic per-class rescale to [0,1] for the fixed-bin accumulator."""
    low, high = maps.min(), maps.max()
    return (maps - low) / (high - low).clamp_min(1e-12)


def score_class(args, batch, device):
    """Extract decoder maps and multi-layer tokens, then mutual-score the batch."""
    model = load_demo_model()
    configure_resolution(model, 672)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    transform = model.model.get_img_transform()
    grid = 672 // model.model.patch_size
    width = model.model.feature_dim
    projection = torch.linalg.qr(torch.randn(width, PROJECTED_DIM,
                                           generator=torch.Generator().manual_seed(0)))[0]
    projection = projection.to(device, torch.float32)
    features = {layer: torch.empty(len(batch), grid * grid, PROJECTED_DIM,
                                   dtype=torch.float16, device=device) for layer in LAYERS}
    decoder_maps, predictor_scores = [], []
    with torch.inference_mode():
        for index, row in enumerate(batch):
            with Image.open(args.dataset / row["image"]) as opened:
                tensor = transform(opened.convert("RGB")).unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                out = model.model.net(tensor, interpolate_pos_encoding=True,
                                      output_hidden_states=True)
                final = out.hidden_states[24][:, 1:, :]
                spatial = final.permute(0, 2, 1).reshape(1, -1, grid, grid)
                logits, _ = model.decoder(spatial)
                summary_token, _ = model.model(tensor)
                predictor_scores.append(float(model.predictor(summary_token).sigmoid()))
            decoder_maps.append(F.avg_pool2d(logits.sigmoid().float(), 5, stride=1,
                                             padding=2)[0, 0].cpu())
            for layer in LAYERS:
                tokens = F.normalize(out.hidden_states[layer][0, 1:, :].float(), dim=-1)
                features[layer][index] = (tokens @ projection).half()
            if (index + 1) % 50 == 0:
                print(f"extracted {index + 1}/{len(batch)}", flush=True)
    del model
    torch.cuda.empty_cache()

    mutual = mutual_scores(features, grid, args.fraction, args.chunk).cpu()
    del features
    torch.cuda.empty_cache()
    return decoder_maps, predictor_scores, mutual


def evaluate(args: argparse.Namespace) -> dict:
    if (args.output / "report.json").exists():
        raise ValueError("Output already has a report; use a new directory")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    summary, image_rows = {}, []
    for category in manifest["categories"]:
        rows = [r for r in manifest["samples"] if r["class"] == category]
        normals = [r for r in rows if r["label"] == "normal"]
        abnormals = [r for r in rows if r["label"] != "normal"]
        if args.normal_limit:
            normals = normals[:args.normal_limit]
        if args.abnormal_limit:
            abnormals = abnormals[:args.abnormal_limit]
        batch = normals + abnormals

        cache = args.output / f"{category}_maps.pt"
        if cache.exists():
            saved = torch.load(cache)
            decoder_maps, predictor_scores, mutual = (
                list(saved["decoder"]), saved["predictor"], saved["mutual"])
            print(f"{category}: resumed maps from {cache.name}", flush=True)
        else:
            decoder_maps, predictor_scores, mutual = score_class(args, batch, device)
            torch.save({"decoder": torch.stack(decoder_maps), "predictor": predictor_scores,
                        "mutual": mutual}, cache)
        decoder = torch.stack(decoder_maps)
        mutual_up = F.interpolate(mutual[:, None], size=decoder.shape[-2:], mode="bilinear",
                                  align_corners=False)[:, 0]
        mutual_up = F.avg_pool2d(mutual_up[:, None], 5, stride=1, padding=2)[:, 0]
        fused = 0.5 * standardize(decoder) + 0.5 * standardize(mutual_up)
        maps = {"baseline": decoder, "mutual": to_unit(mutual_up), "fusion": to_unit(fused)}

        accumulators = {v: PixelAccumulator() for v in VARIANTS}
        per_variant_scores = {v: [] for v in VARIANTS}
        for index, row in enumerate(batch):
            with Image.open(args.dataset / row["image"]) as opened:
                size = (opened.height, opened.width)
            if row["label"] == "normal":
                truth = np.zeros(size, bool)
            else:
                with Image.open(args.dataset / row["mask"]) as opened:
                    truth = np.asarray(opened.convert("L")) > 0
            for variant in VARIANTS:
                full = F.interpolate(maps[variant][index][None, None], size=size,
                                     mode="bilinear", align_corners=False)[0, 0].numpy()
                full = np.clip(full, 0, 1)
                accumulators[variant].add(full, truth, row["label"] != "normal")
                score = float(full.max())
                per_variant_scores[variant].append(score)
                image_rows.append({"class": category, "variant": variant, "image": row["image"],
                                   "label": row["label"], "max_score": score,
                                   "predictor_score": predictor_scores[index]})
        labels = np.array([r["label"] != "normal" for r in batch])
        summary[category] = {}
        for variant in VARIANTS:
            scores = np.array(per_variant_scores[variant])
            metric = accumulators[variant].summary()
            summary[category][variant] = {
                "pixel_auroc_1001_bins": metric["pixel_auroc_1001_bins"],
                "pixel_ap_1001_bins": metric["pixel_ap_1001_bins"],
                "image_auroc_max_map": image_auc(scores[labels], scores[~labels]),
                "image_ap_max_map": image_ap(scores[labels], scores[~labels]),
            }
        predictor = np.array(predictor_scores)
        summary[category]["baseline"]["image_auroc_predictor"] = image_auc(
            predictor[labels], predictor[~labels])
        summary[category]["counts"] = {"normal": int((~labels).sum()), "abnormal": int(labels.sum())}
        print(json.dumps({category: summary[category]}, indent=2), flush=True)

    report = {"dataset_manifest_sha256": sha256(manifest_path), "image_size": 672,
              "layers": LAYERS, "radii": RADII, "projected_dim": PROJECTED_DIM,
              "interval_fraction": args.fraction, "setting": "transductive zero-shot: "
              "unlabeled batch of one class, no normal labels, no masks, no threshold",
              "fusion": "0.5*robust_z(decoder) + 0.5*robust_z(mutual), batch statistics",
              "normal_limit": args.normal_limit, "abnormal_limit": args.abnormal_limit,
              "seconds": time.perf_counter() - started, "by_class": summary,
              "per_image": image_rows}
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--normal-limit", type=int, default=0, help="0 = all")
    parser.add_argument("--abnormal-limit", type=int, default=0, help="0 = all")
    parser.add_argument("--fraction", type=float, default=0.3,
                        help="Share of smallest per-image distances averaged (MuSc interval)")
    parser.add_argument("--chunk", type=int, default=8)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
