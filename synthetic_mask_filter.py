"""Experimental two-threshold synthetic-label filter.

Use high-confidence changed pixels as seeds, then grow through connected
lower-confidence pixels inside the generator's edit support. It is a candidate
mask only: the generated edit and mask still require visual review.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def hysteresis_mask(scores: np.ndarray, support: np.ndarray, *, strong: float = 3.0,
                    min_seed_area: int = 12) -> tuple[np.ndarray, dict]:
    """Grow accepted strong components along connected, weaker evidence."""
    scores = np.asarray(scores, dtype=np.float32)
    support = np.asarray(support, dtype=bool)
    if scores.ndim != 2 or scores.shape != support.shape:
        raise ValueError("scores and support must be aligned 2-D maps")
    if not np.isfinite(scores).all() or not np.isfinite(strong) or strong <= 0:
        raise ValueError("scores and strong threshold must be finite; strong > 0")
    if min_seed_area < 1:
        raise ValueError("min_seed_area must be positive")
    seeds = (scores >= strong) & support
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        seeds.astype(np.uint8), connectivity=8)
    strong_ids = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_seed_area]
    accepted_seeds = np.isin(components, strong_ids)
    if not strong_ids:
        return np.zeros_like(support), {"weak_threshold": None, "seed_pixels": 0,
                                            "mask_pixels": 0, "reason": "no_strong_seed"}
    # Estimate the local non-seed tail inside R rather than selecting a lower
    # threshold separately for each image by looking at its target defect.
    nonseed = scores[support & ~seeds]
    local_tail = float(np.quantile(nonseed, .95)) if nonseed.size else 0.0
    weak = min(strong * .95, max(strong * .25, local_tail * 1.2))
    candidates = ((scores >= weak) & support).astype(np.uint8)
    _, regions = cv2.connectedComponents(candidates, connectivity=8)
    kept = np.unique(regions[accepted_seeds])
    kept = kept[kept > 0]
    mask = np.isin(regions, kept) & support
    return mask, {"weak_threshold": weak, "local_nonseed_q95": local_tail,
                  "seed_pixels": int(accepted_seeds.sum()),
                  "mask_pixels": int(mask.sum()), "reason": "grown_from_strong_seed"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--specification", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.specification.read_text(encoding="utf-8"))
    report = []
    for row in rows:
        folder = args.source / row["name"]
        with np.load(folder / "candidate_maps.npz") as data:
            scores = data["scores"]
            valid = data["valid"].astype(bool)
        x0, y0, x1, y1 = row["edit_support_xyxy"]
        support = np.zeros(scores.shape, dtype=bool)
        support[y0:y1, x0:x1] = True
        mask, metadata = hysteresis_mask(scores, support & valid)
        Image.fromarray(mask.astype(np.uint8) * 255).save(folder / "refined_mask.png")
        with Image.open(folder / "bad.png") as image:
            original = np.asarray(image.convert("RGB"), dtype=np.uint8)
        overlay = original.copy()
        overlay[mask] = (.25 * original[mask] + .75 * np.array([255, 40, 100])).astype(np.uint8)
        Image.fromarray(overlay).save(folder / "refined_overlay.png")
        report.append({"name": row["name"], **metadata,
                       "review_status_unchanged": row["review_status"]})
    (args.source / "refined_filter_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
