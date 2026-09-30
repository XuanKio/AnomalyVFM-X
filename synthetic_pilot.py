"""Build a small, auditable auxiliary dataset from reviewed generated pairs.

Candidate masks are weak labels. A reviewer must explicitly accept each pair.
This script deliberately does not silently turn a difference map into ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(source: Path, destination: Path, specification: Path,
          mask_kind: str = "candidate") -> dict:
    if mask_kind not in {"candidate", "refined"}:
        raise ValueError("mask_kind must be candidate or refined")
    if destination.resolve() == source.resolve():
        raise ValueError("Dataset output must differ from source")
    rows = json.loads(specification.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Specification must contain a nonempty list")
    seen = set()
    report = []
    for row in rows:
        name = row["name"]
        if not isinstance(name, str) or not name.replace("_", "").isalnum() or name in seen:
            raise ValueError(f"Invalid or duplicate category: {name}")
        seen.add(name)
        folder = source / name
        good_path, bad_path = folder / "good.png", folder / "bad.png"
        mask_path = folder / f"{mask_kind}_mask.png"
        with Image.open(good_path) as good_image, Image.open(bad_path) as bad_image, Image.open(mask_path) as mask_image:
            good = good_image.convert("RGB")
            bad = bad_image.convert("RGB")
            mask = np.asarray(mask_image.convert("L")) >= 128
        if good.size != bad.size or mask.shape != (good.height, good.width):
            raise ValueError(f"Misaligned dimensions: {name}")
        x0, y0, x1, y1 = row["edit_support_xyxy"]
        if not (0 <= x0 < x1 <= good.width and 0 <= y0 < y1 <= good.height):
            raise ValueError(f"Invalid edit support: {name}")
        support = np.zeros_like(mask)
        support[y0:y1, x0:x1] = True
        if mask.sum() == 0 or np.any(mask & ~support):
            raise ValueError(f"Missing or off-target candidate mask: {name}")
        accepted = row["review_status"] == "accepted_weak_label"
        if row["review_status"] not in {"accepted_weak_label", "rejected_incomplete_mask"}:
            raise ValueError(f"Review decision required: {name}")
        if accepted:
            for part in ("train/ok", "train/bad", "ground_truth/bad"):
                (destination / part).mkdir(parents=True, exist_ok=True)
            # Matching basenames and paths are required by AuxilaryDataset.
            good.save(destination / "train/ok" / f"{name}.png")
            bad.save(destination / "train/bad" / f"{name}.png")
            Image.fromarray(mask.astype(np.uint8) * 255).save(
                destination / "ground_truth/bad" / f"{name}.png"
            )
        report.append({
            **row,
            "size_wh": list(good.size),
            "mask_pixels": int(mask.sum()),
            "accepted_for_pilot_training": accepted,
            "sha256": {"good": sha256(good_path), "bad": sha256(bad_path),
                       "mask": sha256(mask_path)},
        })
    manifest = {
        "label_type": "weak_aligned_difference_reviewed_not_ground_truth",
        "generator": "OpenAI built-in imagegen image edit; see specification prompts",
        "filter": ("aligned_reference_residual_v1 plus visual review" if mask_kind == "candidate"
                   else "aligned_reference_residual_v1 plus adaptive connected hysteresis and visual review"),
        "mask_kind": mask_kind,
        "accepted_pairs": sum(row["accepted_for_pilot_training"] for row in report),
        "rejected_pairs": sum(not row["accepted_for_pilot_training"] for row in report),
        "pairs": report,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--specification", type=Path, required=True)
    parser.add_argument("--mask-kind", choices=("candidate", "refined"), default="candidate")
    args = parser.parse_args()
    manifest = build(args.source, args.output, args.specification, args.mask_kind)
    print(json.dumps({key: manifest[key] for key in ("accepted_pairs", "rejected_pairs")}, indent=2))


if __name__ == "__main__":
    main()
