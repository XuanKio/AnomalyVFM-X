"""Extract a deterministic, labelled VisA evaluation slice from a local tar prefix.

The archive remains in ignored outputs; no real benchmark images enter Git.
The prefix must include complete categories (images, masks and image_anno.csv).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image


SOURCE_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def prepare(prefix: Path, output: Path, categories: tuple[str, ...], normal_per_class: int) -> dict:
    if not categories or normal_per_class < 1:
        raise ValueError("Require categories and a positive normal sample count")
    metadata = {}
    with tarfile.open(prefix, mode="r:") as archive:
        while len(metadata) < len(categories):
            member = archive.next()
            if member is None:
                raise ValueError("Tar prefix ended before category metadata")
            for category in categories:
                if member.name == f"{category}/image_anno.csv":
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError(f"Unreadable metadata: {category}")
                    metadata[category] = list(csv.DictReader(io.StringIO(stream.read().decode("utf-8-sig"))))
    selected = []
    for category in categories:
        rows = metadata[category]
        healthy = sorted((row for row in rows if row["label"] == "normal"),
                         key=lambda row: row["image"])
        abnormal = sorted((row for row in rows if row["label"] != "normal"),
                          key=lambda row: row["image"])
        if len(healthy) < normal_per_class or not abnormal:
            raise ValueError(f"Insufficient labelled VisA samples in {category}")
        indices = np.linspace(0, len(healthy) - 1, normal_per_class, dtype=int)
        for row in [healthy[int(i)] for i in indices] + abnormal:
            if not row["image"].startswith(f"{category}/Data/Images/"):
                raise ValueError("Unexpected image path in official metadata")
            if row["label"] != "normal" and not row["mask"].startswith(f"{category}/Data/Masks/"):
                raise ValueError("Missing official pixel mask")
            selected.append({"class": category, "label": row["label"],
                             "image": row["image"], "mask": row["mask"] or None})
    needed = {row["image"] for row in selected}
    needed.update(row["mask"] for row in selected if row["mask"])
    found = set()
    with tarfile.open(prefix, mode="r:") as archive:
        while found != needed:
            try:
                member = archive.next()
            except tarfile.ReadError as error:
                raise ValueError(f"Tar prefix lacks {len(needed - found)} selected files") from error
            if member is None:
                raise ValueError(f"Tar prefix lacks {len(needed - found)} selected files")
            if member.name not in needed:
                continue
            # Paths are additionally whitelisted from official CSV and kept
            # beneath output; never call extractall on an external tar.
            target = output / member.name
            if not target.resolve().is_relative_to(output.resolve()):
                raise ValueError(f"Unsafe archive path: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Unreadable archive member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as destination:
                while part := stream.read(1024 * 1024):
                    destination.write(part)
            found.add(member.name)
    for row in selected:
        with Image.open(output / row["image"]) as image:
            row["size_wh"] = list(image.size)
        if row["mask"]:
            with Image.open(output / row["mask"]) as mask:
                if mask.size != tuple(row["size_wh"]):
                    raise ValueError(f"Image/mask dimensions differ: {row['image']}")
    manifest = {"source_url": SOURCE_URL, "source_archive": "VisA_20220922.tar",
                "local_prefix_bytes": prefix.stat().st_size,
                "local_prefix_sha256": sha256(prefix),
                "normal_per_class": normal_per_class,
                "categories": list(categories), "samples": selected}
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=["candle", "capsules"])
    parser.add_argument("--normal-per-class", type=int, default=100)
    args = parser.parse_args()
    result = prepare(args.prefix, args.output, tuple(args.categories), args.normal_per_class)
    print(json.dumps({"categories": result["categories"], "sample_count": len(result["samples"]),
                      "prefix_sha256": result["local_prefix_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
