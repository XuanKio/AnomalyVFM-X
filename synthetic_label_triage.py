"""Triage released synthetic masks against an independent image-change signal.

The output is a review queue and candidate positive/ignore masks. Residuals
are not certified ground truth; this tool never silently relabels training data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from demo_reference import ReferenceMismatch, detect_changes
from synthetic_mask_filter import hysteresis_mask


def triage_label(official: np.ndarray, scores: np.ndarray,
                 valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    official = np.asarray(official, bool)
    if scores.shape != official.shape or valid.shape != official.shape or not official.any():
        raise ValueError("Require aligned nonempty official mask, residual and valid map")
    grown, growth = hysteresis_mask(scores, valid)
    count, components = cv2.connectedComponents(grown.astype(np.uint8), connectivity=8)
    near_label = cv2.dilate(official.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    linked_ids = np.unique(components[near_label])
    linked_ids = linked_ids[linked_ids > 0]
    linked = np.isin(components, linked_ids) if count > 1 else np.zeros_like(official)
    positive = official | linked
    uncertain = grown & ~positive
    residual_count = int(grown.sum())
    official_count = int(official.sum())
    positive_count = int(positive.sum())
    reasons = []
    if official_count < 25:
        reasons.append("tiny_official_mask_review")
    if residual_count == 0:
        reasons.append("no_independent_residual_review")
    if residual_count >= 1000 and int(uncertain.sum()) / residual_count > .8:
        reasons.append("most_residual_outside_label_review")
    if positive_count > max(5000, official_count * 10):
        reasons.append("large_mask_expansion_review")
    return positive, uncertain, {
        "official_pixels": official_count, "candidate_positive_pixels": positive_count,
        "uncertain_pixels": int(uncertain.sum()), "residual_pixels": residual_count,
        "linked_residual_pixels": int(linked.sum()), "growth": growth,
        "review_reasons": reasons, "high_risk": bool(reasons), "requires_review": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Local sample with train/ok, train/bad, ground_truth/bad")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    html_rows = []
    for good_path in sorted((args.dataset / "train/ok").glob("*.png")):
        number = good_path.stem
        bad_path = args.dataset / "train/bad" / f"{number}.png"
        mask_path = args.dataset / "ground_truth/bad" / f"{number}.png"
        with Image.open(good_path) as opened:
            good = opened.convert("RGB")
        with Image.open(bad_path) as opened:
            bad = opened.convert("RGB")
        with Image.open(mask_path) as opened:
            official = np.asarray(opened.convert("L")) > 0
        if good.size != bad.size or official.shape != (bad.height, bad.width):
            raise ValueError(f"Unaligned pair: {number}")
        try:
            detected = detect_changes(good, bad)
            positive, uncertain, info = triage_label(
                official, detected["scores"], detected["valid_mask"])
            info["alignment_correlation"] = detected["metadata"]["alignment_correlation"]
        except ReferenceMismatch as error:
            positive, uncertain = official, np.zeros_like(official)
            info = {"official_pixels": int(official.sum()),
                    "candidate_positive_pixels": int(official.sum()), "uncertain_pixels": 0,
                    "review_reasons": ["alignment_failed_review"], "high_risk": True,
                    "requires_review": True,
                    "reason": str(error)}
        Image.fromarray(positive.astype(np.uint8) * 255).save(args.output / f"{number}_candidate.png")
        Image.fromarray(uncertain.astype(np.uint8) * 255).save(args.output / f"{number}_ignore.png")
        rgb = np.asarray(bad).copy()
        rgb[uncertain] = (.4 * rgb[uncertain] + .6 * np.array([255, 180, 40])).astype(np.uint8)
        rgb[positive] = (.25 * rgb[positive] + .75 * np.array([255, 40, 100])).astype(np.uint8)
        Image.fromarray(rgb).save(args.output / f"{number}_overlay.jpg", quality=90)
        rows.append({"id": number, **info})
        reasons = ", ".join(info["review_reasons"]) or "visual_review_required"
        html_rows.append(f'<tr><th>{number}<br>{reasons}</th><td><img src="{bad_path.resolve().as_uri()}"></td>'
                         f'<td><img src="{number}_overlay.jpg"></td><td>{info["official_pixels"]} → '
                         f'{info["candidate_positive_pixels"]} positive; {info["uncertain_pixels"]} ignore</td></tr>')
        print(f"{number}: {reasons}", flush=True)
    report = {"dataset": str(args.dataset.resolve()), "rule": "linked residual grows official; "
              "unlinked residual is uncertain; review before training", "rows": rows,
              "pairs": len(rows), "requiring_review": sum(row["requires_review"] for row in rows),
              "high_risk": sum(row["high_risk"] for row in rows)}
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    page = '<!doctype html><html lang="vi"><meta charset="utf-8"><title>Kiểm tra nhãn tổng hợp</title><style>body{font:16px system-ui;background:#101827;color:white}table{border-collapse:collapse}td,th{border:1px solid #789;padding:8px}img{width:320px}a{color:#9cf}</style><h1>Hồng: nhãn ứng viên; vàng: vùng thay đổi chưa có nhãn</h1><p>Các vùng vàng cần người xem, không được tự động gọi là lỗi hoặc thêm vào training. Cặp bị đánh dấu cần kiểm duyệt trước khi dùng. <a href="report.json">Báo cáo</a></p><table><tr><th>ID / lý do</th><th>Ảnh lỗi</th><th>Kiểm tra chéo</th><th>Pixel</th></tr>' + ''.join(html_rows) + '</table></html>'
    (args.output / "comparison.html").write_text(page, encoding="utf-8")
    print(json.dumps({"pairs": report["pairs"], "requiring_review": report["requiring_review"],
                      "high_risk": report["high_risk"]}))


if __name__ == "__main__":
    main()
