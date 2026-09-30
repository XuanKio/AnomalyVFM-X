"""Audit whether a pilot head improves real anomaly ranking or just calibration.

Uses saved per-image maximum pixel scores; no model inference or threshold
search on anomalous images. The threshold is fit on candle normals only and
then applied unchanged to capsules as a cross-category check.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def image_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    return float(((positive[:, None] > negative).sum() +
                  .5 * (positive[:, None] == negative).sum()) /
                 (len(positive) * len(negative)))


def image_ap(positive: np.ndarray, negative: np.ndarray) -> float:
    # Scores are continuous in the current report. Group equal scores to make
    # the metric deterministic should quantized scores be used in the future.
    score = np.r_[positive, negative]
    truth = np.r_[np.ones(len(positive), bool), np.zeros(len(negative), bool)]
    order = np.argsort(-score, kind="stable")
    score, truth = score[order], truth[order]
    ends = np.r_[np.flatnonzero(score[1:] != score[:-1]), len(score) - 1]
    true_pos = np.cumsum(truth)[ends]
    precision = true_pos / (ends + 1)
    recall = true_pos / len(positive)
    return float(np.sum(np.diff(np.r_[0, recall]) * precision))


def analyze(report: dict, calibration_class: str = "candle",
            allowed_healthy_alarms: int = 5) -> dict:
    rows = report["per_image"]
    categories = sorted({row["class"] for row in rows})
    variants = sorted({row["variant"] for row in rows})
    if calibration_class not in categories:
        raise ValueError("Calibration class is absent")
    result = {}
    for variant in variants:
        controls = sorted(float(row["max_score"]) for row in rows
                          if row["class"] == calibration_class and
                          row["variant"] == variant and row["label"] == "normal")
        if not 0 <= allowed_healthy_alarms < len(controls):
            raise ValueError("Invalid healthy alarm budget")
        threshold = float(np.nextafter(controls[-allowed_healthy_alarms - 1], np.inf))
        by_class = {}
        for category in categories:
            positive = np.array([row["max_score"] for row in rows
                                 if row["class"] == category and row["variant"] == variant
                                 and row["label"] != "normal"])
            negative = np.array([row["max_score"] for row in rows
                                 if row["class"] == category and row["variant"] == variant
                                 and row["label"] == "normal"])
            if not len(positive) or not len(negative):
                raise ValueError(f"Both normal and anomalous images required: {category}")
            by_class[category] = {
                "image_auroc_exact": image_auc(positive, negative),
                "image_ap_exact": image_ap(positive, negative),
                "healthy_alarm_rate_at_candle_threshold": float((negative >= threshold).mean()),
                "anomalous_detection_rate_at_candle_threshold": float((positive >= threshold).mean()),
                "healthy_count": len(negative), "anomalous_count": len(positive),
            }
        result[variant] = {"threshold_from_candle_normals": threshold,
                           "by_class": by_class}
    return {"source_dataset_manifest_sha256": report["dataset_manifest_sha256"],
            "source_head_sha256": report["head_sha256"],
            "score": "max of the smoothed anomaly mask", "calibration_class": calibration_class,
            "allowed_healthy_alarms_in_calibration": allowed_healthy_alarms,
            "note": "Only two VisA classes; cross-category check is not all-device validation",
            "variants": result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = analyze(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["variants"], indent=2))


if __name__ == "__main__":
    main()
