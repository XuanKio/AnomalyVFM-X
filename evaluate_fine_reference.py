"""Compare the failed single-image pilot with matched-reference fine-defect detection.

The detector never receives the generator edit box. Synthetic edits and their
coarse edit boxes are descriptive evidence, not independent pixel ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image

from compare_reference import control
from demo_reference import ReferenceMismatch, detect_changes
from synthetic_mask_filter import hysteresis_mask


CONTROLS = [
    ("identity", {}),
    ("brightness", {"gain": 1.06, "offset": 2}),
    ("translation", {"shift": (-2.4, 1.8)}),
    ("jpeg85", {"jpeg": 85}),
    ("rotation_photo", {"gain": .97, "offset": -2, "shift": (-2.1, 1.4),
                         "angle": -.3, "jpeg": 95}),
    ("sensor_noise", {"gain": 1.02, "noise": 1.5, "seed": 20260929}),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))
    colored = rgb.copy()
    colored[mask] = (.25 * rgb[mask] + .75 * np.array([255, 40, 100])).astype(np.uint8)
    return Image.fromarray(colored)


def run(source: Path, pilot: Path, specification: Path, output: Path) -> dict:
    rows = json.loads(specification.read_text(encoding="utf-8"))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory to preserve earlier results")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    html_rows = []
    for row in rows:
        name = row["name"]
        folder = source / name
        good_path, bad_path = folder / "good.png", folder / "bad.png"
        with Image.open(good_path) as opened:
            good = opened.convert("RGB")
        with Image.open(bad_path) as opened:
            bad = opened.convert("RGB")
        if good.size != bad.size:
            raise ValueError(f"Unaligned image sizes: {name}")
        checks = []
        for control_name, settings in CONTROLS:
            try:
                checked = detect_changes(good, control(good, **settings))
                grown, _ = hysteresis_mask(checked["scores"], checked["valid_mask"])
                checks.append({"name": control_name, "status": "compared",
                               "mask_pixels": int(grown.sum())})
            except ReferenceMismatch as error:
                checks.append({"name": control_name, "status": "abstained",
                               "reason": str(error)})
        try:
            detected = detect_changes(good, bad)
        except ReferenceMismatch as error:
            results.append({"name": name, "status": "abstained", "reason": str(error),
                            "healthy_controls": checks})
            continue
        grown, growth = hysteresis_mask(detected["scores"], detected["valid_mask"])
        x0, y0, x1, y1 = row["edit_support_xyxy"]
        support = np.zeros(grown.shape, dtype=bool)
        support[y0:y1, x0:x1] = True
        core = detected["mask"]
        overlay(bad, core).save(output / f"{name}_core.png")
        overlay(bad, grown).save(output / f"{name}_grown.png")
        Image.fromarray(grown.astype(np.uint8) * 255).save(output / f"{name}_mask.png")
        baseline = pilot / f"{name}_bad_adapted_overlay.png"
        if baseline.is_file():
            with Image.open(baseline) as opened:
                if opened.size != bad.size:
                    raise ValueError(f"Pilot image size mismatch: {name}")
            pilot_view = f'<img src="{html.escape(baseline.resolve().as_uri())}">'
        else:
            pilot_view = "Không có ảnh pilot"
        results.append({
            "name": name, "status": "compared", "synthetic_pair": True,
            "good_sha256": sha256(good_path), "bad_sha256": sha256(bad_path),
            "core_pixels": int(core.sum()), "grown_pixels": int(grown.sum()),
            "grown_inside_generator_edit_box": int((grown & support).sum()),
            "grown_outside_generator_edit_box": int((grown & ~support).sum()),
            "alignment_correlation": detected["metadata"]["alignment_correlation"],
            "growth": growth, "healthy_controls": checks,
            "mask_review_status": row["review_status"],
        })
        healthy_alarms = sum(c.get("mask_pixels", 0) > 0 for c in checks)
        title = html.escape(name)
        html_rows.append(
            f'<tr><th>{title}<br>{healthy_alarms}/{len(checks)} ảnh lành báo nhầm</th>'
            f'<td>{pilot_view}</td>'
            f'<td><img src="{title}_core.png"><small>{int(core.sum())} px</small></td>'
            f'<td><img src="{title}_grown.png"><small>{int(grown.sum())} px</small></td></tr>'
        )
        print(f"{name}: pilot -> core {int(core.sum())} px -> grown {int(grown.sum())} px; "
              f"healthy alarms {healthy_alarms}/{len(checks)}", flush=True)
    report = {
        "method": "matched_healthy_reference_plus_global_connected_hysteresis",
        "requires_matched_healthy_reference": True,
        "detector_receives_generator_edit_box": False,
        "pixel_ground_truth": False,
        "scope": "Four AI-generated pairs; not a benchmark or a replacement zero-shot model",
        "growth_rule": "strong residual >=3, seed area >=12, connected weak residual; no edit ROI",
        "controls": [name for name, _ in CONTROLS],
        "specification_sha256": sha256(specification), "results": results,
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    page = '''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Vết lỗi nhỏ: ảnh lành đối chiếu</title>
<style>body{font:16px/1.5 system-ui;background:#101827;color:#e7edf8;margin:24px}h1{font-size:26px}
table{border-collapse:collapse}th,td{border:1px solid #567;padding:10px;vertical-align:top}img{width:min(22vw,440px);display:block}
small{display:block}a{color:#9cf}.note{background:#493920;padding:12px}</style>
<h1>Vết lỗi nhỏ: đầu mask thử nghiệm so với ảnh lành đối chiếu</h1>
<p class="note">Cách đối chiếu <strong>cần một ảnh lành của chính thiết bị, cùng góc chụp</strong>. Đây là 4 cặp ảnh AI, chưa chứng minh hiệu quả trên thiết bị thật. Vùng hồng là pixel khác biệt, không phải kết luận lỗi chức năng. Bản mở rộng tìm toàn ảnh; ô chỉnh sửa của generator chỉ dùng để báo cáo.</p>
<table><tr><th>Mẫu</th><th>Đầu mask thử nghiệm</th><th>Đối chiếu: vùng mạnh</th><th>Đối chiếu: mở rộng vết mảnh</th></tr>ROWS</table>
<p>"Ảnh lành báo nhầm" tính trên ảnh gốc và 5 biến đổi sáng, dịch nhẹ, JPEG, xoay nhẹ, nhiễu; không phải tỷ lệ lỗi ngoài thực tế. Mask kim loại mới vẫn có thể thiếu phần đầu/đuôi vết xước. <a href="report.json">Báo cáo đầy đủ</a>.</p></html>'''
    (output / "comparison.html").write_text(page.replace("ROWS", "".join(html_rows)), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--specification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.pilot, args.specification, args.output)


if __name__ == "__main__":
    main()
