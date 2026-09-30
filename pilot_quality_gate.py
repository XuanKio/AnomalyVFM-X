"""Reject pilot models that gain detections by increasing healthy false alarms.

These are necessary local checks, not a substitute for independent real
pixel-labelled validation across product classes.
"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path


def evaluate(report: dict) -> dict:
    pairs = {(row["image"], row["variant"]): row for row in report["samples"]}
    problems = []
    healthy = sorted(name for name, variant in pairs if variant == "baseline" and name.endswith("_good"))
    for name in healthy:
        old = pairs[(name, "baseline")]["mask_pixels_at_0_5"]
        new = pairs[(name, "adapted")]["mask_pixels_at_0_5"]
        if new > old:
            problems.append(f"{name}: ảnh lành tăng vùng báo lỗi {old} → {new} pixel")
    for name in report["weak_train_pairs"]:
        row = pairs[(f"{name}_bad", "adapted")]
        if "weak_label_recall" not in row:
            problems.append(f"{name}_bad: báo cáo cũ thiếu chỉ số trùng mask nhãn yếu")
        elif row["weak_label_recall"] == 0:
            problems.append(f"{name}_bad: mask không trùng nhãn yếu của vết đã sinh")
    for name in report["heldout_rejected_pairs"]:
        row = pairs[(f"{name}_bad", "adapted")]
        if row["mask_pixels_at_0_5"] == 0:
            problems.append(f"{name}_bad: vẫn chưa hiện vết lỗi trên mẫu không dùng để học")
    return {
        "local_checks_passed": not problems,
        "eligible_for_release": False,
        "release_blocker": "Chưa có tập ảnh thật với mask chuẩn độc lập trên nhiều loại thiết bị.",
        "problems": problems,
        "healthy_samples_checked": healthy,
        "threshold": report["threshold"],
    }


def write_gate(experiment: Path, report: dict) -> dict:
    result = evaluate(report)
    (experiment / "gate.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    page = experiment / "comparison.html"
    if page.exists():
        text = page.read_text(encoding="utf-8")
        title = "KHÔNG ĐẠT kiểm tra cục bộ" if result["problems"] else "CHƯA ĐỦ BẰNG CHỨNG để sử dụng"
        items = "".join(f"<li>{html.escape(problem)}</li>" for problem in result["problems"][:5])
        if len(result["problems"]) > 5:
            items += f"<li>Còn {len(result['problems']) - 5} vấn đề trong gate.json.</li>"
        banner = ('<section id="pilot-gate" style="border:2px solid #f66;padding:16px;'
                  'background:#422;margin:20px 0"><strong>' + title + '</strong><ul>'
                  + items + '</ul><p>' + html.escape(result["release_blocker"]) + '</p></section>')
        if 'id="pilot-gate"' in text:
            text = re.sub(r'<section id="pilot-gate".*?</section>', banner, text, count=1, flags=re.S)
        else:
            text = text.replace("<h1>", banner + "<h1>", 1)
        page.write_text(text, encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()
    report = json.loads((args.experiment / "report.json").read_text(encoding="utf-8"))
    result = write_gate(args.experiment, report)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
