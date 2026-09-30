"""Reproducible reference-assisted scratch demonstration with nominal controls."""
from __future__ import annotations

if __name__ == "__main__":
    from demo_bootstrap import launch_web_runtime
    launch_web_runtime(__file__)

import argparse
import hashlib
import html
import io
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from demo_reference import ReferenceConfig, ReferenceMismatch, detect_changes, render_changes


def control(image, *, gain=1., offset=0., shift=(0., 0.), angle=0., jpeg=None, noise=0., seed=0):
    """Photographic nuisance controls from the nominal reference; not real data."""
    rgb = np.array(image)
    h, w = rgb.shape[:2]
    warp = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.)
    warp[:, 2] += shift
    rgb = cv2.warpAffine(rgb, warp, (w, h), borderMode=cv2.BORDER_REFLECT_101).astype(np.float32)
    rgb = rgb * gain + offset
    if noise:
        rgb += np.random.default_rng(seed).normal(0, noise, rgb.shape)
    result = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    if jpeg:
        buffer = io.BytesIO()
        result.save(buffer, format="JPEG", quality=jpeg)
        with Image.open(buffer) as encoded:
            result = encoded.convert("RGB")
    return result


def _font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def figure(image, before, after, box, output, count, has_baseline):
    canvas = Image.new("RGB", (1560, 900), "#101827")
    draw = ImageDraw.Draw(canvas)
    labels = ["Ảnh cần kiểm tra", "Trước: CLIP" if has_baseline else "Chưa cung cấp kết quả CLIP", f"Mới: {count} vùng khác biệt"]
    if box:
        zoom = (max(0, box[0] - 45), max(0, box[1] - 45),
                min(image.width, box[2] + 45), min(image.height, box[3] + 45))
    else:
        zoom = (0, 0, image.width, image.height)
    for column, panel in enumerate([image, before, after]):
        for row in range(2):
            picture = panel.copy() if row == 0 else panel.crop(zoom)
            picture.thumbnail((500, 342), Image.Resampling.LANCZOS)
            x, y = column * 520, row * 402
            draw.text((x + 12, y + 14), labels[column], fill="white", font=_font(23))
            canvas.paste(picture, (x + (520 - picture.width) // 2, y + 52 + (342 - picture.height) // 2))
    draw.text((16, 815), "Hồng: pixel được phát hiện. Khung vàng: vùng thuật toán tự khoanh, không phải ô chỉ dẫn.", fill="#c7d2e5", font=_font(21))
    draw.text((16, 846), "Một cặp ảnh. Bản mới có thêm ảnh lành tham chiếu; không phải kết quả zero-shot.", fill="#c7d2e5", font=_font(21))
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--synthetic-pair", action="store_true", help="Declare that the supplied pair is AI-generated")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new empty output directory to preserve previous experiments")
    config = ReferenceConfig()
    baseline = None
    source_hash = hashlib.sha256(args.image.read_bytes()).hexdigest()
    if args.baseline_dir:
        baseline = json.loads((args.baseline_dir / "report.json").read_text(encoding="utf-8"))
        if baseline["source_sha256"] != source_hash:
            parser.error("Baseline must refer to exactly the same inspected source image")
    with Image.open(args.reference) as opened:
        reference = ImageOps.exif_transpose(opened).convert("RGB")
    with Image.open(args.image) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    output.mkdir(parents=True, exist_ok=True)
    # These settings and control definitions are persisted before target inference.
    controls = [
        ("identity", {}),
        ("brightness", {"gain":1.06,"offset":2}),
        ("translation", {"shift":(-2.4,1.8)}),
        ("jpeg85", {"jpeg":85}),
        ("rotation_photo", {"gain":.97,"offset":-2,"shift":(-2.1,1.4),"angle":-.3,"jpeg":95}),
        ("sensor_noise", {"gain":1.02,"noise":1.5,"seed":20260929}),
    ]
    from dataclasses import asdict
    manifest = {"created_utc":datetime.now(timezone.utc).isoformat(),
                "configuration":asdict(config),"nominal_controls":controls,
                "reference_sha256":hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                "source_sha256":source_hash,
                "code_sha256":{p:hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest()
                               for p in ("demo_reference.py","compare_reference.py")},
                "scope":"Exploratory n=1 reference-assisted change detection; no pixel ground truth; not a benchmark",
                "synthetic_pair":args.synthetic_pair,
                "threshold_selection":"Default threshold 3, persisted before target inference. Nominal controls checked; no target ROI supplied to detector."}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    reference.save(output / "reference.png")
    image.save(output / "input.png")
    checks = []
    for name, settings in controls:
        print("Nominal control:", name, flush=True)
        try:
            checked = detect_changes(reference, control(reference, **settings), config)
            checks.append({"name":name,"status":"compared","regions":len(checked["metadata"]["regions"]),
                           "maximum_residual":checked["metadata"]["maximum_residual"]})
        except ReferenceMismatch as exc:
            checks.append({"name":name,"status":"abstained","reason":str(exc)})
    # Independent nuisance reference with the same color histogram, wrong geometry.
    try:
        checked = detect_changes(ImageOps.mirror(reference), image, config)
        checks.append({"name":"wrong_reference_mirrored","status":"unexpectedly_compared",
                       "regions":len(checked["metadata"]["regions"])})
    except ReferenceMismatch as exc:
        checks.append({"name":"wrong_reference_mirrored","status":"correctly_rejected","reason":str(exc)})
    runs = []
    for index in range(3):
        print(f"Target run {index+1}/3", flush=True)
        result = detect_changes(reference, image, config)
        runs.append(result["metadata"]["seconds"])
    panels = render_changes(result)
    for key,panel in panels.items():
        panel.save(output / f"new_{key}.png")
    np.savez_compressed(output / "maps.npz",scores=result["scores"],mask=result["mask"],valid=result["valid_mask"])
    before = image.copy()
    box = None
    old_info = None
    if baseline:
        box = baseline["configuration"].get("reference_box_xyxy")
        with Image.open(args.baseline_dir / "baseline_mask.png") as saved:
            if saved.size != image.size:
                raise ValueError("Baseline mask size mismatch")
            old_mask = np.array(saved.convert("L")) > 0
        canvas = np.array(before)
        canvas[old_mask] = (canvas[old_mask] * .25 + np.array([255,40,100]) * .75).astype(np.uint8)
        before = Image.fromarray(canvas)
        old_info = {"scope":"Reused saved baseline on identical source; timings from previous session",
                    "baseline_mask_pixels":int(old_mask.sum()),"tiling_mask_area_percent":baseline["map_statistics"]["fused"]["mask_area_percent"],
                    "timings":baseline["timing_medians"],"source_report":str((args.baseline_dir/'report.json').resolve())}
    before.save(output / "before_overlay.png")
    figure(image, before, panels["overlay"], box, output / "comparison.png",len(result["metadata"]["regions"]),old_info is not None)
    report = {**manifest, "detection":result["metadata"], "nominal_control_results":checks,
              "target_run_seconds":runs,"median_seconds":statistics.median(runs),"historical_baseline":old_info,
              "reference_box_reporting_only":box,
              "limitations":["AI-generated pair: generated edits may change other textures too." if args.synthetic_pair else "Source pair is user-supplied; provenance is not independently verified.",
                             "Controls are transformations of one nominal image, not independent real products.",
                             "No exact ground truth; no precision/recall/F1/accuracy claimed.",
                             "Only accepted connected components of >=12 pixels shown; borders excluded; broad changes may be suppressed."]}
    (output / "report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    rows = "".join(f'<tr><td>{html.escape(c["name"])}</td><td>{c["status"]}</td><td>{c.get("regions","—")}</td></tr>' for c in checks)
    regions = html.escape(json.dumps(result["metadata"]["regions"], ensure_ascii=False, indent=2))
    if old_info:
        before_summary = (f'Kết quả CLIP đã lưu: {old_info["baseline_mask_pixels"]} pixel được khoanh; '
                          f'chia ô: {old_info["tiling_mask_area_percent"]:.4f}% diện tích mask. '
                          'Cấu hình và ngưỡng của lần chạy trước nằm trong báo cáo nguồn.')
    else:
        before_summary = 'Chưa cung cấp kết quả CLIP; cột giữa chỉ hiển thị ảnh đầu vào, chưa thể kết luận trước/sau.'
    provenance = 'Cặp ảnh AI n=1' if args.synthetic_pair else 'Một cặp ảnh đầu vào'
    extra_ai_note = 'Ảnh AI chỉnh sửa có thể thay đổi thêm một số chi tiết khác.' if args.synthetic_pair else ''
    page = f'''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Đối chiếu ảnh lành · vùng khác biệt</title><style>
body{{margin:0;background:#101827;color:#e7edf8;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1380px;margin:auto;padding:28px}}h1{{font-size:30px}}
img{{max-width:100%;display:block}}.note{{background:#233449;padding:18px;border-left:4px solid #75dfb0}}table{{border-collapse:collapse;width:100%}}
td,th{{text-align:left;padding:10px;border-bottom:1px solid #354357}}a{{color:#86ccff}}pre{{white-space:pre-wrap;background:#182335;padding:16px}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}@media(max-width:700px){{.pair{{grid-template-columns:1fr}}}}
</style><main><h1>Đối chiếu ảnh lành: {len(result['metadata']['regions'])} vùng khác biệt được khoanh tự động</h1>
<p class="note">Thuật toán tìm trên toàn bộ vùng ảnh khớp, không nhận tọa độ vết xước. Hồng là pixel dự đoán; vàng là khung tự động.
Đây là nhánh thị giác máy tính có thêm ảnh lành tham chiếu, không phải CLIP được huấn luyện lại. {provenance} chưa chứng minh độ chính xác thực tế.</p>
<img src="comparison.png" alt="Ảnh đầu vào, kết quả CLIP nếu được cung cấp và kết quả đối chiếu ảnh lành">
<p>{before_summary} Nhánh mới dùng ngưỡng khác biệt 3,0 (đơn vị riêng, chưa hiệu chỉnh bằng ảnh thực).
Không so trực tiếp giá trị score giữa hai phương pháp. Thời gian trung vị nhánh mới: {statistics.median(runs):.3f} giây trên CPU.</p>
<div class="pair"><div><h2>Ảnh lành cung cấp thêm</h2><img src="reference.png" alt="Ảnh lành tham chiếu"></div>
<div><h2>Mask mới</h2><img src="new_mask.png" alt="Mask khác biệt do thuật toán tạo"></div></div>
<h2>Tất cả vùng được trả về</h2><pre>{regions}</pre><h2>Kiểm tra đối chứng</h2>
<p>Các ảnh lành được biến đổi ánh sáng, vị trí, góc và chất lượng JPEG. Đây là kiểm tra nhiễu tổng hợp, chưa phải tỷ lệ báo nhầm trên dữ liệu thực.</p>
<table><tr><th>Ca</th><th>Trạng thái</th><th>Số vùng</th></tr>{rows}</table>
<p>Giới hạn: cần đúng loại/góc chụp; bỏ qua mép không khớp; thay đổi rộng hoặc lỗi trên cấu trúc nhiều chi tiết có thể bị bỏ sót.
{extra_ai_note} Chưa có nhãn pixel chuẩn để tính F1.</p>
<p><a href="report.json">Báo cáo số liệu</a> · <a href="new_overlay.png">Ảnh khoanh vùng đầy đủ</a> · <a href="maps.npz">Bản đồ float và mask</a></p></main></html>'''
    (output / "comparison.html").write_text(page,encoding="utf-8")
    print(json.dumps({"output":str(output),"regions":result["metadata"]["regions"],"controls":checks,"median_seconds":statistics.median(runs)},indent=2))


if __name__ == "__main__":
    main()
