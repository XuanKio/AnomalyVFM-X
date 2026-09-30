"""Reproducible single-image comparison; descriptive evidence, not a benchmark."""

from __future__ import annotations

if __name__ == "__main__":
    from demo_bootstrap import launch_web_runtime
    launch_web_runtime(__file__)

import argparse
import hashlib
import html
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from demo_fast import MODEL_CACHE_DIR, MODEL_ID, configure_resolution, load_demo_model
from demo_tiling import predict_comparison
from demo_visuals import render_prediction


def describe_map(values, box, threshold):
    """Box statistics are reporting-only; the rectangle is NOT pixel ground truth."""
    y, x = np.unravel_index(int(values.argmax()), values.shape)
    result = {
        "minimum": float(values.min()), "maximum": float(values.max()),
        "maximum_xy": [int(x), int(y)],
        "mask_area_percent": float(100 * (values >= threshold).mean()),
    }
    if box is not None:
        x0, y0, x1, y1 = box
        region = values[y0:y1, x0:x1]
        outside = np.ones(values.shape, dtype=bool)
        outside[y0:y1, x0:x1] = False
        result.update({
            "reference_box_max": float(region.max()),
            "reference_box_mean": float(region.mean()),
            "reference_box_mask_pixels": int((region >= threshold).sum()),
            "outside_box_mask_pixels": int((values[outside] >= threshold).sum()),
            "outside_box_max": float(values[outside].max()) if outside.any() else None,
            "maximum_inside_reference_box": bool(x0 <= x < x1 and y0 <= y < y1),
        })
    return result


def font(size):
    for candidate in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def comparison_figure(image, baseline, proposed, box, output):
    """Scientific result panels; source pixels are never changed for inference."""
    pictures = [image, baseline, proposed]
    labels = ["AI input / reference location", "BEFORE: full image 672", "AFTER: global + tiles"]
    width, height = image.size
    zoom = (0, 0, width, height) if box is None else (
        max(0, box[0] - 60), max(0, box[1] - 60),
        min(width, box[2] + 60), min(height, box[3] + 60),
    )
    canvas = Image.new("RGB", (1500, 854), "#101827")
    draw = ImageDraw.Draw(canvas)
    for row in range(2):
        for column, picture in enumerate(pictures):
            tile = picture.copy()
            if row == 0 and box is not None:
                ImageDraw.Draw(tile).rectangle(box, outline="#ffe066", width=4)
            if row == 1:
                tile = tile.crop(zoom)
            tile.thumbnail((480, 340), Image.Resampling.LANCZOS)
            left, top = column * 500, row * 396
            draw.text((left + 12, top + 12), labels[column], font=font(21), fill="white")
            canvas.paste(tile, (left + (500 - tile.width) // 2, top + 46 + (340 - tile.height) // 2))
    draw.text((14, 798), "Same fixed 0-1 colors. Yellow box = rough reference, NOT pixel ground truth.", font=font(20), fill="#c7d2e5")
    draw.text((14, 826), "One synthetic image; no accuracy claim. Bottom row: identical zoom in all three panels.", font=font(18), fill="#c7d2e5")
    canvas.save(output)


def write_html(output, report):
    metrics = report["map_statistics"]
    rows = []
    entries = [("Điểm vùng lớn nhất", "maximum"),
               ("Điểm lớn nhất trong ô tham chiếu", "reference_box_max"),
               ("Pixel trắng ngoài ô tham chiếu", "outside_box_mask_pixels"),
               ("Diện tích mask (% ảnh)", "mask_area_percent")]
    for label, key in entries:
        vals = []
        for name in ("baseline", "fused"):
            value = metrics[name].get(key)
            vals.append("—" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value)))
        rows.append(f"<tr><td>{label}</td><td>{vals[0]}</td><td>{vals[1]}</td></tr>")
    timings = report["timing_medians"]
    rows.append(f'<tr><td>Thời gian pipeline (giây)</td><td>{timings["baseline_seconds"]:.3f}</td><td>{timings["proposed_seconds"]:.3f}</td></tr>')
    config = html.escape(json.dumps(report["configuration"], ensure_ascii=False, indent=2))
    page = """<!doctype html><html lang="vi"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AnomalyVFM — đối chiếu chia ô</title><style>
body{margin:0;background:#101827;color:#e7edf8;font:16px/1.6 system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:24px}
h1{font-size:28px}p{max-width:1000px}img{max-width:100%;display:block}a{color:#86ccff}
table{border-collapse:collapse;width:100%;margin:24px 0}td,th{padding:10px;text-align:left;border-bottom:1px solid #354357}
.note{border-left:4px solid #ffda75;background:#263044;padding:14px}.slider{position:relative;overflow:hidden}
.slider img{width:100%}.slider .before{position:absolute;inset:0;clip-path:inset(0 50% 0 0)}
input{width:100%}pre{white-space:pre-wrap;background:#182335;padding:18px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
</style><main><h1>Trước / sau: toàn ảnh + các ô chồng lấn</h1>
<p class="note">Ảnh đầu tiên do AI tạo; n = 1. Ô vàng là vị trí vết nứt đánh dấu trước khi chạy model, không phải nhãn vùng chính xác.
Ngưỡng 0,5 chưa hiệu chỉnh. Điểm không phải xác suất. Các số trong ô/ngoài ô chỉ mô tả tín hiệu, không phải precision, recall, F1 hay accuracy.</p>
<img src="comparison.png" alt="Ảnh gốc, trước và sau; hàng dưới phóng to cùng vùng">
<h2>Kéo để đối chiếu</h2><p>Bên trái: bản hiện tại. Bên phải: phương pháp chia ô. Hai ảnh dùng cùng thang màu 0–1.</p>
<div class="slider"><img src="fused_overlay.png" alt="Sau"><img class="before" id="before" src="baseline_overlay.png" alt="Trước"></div>
<label>Vị trí đường chia<input type="range" min="0" max="100" value="50" oninput="document.getElementById('before').style.clipPath='inset(0 '+(100-this.value)+'% 0 0)'"></label>
<table><tr><th>Phép đo mô tả</th><th>Trước</th><th>Sau</th></tr>ROWS</table>
<h2>Mask cùng ngưỡng 0,5</h2><div class="grid"><div>Trước<img src="baseline_mask.png" alt="Mask trước"></div><div>Sau<img src="fused_mask.png" alt="Mask sau"></div></div>
<p>Thời gian là trung vị các lượt sau warm-up, gồm xử lý ảnh, inference, chuyển map về CPU và ghép; không gồm nạp model, mã hóa ảnh hoặc giao diện web.
Phương pháp sau gồm một lượt toàn ảnh cộng các lượt ô. Lượt warm-up không được đưa vào trung vị.</p>
<h2>Cấu hình cố định trước thí nghiệm</h2><pre>CONFIG</pre>
<p><a href="report.json">Số liệu và metadata</a> · <a href="maps.npz">Bản đồ float nguyên độ phân giải</a> · <a href="input.png">Ảnh nguồn đã chuẩn hóa hướng EXIF</a> · <a href="run_manifest.json">Manifest trước inference</a></p>
</main></html>"""
    (output / "comparison.html").write_text(page.replace("ROWS", "".join(rows)).replace("CONFIG", config), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--protocol", type=Path, help="JSON configuration fixed before inference; optional reference_box_xyxy.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tiling_comparison"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20:
        parser.error("--repeats must be between 1 and 20")
    config = dict(input_size=672, tile_size=672, overlap=0.25,
                  smoothing_kernel=5, global_weight=0.25, display_threshold=0.5)
    if args.protocol:
        config.update(json.loads(args.protocol.read_text(encoding="utf-8")))
    threshold = float(config["display_threshold"])
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        parser.error("display_threshold must be finite and in [0,1]")
    with Image.open(args.image) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    if image.width * image.height > 25_000_000:
        parser.error("Input must be at most 25 million pixels")
    box = config.get("reference_box_xyxy")
    if box is not None:
        if (not isinstance(box, list) or len(box) != 4
                or any(type(n) is not int for n in box)
                or not (0 <= box[0] < box[2] <= image.width and 0 <= box[1] < box[3] <= image.height)):
            parser.error("reference_box_xyxy must be four integer coordinates within the EXIF-oriented image")
    output = args.output_dir.resolve()
    if (output / "report.json").exists():
        parser.error("report.json exists; use a new --output-dir to preserve the previous experiment")
    output.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(args.image.read_bytes()).hexdigest()
    root = Path(__file__).resolve().parent
    source_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                     for name in ("demo_tiling.py", "compare_tiling.py", "demo_fast.py", "hf_model.py", "decoder.py", "models/clip.py")}
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    source_path=str(args.image.resolve()), source_sha256=source_hash,
                    original_size=list(image.size), configuration=config,
                    repeats=args.repeats, code_sha256=source_hashes,
                    scope="Single synthetic image, descriptive comparison only; no pixel ground truth.")
    (output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    image.save(output / "input.png")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_demo_model()
    model = model.to(device=device, dtype=torch.bfloat16) if device.type == "cuda" and torch.cuda.is_bf16_supported() else model.to(device)
    model.eval()
    configure_resolution(model, config["input_size"])
    print("Warm-up (excluded from reported timings)...", flush=True)
    tensor = model.model.get_img_transform()(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        model(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    del tensor
    kwargs = {key: config[key] for key in ("input_size", "tile_size", "overlap", "smoothing_kernel", "global_weight")}
    runs = []
    for index in range(args.repeats):
        print(f"Measured run {index + 1}/{args.repeats}", flush=True)
        result = predict_comparison(model, image, device, **kwargs,
                                    progress=lambda done, total: print(f"  tiles {done}/{total}", flush=True))
        runs.append(dict(result["metadata"]))
    stats = {key: describe_map(result[key], box, threshold) for key in ("baseline", "tiled", "fused")}
    medians = {key: statistics.median(run[key] for run in runs)
               for key in ("baseline_seconds", "tiles_seconds", "proposed_seconds")}
    report = dict(manifest)
    report.update({
        "model_id": MODEL_ID, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "model_config": model.config.to_dict(),
        "checkpoint_revisions": {name: (MODEL_CACHE_DIR / name / "refs/main").read_text().strip()
                                 for name in ("models--MaticFuc--anomalyvfm_clip", "models--openai--clip-vit-large-patch14-336")
                                 if (MODEL_CACHE_DIR / name / "refs/main").is_file()},
        "global_image_score_unchanged": result["global_score"],
        "runs": runs, "timing_medians": medians, "map_statistics": stats,
        "maps_from_measured_run": args.repeats,
        "notes": ["No threshold calibration or parameter search on this image.",
                  "ROI is an approximate pre-inference box, not pixel ground truth; outside-ROI pixels are not verified false positives.",
                  "Image score belongs to original whole-image branch; it is not a new fused classifier.",
                  "All PNG heatmaps/overlays use fixed 0..1 scale. Float maps are in maps.npz.",
                  "Timings include transform/inference/map transfer/stitch; exclude loading, file encoding and web UI."],
    })
    np.savez_compressed(output / "maps.npz", **{key: result[key] for key in ("baseline", "tiled", "fused")})
    panels = {}
    for name in ("baseline", "tiled", "fused"):
        panels[name] = render_prediction(image, result[name], threshold)
        for panel in ("heatmap", "overlay", "mask"):
            panels[name][panel].save(output / f"{name}_{panel}.png")
    comparison_figure(image, panels["baseline"]["overlay"], panels["fused"]["overlay"], box, output / "comparison.png")
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(output, report)
    print(json.dumps({"output": str(output), "statistics": stats, "timings": medians}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
