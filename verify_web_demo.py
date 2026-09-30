"""Exercise the local web API on bundled samples; not a benchmark evaluation."""

import argparse
import base64
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--size", type=int, choices=(672,), default=672)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/web_verification"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with urlopen(args.url + "/", timeout=10) as response:
        assert response.status == 200
        assert "Mặt nạ dự đoán" in response.read().decode("utf-8")
    summaries = []
    for name in ("hazelnut_normal.png", "hazelnut_defective.png", "bottle_defective.png"):
        source = root / "demo_images" / name
        boundary = "AnomalyVFMVerificationBoundary"
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"mode\"\r\n\r\nlight\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"size\"\r\n\r\n{args.size}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"threshold\"\r\n\r\n0.5\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            "Content-Type: image/png\r\n\r\n"
        ).encode() + source.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        request = Request(args.url + "/predict", data=body, headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}"
        })
        with urlopen(request, timeout=180) as response:
            result = json.load(response)
        assert result["input_size"] == args.size
        assert result["processing_mode"] == "light"
        assert result["method_id"] == "paper_clip"
        assert result["threshold_calibrated"] is False
        assert 0 <= result["score"] <= 1
        assert 0 <= result["score_range"][0] <= result["score_range"][1] <= 1
        folder = args.output_dir / source.stem
        folder.mkdir(exist_ok=True)
        pngs = dict(result.pop("panels"))
        pngs["result"] = result.pop("image")
        pngs["native_mask"] = result.pop("native_mask")
        assert result["native_mask_shape"] == [96, 96]
        assert result["inference_protocol"] == "authors_predict_single_image"
        pngs["raw_scores"] = result.pop("raw_mask")
        for key, encoded in pngs.items():
            data = base64.b64decode(encoded, validate=True)
            with Image.open(io.BytesIO(data)) as picture:
                assert picture.format == "PNG"
                picture.verify()
            (folder / (key + ".png")).write_bytes(data)
        summaries.append({"sample": name, **result})
        # Windows terminals may still use cp1252; keep logs portable while the
        # UTF-8 report below preserves Vietnamese method labels.
        print(json.dumps(summaries[-1], ensure_ascii=True), flush=True)
    report = {
        "scope": "Local API smoke test on 3 gallery images; no ground-truth masks or benchmark metrics.",
        "samples": summaries,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
