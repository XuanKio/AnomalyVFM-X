"""Authors' predict_single_image.py protocol using the released CLIP checkpoint.

FP32 weights, CUDA BF16 autocast, decoder logits -> mean pool -> float sigmoid.
Web overlays/thresholding are presentation only; native_mask is the authors'
pred.png pixel format. The legacy demo_fast CLI is a separate demo protocol.
"""
from __future__ import annotations

import base64
import io
import math
import threading
import time

import torch
import torch.nn.functional as F
from PIL import Image

from demo_fast import MODEL_ID, configure_resolution, load_demo_model
from demo_visuals import render_prediction


def predict_single_image(model, decoder, predictor, image, feat_size):
    """Mirror the original single-image calculation, without the HF forward.

CPU is a portability fallback; numerical parity is verified on CUDA BF16.
Keep pooling outside autocast, as in the authors' script.
"""
    k = torch.nn.AvgPool2d((5, 5), 1, 2)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                            enabled=image.device.type == "cuda"):
        with torch.no_grad():
            summary, ftrs = model(image)
            ftrs = ftrs.permute(0, 2, 1)
            ftrs = ftrs.reshape(1, -1, feat_size, feat_size)
            mask, c = decoder(ftrs)
            score = predictor(summary).squeeze().sigmoid()
    mask = k(mask)
    return score, mask.float().sigmoid()


class PaperRuntime:
    def __init__(self, size: int = 672, cpu: bool = False) -> None:
        if size != 672:
            raise ValueError("Paper CLIP uses input size 672.")
        self.default_size = size
        self.device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
        print(f"Loading {MODEL_ID} on {self.device}...", flush=True)
        started = time.perf_counter()
        self.model = load_demo_model()
        configure_resolution(self.model, size)
        # Original .cuda() retains FP32 weights; only operations use autocast.
        self.model = self.model.to(device=self.device, dtype=torch.float32)
        self.model.eval()
        self.transform = self.model.model.get_img_transform()
        self.lock = threading.Lock()
        print(f"Model ready in {time.perf_counter() - started:.1f}s", flush=True)

    def predict(self, image: Image.Image, threshold: float = 0.5,
                size: int | None = None) -> dict:
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Ngưỡng cần là số từ 0 đến 1.")
        size = self.default_size if size is None else size
        if size != 672:
            raise ValueError("Chế độ gốc CLIP dùng kích thước 672 × 672.")
        image = image.convert("RGB")
        # Bound display/encoding memory for large phone photos. Inference keeps
        # its explicitly selected resolution; this only bounds the preview size.
        image_for_display = image.copy()
        image_for_display.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        with self.lock:
            configure_resolution(self.model, size)
            self.transform = self.model.model.get_img_transform()
            tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            score, mask = predict_single_image(
                self.model.model, self.model.decoder, self.model.predictor,
                tensor, self.model.feat_size,
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            peak = (
                torch.cuda.max_memory_allocated() / 1024**3
                if self.device.type == "cuda"
                else 0.0
            )
            score_value = float(score.squeeze().cpu())
            mask_cpu = mask.detach().float().cpu()

        values = F.interpolate(
            mask_cpu, size=(image_for_display.height, image_for_display.width),
            mode="bilinear", align_corners=False,
        )[0, 0].numpy().clip(0, 1)
        views = render_prediction(image_for_display, values, threshold)

        def encode(picture: Image.Image) -> str:
            buffer = io.BytesIO()
            picture.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("ascii")

        return {
            "score": score_value,
            "seconds": seconds,
            "vram_gb": peak,
            "image": encode(views["composite"]),
            "panels": {key: encode(views[key]) for key in ("input", "heatmap", "overlay", "mask")},
            "raw_mask": encode(Image.fromarray((values * 255).round().astype("uint8"))),
            # Match save_predictions_with_paths: native grid, truncate uint8.
            "native_mask": encode(Image.fromarray(
                (mask_cpu[0, 0].numpy() * 255).astype("uint8"))),
            "native_mask_shape": list(mask_cpu.shape[-2:]),
            "inference_protocol": "authors_predict_single_image",
            "score_range": [float(values.min()), float(values.max())],
            "mask_area_percent": float((values >= threshold).mean() * 100),
            "threshold": threshold,
            "threshold_calibrated": False,
            "model_id": MODEL_ID,
            "input_size": size,
            "device": str(self.device),
            "smoothing_kernel": 5,
        }
