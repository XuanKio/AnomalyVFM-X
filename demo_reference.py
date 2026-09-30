"""Opt-in aligned-reference change detector for fine visible surface defects.

This is a classical vision branch, not a new CLIP checkpoint or a zero-shot
classifier. Scores are normalized residuals, NOT probabilities. No target ROI
is accepted: the complete overlapping field of view is searched.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps

try:
    import cv2
except ImportError as exc:
    raise ImportError(
        "Chế độ đối chiếu cần OpenCV: cài requirements-reference.txt bằng Python demo."
    ) from exc


class ReferenceMismatch(ValueError):
    """Cannot reliably compare the two images; this does not mean defect-free."""


@dataclass(frozen=True)
class ReferenceConfig:
    threshold: float = 3.0
    min_area: int = 12
    illumination_sigma: float = 12.0
    residual_sigma: float = 0.8
    texture_sigma: float = 3.0
    noise_floor: float = 4.0
    texture_weight: float = 0.5
    alignment_min_correlation: float = 0.97
    alignment_max_side: int = 768
    border_margin: int = 16
    max_pixels: int = 4_000_000

    def __post_init__(self):
        for name in ("threshold", "illumination_sigma", "residual_sigma",
                     "texture_sigma", "noise_floor", "texture_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("min_area", "alignment_max_side", "border_margin", "max_pixels"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (not np.isfinite(self.alignment_min_correlation)
                or not 0 < self.alignment_min_correlation <= 1):
            raise ValueError("alignment_min_correlation must be in (0,1]")


def _rgb(image: Image.Image, config: ReferenceConfig) -> np.ndarray:
    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL image")
    if image.width * image.height > config.max_pixels:
        raise ValueError("Chế độ đối chiếu hỗ trợ tối đa 4 triệu pixel mỗi ảnh.")
    result = np.array(ImageOps.exif_transpose(image).convert("RGB"))
    if min(result.shape[:2]) < max(64, 4 * config.border_margin):
        raise ValueError("Ảnh quá nhỏ để căn chỉnh; mỗi cạnh cần ít nhất 64 pixel.")
    return result


def detect_changes(reference: Image.Image, inspected: Image.Image,
                   config: ReferenceConfig | None = None) -> dict:
    """Search all pixels; return float scores, binary mask and all components.

    Coordinates are on the EXIF-oriented inspected image. Border pixels outside
    reliable overlap are excluded and returned in valid_mask. Raises
    ReferenceMismatch on poor alignment instead of returning a healthy result.
    """
    started = time.perf_counter()
    config = config or ReferenceConfig()
    ref, query = _rgb(reference, config), _rgb(inspected, config)
    if ref.shape != query.shape:
        raise ReferenceMismatch("Hai ảnh cần cùng kích thước và cùng loại thiết bị/góc chụp.")
    h, w = query.shape[:2]
    r = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    q = cv2.cvtColor(query, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    if min(float(r.std()), float(q.std())) < 0.01:
        raise ReferenceMismatch("Ảnh thiếu chi tiết để căn chỉnh tin cậy.")
    scale = min(1.0, config.alignment_max_side / max(h, w))
    # Use actual resize scales; affine translations must map back to native pixels.
    small_w, small_h = max(1, round(w * scale)), max(1, round(h * scale))
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        cc, warp = cv2.findTransformECC(
            cv2.resize(q, (small_w, small_h)), cv2.resize(r, (small_w, small_h)),
            warp, cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-6), None, 5,
        )
    except cv2.error as exc:
        raise ReferenceMismatch("Không căn chỉnh được. Chọn ảnh lành cùng thiết bị và góc chụp.") from exc
    scaling = np.diag([small_w / w, small_h / h, 1.0])
    affine = np.eye(3)
    affine[:2] = warp
    warp = (np.linalg.inv(scaling) @ affine @ scaling)[:2].astype(np.float32)
    if not np.isfinite(warp).all() or not np.isfinite(cc):
        raise ReferenceMismatch("Phép căn chỉnh không hợp lệ.")
    stretch = np.linalg.svd(warp[:, :2], compute_uv=False)
    if (stretch.min() < .9 or stretch.max() > 1.1 or np.linalg.det(warp[:, :2]) <= 0
            or abs(warp[0, 2]) > .15 * w or abs(warp[1, 2]) > .15 * h):
        raise ReferenceMismatch("Hai ảnh chưa khớp đủ tin cậy; không kết luận có/không có lỗi.")
    aligned = cv2.warpAffine(ref.astype(np.float32), warp, (w, h),
                            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_CONSTANT)
    valid = cv2.warpAffine(np.ones((h, w), np.uint8), warp, (w, h),
                          flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    overlap_support = valid.astype(np.float32)
    margin = config.border_margin
    valid = cv2.erode(valid, np.ones((margin * 2 + 1, margin * 2 + 1), np.uint8),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    if float(valid.mean()) < .6:
        raise ReferenceMismatch("Vùng ảnh trùng nhau quá nhỏ để đối chiếu.")
    # A real defect itself can lower raw ECC correlation. Validate the alignment
    # across 99% of the overlapping pixels after excluding the largest global
    # photometric residuals; no location or defect annotation is consulted.
    reference_grey = cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY)[valid]
    query_grey = q[valid] * 255
    if min(float(reference_grey.std()), float(query_grey.std())) < 2.0:
        raise ReferenceMismatch("Vùng trùng nhau thiếu chi tiết để xác nhận căn chỉnh.")
    rn = (reference_grey - reference_grey.mean()) / reference_grey.std()
    qn = (query_grey - query_grey.mean()) / query_grey.std()
    disagreement = np.abs(rn - qn)
    inliers = disagreement <= np.quantile(disagreement, .99)
    quality = float(np.corrcoef(reference_grey[inliers], query_grey[inliers])[0, 1])
    if not np.isfinite(quality) or quality < config.alignment_min_correlation:
        raise ReferenceMismatch("Hai ảnh chưa khớp đủ tin cậy; không kết luận có/không có lỗi.")
    # Slowly varying illumination is removed; local texture accounts for small
    # registration/resampling differences at pre-existing edges.
    residual = query.astype(np.float32) - aligned
    def supported_blur(values, sigma):
        numerator = cv2.GaussianBlur(values * overlap_support[..., None], (0, 0), sigma)
        denominator = cv2.GaussianBlur(overlap_support, (0, 0), sigma)
        return numerator / np.maximum(denominator[..., None], 1e-6)

    # Black warp padding is not an observed change: exclude it from BOTH blur
    # numerators and denominators so it cannot bleed into the valid image.
    residual -= supported_blur(residual, config.illumination_sigma)
    residual = supported_blur(residual, config.residual_sigma)
    mean = cv2.GaussianBlur(aligned, (0, 0), config.texture_sigma)
    variance = cv2.GaussianBlur(aligned * aligned, (0, 0), config.texture_sigma) - mean * mean
    texture = np.sqrt(np.maximum(variance, 0)).mean(axis=2)
    scores = np.sqrt(np.mean(residual * residual, axis=2)) / (
        config.noise_floor + config.texture_weight * texture)
    scores[~valid] = 0
    if not np.isfinite(scores).all():
        raise ReferenceMismatch("Không tính được bản đồ khác biệt hữu hạn.")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (scores >= config.threshold).astype(np.uint8), connectivity=8)
    accepted = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= config.min_area)
    accepted = accepted[accepted != 0]
    keep = np.zeros(count, dtype=bool)
    keep[accepted] = True
    mask = keep[labels]
    # Compute component maxima once, including disjoint off-target components.
    peaks = np.zeros(count, dtype=np.float32)
    np.maximum.at(peaks, labels.ravel(), scores.ravel())
    regions = []
    for label in accepted:
        x, y, rw, rh, area = map(int, stats[label])
        regions.append({"box_xyxy": [x, y, x + rw, y + rh],
                        "area_pixels": area, "peak_residual": float(peaks[label])})
    regions.sort(key=lambda region: region["peak_residual"], reverse=True)
    peak_y, peak_x = np.unravel_index(int(scores.argmax()), scores.shape)
    metadata = {
        "method": "aligned_reference_residual_v1", "requires_reference": True,
        "threshold_calibrated": False, "configuration": asdict(config),
        "alignment_correlation": quality, "raw_ecc_correlation": float(cc),
        "alignment_validation_retained_fraction": float(inliers.mean()),
        "warp_query_to_reference": warp.tolist(),
        "valid_area_percent": float(valid.mean() * 100),
        "maximum_residual": float(scores.max()), "maximum_xy": [int(peak_x), int(peak_y)],
        "regions": regions, "seconds": time.perf_counter() - started,
        "opencv_version": cv2.__version__,
        "warning": "Vùng khác biệt cần kiểm tra; không phải kết luận lỗi chức năng hoặc xác suất.",
    }
    return {"scores": scores, "mask": mask, "valid_mask": valid,
            "aligned_reference": aligned, "input": query, "metadata": metadata}


def render_changes(result: dict) -> dict[str, Image.Image]:
    """Show actual detected mask and all boxes, never a reference-location box."""
    original = result["input"]
    overlay = original.copy()
    mask = result["mask"]
    overlay[mask] = (original[mask] * .25 + np.array([255, 40, 100]) * .75).astype(np.uint8)
    panel = Image.fromarray(overlay)
    draw = ImageDraw.Draw(panel)
    h, w = mask.shape
    for index, region in enumerate(result["metadata"]["regions"], start=1):
        x0, y0, x1, y1 = region["box_xyxy"]
        box = (max(0, x0 - 5), max(0, y0 - 5), min(w - 1, x1 + 4), min(h - 1, y1 + 4))
        draw.rectangle(box, outline=(255, 210, 60), width=2)
        draw.text((box[0], max(0, box[1] - 14)), f"#{index}", fill=(255, 210, 60))
    # Fixed units: black=0, midpoint=threshold, saturated=2*threshold.
    grey = np.clip(result["scores"] / (2 * result["metadata"]["configuration"]["threshold"]), 0, 1)
    heatmap = cv2.applyColorMap(np.uint8(grey * 255), cv2.COLORMAP_INFERNO)[..., ::-1].copy()
    return {"input": Image.fromarray(original),
            "reference": Image.fromarray(np.clip(result["aligned_reference"], 0, 255).astype(np.uint8)),
            "overlay": panel, "mask": Image.fromarray(mask.astype(np.uint8) * 255),
            "heatmap": Image.fromarray(heatmap)}
