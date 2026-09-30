"""Route independent inference methods; resolution is not a method selector."""

from distilled_inference import DistilledRuntime


class ProcessingModes:
    default_mode = "light"
    paper_label = "Nhẹ · phương pháp gốc của bài báo"

    def __init__(self, paper):
        self.paper = paper
        self.distilled = DistilledRuntime()

    def configuration(self):
        return {
            "default_mode": self.default_mode,
            "processing_modes": [
                {"id": "light", "label": self.paper_label, "available": True,
                 "method_id": "paper_clip", "sizes": [672]},
                {"id": "detailed", "label": self.distilled.label,
                 "available": self.distilled.available, "method_id": "distilled",
                 "reason": self.distilled.reason},
            ],
        }

    def predict(self, image, *, mode="light", threshold=0.5, size=None):
        if mode == "light":
            result = self.paper.predict(image, threshold=threshold, size=size)
            return {**result, "processing_mode": mode, "method_id": "paper_clip",
                    "method_label": self.paper_label}
        if mode == "detailed":
            # A missing student is an explicit error, never a fallback to paper.
            result = self.distilled.predict(image, threshold=threshold, size=size)
            return {**result, "processing_mode": mode, "method_id": "distilled",
                    "method_label": self.distilled.label}
        raise ValueError("Chế độ xử lý không hợp lệ; chỉ nhận light hoặc detailed.")
