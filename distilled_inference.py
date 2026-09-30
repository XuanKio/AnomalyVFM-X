"""Separate entry point reserved for the validated distilled method.

No completed distilled checkpoint is available in this project yet. Do not
substitute the paper model, a pilot head, or reference-image differencing here.
When distillation and its evaluation finish, implement the student loader and
inference in this module, without editing paper_inference.py.
"""


class ModeUnavailableError(RuntimeError):
    pass


class DistilledRuntime:
    available = False
    label = "Chi tiết · phương pháp mới sau chưng cất"
    reason = "Chưa có checkpoint chưng cất hoàn tất và được kiểm chứng. Chọn chế độ Nhẹ để phân tích ảnh."

    def predict(self, image, threshold=0.5, size=None):
        raise ModeUnavailableError(self.reason)
