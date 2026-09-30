# Chỉ mục tài liệu AnomalyVFM-X

| Tài liệu | Nội dung |
| --- | --- |
| [OVERVIEW.md](OVERVIEW.md) | Mục tiêu, phạm vi và hai chế độ sử dụng |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Thành phần model, luồng dữ liệu và contract |
| [SETUP.md](SETUP.md) | Cài tự động/thủ công, VS Code và gỡ cài đặt |
| [DEMO.md](DEMO.md) | Web UI, CLI, ảnh mẫu và kịch bản thuyết trình |
| [REFERENCE_DETECTION.md](REFERENCE_DETECTION.md) | Nhánh đối chiếu ảnh lành để tìm vết nhỏ, cách chạy và giới hạn |
| [DATASETS.md](DATASETS.md) | Cấu trúc dataset, loader và thêm dataset mới |
| [SYNTHETIC_DATA.md](SYNTHETIC_DATA.md) | Pipeline sinh dữ liệu ba giai đoạn |
| [TRAINING.md](TRAINING.md) | PEFT training, tham số và checkpoint |
| [EVALUATION.md](EVALUATION.md) | Đánh giá dataset, metrics và kết quả |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Quy trình sửa code và kiểm thử |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Lỗi thường gặp và cách khắc phục |
| [GIT_WORKFLOW.md](GIT_WORKFLOW.md) | Nhánh, commit, push và cộng tác |

Agent phải bắt đầu từ `../AGENTS.md`, sau đó đọc tài liệu của khu vực được giao.
Tài liệu mô tả hai luồng khác nhau:

- **Demo nhẹ** đã được kiểm thử trên Windows, RTX 3050 Laptop 4 GB.
- **Nghiên cứu đầy đủ** bám theo code nghiên cứu nhưng không được kiểm thử toàn
  bộ trên máy demo do yêu cầu dữ liệu/phần cứng lớn.
