# Demo AnomalyVFM-X

## Mở giao diện web

Nhấp đúp `web_demo.cmd`. Chờ khoảng 5 giây để model khởi động; trình duyệt sẽ
mở tại <http://127.0.0.1:7860>.

Kéo một ảnh PNG/JPG vào trang và bấm **Phân tích ảnh**. Kết quả gồm:

- ảnh đầu vào;
- heatmap, trong đó đỏ/vàng là vùng model chú ý nhiều hơn;
- ảnh overlay và anomaly score của toàn ảnh.

Đóng cửa sổ terminal hoặc bấm `Ctrl+C` để dừng server.

## Ba ảnh nên trình diễn

1. `demo_images/hazelnut_normal.png` — mẫu bình thường, score đo được khoảng 0.44.
2. `demo_images/hazelnut_defective.png` — hạt phỉ có vùng lỗi, score khoảng 0.57.
3. `demo_images/bottle_defective.png` — chai bị lỗi, score khoảng 0.66.

Điểm số không phải phần trăm xác suất và ngưỡng phân loại chính thức phải được
hiệu chỉnh theo dataset. Demo này tập trung vào zero-shot anomaly localization.

## Chạy bằng terminal (phương án dự phòng)

```powershell
cd <thu-muc-clone>\AnomalyVFM-X
.\demo.cmd demo_images\hazelnut_normal.png demo_images\hazelnut_defective.png
```

Ảnh kết quả được lưu trong thư mục `outputs`.

## Cấu hình tối ưu đã dùng

- Backbone/checkpoint: `MaticFuc/anomalyvfm_clip`.
- Python 3.10, PyTorch 2.10 + CUDA 12.8.
- BF16 và kích thước ảnh 336 để vừa GPU RTX 3050 4 GB.
- Model chạy từ cache offline; không tải dataset huấn luyện 93.8 GB.
- Đo thực tế: khoảng 0.68 GB peak VRAM và 0.1–1.0 giây/ảnh.

Mã nguồn gốc: <https://github.com/MaticFuc/AnomalyVFM>

Ảnh mẫu MVTec AD được lấy từ gallery phục vụ demo học thuật:
<https://huggingface.co/datasets/ari1268/mvtec-demo-gallery>. MVTec AD được phát
hành theo giấy phép CC BY-NC-SA 4.0 cho mục đích phi thương mại.
