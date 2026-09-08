# Tổng quan dự án

## Bài toán

AnomalyVFM nhận một ảnh và thực hiện zero-shot visual anomaly detection: phát
hiện khả năng ảnh chứa bất thường và định vị vùng đáng ngờ mà không cần ảnh train
từ chính lớp đối tượng đang kiểm tra.

Đầu ra gồm:

- `anomaly_score`: một điểm vô hướng cho toàn ảnh;
- `anomaly_mask`: bản đồ điểm theo pixel/patch để tạo heatmap.

Score không phải xác suất. Muốn biến score thành nhãn normal/anomaly cần hiệu
chỉnh threshold trên validation set của domain mục tiêu.

## Hai chế độ sử dụng

### Demo nhẹ

Mục tiêu là chạy nhanh, ít dung lượng và dễ trình bày:

- checkpoint `MaticFuc/anomalyvfm_clip`;
- input 336×336 thay cho cấu hình benchmark 672×672;
- BF16 trên GPU hỗ trợ;
- giao diện local web tại `127.0.0.1:7860`;
- không cài FLUX, ADEval, OpenCV hoặc full dataset.

Đo trên máy Xuân: model khởi động khoảng 4 giây sau khi đã cache, inference
khoảng 0.1–1.0 giây/ảnh và peak VRAM khoảng 0.68 GB.

### Nghiên cứu đầy đủ

Bao gồm:

1. sinh ảnh normal tổng hợp;
2. inpainting để tạo anomaly;
3. lọc cặp ảnh và sinh mask bằng feature difference;
4. PEFT backbone bằng LoRA/DoRA;
5. đánh giá trên các benchmark công nghiệp/y tế.

Luồng này dùng `requirements.txt`, dataset lớn, nhiều checkpoint và GPU mạnh.
Nó không nằm trong smoke test mặc định của repository.

## Bản đồ mã nguồn

| Thành phần | Vai trò |
| --- | --- |
| `hf_model.py` | Wrapper tải/lưu checkpoint Hugging Face |
| `models/` | Adapter cho RADIO, DINOv2/v3, CLIP, SigLIP2, TIPSv2 |
| `peft_local/` | LoRA, DoRA, VPT và wrapper chèn adapter |
| `decoder.py` | Decoder mask và predictor score |
| `demo_fast.py` | Inference ảnh đơn tối ưu cho GPU nhỏ |
| `web_demo.py` | Web server local, upload và hiển thị kết quả |
| `datasets/` | Loader các benchmark đánh giá |
| `generate_dataset.py` | Sinh/lọc dữ liệu tổng hợp |
| `train.py` | Huấn luyện adapter, decoder và predictor |
| `test.py` | Đánh giá nhiều dataset và metrics |
| `results/` | Kết quả tham chiếu có sẵn |

## Phạm vi quản lý

Repository được quản lý độc lập tại `XuanKio/AnomalyVFM-X`. Không cấu hình
upstream remote. Khi cần lấy thay đổi từ nguồn nghiên cứu, Xuân sẽ chỉ định quy
trình riêng cho lần đó.
