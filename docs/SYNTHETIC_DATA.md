# Sinh dữ liệu anomaly tổng hợp

## Mục tiêu

Pipeline tạo auxiliary dataset đa dạng mà không lấy mẫu train từ domain đánh
giá. Có ba giai đoạn:

1. `generate`: sinh ảnh đối tượng không lỗi;
2. `generate_anom`: inpaint một vùng để tạo lỗi;
3. `filter`: so feature ảnh normal/anomaly, loại cặp kém và tạo mask.

## Cảnh báo tài nguyên

Đây không phải luồng demo. FLUX, foreground segmentor, filter backbone và dataset
có thể cần hàng chục đến hơn 100 GB. Dataset được phát hành sẵn khoảng 93.8 GB.
Không chạy trên RTX 3050 4 GB nếu chưa có kế hoạch offload/phần cứng khác.

## Prerequisite

- môi trường full theo `SETUP.md`;
- Hugging Face authentication cho model/dataset gated nếu có;
- foreground segmentor tại `pretrained_models/isnet-general-use.pth`;
- nếu dùng DINOv3: clone DINOv3 riêng, cấu hình path trong `models/dinov3.py` và
  đặt checkpoint ViT-L theo hướng dẫn nguồn;
- đủ disk cho ảnh trung gian normal, anomaly và mask.

Không commit access token, checkpoint hoặc output synthetic.

## Lệnh cơ bản

Luôn thử với `--n-img 1` trước:

```powershell
python generate_dataset.py --mode generate --n-img 1 --out-path ./synthetic_dataset
python generate_dataset.py --mode generate_anom --n-img 1 --out-path ./synthetic_dataset
python generate_dataset.py --mode filter --n-img 1 --out-path ./synthetic_dataset
```

Sau khi xác minh cấu trúc/output mới tăng số lượng. Tham số chính:

| Flag | Mặc định | Vai trò |
| --- | --- | --- |
| `--img-gen-model` | `flux` | Backend generate/inpaint |
| `--object-data` | `default` | Danh sách object/prompt |
| `--image-size` | 1024 | Độ phân giải sinh ảnh |
| `--n-img` | 1 | Số mẫu trước filter |
| `--out-path` | `./synthetic_dataset/` | Thư mục output |
| `--filter-model` | `dinov3` | Backbone tạo feature difference |
| `--filter-thr` | 0.25 | Ngưỡng filter/mask |
| `--mode` | `generate` | Giai đoạn pipeline |

## Download dataset phát hành sẵn

```powershell
python download_dataset.py
```

Lệnh yêu cầu authentication và đủ disk; kiểm tra dung lượng trước. Không chạy
trong CI hoặc setup demo.

## Checklist xác minh

- normal/anomaly cùng kích thước và alignment;
- mask binary/continuous đúng quy ước training;
- không có ảnh hỏng hoặc file zero-byte;
- seed, model revision, prompt config và filter threshold được ghi lại;
- kiểm tra thủ công một grid nhỏ trước khi scale;
- thống kê tỷ lệ cặp bị filter để phát hiện pipeline sai.
