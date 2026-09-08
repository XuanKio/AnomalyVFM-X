# Đánh giá model

## Hai cấp độ

### Inference ảnh đơn

Demo nhẹ dùng `demo_fast.py` và checkpoint Hugging Face. Nó phù hợp trình diễn,
không tính AUROC/AP/F1/AUPRO.

Script nghiên cứu gốc:

```powershell
python predict_single_image.py --image-path test.png `
  --model clip --image-size 672 `
  --model-path ./pretrained_models/anomalyvfm_clip.pkl
```

Script này cần môi trường full và checkpoint `.pkl` đúng model/PEFT config.

### Đánh giá benchmark

```powershell
python test.py --model clip --image-size 672 `
  --model-path ./pretrained_models/anomalyvfm_clip.pkl `
  --datasets mvtec_ad visa --save-images
```

Dataset name/path được lấy từ `datasets/dataset.py`.

## Metrics

- `I-AUROC`: AUROC image-level;
- `I-F1`: F1 image-level tại ngưỡng được chọn;
- `I-AP`: average precision image-level;
- `P-AUROC`: AUROC pixel-level;
- `P-F1`: F1 pixel-level;
- `P-AP`: average precision pixel-level;
- `AUPRO-0.3`: area under per-region overlap tới FPR 0.3.

`mean-kernel-size` mặc định 5 làm mượt mask và ổn định pixel metric. Mọi bảng so
sánh phải ghi giá trị này.

## Tính toàn vẹn kết quả

- Không so score demo 336 trực tiếp với bảng benchmark 672/768.
- Checkpoint, backbone, PEFT type/rank và image size phải khớp.
- Không chọn threshold trên test set nếu tuyên bố khả năng tổng quát hóa.
- Ghi per-dataset/per-class trước khi tính trung bình.
- Giữ raw log/config; không chỉnh số trong `results/` bằng tay.

## Checklist

1. Xác minh data path và số lượng sample/class.
2. Chạy một class nhỏ, bật save image và kiểm tra mask alignment.
3. Chạy full dataset với seed/config cố định.
4. Kiểm tra NaN, class thiếu, mask rỗng bất thường.
5. Xuất command, commit hash và environment cùng metric.
