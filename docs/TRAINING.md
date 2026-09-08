# Huấn luyện AnomalyVFM

## Phạm vi

`train.py` freeze backbone gốc, chèn PEFT adapter và tối ưu adapter cùng decoder,
predictor trên auxiliary synthetic dataset. Đây là nghiên cứu đầy đủ, không nằm
trong setup demo.

## Prerequisite

- môi trường full;
- synthetic dataset đúng cấu trúc `AuxilaryDataset`;
- checkpoint backbone tải được;
- GPU đủ VRAM cho backbone và image size;
- nếu bật evaluate, các dataset benchmark đã được cấu hình.

## Smoke run

Trước một job dài, dùng output riêng, ít step, batch nhỏ và tắt evaluation:

```powershell
python train.py --model clip --image-size 672 `
  --data-path ./synthetic_dataset `
  --batch-size 1 --accumulation-steps 1 `
  --train-steps 2 --test-steps 2 --no-evaluate `
  --out-path ./experiments/smoke_clip
```

Smoke run vẫn có thể không vừa GPU 4 GB. Không tự giảm image size cho benchmark
mà không ghi rõ vì feature grid và kết quả sẽ khác paper.

## Tham số chính

| Flag | Mặc định | Ghi chú |
| --- | ---: | --- |
| `--model` | `radio` | radio/dinov3/dinov2/clip/siglip2/tipsv2 |
| `--peft-type` | `dora` | Adapter phải khớp checkpoint khi load |
| `--peft-rank` | 64 | Rank nhỏ hơn giảm tham số nhưng đổi checkpoint |
| `--batch-size` | 32 | Total batch, phải phù hợp accumulation |
| `--accumulation-steps` | 4 | Gradient accumulation |
| `--learning-rate` | 1e-4 | Cấu hình optimizer |
| `--weight-decay` | 1e-2 | Với Muon theo comment code nên bằng 0 |
| `--optimizer` | `adamw` | Cấu hình đã được tác giả kiểm thử chính |
| `--scheduler` | `none` | none/cos/multisteplr/exp |
| `--train-steps` | 200 | Số iteration |
| `--test-steps` | 100 | Chu kỳ đánh giá |
| `--evaluate` | true | Dùng `--no-evaluate` để tắt |
| `--seed` | 12 | Ghi lại để tái lập |

Image size tham chiếu: RADIO/DINOv3/SigLIP2 dùng 768; DINOv2/CLIP dùng 672.

## Command tham chiếu

```powershell
python train.py --model radio --image-size 768 --train-steps 200 --seed 12
python train.py --model dinov3 --image-size 768 --train-steps 500 --seed 12
python train.py --model siglip2 --image-size 768 --train-steps 300 --seed 12
python train.py --model dinov2 --image-size 672 --train-steps 500 --seed 12
python train.py --model clip --image-size 672 --train-steps 500 --seed 12
```

## Theo dõi và artifact

Mỗi experiment cần lưu:

- command/config đầy đủ;
- git commit hash;
- model/checkpoint revision;
- dataset version và seed;
- GPU, PyTorch/CUDA version;
- loss/metric và checkpoint cuối.

Không commit checkpoint vào Git thường. Dùng release asset, model registry hoặc
Git LFS chỉ sau khi có quyết định quản lý dung lượng.
