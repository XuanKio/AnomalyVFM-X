# Huấn luyện AnomalyVFM

## Phạm vi

`train.py` freeze backbone gốc, chèn PEFT adapter và tối ưu adapter cùng decoder,
predictor trên auxiliary synthetic dataset. Đây là nghiên cứu đầy đủ, không nằm
trong setup demo.

Mask lỗi mảnh được giảm kích thước bằng `preserve_defects_at_decoder_scale`
(`mask_ops.py`): một ô decoder được đánh dấu nếu vùng nguồn tương ứng có pixel
lỗi. Nội suy bilinear rồi nhị phân hóa có thể xóa vết xước dưới một ô. Cách mới
giữ recall của nhãn nhỏ nhưng mở rộng đường biên tối đa cỡ một ô decoder; không
thay thế mask thật ở độ phân giải cao và phải đánh giá lại IoU trên dữ liệu thật.

Với **nhãn đã được duyệt**, có thể thêm `ground_truth/ignore/<id>.png` bên cạnh
`ground_truth/bad/<id>.png`; pixel trắng trong ảnh `ignore` có nghĩa là chưa
biết đúng/sai. `AuxilaryDataset` giảm kích thước vùng ignore bằng max-pooling
cùng mask lỗi, trả `valid_mask`; `train.py` không tính pixel loss ở vùng đó.
Nếu không có file ignore, mọi pixel giữ nguyên cách huấn luyện cũ. Không lấy
vùng sai khác tự động làm ignore/positive mà chưa duyệt: việc bỏ giám sát ở
vùng rộng có thể làm model kém đi. Chế độ ZIP từ xa hiện không có ignore mask,
nên vẫn dùng loss gốc. Cơ chế này mới qua unit test, **chưa chứng minh tăng
độ chính xác** bằng một vòng train và tập test thật.

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

## Thử nghiệm đầu định vị với pilot nhỏ

Dataset sinh phát hành sẵn trên Hugging Face có thể đọc trực tiếp qua HTTP
Range bằng `train.py --remote-data-url ... --remote-network-budget-gib N` mà
không tải ZIP 93,8 GB xuống ổ đĩa. Xem lệnh và giới hạn đã xác minh trong
`docs/SYNTHETIC_DATA.md`. Bộ nạp từ xa đã smoke-test hai mẫu/nhãn, còn toàn
bộ vòng train trên GPU 4 GB chưa được xác nhận; streaming tiết kiệm ổ đĩa chứ
không giảm VRAM cần cho backbone và decoder.

`pilot_decoder_adapt.py` là thí nghiệm riêng, không thay `train.py` hay web demo.
Nó lấy ảnh từ dataset yếu ở trên, cache feature CLIP 672 và phần thân decoder,
rồi chỉ tối ưu lớp `final` 1×1 tạo mask. Lớp dự đoán ảnh và các trọng số còn lại
được giữ nguyên. Cặp kim loại bị loại khỏi tập huấn luyện; ảnh xe thật và hạt phỉ
lành được dùng để đối chiếu đầu ra cùng ngưỡng 0,50. Chạy trên máy đã có
checkpoint demo cục bộ:

```powershell
& 'C:\Users\Admin\AppData\Local\AnomalyVFM-X\venv\Scripts\python.exe' pilot_decoder_adapt.py `
  --dataset outputs/synthetic_pilot_20260930/dataset `
  --source outputs/synthetic_pilot_20260930 `
  --scooter 'C:\Users\Admin\Downloads\OIP (1).webp' `
  --output outputs/synthetic_pilot_20260930/head_672 `
  --size 672 --steps 40 --seed 12
```

Đầu ra gồm `experimental_head.pt`, `report.json`, ảnh overlay/mask và
`comparison.html`. Đây chỉ là kiểm tra khả năng học từ vài nhãn yếu, không phải
fine-tune tương đương bài báo và không chứng minh cải thiện trên thiết bị khác.
Kết quả đầu tiên đã tô được vết trên ảnh xe nhưng báo nhầm rất lớn trên ảnh hạt
phỉ lành, và vẫn bỏ sót vết kim loại chưa thấy khi huấn luyện. Vì vậy checkpoint
pilot bị giữ riêng, không được dùng cho demo mặc định.

Chạy cổng kiểm tra trên từng output trước khi cân nhắc dùng đầu định vị:

```powershell
& 'C:\Users\Admin\AppData\Local\AnomalyVFM-X\venv\Scripts\python.exe' pilot_quality_gate.py `
  outputs/synthetic_pilot_20260930/head_672_final
```

`gate.json` ghi mọi ảnh lành mà vùng báo lỗi tăng so với gốc và mọi mẫu lỗi chưa
được định vị. `comparison.html` hiển thị cảnh báo đỏ nếu thất bại.
`pilot_decoder_adapt.py` cũng tự gọi cổng kiểm tra sau mỗi lần chạy. Với
`cable_good`, đầu mask thử nghiệm báo 7.882 pixel trên đầu nối nhựa trắng mặc dù
ảnh lành; `cable_bad` được tô ở vị trí khác. Cổng kiểm tra không chọn ngưỡng
riêng cho từng ảnh và vẫn yêu cầu tập ảnh thật có mask chuẩn trước phát hành.

Đã thử thêm nhãn hai ngưỡng và hard-negative mining trên 2% ô điểm cao nhất của
mọi ảnh lành (`--hard-negative-fraction 0.02 --hard-negative-weight 0.5`). Bản này
đưa `cable_good` về 0 pixel báo nhầm ở ngưỡng 0,50, vẫn nhận vùng lỗi trên
`cable_bad`, nhưng `hazelnut_good` còn 66.210 pixel báo nhầm, `pcb_bad` không
trùng nhãn yếu và `metal_plate_bad` vẫn không có mask. Ba phép thử chỉ chỉnh đầu
mask đã không đạt cổng kiểm tra; giả định vài cặp ảnh AI cộng với đầu 1×1 cố
định backbone đủ cho mọi thiết bị là không có bằng chứng. Hướng tiếp theo cần
tăng thực sự độ đa dạng dữ liệu, lọc cặp sinh sai, huấn luyện adapter bên trong
backbone kèm loss xử lý nhãn yếu như paper, rồi đánh giá trên tập ảnh thật độc lập.
Chỉ cân nhắc đưa vào demo sau khi kiểm tra trên ảnh thật có mask chuẩn tách biệt
theo thiết bị, đo cả recall/IoU lẫn báo nhầm trên ảnh lành, và kiểm chứng ít
nhất nhiều loại sản phẩm. Checklist: cùng checkpoint, kích thước, ngưỡng và hậu
xử lý trước/sau; cặp kim loại và xe thật không tham gia train; xem toàn bộ
overlay và `report.json`; không chọn ngưỡng riêng cho từng ảnh.
