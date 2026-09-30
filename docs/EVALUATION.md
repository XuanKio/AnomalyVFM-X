# Đánh giá model

## Hai cấp độ

### Inference ảnh đơn

CLI nhẹ dùng `demo_fast.py` ở 336; web dùng cùng checkpoint CLIP, mặc định 672.
Cả hai phù hợp trình diễn, không tính AUROC/AP/F1/AUPRO. Mask nhị phân của web
được tạo bằng ngưỡng hiển thị chưa hiệu chuẩn, không được dùng làm nhãn GT.

Sau khi khởi động web, `python verify_web_demo.py` kiểm tra ba ảnh gallery,
lưu PNG và `report.json` vào `outputs/web_verification`. Ba ảnh không có mask
tham chiếu; báo cáo này chỉ xác minh pipeline, không chứng minh chất lượng bằng paper.

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

### Lát cắt VisA có nhãn thật

`prepare_visa_slice.py` trích từ prefix 320 MiB của TAR VisA chính thức: 100 ảnh
lành và 100 ảnh lỗi/mask cho từng lớp `candle`, `capsules`. `evaluate_visa_slice.py`
so baseline với hai đầu mask thử nghiệm ở cùng độ phân giải 672 và ngưỡng 0,50.
Báo cáo nằm tại `outputs/visa_slice_20260930/evaluation_672/report.json` (400 ảnh).
Hai lớp này không đại diện mọi thiết bị. Bản hard-negative tăng P-AUROC candle
0,9840→0,9856 và capsules 0,9577→0,9638, nhưng ảnh lành báo nhầm 3% và 1%
so với 0% của baseline. AUROC ở mức ảnh gần như không tăng; nhiều cải thiện mask
tại ngưỡng 0,50 là thay đổi thang điểm. Do đó chưa đưa đầu mask thử nghiệm vào
web demo. [Nguồn VisA](https://github.com/amazon-science/spot-diff) và hash dữ
liệu/đầu mask được lưu trong báo cáo.

`analyze_visa_ranking.py` kiểm tra lại điểm lớn nhất từng ảnh bằng AUROC/AP ở
mức ảnh, rồi chọn ngưỡng chỉ từ 100 ảnh lành candle với ngân sách 5 báo nhầm.
Áp nguyên ngưỡng sang capsules: baseline và đầu hard-negative đều phát hiện
41/100 ảnh lỗi, cùng báo nhầm 4/100 ảnh lành. AUROC ảnh candle baseline
0,8516 so với hard-negative 0,8481; capsules 0,7339 so với 0,7379. Đây là
bằng chứng thay đầu mask chưa cải thiện phân biệt ảnh lỗi/lành ổn định, dù F1
pixel tại ngưỡng 0,5 thay đổi nhiều. File audit:
`outputs/visa_slice_20260930/evaluation_672/ranking_audit.json`.

`diagnose_small_defect_scores.py` đo riêng 4 cặp ảnh AI và 3 kernel làm mượt
(1, 3, 5). Với đầu hard-negative, `pcb_bad` ở kernel 1 tạo 712 pixel mask
**sai chỗ**; điểm cao nhất trên nhãn xước yếu chỉ 0,260, thấp hơn điểm cao
nhất của `pcb_good` (0,500). Ở kernel 5, vết kim loại xếp cao trong ảnh lỗi
nhưng điểm tối đa chỉ 0,054 nên không vượt ngưỡng 0,5. Bỏ làm mượt hoặc hạ
ngưỡng toàn cục không giải quyết đồng thời hai trường hợp. Nhãn AI yếu và mask
kim loại chưa đủ chiều dài; phép đo này là chẩn đoán, không phải benchmark thật.

`evaluate_normal_memory.py` thử thêm bộ nhớ đặc trưng CLIP từ 10 ảnh lành mỗi
lớp trên 380 ảnh VisA còn lại (90 lành và 100 lỗi cho mỗi lớp). Kết quả
`outputs/visa_slice_20260930/normal_memory_full/report.json`: candle giảm
P-AP 0,3958→0,0650 và I-AUROC 0,8621→0,7583; capsules I-AUROC tăng
0,7363→0,8213 nhưng P-AP giảm 0,3512→0,2545. Bộ nhớ đặc trưng một tầng
không cải thiện định vị ổn định qua hai lớp nên không đưa vào demo. Đây là
thử nghiệm có ảnh lành cùng lớp, không còn là zero-shot thuần túy.

Các phép thử trên cho thấy chưa có bằng chứng đầu mask huấn luyện trên vài
cặp ảnh AI học được quy luật khuyết tật tổng quát. Trước khi huấn luyện lại,
cần kiểm tra chất lượng mask ảnh sinh, rồi so backbone/adaptation có nhiều mức
đặc trưng và đo P-AP/I-AUROC trên tập thật độc lập ở cùng mức báo nhầm ảnh lành.

`evaluate_feature_layers.py` đã kiểm tra trực tiếp đặc trưng CLIP tầng 6, 12,
18, 24 với 10 ảnh lành làm bộ nhớ và 380 ảnh VisA còn lại. Báo cáo
`outputs/visa_slice_20260930/feature_layer_full/report.json` dùng cùng mask,
kích thước 672 và không chọn ngưỡng trên ảnh test. P-AP của decoder gốc là
0,396/0,351 trên candle/capsules; mọi tầng so khớp patch riêng đều thấp hơn
trên cả hai lớp (tốt nhất 0,189/0,308). Tầng 12/24 giúp I-AUROC capsules
0,736→0,809/0,813 nhưng giảm P-AP và candle. Do đó không ghép nhánh bộ nhớ
vào model chạy thật. Điều này **không bác bỏ** decoder đa tầng được huấn luyện
đúng cách, chỉ bác bỏ giả thuyết rằng khoảng cách cosine đơn giản trên tầng
giữa đã đủ thay decoder của checkpoint gốc.

## Checklist

1. Xác minh data path và số lượng sample/class.
2. Chạy một class nhỏ, bật save image và kiểm tra mask alignment.
3. Chạy full dataset với seed/config cố định.
4. Kiểm tra NaN, class thiếu, mask rỗng bất thường.
5. Xuất command, commit hash và environment cùng metric.

### Chấm chéo giữa ảnh (transductive, ý tưởng MuSc)

`evaluate_mutual_scoring.py` thử một nhánh không cần huấn luyện, không cần ảnh
lành có nhãn: trong một lô ảnh cùng lớp, mỗi patch được so với mọi patch của
các ảnh khác; điểm là trung bình 30% khoảng cách nhỏ nhất (tầng CLIP 6/12/18/24,
lân cận r = 1/3/5, chiếu ngẫu nhiên 1024→256 chiều). Bản đồ được ghép 50/50 với
decoder gốc sau chuẩn hóa robust-z bằng thống kê của chính lô. Mọi quy tắc chốt
trước khi đo. Báo cáo: `outputs/visa_slice_20260930/mutual_full_r135/report.json`
(400 ảnh, 672, khoảng 17 phút trên RTX 3050 4 GB).

| Lớp | Biến thể | P-AP | I-AUROC (max map) |
| --- | --- | ---: | ---: |
| candle | decoder gốc | 0,393 | 0,852 |
| candle | chấm chéo | 0,254 | 0,896 |
| candle | ghép | 0,371 | 0,913 |
| capsules | decoder gốc | 0,349 | 0,734 |
| capsules | chấm chéo | 0,426 | 0,811 |
| capsules | ghép | 0,413 | 0,793 |

P-AP trung bình hai lớp: 0,371 (gốc) → 0,392 (ghép), nhưng candle giảm. Đầu
dự đoán ảnh (`predictor`) của checkpoint gốc đã đạt I-AUROC 0,910/0,910, nên
mức tăng I-AUROC ở bảng chỉ đúng khi so với điểm max của bản đồ; ghép thứ hạng
predictor + chấm chéo cho 0,921/0,903, không hơn rõ. Kết luận: nhánh chấm chéo
giúp định vị capsules (vật thể lặp, vị trí ngẫu nhiên), chưa cải thiện đồng
đều; cần một lô ảnh cùng loại nên không áp được cho ảnh đơn trên web.
