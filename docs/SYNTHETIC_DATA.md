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

Dataset [Hugging Face của tác giả](https://huggingface.co/datasets/MaticFuc/AnomalyVFM_Synthetic_Dataset)
hiện là một ZIP 93,8 GB. `download_dataset.py` tải nguyên ZIP rồi giải nén;
`train.py`/`AuxilaryDataset` cũng đang dùng `glob` và đường dẫn ảnh cục bộ.
Vì thế `streaming=True` không tự nối được vào mã huấn luyện này. Máy chủ hỗ trợ
HTTP Range; có thể đọc ZIP từ xa theo từng ảnh, trả giá bằng lưu lượng mạng và
độ trễ. `remote_synthetic_sample.py` kiểm tra con đường đó với giới hạn số byte:

```powershell
python remote_synthetic_sample.py `
  --output outputs/remote_synthetic_sample_new --count 2 --max-network-mib 64
```

Lệnh chỉ lưu hai bộ ảnh lành/ảnh lỗi/mask vào thư mục output. Trong lượt thử
30/09/2026, nó đọc 25.019.156 byte
qua mạng để lấy hai bộ ba 1024×1024, kiểm tra CRC ZIP và PNG, thay vì tải
93.805.724.049 byte. Dataset card hiện không xem trước mẫu được bằng Dataset
Viewer; đọc trực tiếp ZIP bằng Range đã chạy. Không nhập token vào command/log.

Để **huấn luyện mà không lưu ảnh trên máy**, `train.py` có chế độ tùy chọn
`--remote-data-url`: `RemoteAuxilaryDataset` đọc ảnh/mask trực tiếp từ ZIP,
ghép theo cùng mã, giữ giao diện dataset như bản local. Ví dụ cấu hình cho
**môi trường huấn luyện đầy đủ của repo** (không phải môi trường demo nhẹ):

```powershell
python train.py --model clip --image-size 672 --no-evaluate `
  --remote-data-url "https://huggingface.co/datasets/MaticFuc/AnomalyVFM_Synthetic_Dataset/resolve/main/dataset.zip" `
  --remote-network-budget-gib 4
```

Đã smoke-test bộ nạp: 56.406 mẫu ảnh, lấy trực tiếp một mẫu lành và một mẫu lỗi
với mask 600 pixel bằng 18.049.061 byte truyền qua mạng, **không tạo file ảnh**.
Chưa chạy vòng huấn luyện đầy đủ trên máy GPU 4 GB. Môi trường demo hiện tại
thiếu ít nhất `timm` khi import `utils.py`; muốn chạy lệnh trên phải cài môi
trường training theo `requirements.txt` mà vẫn giữ tương thích torch/CUDA.
Cấu hình mặc định của paper có thể vượt VRAM. Bộ nạp dùng 0 worker để tránh
mỗi worker tải lại chỉ mục;
đọc ngẫu nhiên qua mạng sẽ chậm hơn file local và mỗi ảnh vẫn phải truyền về
RAM. Giới hạn byte ngắt rõ ràng khi hết ngân sách; tăng ngân sách chỉ khi đã
ước lượng số bước huấn luyện và đường truyền.

## Checklist xác minh

- normal/anomaly cùng kích thước và alignment;
- mask binary/continuous đúng quy ước training;
- không có ảnh hỏng hoặc file zero-byte;
- seed, model revision, prompt config và filter threshold được ghi lại;
- kiểm tra thủ công một grid nhỏ trước khi scale;
- thống kê tỷ lệ cặp bị filter để phát hiện pipeline sai.

## Pilot ảnh sinh trên máy 4 GB

`pilot_specification.json` ghi loại bề mặt, prompt ảnh lành/ảnh sửa, vùng dự kiến
sửa và quyết định duyệt. Ảnh được tạo/chỉnh bằng OpenAI built-in imagegen, đặt
trong `outputs/synthetic_pilot_20260930/<loại>/good.png` và `bad.png`. Cặp PCB
được dùng lại từ lượt tạo trước nên không có prompt gốc đầy đủ. Bộ lọc
`demo_reference.detect_changes` tạo `candidate_mask.png` bằng đối chiếu hai ảnh
đã căn chỉnh. Đây là **nhãn yếu**, không phải ground truth được vẽ tay. Người
duyệt phải đánh dấu `accepted_weak_label` hoặc `rejected_incomplete_mask` trong
specification sau khi xem ảnh chồng mask.

```powershell
& 'C:\Users\Admin\AppData\Local\AnomalyVFM-X\venv\Scripts\python.exe' synthetic_pilot.py `
  --source outputs/synthetic_pilot_20260930 `
  --output outputs/synthetic_pilot_20260930/dataset `
  --specification pilot_specification.json
```

Đầu ra là `train/ok`, `train/bad`, `ground_truth/bad` tương thích
`AuxilaryDataset`, cùng `manifest.json` có SHA-256 và trạng thái duyệt. Pilot
hiện có ba cặp được chấp nhận (thân xe, cáp, PCB); cặp kim loại bị loại vì mask
chỉ bắt được đầu vết xước. Tỉ lệ loại 1/4 cho thấy không được dùng mọi ảnh sinh
và mask tự động làm nhãn huấn luyện. Phép đối chiếu có thể bỏ sót vết mảnh hoặc
nhầm khác biệt ánh sáng; duyệt độc lập và dữ liệu thật có nhãn vẫn cần thiết.

Checklist riêng: xác nhận 3 cặp trong manifest, xem overlay của từng cặp,
kiểm tra cặp kim loại không xuất hiện trong `train`, và dùng
`AuxilaryDataset` đọc lại sáu ảnh cùng ba mask. Không suy rộng kết quả pilot
thành chất lượng của pipeline 10.000 ảnh trong bài báo.

### Kiểm tra chéo nhãn tổng hợp của tác giả

`remote_synthetic_sample.py` đã lấy 20 cặp cách đều ID từ ZIP Hugging Face
với ngân sách truyền 128 MiB. Trong mẫu thăm dò này, bốn mask gốc có diện
tích 12, 22, 12 và 313 pixel trong khi ảnh lỗi/ảnh lành cho thấy vùng thay đổi
rộng hơn. Không suy rộng tỉ lệ lỗi nhãn từ 20 cặp sang toàn bộ dataset.

`synthetic_label_triage.py` tính sai khác giữa ảnh lành và ảnh lỗi bằng một
phương pháp độc lập, nối thành phần liền với mask gốc thành **nhãn ứng viên**,
và đánh dấu vùng sai khác không nối được là **ignore** để người duyệt xem.
Nhãn này không được tự động đưa vào tập train vì sai khác cũng có thể đến từ
ánh sáng, hình học, hoặc thay đổi do trình sinh ảnh. Mỗi cặp đều yêu cầu duyệt;
các cặp mask rất nhỏ, lệch căn chỉnh hoặc mở rộng quá lớn được ưu tiên.

```powershell
& 'C:\Users\Admin\AppData\Local\AnomalyVFM-X\venv\Scripts\python.exe' synthetic_label_triage.py `
  --dataset outputs/remote_synthetic_sample_20260930/stage1_audit_20 `
  --output outputs/remote_synthetic_sample_20260930/stage1_triage_run2
```

Ảnh chồng và báo cáo nằm trong `comparison.html` và `report.json` ở thư mục
output. Đây là công cụ kiểm duyệt dữ liệu, chưa phải kết quả nâng cấp model.

### Bộ lọc hai ngưỡng thử nghiệm

`synthetic_mask_filter.py` đọc bản đồ sai khác đã căn chỉnh và vùng sửa do bước
sinh ảnh cung cấp. Nó giữ các vùng điểm cao làm hạt giống, rồi nối các pixel
điểm thấp hơn nếu liền mạch trong vùng sửa. Ngưỡng thấp được ước lượng từ đuôi
điểm của vùng không phải hạt giống, không chọn theo vị trí vết trên ảnh kiểm tra.

```powershell
& 'C:\Users\Admin\AppData\Local\AnomalyVFM-X\venv\Scripts\python.exe' synthetic_mask_filter.py `
  --source outputs/synthetic_pilot_20260930 `
  --specification pilot_specification.json
```

Đầu ra `refined_mask.png`, `refined_overlay.png` cho mỗi loại và
`refined_filter_report.json`. Dùng `synthetic_pilot.py --mask-kind refined`
với một thư mục output **mới** để so sánh. Trên bốn cặp pilot, cách này giữ vết
thân xe, cáp và PCB, đồng thời kéo dài mask kim loại từ 162 lên 887 pixel. Mask
kim loại vẫn thiếu một đoạn vết nhìn thấy nên cặp đó vẫn bị loại. Đây là nhánh
thử nghiệm dựa trên sai khác ảnh đã căn chỉnh; nó chưa chứng minh thay thế bộ
lọc feature DINOv2/DINOv3 của bài báo trên dữ liệu lớn.
