# Hướng dẫn dành cho agent

## Mục tiêu repository

Repository `AnomalyVFM-X` là bản quản lý độc lập của Xuân cho môn học máy. Hai
luồng sử dụng được tách rõ:

1. **Demo nhẹ**: dự đoán ảnh đơn bằng checkpoint CLIP, ưu tiên Windows và GPU
   laptop 4 GB.
2. **Nghiên cứu đầy đủ**: sinh dữ liệu, huấn luyện và đánh giá theo mã nguồn
   AnomalyVFM. Luồng này cần dataset và phần cứng lớn, không được tự động chạy.

Remote Git duy nhất dùng để quản lý là
`git@github.com:XuanKio/AnomalyVFM-X.git`. Không tự thêm upstream remote. Các
đường dẫn paper/model trong tài liệu chỉ nhằm trích dẫn học thuật.

## Bản đồ trách nhiệm

| Khu vực | Tệp/thư mục chính | Tài liệu bắt buộc đọc |
| --- | --- | --- |
| Demo CLI | `demo_fast.py`, `demo.cmd` | `docs/DEMO.md` |
| Demo web | `web_demo.py`, `web_demo.cmd` | `docs/DEMO.md` |
| Cài đặt | `setup_demo.cmd`, `demo_env.cmd`, `requirements.txt` | `docs/SETUP.md` |
| Kiến trúc/model | `hf_model.py`, `decoder.py`, `models/` | `docs/ARCHITECTURE.md`, `models/AGENTS.md` |
| PEFT | `peft_local/` | `docs/ARCHITECTURE.md`, `peft_local/AGENTS.md` |
| Dataset/evaluation loader | `datasets/`, `aux_dataset.py` | `docs/DATASETS.md`, `datasets/AGENTS.md` |
| Sinh dữ liệu | `generate_dataset.py`, `image_gen_models/`, `flux/`, `flux2/` | `docs/SYNTHETIC_DATA.md` và AGENTS cục bộ |
| Huấn luyện | `train.py`, `utils.py` | `docs/TRAINING.md` |
| Đánh giá | `test.py`, `predict_single_image.py`, `results/` | `docs/EVALUATION.md` |
| Quy trình phát triển | toàn repo | `docs/DEVELOPMENT.md`, `docs/GIT_WORKFLOW.md` |

AGENTS ở thư mục con bổ sung quy tắc cho phạm vi đó và có hiệu lực cùng file
này.

## Quy tắc vận hành

- Mặc định chỉ làm việc với luồng demo nhẹ. Không tải full dataset, checkpoint
  khác, FLUX, DINOv3 repo hoặc chạy training nếu Xuân chưa yêu cầu rõ.
- Không commit môi trường Python, cache model, checkpoint, dataset hay output.
- Giữ Python demo ở phiên bản 3.10 và cặp `torch==2.10.0`,
  `torchvision==0.25.0` cho đến khi có lượt nâng cấp được kiểm thử riêng.
- `demo_fast.py` và `web_demo.py` phải cùng dùng checkpoint
  `MaticFuc/anomalyvfm_clip`, input mặc định 336 và BF16 trên GPU hỗ trợ.
- Kích thước 336 là chế độ demo tiết kiệm bộ nhớ, không phải cấu hình benchmark
  trong paper. Tài liệu phải nói rõ khi báo cáo kết quả.
- Không diễn giải anomaly score là xác suất. Không gán nhãn good/bad bằng ngưỡng
  tùy ý trong UI.
- Giữ server web bind vào `127.0.0.1` theo mặc định. Không mở `0.0.0.0`, tạo
  public tunnel hoặc upload ảnh ra dịch vụ ngoài nếu chưa được yêu cầu.
- Mọi thay đổi hành vi phải cập nhật tài liệu tương ứng và mục liên quan trong
  README nếu ảnh hưởng người dùng.

## Quy trình trước khi sửa

1. Đọc tài liệu của khu vực trong bảng trên.
2. Chạy `git status --short --branch`; không ghi đè thay đổi chưa rõ nguồn gốc.
3. Xác định thay đổi thuộc demo nhẹ hay nghiên cứu đầy đủ.
4. Ước lượng tác động dung lượng/VRAM trước mọi download lớn hơn 500 MB.

## Kiểm thử tối thiểu

Sau thay đổi Python demo:

```powershell
python -m py_compile demo_fast.py web_demo.py
.\demo.cmd demo_images\hazelnut_normal.png
```

Sau thay đổi web, kiểm tra thêm:

- `GET http://127.0.0.1:7860/` trả HTTP 200;
- chọn được ba ảnh mẫu;
- `POST /predict` trả score, thời gian, VRAM và ảnh PNG base64;
- `Ctrl+C` dừng server và giải phóng cổng 7860.

Sau thay đổi model/dataset/training, làm theo checklist riêng trong tài liệu khu
vực. Nếu thiếu dataset hoặc GPU, ghi rõ phần chưa thể kiểm thử thay vì tải hoặc
chạy ngầm.

## Git và phát hành

- Nhánh chính: `main`.
- Remote quản lý: `origin` → `git@github.com:XuanKio/AnomalyVFM-X.git`.
- Commit phải nhỏ, mô tả đúng phạm vi và không chứa artifact lớn.
- Không force-push sau khi repository đã có cộng tác viên nếu Xuân chưa yêu cầu.
- Trước khi push: chạy kiểm thử liên quan, `git diff --check`, kiểm tra file lớn
  và xác nhận không có secret/token.
