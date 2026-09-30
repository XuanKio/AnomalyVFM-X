# Cài đặt và thiết lập môi trường

## Cấu hình demo đề nghị

- Windows 10/11 64-bit;
- Git và `uv` trong PATH;
- 8 GB dung lượng trống;
- RAM từ 8 GB, khuyến nghị 16 GB;
- NVIDIA GPU từ 4 GB VRAM, hoặc CPU nếu chấp nhận chạy chậm.

Không cần cài CUDA Toolkit riêng cho demo; wheel PyTorch CUDA chứa runtime cần
thiết. NVIDIA driver vẫn phải đủ mới cho CUDA 12.8.

## Cài tự động

```powershell
git clone git@github.com:XuanKio/AnomalyVFM-X.git
cd AnomalyVFM-X
.\setup_demo.cmd
```

Script thực hiện:

1. kiểm tra `uv`;
2. tạo Python 3.10 tại `%LOCALAPPDATA%\AnomalyVFM-X\venv`;
3. phát hiện NVIDIA để chọn PyTorch CUDA 12.8 hoặc CPU;
4. cài dependency inference tối thiểu;
5. tải checkpoint CLIP vào `%LOCALAPPDATA%\AnomalyVFM-X\huggingface`;
6. chạy smoke test với `test.png`.

Lần cài đầu tải khoảng 6–8 GB và có thể mất vài phút. Không đóng terminal khi
file PyTorch/checkpoint đang tải.

## Cài `uv`

Làm theo hướng dẫn chính thức tại
<https://docs.astral.sh/uv/getting-started/installation/>. Sau khi cài, đóng và
mở lại VS Code/terminal rồi kiểm tra:

```powershell
uv --version
```

## Thiết lập VS Code

1. Mở folder repository.
2. `Ctrl+Shift+P` → **Python: Select Interpreter**.
3. Chọn/nhập:

```text
%LOCALAPPDATA%\AnomalyVFM-X\venv\Scripts\python.exe
```

Sau khi cài, mở `web_demo.py` và bấm **Run Python File**. File tự chuyển sang
Python demo đã cài trên Windows, kể cả khi VS Code chọn Python 3.13, rồi dùng
cache offline. Không cần đặt biến môi trường; `web_demo.cmd` vẫn dùng được.
Web hiện mặc định CLIP 672, có lựa chọn nhẹ 336 trong giao diện. Không tải thêm
checkpoint; mặt nạ trắng/đen là dự đoán theo ngưỡng chưa hiệu chuẩn. Chi tiết
và giới hạn được ghi ở [DEMO.md](DEMO.md).
Nếu máy đã cài bản cũ, cache đầy đủ tại
`%USERPROFILE%\.cache\huggingface-anomalyvfm\hub` được tự nhận mà không cần
liên kết sang AppData. Terminal in đường dẫn cache sau khi kiểm tra đủ hai model.

## Kiểm tra cài đặt

```powershell
.\demo.cmd demo_images\hazelnut_normal.png
```

Kỳ vọng terminal in:

- `Loading MaticFuc/anomalyvfm_clip on cuda` hoặc `cpu`;
- `Model ready`;
- score, inference time, VRAM và đường dẫn output.

## Môi trường nghiên cứu đầy đủ

Chỉ dùng khi cần train/generate/evaluate:

```powershell
uv venv .venv-full --python 3.10
uv pip install --python .venv-full\Scripts\python.exe -r requirements.txt
uv pip install --python .venv-full\Scripts\python.exe -e .\flux
uv pip install --python .venv-full\Scripts\python.exe -e .\flux2
```

Luồng đầy đủ có thể phát sinh yêu cầu compiler, authentication và model/dataset
rất lớn. Không trộn môi trường full với môi trường demo.

## Gỡ cài đặt demo

Đóng web server trước, sau đó xóa thư mục runtime:

```powershell
Remove-Item -LiteralPath "$env:LOCALAPPDATA\AnomalyVFM-X" -Recurse
```

Thao tác này xóa môi trường và model cache nhưng không xóa repository. Kiểm tra
đường dẫn trước khi xác nhận lệnh.
