# Xử lý lỗi thường gặp

## `uv is not installed`

Cài `uv` theo link trong `SETUP.md`, mở terminal mới và chạy `uv --version`.

## Demo báo môi trường chưa cài

Chạy:

```powershell
.\setup_demo.cmd
```

Không gọi `python web_demo.py` bằng Python hệ thống 3.12/3.13.

## `torch.cuda.is_available() == False`

- kiểm tra `nvidia-smi`;
- cập nhật NVIDIA driver;
- xác minh setup đã chọn index `cu128`, không phải `cpu`;
- kiểm tra bằng Python trong runtime:

```powershell
& "$env:LOCALAPPDATA\AnomalyVFM-X\venv\Scripts\python.exe" `
  -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Nếu không có NVIDIA, chạy CPU vẫn được nhưng chậm.

## CUDA out of memory

- đóng game/app dùng GPU;
- giữ `--size 336`;
- xử lý từng ảnh hoặc dùng web (server khóa request tuần tự);
- dùng `--cpu` làm phương án cuối.

Không giảm dưới 336 nếu chưa cập nhật code/đánh giá vì size phải chia hết patch 14.

## Offline mode không tìm thấy model

`demo_env.cmd` cố ý bật offline cho buổi trình bày. Nếu cache bị xóa, chạy lại
`setup_demo.cmd` khi có mạng để tải model, rồi thử lại.

## Lần đầu tải rất chậm hoặc timeout

Checkpoint và backbone vài GB. `hf_xet` đã nằm trong setup để tải ổn định hơn.
Giữ terminal mở; lần chạy sau dùng cache. Kiểm tra disk trống trước khi retry.

## Port 7860 đã được sử dụng

Tìm/dừng phiên `web_demo.py` cũ bằng `Ctrl+C`. Hoặc chạy trực tiếp port khác:

```powershell
& "$env:LOCALAPPDATA\AnomalyVFM-X\venv\Scripts\python.exe" `
  web_demo.py --port 7861
```

## Trang mở nhưng phân tích lỗi

- dùng ảnh PNG/JPG/WEBP hợp lệ;
- thử `demo_images/hazelnut_normal.png` để tách lỗi input khỏi model;
- xem thông báo terminal;
- chạy CLI để lấy traceback rõ hơn.

## Normal vẫn có vùng đỏ

Visualization min-max riêng từng ảnh, nên mọi heatmap đều có một vùng nóng nhất.
Đây không tự động nghĩa là defective. Dùng score và threshold đã hiệu chỉnh trên
dataset, không kết luận chỉ từ màu.

## Muốn giải phóng dung lượng

Runtime nằm tại `%LOCALAPPDATA%\AnomalyVFM-X`. Đóng server, xác minh đúng đường
dẫn rồi xóa thư mục này. Repository và code vẫn còn, nhưng lần chạy sau phải setup
và tải model lại.
