# Quy tắc khu vực flux

Thư mục này là dependency nguồn phục vụ sinh dữ liệu, không thuộc demo nhẹ.

- Tránh sửa nếu thay đổi có thể thực hiện ở adapter trong `image_gen_models/`.
- Không chạy download model hoặc demo FLUX trên máy 4 GB VRAM.
- Giữ nguyên license và cấu trúc package vendored.
- Nếu đồng bộ phiên bản, thực hiện trong commit riêng và ghi nguồn/version trong
  `../docs/SYNTHETIC_DATA.md`.
