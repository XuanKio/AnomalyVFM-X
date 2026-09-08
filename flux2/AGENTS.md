# Quy tắc khu vực flux2

Thư mục này là dependency nguồn thử nghiệm cho sinh dữ liệu, không thuộc demo
nhẹ.

- Tránh sửa trực tiếp; ưu tiên adapter ở `image_gen_models/`.
- Không chạy hoặc tải checkpoint FLUX.2 nếu chưa có yêu cầu rõ và kiểm tra phần
  cứng/dung lượng.
- Giữ nguyên license, pyproject và cấu trúc package vendored.
- Mọi lần cập nhật dependency phải ở commit riêng và có ghi chú tương thích.
