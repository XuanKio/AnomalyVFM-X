# Quy tắc khu vực PEFT

Đọc `../docs/ARCHITECTURE.md` và `ADDING_NEW_PEFTS.md` trước khi sửa.

- Giữ tương thích state dict của checkpoint hiện có.
- Không đổi tên module/wrapper hoặc target QKV nếu chưa có kế hoạch migration.
- Xác minh chỉ tham số adapter dự kiến có `requires_grad=True` trong training.
- Kiểm tra LoRA/DoRA với ít nhất một forward pass và shape không đổi.
- Rank mặc định của checkpoint demo là 64; đổi rank sẽ làm checkpoint không load
  được nếu không có checkpoint tương ứng.
