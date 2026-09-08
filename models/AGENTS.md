# Quy tắc khu vực models

Đọc `../docs/ARCHITECTURE.md` trước khi sửa.

- Giữ interface chung: `feature_dim`, `patch_size`, `get_img_transform()`,
  `add_peft()` và `forward()` trả `(summary, features)`.
- Backbone mới phải được thêm vào `BACKBONES` và `FeatureExtractor` trong
  `model.py`, kèm tài liệu kích thước ảnh/patch và nguồn checkpoint.
- Không tự tải backbone lớn để thử nghiệm. Nêu dung lượng và xin phạm vi rõ nếu
  download vượt 500 MB.
- Kiểm tra shape token, `feat_size = image_size // patch_size`, dtype và thiết bị.
- Thay đổi transform có thể làm sai kết quả pretrained; phải ghi rõ và so sánh
  ít nhất một ảnh mẫu trước/sau.
