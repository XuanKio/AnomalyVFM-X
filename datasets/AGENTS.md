# Quy tắc khu vực datasets

Đọc `../docs/DATASETS.md` và `ADDING_NEW_DATASET.md` trước khi sửa loader.

- Không commit ảnh/dataset thật vào repo.
- Giữ đường dẫn gốc tập trung ở `DATASET_ROOT` và `DATASET_RESOURCES`.
- Loader phải trả nhất quán image, mask, label, class và path theo contract hiện
  có của `test.py`.
- Normal sample phải có mask rỗng/zero đúng shape; không tạo nhãn từ tên file nếu
  dataset đã có metadata chính thức.
- Dataset mới phải có fixture nhỏ hoặc kiểm thử shape không phụ thuộc full data.
- Không đổi class list hay path mặc định mà không cập nhật `docs/DATASETS.md`.
