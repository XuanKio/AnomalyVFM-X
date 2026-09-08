# Quy tắc khu vực results

- Các bảng/kết quả có sẵn được xem là dữ liệu tham chiếu của nghiên cứu.
- Không sửa số liệu thủ công để khớp kỳ vọng.
- Kết quả local mới phải ghi model, checkpoint, image size, seed, dataset version
  và command tạo ra kết quả.
- Artifact ảnh lớn và log chạy thử để trong thư mục ignored `outputs/` hoặc
  `experiments/`, không commit vào `results/` nếu chưa được duyệt.
