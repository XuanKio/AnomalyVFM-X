# Dataset và loader

## Hai loại dữ liệu trong repository

`demo_images/` chỉ chứa vài ảnh MVTec AD nhỏ để kiểm tra/demo. Đây không phải
evaluation set và không được dùng báo cáo metric.

Dataset benchmark đầy đủ đặt ngoài Git trong `data/` theo
`datasets/dataset.py`. `DATASET_ROOT` mặc định là `./data/`.

## Dataset được đăng ký

### Công nghiệp — paper chính

- `mvtec_ad`, `visa`, `real_iad`, `mpdd`, `btad`;
- `ksdd`, `ksdd2`, `dagm`, `dtd`.

### Y tế

- image-level: `headct`, `brainmri`, `br35h`;
- pixel-level: `isic`, `cvc_colondb`, `cvc_clinicdb`, `kvasir`, `endo`,
  `thyro`.

### Ngoài paper chính

- 2D: `real_iad_variety`, `goods_ad`, `rsdd`;
- RGB từ benchmark 3D: `mvtec_3d`, `eyecandies`, `real_iad_3d`.

## Cấu hình đường dẫn

Mỗi entry `DatasetProperties` chứa:

```text
name, path, init_class, class_names
```

Ví dụ MVTec AD được tìm tại `./data/mvtec`. Không hard-code đường dẫn tuyệt đối
của một máy vào loader. Nếu cần root khác, ưu tiên thêm cấu hình CLI/env chung
thay vì sửa từng class.

## Contract loader

Trước khi thay đổi, đọc cách `test.py` unpack batch của DataLoader. Các loader
phải thống nhất:

- ảnh đã transform về tensor đúng shape/dtype;
- label image-level normal/anomaly;
- pixel mask cho dataset có ground truth;
- class name và original path để ghi kết quả;
- normal mask là zero/rỗng đúng quy ước hiện tại.

## Thêm dataset mới

1. Đọc `datasets/ADDING_NEW_DATASET.md`.
2. Tạo loader trong `datasets/<name>.py`.
3. Import loader trong `datasets/dataset.py`.
4. Thêm `DatasetProperties` với path/class list.
5. Tạo fixture tối thiểu: một normal và một anomaly/mask nếu giấy phép cho phép.
6. Kiểm tra batch shape và path, sau đó chạy `test.py -d <name>` khi có data.
7. Cập nhật tài liệu này và README nếu dataset là tính năng người dùng.

## Dung lượng và giấy phép

- Không commit full dataset, archive tải về hoặc extracted data.
- Kiểm tra license từng dataset trước khi chia sẻ ảnh mẫu.
- MVTec sample trong `demo_images/` dùng cho học thuật phi thương mại và có
  attribution trong `demo_images/README.md`.
- Ghi dataset version/hash khi tạo kết quả so sánh.
