# Hướng dẫn demo CLI và web

## Web demo

```powershell
.\web_demo.cmd
```

Sau khi model load, trình duyệt mở <http://127.0.0.1:7860>. Có thể:

- kéo thả PNG/JPG/JPEG/WEBP;
- chọn một trong ba ảnh mẫu;
- xem score, inference time, peak VRAM, heatmap và overlay.

Dừng bằng `Ctrl+C` trong terminal. Web chỉ chạy local; ảnh không được upload tới
dịch vụ bên ngoài.

## CLI

Một ảnh:

```powershell
.\demo.cmd test.png
```

Nhiều ảnh trong một lần load model:

```powershell
.\demo.cmd demo_images\hazelnut_normal.png `
  demo_images\hazelnut_defective.png `
  demo_images\bottle_defective.png
```

Gọi Python trực tiếp khi môi trường đã activate:

```powershell
python demo_fast.py test.png --size 336 --output-dir outputs
```

Tùy chọn:

| Flag | Ý nghĩa |
| --- | --- |
| `images` | Một hoặc nhiều đường dẫn ảnh |
| `--size 336` | Nhanh, ít VRAM, mặc định demo |
| `--size 672` | Độ phân giải cấu hình CLIP gốc, tốn VRAM hơn |
| `--cpu` | Buộc chạy CPU |
| `--output-dir` | Thư mục ảnh kết quả |

## Ảnh mẫu và số đo tham khảo

| Ảnh | Loại | Score đã đo trên máy Xuân |
| --- | --- | ---: |
| `hazelnut_normal.png` | normal | 0.4375 |
| `hazelnut_defective.png` | defective | 0.5664 |
| `bottle_defective.png` | defective | 0.6641 |

Các số này chỉ kiểm tra pipeline và minh họa so sánh. Không dùng ba mẫu để tuyên
bố accuracy hoặc chọn threshold khoa học.

## Kịch bản trình bày 2–3 phút

1. Nêu bài toán: phát hiện và định vị lỗi chưa thấy trong domain train.
2. Chạy ảnh normal, giải thích score toàn ảnh và heatmap theo vùng.
3. Chạy ảnh defective cùng loại để so sánh score/vị trí đỏ.
4. Chạy chai lỗi để cho thấy zero-shot chuyển sang loại vật khác.
5. Nêu giới hạn: size 336 là bản tối ưu demo; benchmark paper dùng cấu hình lớn
   hơn và nhiều dataset chuẩn.

## Heatmap

Heatmap được scale min-max riêng cho từng ảnh để vùng tương đối dễ nhìn. Vì vậy
một ảnh normal vẫn có vùng đỏ nhất. Khi so sánh normal/anomaly phải nhìn đồng
thời global score, hình dạng vùng chú ý và dữ liệu đánh giá có threshold chuẩn.
