# Hướng dẫn demo CLI và web

## Web demo

Mở `web_demo.py` trong VS Code và bấm **Run Python File**. File tự chuyển sang
Python demo 3.10 đã cài, đặt cache offline và mở trình duyệt; kể cả khi VS Code
đang chọn Python 3.13. Không cần activate môi trường hoặc nhập biến môi trường.

```powershell
python .\web_demo.py
```

Sau khi cài đặt, có thể chạy file này từ terminal tích hợp của VS Code
hoặc nháy đúp `web_demo.cmd` trong File Explorer. Launcher tự chọn Python,
cache model và chế độ offline; không cần activate môi trường thủ công.

Cũng có thể dùng `web_demo.cmd` hoặc cấu hình **AnomalyVFM Web Demo** với `F5`.
`demo_bootstrap.py` xử lý chọn Python trước khi import thư viện. Model đọc
`config.json` và checkpoint từ cache local, không tự tải lại. Nếu môi trường
hoặc checkpoint bị thiếu, chạy `setup_demo.cmd` để cài/khôi phục trước.

Launcher kiểm tra import thư viện bằng từng Python: interpreter đang chạy,
môi trường cũ `%USERPROFILE%\.venvs\anomalyvfm-demo`, rồi môi trường do bộ cài
tạo. Đường dẫn nào kiểm tra thành công sẽ được giữ nguyên, không đổi qua junction.
Nếu tất cả đều hỏng, terminal liệt kê lỗi từng môi trường để sửa đúng nơi.

Cache model được kiểm tra riêng: ưu tiên `%USERPROFILE%\.cache\huggingface-anomalyvfm\hub`,
sau đó `%LOCALAPPDATA%\AnomalyVFM-X\huggingface\hub`, rồi cache Hugging Face mặc định.
Chỉ chọn nơi có đủ `config.json` và trọng số của cả CLIP lẫn AnomalyVFM theo
revision đã cache. Terminal in `Demo model cache:` để xác định nơi thực sự dùng.
Không cần junction trong AppData để dùng được cache cũ.

Sau khi model load, trình duyệt mở <http://127.0.0.1:7860>. Có thể:

- kéo thả PNG/JPG/JPEG/WEBP;
- chọn một trong ba ảnh mẫu;
- chọn **Nhẹ · phương pháp gốc của bài báo**; **Chi tiết** dành riêng cho phương
  pháp mới sau chưng cất, hiện bị khóa vì chưa có checkpoint hoàn tất;
- dùng cấu hình CLIP gốc 672 × 672, không có lựa chọn 336 trên web;
- xem score, inference time, peak VRAM và bốn ảnh riêng: ảnh gốc / heatmap /
  overlay / mặt nạ dự đoán;
- chọn ngưỡng từ 0 đến 1 rồi bấm phân tích lại để cập nhật mặt nạ;
- lưu bảng bốn ảnh, mặt nạ PNG, bản đồ preview hoặc **pred.png gốc 96×96**.

Ngưỡng 0,50 là giá trị minh họa **chưa hiệu chuẩn**, không dùng kết luận sản phẩm
good/bad. Vùng trắng là điểm dự đoán vượt ngưỡng, không phải đáp án ground truth.
Mặt nạ toàn đen chỉ có nghĩa không điểm nào vượt ngưỡng, không chứng minh ảnh lành.
Ảnh chỉ được chuyển RGB theo script gốc, không tự xoay EXIF. Preview giới hạn
cạnh dài 1600 pixel; inference luôn 672. Upload tối đa 25 MB và 25 triệu pixel.
CPU có thể chạy qua `python web_demo.py --cpu`; đây là fallback FP32, không
phải môi trường CUDA BF16 đã kiểm tra parity.

Dừng bằng `Ctrl+C` trong terminal. Web chỉ chạy local; ảnh không được upload tới
dịch vụ bên ngoài.

`paper_inference.py` thực hiện luồng **ảnh đơn CLIP** theo
[predict_single_image.py của tác giả](https://github.com/MaticFuc/AnomalyVFM/blob/main/predict_single_image.py):
trọng số FP32, transform backbone gốc, autocast CUDA BF16, decoder logits →
AvgPool2d(5,1,2) ngoài autocast → chuyển float → sigmoid. Không gọi HF forward
vì wrapper đó đã sigmoid mask. Checkpoint vẫn là `MaticFuc/anomalyvfm_clip`
trong cache; không thay decoder/backbone/DoRA/predictor, không train lại.

`native_mask` là PNG 96×96, nhân 255 rồi ép uint8 không làm tròn, như `pred.png`
của tác giả. `raw_mask` là bản đồ nội suy bilinear theo kích thước preview để
hiển thị; không phải bản đồ native. Ngưỡng người dùng chỉ tạo mask nhị phân
hiển thị, không đổi score hoặc native map. Script đánh giá batch `test.py` có
thứ tự hậu xử lý khác; phạm vi khôi phục ở đây là **script dự đoán ảnh đơn**,
không khẳng định tái lập toàn bộ benchmark hoặc kết quả RADIO.

`processing_modes.py` định tuyến `mode=light|detailed`. Chi tiết nằm riêng ở
`distilled_inference.py`, trả 503 tới khi có student đã chưng cất và kiểm chứng.
Thiếu `mode` mặc định là light; size khác 672 trả 400. Không có UI hoặc endpoint
`/predict-reference` nữa (404); nghiên cứu đối chiếu chỉ còn ở CLI riêng.

Kiểm tra parity GPU (dùng Python demo):

```powershell
python verify_paper_parity.py --output-dir outputs/paper_parity_moi
```

Runner thực thi nguyên các câu lệnh transform/suy luận/hậu xử lý trích bằng
AST từ `predict_single_image.py`, so score, map float và pixel PNG trên ba ảnh
mẫu. Hai nhánh dùng cùng trọng số HF đã cache; không xác minh một file PKL
độc lập. Lưu report, hash nguồn, map float và cặp ảnh kết quả.

Kiểm tra cũ `outputs/mode_separation_20260930/regression.json` chỉ chứng minh
bảo toàn **demo tùy biến trước đây**, không phải parity với script tác giả.

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

`demo_fast.py` và `web_demo.py` tự chọn cache đầy đủ theo thứ tự ở trên
khi chạy trực tiếp. CLI cần Python demo 3.10;
riêng `web_demo.py` tự chuyển interpreter trên Windows.

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

Các số ở bảng trên là CLI nhẹ 336 cũ. **Web tùy biến cũ, trước khi khôi phục
script tác giả**, kiểm tra ngày 10/09/2026 trên RTX
3050 Laptop 4 GB dùng khoảng 0,78 GiB VRAM, khoảng 1,0 giây/ảnh sau lần đầu.
Score tương ứng normal / hazelnut defective / bottle defective: 0,4082 / 0,4746 /
0,5391. Tại ngưỡng 0,50, ảnh hạt phỉ lành vẫn có vùng trắng, ảnh hạt phỉ lỗi lại
có mặt nạ rỗng. Đây là giới hạn đo được, không che đi bằng đổi màu hay chọn
ngưỡng riêng cho từng mẫu. Bộ ba ảnh không có mask GT nên chưa tính IoU/F1/AUROC.

Sau khôi phục ngày 30/09/2026: ba score lần lượt là 0,408203125 / 0,474609375 /
0,54296875; peak VRAM khoảng 1,92 GiB. Cả score, map float và pixel PNG native
trùng code ảnh đơn tác giả khi dùng cùng checkpoint HF; max_abs_error=0 trên
cả ba ảnh. Bằng chứng: `outputs/paper_restore_20260930/parity/report.json`.

## Kịch bản trình bày 2–3 phút

1. Nêu bài toán: phát hiện và định vị lỗi chưa thấy trong domain train.
2. Chạy ảnh normal, giải thích score toàn ảnh và heatmap theo vùng.
3. Chạy ảnh defective cùng loại để so sánh score/vị trí đỏ.
4. Chạy chai lỗi để cho thấy zero-shot chuyển sang loại vật khác.
5. Nêu giới hạn: web CLIP không phải RADIO trong paper; mask là dự đoán theo
   ngưỡng chưa hiệu chuẩn, không phải GT. Cấu hình lớn hơn không đảm bảo mọi ảnh tốt hơn.

## Heatmap

Web dùng thang màu cố định 0–1 cho mọi ảnh. Logits từ decoder được làm mượt
trung bình 5×5 rồi chuyển float và sigmoid, theo `predict_single_image.py`.
Sau khi giữ bản đồ native, nội suy bilinear lên ảnh preview. Không sửa
decoder/trọng số, không xóa vùng dự đoán để
ép khớp ảnh mẫu. Score toàn ảnh là nhánh riêng, không bị thay đổi bởi ngưỡng.

CLI `demo_fast.py` vẫn giữ hiển thị min-max cũ để đối chiếu; không dùng màu giữa
hai chế độ làm bằng chứng chất lượng. Điểm không phải xác suất và mask không phải GT.

## Checklist xác minh web

Khởi động web trước, rồi trong terminal khác:

```powershell
python -m unittest discover -s tests
python verify_web_demo.py --output-dir outputs/web_verification
```

Kiểm tra HTTP 200, ba ảnh mẫu, bốn PNG trả về, ngưỡng/chế độ đúng và không có NaN.
`report.json` cùng ảnh được lưu vào output. Đây là smoke test, không phải benchmark.

## So sánh toàn ảnh và chia ô chồng lấn

`compare_tiling.py` chạy hai phương pháp trên đúng một ảnh bằng checkpoint demo
offline. `demo_tiling.py` tạo các ô phủ kín ảnh, chạy tuần tự trên GPU và ghép
bản đồ trên CPU. Không huấn luyện lại; không đổi mặc định CLI cũ hoặc web.

```powershell
python compare_tiling.py duong_dan_anh.png --output-dir outputs/tiling_comparison
```

Windows tự chọn Python demo giống `web_demo.py`. Đầu vào là ảnh RGB hoặc ảnh
chuyển được sang RGB; giới hạn CLI là 25 triệu pixel. Đầu ra gồm ảnh nguồn,
heatmap/overlay/mask cho baseline, tiled và fused, `maps.npz` chứa map float,
`run_manifest.json` trước inference, `report.json`, `comparison.png` và
`comparison.html` có thanh kéo trước/sau. Mở HTML bằng trình duyệt.

Cấu hình mặc định: đầu vào model 672, ô nguồn 672, overlap 25%, làm mượt 5×5
cho cả hai nhánh, bản đồ cuối = 25% toàn ảnh + 75% bản đồ ô đã ghép. Cửa sổ
ghép có trọng số dương tới tận mép, không để vùng ảnh không được phủ. Cùng
thang màu 0–1, cùng ngưỡng minh họa 0,5. Điểm toàn ảnh giữ nguyên nhánh gốc;
không diễn giải nó thành điểm phân loại mới của phương pháp ghép.

`--protocol path.json` cho phép cố định cấu hình trước chạy; JSON nhận
`input_size`, `tile_size`, `overlap`, `smoothing_kernel`, `global_weight`,
`display_threshold`. `reference_box_xyxy: [x0,y0,x1,y1]` là tùy chọn chỉ dùng
cho báo cáo, tọa độ trên ảnh sau EXIF transpose, cạnh phải/dưới loại trừ. Không
đưa ô này vào inference hoặc chọn tile. Nếu không có ô tham chiếu thì bỏ các
số đo trong/ngoài ô và phần phóng to dùng cả ảnh.

Mặc định warm-up một lượt rồi đo 3 lượt (`--repeats`), báo trung vị. Thời gian
gồm transform/inference/chuyển map về CPU/ghép, không gồm nạp model, ghi PNG
hoặc giao diện. Phương pháp sau phải tính cả lượt toàn ảnh và mọi lượt tile.
`--cpu` chạy trên CPU. Không ghi đè một thư mục đã có `report.json`; chọn output
mới khi đổi cấu hình để giữ lịch sử đối chiếu.

Giới hạn: ngưỡng và trọng số chưa hiệu chỉnh; ảnh AI không tự có ground truth.
Ô tham chiếu thô không dùng để tính IoU/F1/accuracy; số pixel ngoài ô không đồng
nghĩa false positives đã được xác minh. Một ảnh chỉ minh họa tín hiệu và chi phí.
Giữ ảnh đầu tiên, không chọn lại ảnh hoặc chỉnh tham số sau khi thấy kết quả rồi
tuyên bố đó là phép so sánh độc lập. Chia ô có thể chậm hơn và vẫn báo nhầm;
ảnh nguồn mờ/nhỏ không thể được khôi phục chi tiết bằng chia ô.

Checklist: chạy `python -m unittest discover -s tests`, mở HTML kiểm tra hai
ảnh cùng tọa độ và thang màu, đối chiếu SHA-256/cấu hình trong manifest và xem
metadata về số ô/thời gian/bộ nhớ. Phải có dataset với mask chuẩn riêng trước
khi báo cáo cải thiện độ chính xác.

## Thử nghiệm đối chiếu ảnh lành (CLI riêng)

Thẻ đối chiếu và `/predict-reference` đã bị gỡ khỏi web. Các script nghiên cứu
cũ còn ở `compare_reference.py` và `demo_reference.py`, không được gọi bởi
chế độ Nhẹ hoặc Chi tiết. Xem [REFERENCE_DETECTION.md](REFERENCE_DETECTION.md)
để chạy lại thí nghiệm có thêm ảnh lành; đây không phải phương pháp ảnh đơn gốc.
