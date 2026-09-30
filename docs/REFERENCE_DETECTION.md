# Định vị vết nhỏ bằng ảnh lành tham chiếu

## Mục đích và phạm vi

Thử nghiệm CLI **Đối chiếu ảnh lành · tìm vết nhỏ** tìm các vùng khác biệt giữa ảnh cần
kiểm tra và ảnh lành cùng sản phẩm, cùng góc chụp. Đây là nhánh thị giác máy tính
cổ điển chạy CPU, có thêm ảnh tham chiếu. Không phải checkpoint CLIP mới,
không huấn luyện lại AnomalyVFM và không có bằng chứng độ chính xác trên mọi
thiết bị. Vùng được đánh dấu là thay đổi cần kiểm tra, chưa xác nhận lỗi chức năng.

| Thành phần | Trách nhiệm |
| --- | --- |
| `demo_reference.py` | `ReferenceConfig`, căn chỉnh, kiểm tra độ khớp, tính residual, mask và vùng liên thông |
| `compare_reference.py` | Runner lưu cấu hình, đối chứng tổng hợp và báo cáo so sánh |
| Web | Đã gỡ thẻ và endpoint `/predict-reference` ngày 30/09/2026; không thuộc phương pháp gốc |
| `requirements-reference.txt` | OpenCV tùy chọn cho môi trường demo hiện có |

## Cài đặt và chạy trên Windows

Cài thêm `opencv-python-headless==4.12.0.88` vào **Python demo 3.10 hiện có**.
Thay đường dẫn ví dụ dưới đây bằng interpreter demo thực sự đang dùng; không
nâng `torch==2.10.0` hoặc `torchvision==0.25.0`. Không cài song song nhiều gói
OpenCV cùng cung cấp module `cv2`.

```powershell
uv pip install --python "C:\duong-dan-moi-truong-demo\Scripts\python.exe" -r requirements-reference.txt
```

Web không còn chức năng này. Dùng runner nghiên cứu riêng dưới đây.

Runner dùng hai tên ảnh do người chạy cung cấp, không kèm ảnh AI trong Git:

```powershell
python compare_reference.py `
  --reference "D:\duong-dan-anh\anh_lanh.png" `
  --image "D:\duong-dan-anh\anh_can_kiem_tra.png" `
  --baseline-dir "D:\duong-dan-ket-qua\tiling_run" `
  --output-dir "outputs\reference_comparison_moi"
```

`--baseline-dir` là tùy chọn, chỉ truyền khi có kết quả `compare_tiling.py` trên
**đúng file ảnh cần kiểm tra**: runner đối chiếu SHA-256 và kích thước mask.
Baseline được đọc từ báo cáo cũ, không chạy lại CLIP trong lệnh này. Chọn thư
mục đầu ra mới hoặc rỗng để giữ lịch sử. Trên Windows, runner chọn Python demo
qua `demo_bootstrap.py` giống web.

Thêm `--synthetic-pair` nếu cả cặp ảnh được tạo bằng AI để báo cáo ghi đúng
nguồn ảnh. Khi không có baseline, cột giữa được ghi rõ chưa cung cấp kết quả
CLIP; runner không suy diễn kết quả trước khi cải thiện.

## Đầu vào và đầu ra

Hai ảnh phải cùng kích thước sau xoay EXIF, tối đa 4.000.000 pixel mỗi ảnh và
mỗi cạnh ít nhất 64 pixel. Ảnh lành cần mô tả cùng sản phẩm, cùng góc chụp và có
đủ chi tiết để căn chỉnh. Không đưa tọa độ vết lỗi vào `detect_changes`.

API web cũ đã bị gỡ; `/predict-reference` trả 404. Detector CLI vẫn trả
metadata, vùng liên thông và map; lỗi căn chỉnh không được diễn giải là ảnh lành.

Runner lưu `manifest.json` trước suy luận, `report.json`, `maps.npz`,
`reference.png`, `input.png`, các `new_*.png`, `before_overlay.png`,
`comparison.png` và `comparison.html`. Manifest ghi hash ảnh, hash code, cấu
hình và đối chứng để truy vết. `maps.npz` giữ score float, mask và vùng hợp lệ.
Màu hồng là pixel mask; khung vàng được suy ra từ vùng liên thông. Ô tham chiếu
cũ, nếu có, chỉ phục vụ báo cáo/phóng to; không đi vào bộ phát hiện.

## Vì sao phương pháp có thể thấy vết xước nhỏ

1. **Căn chỉnh affine bằng ECC:** đưa ảnh lành về tọa độ ảnh cần kiểm tra, giảm
   sai khác do dịch chuyển hoặc xoay nhỏ. ECC được tính trên ảnh thu nhỏ cạnh
   dài tối đa 768; residual được tính lại ở độ phân giải ảnh gốc.
2. **Kiểm tra độ khớp trên vùng chung:** bỏ 1% pixel có sai khác chuẩn hóa lớn
   nhất rồi yêu cầu tương quan trên phần còn lại ít nhất 0,97. Điều này tránh
   để một vết nhỏ tự làm hỏng đánh giá căn chỉnh. Gate này vẫn không chứng minh
   hai ảnh là cùng sản phẩm.
3. **Tách thay đổi ánh sáng chậm:** trừ thành phần residual làm mượt, rồi lọc
   nhiễu nhẹ. Blur chuẩn hóa theo vùng có dữ liệu để pixel đệm đen sau warp
   không lan vào vùng ảnh hợp lệ.
4. **Giảm ảnh hưởng cạnh/hoa văn có sẵn:** chia residual cho nền nhiễu cộng
   thành phần độ biến thiên cục bộ của ảnh lành. Chi tiết mới trên nền ít hoa
   văn có thể nổi bật hơn sai số nội suy ở các cạnh cũ.
5. **Tạo vùng tự động:** ngưỡng residual cố định 3,0, giữ thành phần liên thông
   từ 12 pixel; loại dải mép 16 pixel quanh vùng không khớp. Trả mọi vùng đạt
   điều kiện, không chỉ vùng gần vết đã biết.

Residual là đơn vị riêng của thuật toán, **không phải xác suất**. Ngưỡng 3,0
chưa hiệu chỉnh trên bộ ảnh thực; không so số này trực tiếp với score CLIP
hay ngưỡng minh họa 0,5. Ngưỡng giao diện CLIP không điều khiển nhánh đối chiếu.

Ý tưởng dùng ảnh lành đã căn chỉnh có cơ sở trong nghiên cứu
[RegAD, ECCV 2022](https://arxiv.org/abs/2207.07361): nghiên cứu so sánh đặc
trưng đã đăng ký giữa ảnh kiểm tra và ảnh hỗ trợ lành. Module hiện tại chỉ
lấy cảm hứng từ nguyên lý căn chỉnh, **không triển khai RegAD** và không kế
thừa số benchmark của paper. Chi tiết phép căn chỉnh ECC tham khảo
[tài liệu OpenCV](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html).

## Bằng chứng local và giới hạn

Trên cặp ảnh AI liên quan đã dùng ngày 29/09/2026, baseline CLIP 672 và chia ô
đều có mask rỗng tại ngưỡng minh họa 0,5. Nhánh đối chiếu tìm được vùng tự động
`[1019, 466, 1097, 529]`, diện tích 491 pixel, đỉnh residual tại `[1048, 490]`.
Đây là kết quả thăm dò trên **một cặp ảnh** có thêm thông tin ảnh lành, không
phải so sánh công bằng hai model cùng đầu vào. Dùng báo cáo runner của từng
lần chạy làm nguồn cho số đo thời gian và cấu hình cuối cùng.

Sáu đối chứng lành tổng hợp từ một ảnh (đồng nhất, sáng, dịch chuyển, JPEG,
xoay kèm thay đổi chụp, nhiễu cảm biến) đã không tạo vùng; ảnh tham chiếu lật
gương bị từ chối. Đây là kiểm tra độ ổn định với nhiễu mô phỏng, không phải
sáu sản phẩm thực độc lập và không cho phép tính tỷ lệ báo nhầm thực tế.

Cặp AI có thay đổi ngoài vết xước; chưa có nhãn pixel chuẩn nên chưa báo cáo
IoU, F1, precision, recall hoặc accuracy. Phương pháp có thể bỏ sót thay đổi
rộng do bước khử ánh sáng, lỗi trên hoa văn nhiều chi tiết, lỗi sát mép và
vùng nhỏ hơn 12 pixel. Khác góc nhìn lớn, biến dạng, phản xạ hoặc ảnh lành
không phù hợp có thể gây từ chối hay vùng khác biệt giả. Cần bộ ảnh sản phẩm
thực độc lập, có nhãn, trước khi chọn cấu hình triển khai.

### Kiểm tra vết mảnh trên bốn cặp AI

`evaluate_fine_reference.py` kiểm tra thêm bước mở rộng vùng mạnh sang các pixel
khác biệt yếu **liền kề trên toàn ảnh**. Vùng sửa trong prompt sinh ảnh không đi vào
bộ phát hiện, chỉ dùng đối chiếu vị trí khi viết báo cáo. Chạy với Python demo:

```powershell
python evaluate_fine_reference.py `
  --source outputs/synthetic_pilot_20260930 `
  --pilot outputs/synthetic_pilot_20260930/head_672_hardneg `
  --specification pilot_specification.json `
  --output outputs/synthetic_pilot_20260930/fine_reference_new
```

Trên cặp PCB và kim loại, mask đối chiếu ban đầu có 491 và 162 pixel; mở rộng
liền kề được 3.061 và 945 pixel. Tất cả pixel được đánh dấu đều ở trong vùng
generator được yêu cầu sửa. Bốn ảnh lành cùng 5 phép biến đổi mô phỏng cho mỗi
ảnh không tạo mask (0/24); đây **không phải** tỷ lệ báo nhầm trên ảnh thật.
Mask kim loại vẫn chưa phủ hết đường xước. Cách này đòi ảnh lành của **chính
thiết bị, cùng góc chụp**; không thay thế detector ảnh đơn hoặc chứng minh
chất lượng tổng quát.

## Checklist xác minh

- [ ] Đúng Python demo và OpenCV tùy chọn; nhánh CLIP cũ vẫn khởi động.
- [ ] Hai ảnh đúng loại/góc chụp, cùng kích thước; kết quả hiển thị đủ mọi vùng.
- [ ] Đối chứng lành không tạo vùng; ảnh tham chiếu sai bị từ chối và không bị gọi là lành.
- [ ] Runner lưu đúng metadata, maps.npz và các PNG; web không có nhánh đối chiếu.
- [ ] Đọc manifest/report, mask gốc và vùng bị loại ở mép; dùng dữ liệu có nhãn độc lập trước khi tuyên bố độ chính xác.
