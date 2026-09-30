# Quyết định nghiên cứu định vị vết lỗi nhỏ

## Kết luận từ dữ liệu hiện có (30/09/2026)

Mục tiêu là tăng khả năng định vị vết nhỏ trên nhiều loại thiết bị mà không
tăng báo nhầm trên ảnh lành. Chưa có biến thể nào chứng minh được điều đó;
checkpoint web tiếp tục là bản CLIP do tác giả phát hành. VisA candle/capsules
là hai lớp có mask thật đã kiểm tra, không đại diện toàn bộ thiết bị.

| Giả thuyết | Kiểm tra độc lập | Quyết định |
| --- | --- | --- |
| Đổi ngưỡng/làm mượt sẽ tìm được lỗi | PCB điểm cao ở sai chỗ; vết kim loại điểm rất thấp | Không dùng làm cải thiện model |
| Chỉnh đầu mask bằng vài ảnh AI đủ tổng quát | Cùng mức báo nhầm, baseline và hard-negative đều nhận 41/100 ảnh lỗi capsules | Giữ riêng checkpoint pilot |
| So patch với ảnh lành ở tầng giữa tốt hơn | 380 ảnh VisA: P-AP decoder gốc 0,396/0,351; mọi tầng riêng thấp hơn cả hai lớp | Không đưa bộ nhớ đặc trưng vào web |
| Nhãn ảnh sinh có thể thiếu vùng lỗi | Mẫu thăm dò 20 cặp có mask 12–22 pixel dù thay đổi thấy rộng hơn | Ưu tiên duyệt nhãn trước huấn luyện lại |

Số liệu và hash nằm trong `docs/EVALUATION.md`,
`outputs/visa_slice_20260930/feature_layer_full/report.json`, và
`outputs/remote_synthetic_sample_20260930/stage1_triage_run2/report.json`.

## So với phương pháp gốc

[AnomalyVFM](https://arxiv.org/html/2601.20524v2) tạo ảnh lành, inpaint lỗi,
rồi lọc bằng sai khác đặc trưng. Mô hình chèn adapter vào các block transformer,
dùng đặc trưng tầng cuối với decoder convolution và loss pixel có trọng số tự
tin. Paper báo hiệu quả cao trên nhiều benchmark, nhưng không chứng minh mọi
vết xước nhỏ hoặc mọi thiết bị được phát hiện.

Điểm thử nghiệm cần thay đổi **trước** khi train model mới là cách xử lý nhãn
thiếu: pixel đổi đáng kể nhưng không nằm trong mask gốc không nên tự động bị
coi là nền. `synthetic_label_triage.py` đưa vùng đó vào hàng đợi duyệt;
`ground_truth/ignore` cho phép loại pixel chưa chắc khỏi loss. Đây là phần bổ
sung có chủ đích cho giám sát nhãn yếu, không thay confidence loss của paper.
Ignore không tạo thêm nhãn dương và tự nó không bảo đảm cải thiện.

## Thí nghiệm tiếp theo cần đạt

1. Duyệt cặp ảnh sinh theo nhóm thiết bị/bề mặt, lưu mask dương đã xác nhận và
   vùng ignore; tách nguồn ảnh sinh khỏi tập kiểm tra ảnh thật.
2. Train hai bản **cùng kiến trúc, seed, số bước, dữ liệu**: loss gốc và loss có
   ignore. Chỉ thêm decoder đa tầng như một thí nghiệm riêng sau khi kiểm tra
   nhãn, để biết cải thiện đến từ đâu.
3. So trên ảnh thật có mask độc lập: P-AP, khả năng bắt từng vết nhỏ, I-AUROC
   và số ảnh lành báo nhầm theo từng lớp, cùng một ngân sách báo nhầm. Công bố
   cả lớp kém đi và thời gian/VRAM; chỉ đổi web khi cải thiện ổn định.

Máy GPU 4 GB chưa chạy được vòng train đầy đủ của paper trong môi trường demo.
Bộ nạp ZIP Hugging Face từ xa chỉ tiết kiệm ổ đĩa, không giảm VRAM. Vì vậy
hiện mới xác minh code loss/loader bằng unit test, chưa có kết quả A/B sau
huấn luyện để tuyên bố phương pháp mới tốt hơn.
