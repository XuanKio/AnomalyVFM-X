# Kiến trúc và luồng dữ liệu

## Luồng inference

```text
PIL image
   │ resize + normalize theo backbone
   ▼
Vision Foundation Model + PEFT adapters
   ├── summary token ───────────────► SimplePredictor ─► anomaly score
   └── patch features ─► reshape 2D ► SimpleDecoder   ─► anomaly mask
```

`FeatureExtractor` trong `models/model.py` chọn adapter backbone theo enum
`BACKBONES`. Mỗi adapter phải cung cấp:

- `feature_dim`: chiều channel của feature;
- `patch_size`: kích thước patch;
- `get_img_transform()`: resize/normalize đúng backbone;
- `add_peft(rank, peft_type)`: chèn PEFT;
- `forward(x) -> (summary, features)`.

`features` có dạng `[batch, tokens, feature_dim]`. Wrapper đổi thành
`[batch, feature_dim, Hpatch, Wpatch]`, trong đó `Hpatch = image_size /
patch_size`, rồi đưa qua decoder convolution.

## Wrapper Hugging Face

`AnomalyVFM` trong `hf_model.py`:

1. đọc `AnomalyVFMConfig` từ model repository;
2. khởi tạo backbone và adapter đúng rank/type;
3. tạo `SimpleDecoder` và `SimplePredictor`;
4. nạp `model.safetensors`;
5. chạy forward dưới autocast BF16 và `no_grad`.

Checkpoint demo CLIP có cấu hình gốc image size 672, DoRA rank 64. CLI demo cũ cập
nhật `model.model.H` và `model.feat_size` về 336/patch size sau khi load. Trọng
số không đổi, nhưng độ phân giải khác benchmark nên kết quả chỉ dùng minh họa.

## Decoder và predictor

`SimpleDecoder` gồm bottleneck convolution, GroupNorm/ReLU, một hoặc nhiều block
upsample bilinear và convolution 1×1 cuối. Nó trả `(mask, confidence)`. Wrapper
Hugging Face hiện chỉ trả mask sigmoid cho người dùng.

`SimplePredictor` là linear head trên summary token, sau đó sigmoid để tạo score.
RADIO ghép ba summary feature nên predictor dùng `3 * feature_dim`; backbone còn
lại dùng `feature_dim`.

## PEFT

`peft_local/` bọc projection/attention layer bằng LoRA, DoRA hoặc VPT. Rank và
loại adapter là một phần của checkpoint contract. Thay đổi tên module, target
layer hoặc rank sẽ làm state dict không tương thích nếu không migration.

## Web demo

```text
Browser ──multipart POST /predict──► ThreadingHTTPServer
                                      │
                                      ├─ PIL decode
                                      ├─ ProcessingModes: light → PaperRuntime
                                      │                   detailed → DistilledRuntime
                                      ├─ heatmap + overlay
Browser ◄──── JSON + PNG base64 ──────┘
```

Model được load một lần khi server khởi động; khóa bảo vệ inference. Web Nhẹ
cố định CLIP 672. `paper_inference.predict_single_image` dùng submodules đã nạp
trọng số FP32, transform gốc, autocast CUDA BF16, decoder logits → pool5 →
float sigmoid, đúng `predict_single_image.py` của tác giả. Pool nằm ngoài
autocast. Không gọi `hf_model.forward` vì nó sigmoid trước khi trả mask.

Bản đồ native 96×96 được trả ở `native_mask` (uint8 truncate). Sau đó mới
nội suy bilinear để tạo preview và mask nhị phân trong `demo_visuals.py`.
Threshold không đổi bản đồ native/score. `raw_mask` vẫn là PNG preview.
Không EXIF transpose trong nhánh gốc. CPU là fallback FP32 riêng về số học.
`verify_paper_parity.py` so bằng các câu lệnh trích nguyên từ script tác giả,
cùng checkpoint HF, không phải xác minh trọng số PKL độc lập.

`web_demo.Runtime` là alias của `PaperRuntime`. `distilled_inference.py` giữ
phương pháp mới tách riêng, hiện chưa có student nên trả 503. CLI `demo_fast.py`
giữ demo cũ 336 và HF forward, không dùng làm đối chứng script ảnh đơn gốc.

## Contract cần giữ ổn định

- CLI mặc định: `demo_fast.py test.png`, size 336.
- Output CLI: `outputs/<stem>_result.png`.
- Web `POST /predict`: multipart field tên `image`.
- Web nhận thêm `size` (chỉ 672; giá trị khác trả 400) và `threshold` (0–1, mặc định minh họa 0,50).
- `mode=light` mặc định gọi nhánh paper; `mode=detailed` hiện trả 503 chưa sẵn
  sàng. Mode không hợp lệ trả 400; không tự rơi về phương pháp khác.
- JSON giữ `score`, `seconds`, `vram_gb`, `image` (composite bốn cột); thêm
  `panels`, `raw_mask`, `score_range`, `mask_area_percent`, `threshold`,
  `native_mask`, `native_mask_shape`, `inference_protocol`,
  `threshold_calibrated=false`, `model_id`, `input_size`, `device`, `smoothing_kernel`.
- `GET /config` trả model/chế độ mặc định/device và trạng thái `processing_modes`.
  Response dự đoán thêm `processing_mode`, `method_id`, `method_label`.
  CLI visualization vẫn ba cột.
- Mask trắng/đen là dự đoán chưa hiệu chuẩn; không gán good/bad hay gọi nó là GT.

## Thử nghiệm đối chiếu độc lập

Web không còn UI, import detector hoặc route `/predict-reference` (404).
`demo_reference.py` và `compare_reference.py` giữ thí nghiệm CPU ngoài web,
không được gọi trong chế độ gốc hoặc nhánh chưng cất. Xem
[REFERENCE_DETECTION.md](REFERENCE_DETECTION.md) về cách chạy và giới hạn.
