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

Checkpoint demo CLIP có cấu hình gốc image size 672, DoRA rank 64. Demo nhẹ cập
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
                                      ├─ Runtime.predict (khóa tuần tự GPU)
                                      ├─ heatmap + overlay
Browser ◄──── JSON + PNG base64 ──────┘
```

Model được load một lần khi server khởi động. `threading.Lock` ngăn hai request
cùng dùng GPU 4 GB. Server bind `127.0.0.1`, không được truy cập từ máy khác.

## Contract cần giữ ổn định

- CLI mặc định: `demo_fast.py test.png`, size 336.
- Output CLI: `outputs/<stem>_result.png`.
- Web `POST /predict`: multipart field tên `image`.
- JSON web: `score`, `seconds`, `vram_gb`, `image`.
- Visualization: ba cột Input / Anomaly heatmap / Overlay.
- Không tự gán nhãn binary bằng threshold mặc định.
