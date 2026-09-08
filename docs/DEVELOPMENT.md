# Phát triển và kiểm thử

## Bắt đầu một thay đổi

```powershell
git status --short --branch
git switch -c feature/<ten-ngan>
```

Đọc `AGENTS.md` ở root và AGENTS gần nhất trong thư mục sẽ sửa. Không trộn thay
đổi demo nhẹ với nâng cấp research dependency trong cùng commit.

## Kiểm thử demo

```powershell
& "$env:LOCALAPPDATA\AnomalyVFM-X\venv\Scripts\python.exe" `
  -m py_compile demo_fast.py web_demo.py
.\demo.cmd demo_images\hazelnut_normal.png
```

Khởi động web:

```powershell
.\web_demo.cmd
```

Trong terminal thứ hai:

```powershell
Invoke-WebRequest http://127.0.0.1:7860/ -UseBasicParsing
Invoke-RestMethod http://127.0.0.1:7860/predict -Method Post `
  -Form @{image=Get-Item .\demo_images\hazelnut_normal.png}
```

## Kiểm tra trước commit

```powershell
git diff --check
git status --short
git diff --stat
```

Kiểm tra riêng:

- không có token, `.env`, credential hoặc path cá nhân mới;
- không có `.venv`, cache, checkpoint, dataset/output lớn;
- README/docs khớp hành vi;
- file ảnh mới có nguồn/license;
- server vẫn bind local và xử lý input lỗi hợp lý.

## Nâng dependency

1. Tạo nhánh/commit riêng.
2. Ghi phiên bản cũ/mới và lý do.
3. Kiểm tra CUDA availability, model load offline, CLI và web.
4. Đo lại startup, inference, VRAM và disk.
5. Cập nhật `setup_demo.cmd`, `requirements.txt` (nếu full env) và docs liên quan.

Không nâng PyTorch/Transformers tự động chỉ vì có bản mới; checkpoint/custom
code có thể phụ thuộc hành vi phiên bản cũ.

## Định nghĩa hoàn thành

Một thay đổi hoàn thành khi code chạy, kiểm thử tương xứng đã qua, docs cập nhật,
artifact lớn không bị track và commit có thể được người khác clone/tái hiện.
