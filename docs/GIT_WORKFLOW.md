# Git workflow

## Remote chính

Repository chỉ dùng:

```text
origin  git@github.com:XuanKio/AnomalyVFM-X.git
```

Không thêm `upstream` tự động. Khi Xuân muốn lấy thay đổi từ nguồn khác, thực
hiện như một nhiệm vụ riêng có review.

## Clone và kiểm tra

```powershell
git clone git@github.com:XuanKio/AnomalyVFM-X.git
cd AnomalyVFM-X
git remote -v
git status --short --branch
```

## Nhánh

- `main`: trạng thái demo chạy được;
- `feature/<name>`: tính năng;
- `fix/<name>`: sửa lỗi;
- `docs/<name>`: tài liệu;
- `experiment/<name>`: thử nghiệm chưa chắc merge.

```powershell
git switch main
git pull --ff-only origin main
git switch -c feature/example
```

## Commit

Commit theo một mục tiêu, ví dụ:

```text
docs: add training and dataset runbooks
feat(demo): add local browser interface
fix(setup): select CPU wheel without NVIDIA
```

Trước commit:

```powershell
git diff --check
git status --short
git diff --stat
```

Sau đó:

```powershell
git add <cac-file-can-thiet>
git commit -m "<type>: <mo-ta>"
git push -u origin <ten-nhanh>
```

## Đồng bộ khi làm nhóm

- pull bằng `--ff-only` trên main để tránh merge ngoài ý muốn;
- làm việc trên nhánh riêng và review diff trước merge;
- không force-push nhánh người khác đang dùng;
- không commit output/model/data chỉ để chuyển file cho nhau.

## File lớn và secret

Git thường không dùng cho:

- môi trường Python và cache;
- `*.pkl`, `*.safetensors`, checkpoint backbone;
- dataset/archive;
- generated images hàng loạt;
- token Hugging Face, SSH key, `.env`.

Nếu cần phát hành checkpoint, quyết định rõ Git LFS, GitHub Release hay Hugging
Face model repository trước khi upload.

## Release demo

Một release nên ghi:

- commit/tag;
- phiên bản Python/PyTorch/Transformers;
- model ID/revision và image size;
- cấu hình máy đã test;
- lệnh setup/run;
- giới hạn đã biết.
