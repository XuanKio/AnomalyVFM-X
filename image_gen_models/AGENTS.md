# Quy tắc khu vực image generation

Đọc `../docs/SYNTHETIC_DATA.md` trước khi sửa.

- Đây là luồng nặng, không được chạy mặc định trên laptop demo 4 GB VRAM.
- Model mới phải triển khai cùng contract generate/inpaint mà `Generator` dùng.
- Không hard-code token Hugging Face, API key hoặc đường dẫn máy cá nhân.
- Ghi rõ license, VRAM, dung lượng tải và authentication của model mới.
- Test logic bằng mock/fixture nhỏ khi không có phần cứng phù hợp.
