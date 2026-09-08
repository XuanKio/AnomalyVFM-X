"""Tiny local web interface for the lightweight AnomalyVFM demo."""

from __future__ import annotations

import argparse
import base64
import cgi
import io
import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import torch
from PIL import Image

from demo_fast import MODEL_ID, build_visualization, configure_resolution
from hf_model import AnomalyVFM


PAGE = """<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AnomalyVFM Demo</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, Segoe UI, sans-serif; }
    body { margin: 0; background: #0b1020; color: #e8ecf8; }
    main { width: min(1080px, calc(100% - 32px)); margin: 36px auto; }
    h1 { margin: 0 0 8px; font-size: clamp(28px, 5vw, 46px); }
    .sub { color: #a8b2ce; margin-bottom: 24px; }
    .card { background: #151c31; border: 1px solid #293451; border-radius: 18px;
            padding: 22px; box-shadow: 0 18px 50px #0006; }
    #drop { display: grid; place-items: center; min-height: 170px; cursor: pointer;
            border: 2px dashed #6176ad; border-radius: 14px; text-align: center;
            transition: .2s; background: #10172a; }
    #drop.over { border-color: #62d8ff; background: #14243a; }
    input { display: none; }
    button { margin-top: 14px; padding: 11px 18px; border: 0; border-radius: 10px;
             background: #6d7cff; color: white; font-weight: 700; cursor: pointer; }
    button:disabled { opacity: .45; cursor: wait; }
    .examples { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-top: 12px; }
    .examples button { margin: 0; padding: 8px 11px; background: #263352; font-weight: 500; }
    #status { min-height: 24px; margin: 15px 0 0; color: #9ee8ff; }
    #result { display: none; margin-top: 20px; }
    #result img { display: block; width: 100%; border-radius: 12px; background: white; }
    .metrics { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }
    .pill { padding: 8px 12px; background: #202b47; border-radius: 999px; }
    .note { color: #8894b2; font-size: 13px; margin-top: 18px; }
  </style>
</head>
<body><main>
  <h1>AnomalyVFM</h1>
  <div class="sub">Zero-shot visual anomaly detection — CVPR 2026</div>
  <section class="card">
    <label id="drop" for="file">
      <div><strong>Kéo ảnh vào đây hoặc bấm để chọn</strong><br>
      <span style="color:#8f9aba">PNG, JPG, JPEG, WEBP</span></div>
    </label>
    <input id="file" type="file" accept="image/png,image/jpeg,image/webp">
    <div class="examples"><span>Ảnh mẫu:</span>
      <button class="sample" data-name="hazelnut_normal.png">Hạt phỉ bình thường</button>
      <button class="sample" data-name="hazelnut_defective.png">Hạt phỉ bị lỗi</button>
      <button class="sample" data-name="bottle_defective.png">Chai bị lỗi</button>
    </div>
    <button id="run" disabled>Phân tích ảnh</button>
    <div id="status"></div>
    <div id="result">
      <div class="metrics">
        <span class="pill" id="score"></span>
        <span class="pill" id="speed"></span>
        <span class="pill" id="vram"></span>
      </div>
      <img id="visual" alt="AnomalyVFM result">
    </div>
    <div class="note">Đỏ/vàng = vùng model chú ý nhiều hơn. Điểm anomaly càng cao,
    ảnh càng có khả năng chứa bất thường.</div>
  </section>
</main>
<script>
const drop = document.querySelector('#drop'), file = document.querySelector('#file');
const run = document.querySelector('#run'), status = document.querySelector('#status');
let selected;
function choose(f) { selected=f; run.disabled=!f; drop.querySelector('strong').textContent=f?.name || 'Chọn ảnh'; }
file.onchange = () => choose(file.files[0]);
drop.ondragover = e => { e.preventDefault(); drop.classList.add('over'); };
drop.ondragleave = () => drop.classList.remove('over');
drop.ondrop = e => { e.preventDefault(); drop.classList.remove('over'); choose(e.dataTransfer.files[0]); };
document.querySelectorAll('.sample').forEach(button => button.onclick = async () => {
  const name=button.dataset.name, res=await fetch('/sample/'+name), blob=await res.blob();
  choose(new File([blob], name, {type: blob.type}));
});
run.onclick = async () => {
  if (!selected) return;
  run.disabled=true; status.textContent='Model đang phân tích...';
  document.querySelector('#result').style.display='none';
  const body=new FormData(); body.append('image', selected);
  try {
    const res=await fetch('/predict', {method:'POST', body}); const data=await res.json();
    if (!res.ok) throw new Error(data.error || 'Không thể xử lý ảnh');
    document.querySelector('#score').textContent=`Anomaly score: ${data.score.toFixed(4)}`;
    document.querySelector('#speed').textContent=`Inference: ${data.seconds.toFixed(2)} giây`;
    document.querySelector('#vram').textContent=`Peak VRAM: ${data.vram_gb.toFixed(2)} GB`;
    document.querySelector('#visual').src='data:image/png;base64,'+data.image;
    document.querySelector('#result').style.display='block'; status.textContent='Hoàn tất';
  } catch(e) { status.textContent='Lỗi: '+e.message; }
  finally { run.disabled=false; }
};
</script></body></html>"""


class Runtime:
    def __init__(self, size: int) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {MODEL_ID} on {self.device}...", flush=True)
        started = time.perf_counter()
        self.model = AnomalyVFM.from_pretrained(MODEL_ID)
        configure_resolution(self.model, size)
        if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
            self.model = self.model.to(device=self.device, dtype=torch.bfloat16)
        else:
            self.model = self.model.to(self.device)
        self.model.eval()
        self.transform = self.model.model.get_img_transform()
        self.lock = threading.Lock()
        print(f"Model ready in {time.perf_counter() - started:.1f}s", flush=True)

    def predict(self, image: Image.Image) -> dict[str, float | str]:
        with self.lock:
            tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.inference_mode():
                score, mask = self.model(tensor)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            peak = (
                torch.cuda.max_memory_allocated() / 1024**3
                if self.device.type == "cuda"
                else 0.0
            )

        score_value = float(score.squeeze().cpu())
        visual = build_visualization(image, mask[0], score_value)
        buffer = io.BytesIO()
        visual.save(buffer, format="PNG", optimize=True)
        return {
            "score": score_value,
            "seconds": seconds,
            "vram_gb": peak,
            "image": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }


def make_handler(runtime: Runtime) -> type[BaseHTTPRequestHandler]:
    sample_dir = Path(__file__).resolve().parent / "demo_images"
    samples = {
        name: sample_dir / name
        for name in (
            "hazelnut_normal.png",
            "hazelnut_defective.png",
            "bottle_defective.png",
        )
    }

    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, status: int, content_type: str, data: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path.startswith("/sample/"):
                name = unquote(self.path.removeprefix("/sample/"))
                sample = samples.get(name)
                if sample and sample.is_file():
                    self.send_bytes(200, "image/png", sample.read_bytes())
                else:
                    self.send_bytes(404, "text/plain; charset=utf-8", b"Not found")
                return
            if self.path != "/":
                self.send_bytes(404, "text/plain; charset=utf-8", b"Not found")
                return
            self.send_bytes(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))

        def do_POST(self) -> None:
            if self.path != "/predict":
                self.send_bytes(404, "application/json", b'{"error":"Not found"}')
                return
            try:
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                        "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                    },
                )
                upload = form["image"]
                image = Image.open(io.BytesIO(upload.file.read())).convert("RGB")
                payload = runtime.predict(image)
                data = json.dumps(payload).encode("utf-8")
                self.send_bytes(200, "application/json", data)
            except Exception as exc:
                data = json.dumps({"error": str(exc)}).encode("utf-8")
                self.send_bytes(400, "application/json", data)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Local AnomalyVFM browser demo")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--size", type=int, choices=(336, 672), default=336)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    runtime = Runtime(args.size)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(runtime))
    url = f"http://127.0.0.1:{args.port}"
    print(f"Web demo: {url}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
