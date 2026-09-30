"""Local AnomalyVFM inference with explicit, uncalibrated prediction views."""

from __future__ import annotations

if __name__ == "__main__":
    from demo_bootstrap import launch_web_runtime

    launch_web_runtime(__file__)

import argparse
import cgi
import io
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from PIL import Image

from paper_inference import MODEL_ID, PaperRuntime
from processing_modes import ProcessingModes
from distilled_inference import ModeUnavailableError


PAGE = """<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AnomalyVFM Demo</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, Segoe UI, sans-serif; }
    body { margin: 0; background: #0b1020; color: #e8ecf8; }
    main { width: min(1440px, calc(100% - 32px)); margin: 28px auto; }
    h1 { margin: 0 0 8px; font-size: clamp(28px, 5vw, 46px); }
    .sub { color: #a8b2ce; margin-bottom: 24px; }
    .card { background: #151c31; border: 1px solid #293451; border-radius: 18px;
            padding: 22px; box-shadow: 0 18px 50px #0006; }
    #drop { display: grid; place-items: center; min-height: 170px; cursor: pointer;
            border: 2px dashed #6176ad; border-radius: 14px; text-align: center;
            transition: .2s; background: #10172a; }
    #drop.over { border-color: #62d8ff; background: #14243a; }
    #file { display: none; }
    select, input[type=number] { color: #e8ecf8; background: #10172a;
      border: 1px solid #6176ad; padding: 9px; border-radius: 8px; }
    .controls { display: flex; flex-wrap: wrap; gap: 18px; margin-top: 18px; }
    .controls label { display: grid; gap: 6px; font-size: 14px; }
    button { margin-top: 14px; padding: 11px 18px; border: 0; border-radius: 10px;
             background: #6d7cff; color: white; font-weight: 700; cursor: pointer; }
    button:disabled { opacity: .45; cursor: wait; }
    .examples { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-top: 12px; }
    .examples button { margin: 0; padding: 8px 11px; background: #263352; font-weight: 500; }
    #status { min-height: 24px; margin: 15px 0 0; color: #9ee8ff; }
    #result { display: none; margin-top: 20px; }
    .panels { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
    figure { margin: 0; min-width: 0; }
    figure img { display: block; width: 100%; aspect-ratio: 1; object-fit: contain;
      background: #050812; border-radius: 8px; }
    figcaption { margin: 9px 0; font-size: 14px; }
    .legend { width: min(320px, 100%); margin: 16px 0; }
    .gradient { height: 12px; background: linear-gradient(to right,
      #000080 0%, #0000ff 12.5%, #00ffff 37.5%, #ffff00 62.5%, #ff0000 87.5%, #800000 100%); }
    .legend-labels { display: flex; justify-content: space-between; font-size: 12px; }
    .downloads { display: flex; gap: 18px; flex-wrap: wrap; margin-top: 16px; }
    a { color: #9ee8ff; }
    @media (max-width: 1000px) { .panels { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    @media (max-width: 540px) { .panels { grid-template-columns: 1fr; } .card { padding: 14px; } }
    .metrics { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }
    .pill { padding: 8px 12px; background: #202b47; border-radius: 999px; }
    .note { color: #b1bdd9; font-size: 13px; line-height: 1.6; margin-top: 14px; }
  </style>
</head>
<body><main>
  <h1>AnomalyVFM</h1>
  <div class="sub">Ảnh đầu vào → điểm bất thường theo vùng → mặt nạ dự đoán</div>
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
    <div class="controls">
      <label for="processing-mode">Chế độ xử lý
        <select id="processing-mode">
          <option value="light">Nhẹ · phương pháp gốc của bài báo</option>
          <option value="detailed" disabled>Chi tiết · sau chưng cất (chưa sẵn sàng)</option>
        </select>
      </label>
      <label for="threshold">Ngưỡng hiển thị mặt nạ
        <input id="threshold" type="number" min="0" max="1" step="0.01" value="0.50">
      </label>
    </div>
    <div class="note" id="method-note">Nhẹ dùng CLIP 672 × 672 theo script dự đoán ảnh đơn của tác giả.
      Chi tiết dành cho phương pháp mới sau chưng cất; hiện chưa có checkpoint hoàn tất để sử dụng.</div>
    <div class="note">Heatmap, overlay và mặt nạ nhị phân là phần hiển thị của web. Bản đồ gốc tải ở pred.png.
      Ngưỡng 0,50 là giá trị minh họa, chưa được hiệu chuẩn.
      Giảm ngưỡng để lấy thêm vùng nghi ngờ; tăng ngưỡng để lấy ít vùng hơn.
      Sau khi đổi ngưỡng, bấm phân tích lại. Không dùng mặt nạ này làm đáp án chuẩn.</div>
    <button id="run" disabled>Phân tích ảnh</button>
    <div id="status" role="status" aria-live="polite"></div>
    <div id="result">
      <div class="metrics">
        <span class="pill" id="score"></span>
        <span class="pill" id="speed"></span>
        <span class="pill" id="vram"></span>
        <span class="pill" id="mode"></span>
      </div>
      <div class="panels">
        <figure><img id="input-view" alt="Ảnh đầu vào"><figcaption>1. Ảnh gốc</figcaption></figure>
        <figure><img id="heatmap-view" alt="Bản đồ điểm bất thường, thang màu cố định từ 0 đến 1"><figcaption>2. Heatmap · điểm theo vùng</figcaption></figure>
        <figure><img id="overlay-view" alt="Bản đồ điểm bất thường chồng lên ảnh gốc"><figcaption>3. Heatmap chồng lên ảnh</figcaption></figure>
        <figure><img id="mask-view" alt="Mặt nạ dự đoán tại ngưỡng đã chọn, không phải ground truth"><figcaption id="mask-label">4. Mặt nạ dự đoán · không phải GT</figcaption></figure>
      </div>
      <div class="legend"><div class="gradient"></div><div class="legend-labels"><span>0 · thấp</span><span>0,5</span><span>1 · cao</span></div></div>
      <div class="note" id="range"></div>
      <div class="downloads"><a id="download-result" download="anomalyvfm_result.png">Lưu bảng kết quả</a>
        <a id="download-mask" download="predicted_mask.png">Lưu mặt nạ dự đoán</a>
        <a id="download-native" download="pred.png">Lưu pred.png gốc (96 × 96)</a>
        <a id="download-score" download="anomaly_scores.png">Lưu bản đồ điểm xám (0–255)</a></div>
    </div>
    <div class="note">Màu dùng cùng thang điểm 0–1 cho mọi ảnh, không tự kéo giãn từng ảnh.
      Điểm bất thường không phải phần trăm xác suất. Đây là dự đoán của CLIP,
      không cam kết trùng kết quả RADIO trong bài báo. Ảnh chỉ được xử lý trên máy này.</div>
  </section>
</main>
<script>
const drop = document.querySelector('#drop'), file = document.querySelector('#file');
const run = document.querySelector('#run'), status = document.querySelector('#status');
let selected, busy=false;
function choose(f) { if (busy) return; selected=f; run.disabled=!f; drop.querySelector('strong').textContent=f?.name || 'Chọn ảnh';
  document.querySelector('#result').style.display='none'; status.textContent=''; }
file.onchange = () => choose(file.files[0]);
drop.ondragover = e => { e.preventDefault(); drop.classList.add('over'); };
drop.ondragleave = () => drop.classList.remove('over');
drop.ondrop = e => { e.preventDefault(); drop.classList.remove('over'); choose(e.dataTransfer.files[0]); };
document.querySelectorAll('.sample').forEach(button => button.onclick = async () => {
  try {
    const name=button.dataset.name, res=await fetch('/sample/'+name);
    if (!res.ok) throw new Error('Không đọc được ảnh mẫu');
    const blob=await res.blob(); choose(new File([blob], name, {type: blob.type}));
  } catch(e) { status.textContent=e.message; }
});
fetch('/config').then(res=>res.json()).then(data=>{
  const selector=document.querySelector('#processing-mode');
  for(const mode of data.processing_modes) {
    const option=selector.querySelector(`option[value="${mode.id}"]`);
    option.disabled=!mode.available;
    option.textContent=mode.label+(mode.available ? '' : ' (chưa sẵn sàng)');
  }
  selector.value=data.default_mode;
})
  .catch(()=>{status.textContent='Không đọc được cấu hình server. Hãy tải lại trang.';});
document.querySelector('#processing-mode').onchange=()=>{
  document.querySelector('#result').style.display='none'; status.textContent='';
};
run.onclick = async () => {
  if (!selected || busy) return;
  const threshold=Number(document.querySelector('#threshold').value);
  if (!document.querySelector('#threshold').value || !Number.isFinite(threshold) || threshold<0 || threshold>1) {
    status.textContent='Ngưỡng cần là một số từ 0 đến 1.'; return;
  }
  busy=true; run.disabled=true; status.textContent='Model đang phân tích...';
  document.querySelectorAll('.sample, #processing-mode, #threshold, #file').forEach(el=>el.disabled=true);
  document.querySelector('#result').style.display='none';
  const body=new FormData(); body.append('image', selected);
  body.append('mode', document.querySelector('#processing-mode').value);
  body.append('threshold', String(threshold));
  try {
    const res=await fetch('/predict', {method:'POST', body}); const data=await res.json();
    if (!res.ok) throw new Error(data.error || 'Không thể xử lý ảnh');
    document.querySelector('#score').textContent=`Anomaly score: ${data.score.toFixed(4)}`;
    document.querySelector('#speed').textContent=`Inference: ${data.seconds.toFixed(2)} giây`;
    document.querySelector('#vram').textContent=`Peak VRAM: ${data.vram_gb.toFixed(2)} GB`;
    document.querySelector('#mode').textContent=`${data.method_label} · ${data.input_size} · ${data.device}`;
    for (const key of ['input','heatmap','overlay','mask']) {
      document.querySelector('#'+key+'-view').src='data:image/png;base64,'+data.panels[key];
    }
    document.querySelector('#mask-label').textContent=`4. Mặt nạ dự đoán · ngưỡng ${data.threshold.toFixed(2)} · không phải GT`;
    document.querySelector('#range').textContent=`Điểm vùng: ${data.score_range[0].toFixed(4)}–${data.score_range[1].toFixed(4)}. `+
      `Vùng trắng chiếm ${data.mask_area_percent.toFixed(2)}% ảnh tại ngưỡng này; không phải tỷ lệ lỗi đã xác nhận.`;
    document.querySelector('#download-result').href='data:image/png;base64,'+data.image;
    document.querySelector('#download-mask').href='data:image/png;base64,'+data.panels.mask;
    document.querySelector('#download-native').href='data:image/png;base64,'+data.native_mask;
    document.querySelector('#download-score').href='data:image/png;base64,'+data.raw_mask;
    document.querySelector('#result').style.display='block'; status.textContent='Hoàn tất';
  } catch(e) { status.textContent='Lỗi: '+e.message; }
  finally { busy=false; run.disabled=false; document.querySelectorAll('.sample, #processing-mode, #threshold, #file').forEach(el=>el.disabled=false); }
};
</script></body></html>"""


# Compatibility for callers that construct the released-paper runtime.
Runtime = PaperRuntime


def make_handler(runtime: Runtime) -> type[BaseHTTPRequestHandler]:
    modes = ProcessingModes(runtime)
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
            if self.path == "/config":
                self.send_bytes(200, "application/json", json.dumps({
                    "model_id": MODEL_ID, "input_size": runtime.default_size,
                    "device": str(runtime.device),
                    **modes.configuration(),
                }).encode("utf-8"))
                return
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
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 25 * 1024 * 1024:
                    raise ValueError("Ảnh tải lên cần nhỏ hơn 25 MB.")
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
                with Image.open(io.BytesIO(upload.file.read())) as image:
                    if image.width * image.height > 25_000_000:
                        raise ValueError("Ảnh cần nhỏ hơn 25 triệu pixel.")
                    threshold = float(form.getfirst("threshold", "0.5"))
                    size = int(form.getfirst("size", str(runtime.default_size)))
                    if size != 672:
                        raise ValueError("Chế độ gốc CLIP dùng kích thước 672 × 672.")
                    mode = form.getfirst("mode", "light")
                    payload = modes.predict(image, mode=mode, threshold=threshold, size=size)
                data = json.dumps(payload).encode("utf-8")
                self.send_bytes(200, "application/json", data)
            except ModeUnavailableError as exc:
                data = json.dumps({"error": str(exc), "mode": "detailed",
                                   "available": False}).encode("utf-8")
                self.send_bytes(503, "application/json", data)
            except Exception as exc:
                data = json.dumps({"error": str(exc)}).encode("utf-8")
                self.send_bytes(400, "application/json", data)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Local AnomalyVFM browser demo")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--size", type=int, choices=(672,), default=672)
    parser.add_argument("--cpu", action="store_true", help="Run on CPU without lowering resolution")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    runtime = Runtime(args.size, cpu=args.cpu)
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
