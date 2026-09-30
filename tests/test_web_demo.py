"""Local HTTP and CPU-only runtime contracts without loading model checkpoints."""

import base64
import http.client
import io
import json
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from PIL import Image

import web_demo
import paper_inference


def _png_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (5, 3), (20, 80, 140)).save(buffer, format="PNG")
    return buffer.getvalue()


def _multipart(fields=None, image_data=None, include_image=True):
    boundary = "anomalyvfm-test-boundary"
    chunks = []
    for name, value in (fields or {}).items():
        chunks.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n'
            f"\r\n{value}\r\n".encode("utf-8")
        )
    if include_image:
        chunks.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; '
            'filename="test.png"\r\nContent-Type: image/png\r\n\r\n'.encode("ascii")
        )
        chunks.append(_png_bytes() if image_data is None else image_data)
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("ascii"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _decode_png(encoded):
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as picture:
        return picture.copy()


class WebHandlerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = Mock(default_size=672, device="cpu")
        cls.server = web_demo.ThreadingHTTPServer(
            ("127.0.0.1", 0), web_demo.make_handler(cls.runtime)
        )
        cls.thread = threading.Thread(
            target=cls.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self):
        self.runtime.predict.reset_mock()
        self.runtime.predict.side_effect = None
        self.runtime.predict.return_value = {"score": 0.375, "threshold_calibrated": False}

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=5)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def post_image(self, fields=None, image_data=None, include_image=True):
        body, content_type = _multipart(fields, image_data, include_image)
        return self.request("POST", "/predict", body, {"Content-Type": content_type})

    def test_root_exposes_both_modes_and_four_prediction_views(self):
        status, headers, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["Content-Type"])
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(int(headers["Content-Length"]), len(body))
        page = body.decode("utf-8")
        self.assertNotIn('id="size"', page)
        self.assertIn("672 × 672", page)
        self.assertIn('id="download-native"', page)
        self.assertIn('id="processing-mode"', page)
        self.assertIn('<option value="light">', page)
        self.assertIn('<option value="detailed" disabled>', page)
        self.assertIn("chưng cất", page)
        for panel in ("input", "heatmap", "overlay", "mask"):
            self.assertIn(f'id="{panel}-view"', page)
        self.assertIn("chưa được hiệu chuẩn", page)

    def test_config_exposes_default_resolution_and_cpu(self):
        status, _, body = self.request("GET", "/config")
        self.assertEqual(status, 200)
        config = json.loads(body)
        self.assertEqual(config["input_size"], 672)
        self.assertEqual(config["device"], "cpu")
        self.assertEqual(config["model_id"], "MaticFuc/anomalyvfm_clip")
        self.assertEqual(config["default_mode"], "light")
        modes = {mode["id"]: mode for mode in config["processing_modes"]}
        self.assertTrue(modes["light"]["available"])
        self.assertFalse(modes["detailed"]["available"])

    def test_three_samples_return_decodable_png(self):
        for name in ("hazelnut_normal.png", "hazelnut_defective.png", "bottle_defective.png"):
            with self.subTest(name=name):
                status, headers, body = self.request("GET", "/sample/" + name)
                self.assertEqual(status, 200)
                self.assertEqual(headers["Content-Type"], "image/png")
                with Image.open(io.BytesIO(body)) as picture:
                    self.assertEqual(picture.format, "PNG")
                    picture.verify()

    def test_unknown_routes_and_sample_traversal_return_404(self):
        for path in ("/missing", "/sample/missing.png", "/sample/%2e%2e%2fweb_demo.py"):
            with self.subTest(path=path):
                self.assertEqual(self.request("GET", path)[0], 404)
        self.assertEqual(self.request("POST", "/missing")[0], 404)
        self.runtime.predict.assert_not_called()

    def test_valid_upload_passes_image_and_explicit_options(self):
        captured = {}

        def predict(image, threshold, size):
            captured.update(size=image.size, pixel=image.convert("RGB").getpixel((0, 0)))
            return {"score": 0.375, "input_size": size, "threshold": threshold,
                    "threshold_calibrated": False}

        self.runtime.predict.side_effect = predict
        status, headers, body = self.post_image({"size": "672", "threshold": "0.73"})
        self.assertEqual(status, 200)
        self.assertIn("application/json", headers["Content-Type"])
        payload = json.loads(body)
        self.assertEqual(payload["input_size"], 672)
        self.assertEqual(payload["threshold"], 0.73)
        self.assertEqual(payload["processing_mode"], "light")
        self.assertEqual(payload["method_id"], "paper_clip")
        self.assertIs(payload["threshold_calibrated"], False)
        self.assertEqual(captured, {"size": (5, 3), "pixel": (20, 80, 140)})
        self.assertEqual(self.runtime.predict.call_count, 1)

    def test_legacy_upload_without_options_uses_server_defaults(self):
        status, _, _ = self.post_image()
        self.assertEqual(status, 200)
        self.assertEqual(self.runtime.predict.call_args.kwargs, {"threshold": 0.5, "size": 672})

    def test_detailed_is_unavailable_and_never_calls_paper_inference(self):
        for size in (672,):
            status, _, body = self.post_image({"mode": "detailed", "size": str(size)})
            self.assertEqual(status, 503)
            payload = json.loads(body)
            self.assertFalse(payload["available"])
            self.assertEqual(payload["mode"], "detailed")
        self.runtime.predict.assert_not_called()

    def test_light_uses_paper_at_original_resolution(self):
        for size in (672,):
            status, _, body = self.post_image({"mode": "light", "size": str(size)})
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["method_id"], "paper_clip")
            self.assertEqual(self.runtime.predict.call_args.kwargs["size"], size)

    def test_unknown_mode_is_rejected_without_inference(self):
        status, _, body = self.post_image({"mode": "anything_else"})
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(body))
        self.runtime.predict.assert_not_called()

    def test_invalid_or_missing_image_returns_json_without_inference(self):
        for options in ({"image_data": b"not an image"}, {"include_image": False}):
            with self.subTest(options=options):
                status, _, body = self.post_image({"threshold": "0.5"}, **options)
                self.assertEqual(status, 400)
                self.assertIn("error", json.loads(body))
        self.runtime.predict.assert_not_called()

    def test_invalid_numeric_fields_return_json_without_inference(self):
        for fields in ({"size": "336"}, {"size": "336.5"}, {"size": "oops"}, {"threshold": "oops"}):
            with self.subTest(fields=fields):
                status, _, body = self.post_image(fields)
                self.assertEqual(status, 400)
                self.assertIn("error", json.loads(body))
        self.runtime.predict.assert_not_called()

    def test_invalid_content_lengths_rejected_before_body_read(self):
        for length in ("0", "-1", "not-a-number", str(25 * 1024 * 1024 + 1)):
            with self.subTest(length=length):
                status, _, body = self.request(
                    "POST", "/predict", b"", {"Content-Length": length}
                )
                self.assertEqual(status, 400)
                self.assertIn("error", json.loads(body))
        self.runtime.predict.assert_not_called()

    def test_runtime_failure_returns_error_response(self):
        self.runtime.predict.side_effect = ValueError("Invalid inference options")
        status, _, body = self.post_image()
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body), {"error": "Invalid inference options"})


class RuntimeTests(unittest.TestCase):
    def test_invalid_options_fail_before_model_or_transform_access(self):
        runtime = web_demo.Runtime.__new__(web_demo.Runtime)
        runtime.default_size = 672
        # Intentionally no model, transform or lock: validation must precede all of them.
        image = Image.new("RGB", (5, 3))
        for threshold in (float("nan"), float("inf"), -0.001, 1.001):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                runtime.predict(image, threshold=threshold)
        for size in (0, 224, 336, 673, "672"):
            with self.subTest(size=size), self.assertRaises(ValueError):
                runtime.predict(image, size=size)

    def test_invalid_initial_resolution_does_not_load_checkpoint(self):
        with patch.object(paper_inference, "load_demo_model") as load:
            with self.assertRaises(ValueError):
                web_demo.Runtime(size=224)
        load.assert_not_called()

    def test_fake_cpu_model_outputs_calibration_metadata_and_usable_views(self):
        fake_model = Mock()
        fake_model.to.return_value = fake_model
        transform = Mock(return_value=torch.zeros((3, 5, 5)))
        fake_model.model.get_img_transform.return_value = transform
        fake_model.feat_size = 5
        fake_model.model.return_value = (torch.zeros((1, 2)), torch.zeros((1, 25, 2)))
        logits = torch.full((1, 1, 5, 5), 2.0)
        logits[0, 0, 2, 2] = -10.0
        fake_model.decoder.return_value = (logits, None)
        fake_model.predictor.return_value = torch.logit(torch.tensor([0.375]))

        with patch.object(paper_inference, "load_demo_model", return_value=fake_model), \
             patch.object(paper_inference, "configure_resolution") as configure, \
             patch.object(torch.cuda, "is_available", return_value=False), \
             patch.object(torch.cuda, "reset_peak_memory_stats") as reset_gpu, \
             patch("builtins.print"):
            runtime = web_demo.Runtime()
            self.assertEqual(runtime.default_size, 672)
            self.assertEqual(runtime.device.type, "cpu")
            image = Image.new("RGB", (5, 5), "white")
            result = runtime.predict(image, threshold=0.5)
            higher = runtime.predict(image, threshold=0.8)

        self.assertEqual([call.args[1] for call in configure.call_args_list], [672, 672, 672])
        reset_gpu.assert_not_called()
        fake_model.assert_not_called()
        self.assertEqual(fake_model.model.call_count, 2)
        fake_model.to.assert_called_once_with(device=torch.device("cpu"), dtype=torch.float32)
        self.assertAlmostEqual(result["score"], 0.375, places=6)
        self.assertEqual(result["vram_gb"], 0.0)
        self.assertGreaterEqual(result["seconds"], 0.0)
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(result["input_size"], 672)
        self.assertIs(result["threshold_calibrated"], False)
        self.assertEqual(result["smoothing_kernel"], 5)
        expected = torch.nn.AvgPool2d(5, 1, 2)(logits).float().sigmoid()[0, 0].numpy()
        wrong_order = torch.nn.AvgPool2d(5, 1, 2)(logits.sigmoid())[0, 0].numpy()
        self.assertFalse(np.allclose(expected, wrong_order))
        self.assertAlmostEqual(result["score_range"][0], float(expected.min()), places=6)
        self.assertAlmostEqual(result["score_range"][1], float(expected.max()), places=6)
        np.testing.assert_array_equal(_decode_png(result["panels"]["mask"]), (expected >= .5).astype('uint8') * 255)
        np.testing.assert_array_equal(_decode_png(higher["panels"]["mask"]), (expected >= .8).astype('uint8') * 255)
        np.testing.assert_array_equal(_decode_png(result["native_mask"]), (expected * 255).astype('uint8'))
        self.assertEqual(result["native_mask"], higher["native_mask"])
        self.assertEqual(result["native_mask_shape"], [5, 5])
        self.assertEqual(result["inference_protocol"], "authors_predict_single_image")
        for key in ("input", "heatmap", "overlay", "mask"):
            panel = _decode_png(result["panels"][key])
            self.assertEqual(panel.size, (5, 5))
            self.assertEqual(panel.mode, "L" if key == "mask" else "RGB")
        self.assertEqual(_decode_png(result["raw_mask"]).mode, "L")
        self.assertEqual(_decode_png(result["image"]).mode, "RGB")


if __name__ == "__main__":
    unittest.main()
