"""CPU-only spatial and math contracts; no checkpoint or network required."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import demo_tiling


class FakeModel:
    def __init__(self, mode="pixels"):
        self.mode = mode
        self.calls = 0
        self.saw_grad = []
        self.saw_modes = []
        self.model = SimpleNamespace(get_img_transform=lambda: self.transform)

    def eval(self):
        return self

    def transform(self, image):
        self.saw_modes.append(image.mode)
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255

    def __call__(self, tensor):
        self.calls += 1
        self.saw_grad.append(torch.is_grad_enabled())
        mask = tensor[:, :1]
        if self.mode == "constant":
            mask = torch.full_like(mask, 0.7)
        elif self.mode == "nan":
            mask = torch.full_like(mask, float("nan"))
        elif self.mode == "shape":
            mask = mask[0]
        elif self.mode == "different":
            mask = torch.full_like(mask, 0.8 if self.calls == 1 else 0.2)
        elif self.mode == "decoder":
            mask = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4) / 12
        score = torch.tensor([0.6 if self.calls == 1 else 0.1])
        return score, mask


class TileGeometryTests(unittest.TestCase):
    def test_coverage_including_edges_small_rectangular_and_singleton(self):
        for width, height in [(1, 1), (1, 31), (33, 1), (5, 7), (19, 31), (32, 32)]:
            for overlap in [0, 0.25, 0.9]:
                with self.subTest(size=(width, height), overlap=overlap):
                    boxes = demo_tiling.tile_boxes(width, height, 8, overlap)
                    counts = np.zeros((height, width), dtype=np.int32)
                    self.assertEqual(len(boxes), len(set(boxes)))
                    for left, top, right, bottom in boxes:
                        self.assertTrue(0 <= left < right <= width)
                        self.assertTrue(0 <= top < bottom <= height)
                        self.assertLessEqual(right - left, 8)
                        self.assertLessEqual(bottom - top, 8)
                        counts[top:bottom, left:right] += 1
                    self.assertTrue(np.all(counts > 0))

    def test_axis_final_anchor_and_no_duplicate(self):
        self.assertEqual(demo_tiling.axis_starts(20, 8, 0.25), [0, 6, 12])
        self.assertEqual(demo_tiling.axis_starts(21, 8, 0.25), [0, 6, 12, 13])
        self.assertEqual(demo_tiling.axis_starts(3, 8), [0])

    def test_feather_is_finite_positive_and_symmetric(self):
        for width, height in [(1, 1), (1, 8), (8, 9), (672, 672)]:
            weights = demo_tiling.feather_weights(width, height)
            self.assertEqual(weights.shape, (height, width))
            self.assertEqual(weights.dtype, np.float32)
            self.assertTrue(np.isfinite(weights).all())
            self.assertTrue((weights > 0).all())
            np.testing.assert_array_equal(weights, weights[::-1, ::-1])

    def test_invalid_geometry(self):
        for args in [(0, 8, 0.25), (8, 0, 0.25), (8.5, 8, 0.25),
                     (8, True, 0.25), (8, 8, -0.1), (8, 8, 1),
                     (8, 8, float("nan")), (8, 8, "x")]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                demo_tiling.axis_starts(*args)


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.configure = patch("demo_tiling.configure_resolution").start()
        self.addCleanup(patch.stopall)

    def predict(self, image, model=None, **kwargs):
        return demo_tiling.predict_comparison(
            model or FakeModel(), image, torch.device("cpu"), tile_size=8,
            smoothing_kernel=kwargs.pop("smoothing_kernel", 1), **kwargs)

    def test_constant_reconstruction_and_global_score_unchanged(self):
        progress = []
        model = FakeModel("constant")
        result = self.predict(Image.new("RGB", (23, 15)), model,
                              progress=lambda done, total: progress.append((done, total)))
        for key in ["baseline", "tiled", "fused"]:
            self.assertEqual(result[key].shape, (15, 23))
            self.assertEqual(result[key].dtype, np.float32)
            np.testing.assert_allclose(result[key], 0.7, atol=2e-7)
        self.assertAlmostEqual(result["global_score"], 0.6)
        meta = result["metadata"]
        self.assertEqual(model.calls, meta["tile_count"] + 1)
        self.assertEqual(progress[0], (0, meta["tile_count"]))
        self.assertEqual(progress[-1], (meta["tile_count"], meta["tile_count"]))
        self.assertFalse(any(model.saw_grad))
        self.assertIsNone(meta["peak_allocated_gb"])
        self.assertEqual(meta["proposed_seconds"], meta["baseline_seconds"] + meta["tiles_seconds"])
        self.assertGreaterEqual(meta["tiles_seconds"], 0)
        self.configure.assert_called_once_with(model, 672)

    def test_source_coordinate_ramp_and_edges_survive_stitching(self):
        yy, xx = np.mgrid[:17, :29]
        pixels = np.repeat(((yy * 4 + xx * 3) % 256).astype(np.uint8)[..., None], 3, axis=2)
        result = self.predict(Image.fromarray(pixels))
        expected = pixels[..., 0].astype(np.float32) / 255
        for key in ["baseline", "tiled", "fused"]:
            np.testing.assert_allclose(result[key], expected, atol=2e-7)

    def test_web_mean_pool_then_interpolate_math(self):
        image = Image.fromarray(np.arange(35, dtype=np.uint8).reshape(5, 7)).convert("RGB")
        result = self.predict(image, FakeModel("decoder"), smoothing_kernel=5)
        tensor = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4) / 12
        expected = F.interpolate(F.avg_pool2d(tensor, 5, 1, 2), (5, 7),
                                 mode="bilinear", align_corners=False)[0, 0].numpy()
        np.testing.assert_array_equal(result["baseline"], expected)
        np.testing.assert_allclose(result["fused"], expected, atol=1e-7)

    def test_constant_on_singleton_and_small_sources(self):
        for size in [(1, 1), (1, 17), (17, 1), (3, 5)]:
            with self.subTest(size=size):
                result = self.predict(Image.new("RGB", size), FakeModel("constant"))
                for key in ["baseline", "tiled", "fused"]:
                    self.assertEqual(result[key].shape, size[::-1])
                    self.assertTrue(np.isfinite(result[key]).all())
                    np.testing.assert_allclose(result[key], 0.7, atol=2e-7)

    def test_explicit_fusion_weight_does_not_change_image_score(self):
        for weight in [0, 0.25, 1]:
            result = self.predict(Image.new("RGB", (11, 13)), FakeModel("different"),
                                  global_weight=weight)
            np.testing.assert_allclose(result["fused"], weight * 0.8 + (1 - weight) * 0.2,
                                       atol=2e-7)
            self.assertAlmostEqual(result["global_score"], 0.6)

    def test_exif_rotation_and_rgb_conversion(self):
        image = Image.new("L", (3, 7), 42)
        image.getexif()[274] = 6
        model = FakeModel()
        result = self.predict(image, model)
        self.assertEqual(result["baseline"].shape, (3, 7))
        self.assertEqual(result["metadata"]["original_size"], [7, 3])
        self.assertEqual(set(model.saw_modes), {"RGB"})

    def test_rejects_invalid_parameters_and_nonfinite_model_outputs(self):
        image = Image.new("RGB", (5, 7))
        for kwargs in [{"input_size": 99}, {"input_size": True},
                       {"smoothing_kernel": 2}, {"smoothing_kernel": 0},
                       {"global_weight": float("inf")}, {"global_weight": -0.1},
                       {"global_weight": 1.1}, {"overlap": 1}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.predict(image, **kwargs)
        for mode in ["nan", "shape"]:
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.predict(image, FakeModel(mode))
        with self.assertRaises(TypeError):
            self.predict("not an image")
        with self.assertRaises(TypeError):
            self.predict(image, progress="not callable")


if __name__ == "__main__":
    unittest.main()
