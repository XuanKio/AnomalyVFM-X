"""Regression checks for score-faithful anomaly visualization; no model needed."""

import unittest

import numpy as np
from PIL import Image

from demo_visuals import render_prediction


class PredictionVisualsTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (5, 3), (100, 120, 140))

    def test_constant_low_scores_stay_blue_and_mask_empty(self):
        for score in (0.0, 0.1):
            with self.subTest(score=score):
                result = render_prediction(self.image, np.full((3, 5), score))
                heatmap = np.asarray(result["heatmap"])
                self.assertEqual(len(np.unique(heatmap.reshape(-1, 3), axis=0)), 1)
                self.assertGreater(int(heatmap[0, 0, 2]), int(heatmap[0, 0, 0]))
                self.assertEqual(np.count_nonzero(result["mask"]), 0)

    def test_equal_scores_have_equal_colors_across_images(self):
        low_context = np.full((3, 5), 0.1)
        high_context = np.full((3, 5), 0.9)
        low_context[1, 2] = high_context[1, 2] = 0.4
        low_result = render_prediction(self.image, low_context)
        high_result = render_prediction(self.image, high_context)
        self.assertEqual(
            low_result["heatmap"].getpixel((2, 1)),
            high_result["heatmap"].getpixel((2, 1)),
        )

    def test_threshold_changes_area_and_includes_equal_values(self):
        values = np.array([[0.0, 0.2, 0.5, 0.8, 1.0]] * 3)
        low = render_prediction(self.image, values, threshold=0.2)
        high = render_prediction(self.image, values, threshold=0.8)
        np.testing.assert_array_equal(np.asarray(low["mask"]), (values >= 0.2) * 255)
        np.testing.assert_array_equal(np.asarray(high["mask"]), (values >= 0.8) * 255)
        self.assertEqual(np.count_nonzero(low["mask"]), 12)
        self.assertEqual(np.count_nonzero(high["mask"]), 6)
        for key in ("input", "heatmap", "overlay"):
            np.testing.assert_array_equal(np.asarray(low[key]), np.asarray(high[key]))

    def test_threshold_endpoints(self):
        values = np.zeros((3, 5))
        values[1, 2] = 1.0
        self.assertEqual(np.count_nonzero(render_prediction(self.image, values, 0)["mask"]), 15)
        self.assertEqual(np.count_nonzero(render_prediction(self.image, values, 1)["mask"]), 1)

    def test_nonsquare_alignment_and_original_dimensions(self):
        values = np.zeros((3, 5))
        values[1, 4] = 1.0
        result = render_prediction(self.image, values)
        self.assertEqual(set(result), {"input", "heatmap", "overlay", "mask", "composite"})
        for key in ("input", "heatmap", "overlay", "mask"):
            self.assertEqual(result[key].size, (5, 3))
        self.assertEqual(result["mask"].getpixel((4, 1)), 255)
        self.assertEqual(np.count_nonzero(result["mask"]), 1)
        red, _, blue = result["heatmap"].getpixel((4, 1))
        self.assertGreater(red, blue)
        composite = result["composite"]
        column_width = composite.width // 4
        title_height = composite.height - self.image.height
        self.assertGreater(title_height, 0)
        for index, key in enumerate(("input", "heatmap", "overlay", "mask")):
            left = column_width * index + (column_width - self.image.width) // 2
            cropped = composite.crop((left, title_height, left + 5, title_height + 3))
            np.testing.assert_array_equal(np.asarray(cropped), np.asarray(result[key].convert("RGB")))

    def test_modes_and_overlay_blend(self):
        result = render_prediction(self.image.convert("RGBA"), np.full((3, 5), 0.4))
        for key in ("input", "heatmap", "overlay", "composite"):
            self.assertEqual(result[key].mode, "RGB")
        self.assertEqual(result["mask"].mode, "L")
        expected = Image.blend(result["input"], result["heatmap"], alpha=0.45)
        np.testing.assert_array_equal(np.asarray(result["overlay"]), np.asarray(expected))

    def test_does_not_modify_inputs(self):
        values = np.linspace(0, 1, 15).reshape(3, 5)
        before_values = values.copy()
        before_image = self.image.tobytes()
        render_prediction(self.image, values)
        np.testing.assert_array_equal(values, before_values)
        self.assertEqual(self.image.tobytes(), before_image)

    def test_invalid_map_shape(self):
        for shape in ((5, 3), (3, 5, 1), (15,), (0, 5)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                render_prediction(self.image, np.zeros(shape))

    def test_invalid_map_range_or_nonfinite(self):
        for invalid in (-0.001, 1.001, np.nan, np.inf, -np.inf):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                values = np.zeros((3, 5))
                values[1, 2] = invalid
                render_prediction(self.image, values)

    def test_invalid_map_dtype(self):
        for dtype in (complex, object, str, bool):
            with self.subTest(dtype=dtype), self.assertRaises(TypeError):
                render_prediction(self.image, np.zeros((3, 5), dtype=dtype))

    def test_invalid_threshold_range_or_nonfinite(self):
        for threshold in (-0.01, 1.01, np.nan, np.inf, -np.inf):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                render_prediction(self.image, np.zeros((3, 5)), threshold)

    def test_invalid_threshold_type(self):
        for threshold in ("0.5", None, True, 0.5j):
            with self.subTest(threshold=threshold), self.assertRaises(TypeError):
                render_prediction(self.image, np.zeros((3, 5)), threshold)

    def test_invalid_image(self):
        with self.assertRaises(TypeError):
            render_prediction(None, np.zeros((3, 5)))
        with self.assertRaises(ValueError):
            render_prediction(Image.new("RGB", (0, 3)), np.zeros((3, 0)))


if __name__ == "__main__":
    unittest.main()
