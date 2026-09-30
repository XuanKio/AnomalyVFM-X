"""Independent CPU behavior checks on procedural images, with no model or ROI.

The generated surfaces are test fixtures, not evidence of industrial accuracy.
Expected defect masks are supplied only to assertions after full-image inference.
"""
import io
import unittest

try:
    import cv2
except ModuleNotFoundError as exc:
    if exc.name != 'cv2':
        raise
    raise unittest.SkipTest('Optional reference tests require requirements-reference.txt') from exc
import numpy as np
from PIL import Image

from demo_reference import ReferenceMismatch, detect_changes


def textured_surface(width=385, height=289, seed=812):
    rng = np.random.default_rng(seed)
    field = cv2.GaussianBlur(
        rng.normal(size=(height, width)).astype(np.float32), (0, 0), 3
    )
    field = 110 + field / field.std() * 22
    rgb = np.stack([field * .85, field, field * .75], axis=2)
    rgb = np.clip(rgb, 20, 220).astype(np.uint8)
    for _ in range(35):
        x = int(rng.integers(20, width - 20))
        y = int(rng.integers(20, height - 20))
        color = tuple(int(v) for v in rng.integers(45, 190, size=3))
        cv2.circle(rgb, (x, y), int(rng.integers(4, 12)), color, -1)
    return rgb


def add_scratch(rgb, x, y):
    changed = rgb.copy()
    truth = np.zeros(rgb.shape[:2], np.uint8)
    cv2.line(truth, (x, y), (x + 32, y + 19), 1, 4)
    changed[truth > 0] = (240, 245, 235)
    return changed, truth > 0


def compare(reference, inspected):
    return detect_changes(Image.fromarray(reference), Image.fromarray(inspected))


class ReferenceDetectionTests(unittest.TestCase):
    def assert_localizes(self, result, truth):
        mask = result['mask']
        self.assertTrue(mask.any(), 'No changed pixels were detected')
        recall = float((mask & truth).sum() / truth.sum())
        nearby = cv2.dilate(truth.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        nearby_precision = float((mask & nearby).sum() / mask.sum())
        self.assertGreater(recall, .55, f'Changed-region recall = {recall:.3f}')
        self.assertGreater(nearby_precision, .85,
                           f'Detected pixels near actual change = {nearby_precision:.3f}')
        self.assertTrue(result['metadata']['regions'])
        self.assertFalse(mask[~result['valid_mask']].any())

    def test_detects_defect_at_multiple_unprovided_positions(self):
        ref = textured_surface()
        for x, y in [(65, 60), (180, 125), (290, 205)]:
            with self.subTest(position=(x, y)):
                query, truth = add_scratch(ref, x, y)
                self.assert_localizes(compare(ref, query), truth)

    def test_healthy_identity(self):
        ref = textured_surface()
        result = compare(ref, ref)
        self.assertFalse(result['mask'].any())
        self.assertEqual(result['metadata']['regions'], [])

    def test_healthy_illumination_change(self):
        ref = textured_surface()
        gradient = np.linspace(-8, 8, ref.shape[1], dtype=np.float32)[None, :, None]
        query = np.clip(ref.astype(np.float32) + 12 + gradient, 0, 255).astype(np.uint8)
        self.assertFalse(compare(ref, query)['mask'].any())

    def test_healthy_translation(self):
        ref = textured_surface()
        query = cv2.warpAffine(ref, np.float32([[1, 0, 3.5], [0, 1, -2.25]]),
                               (ref.shape[1], ref.shape[0]), borderMode=cv2.BORDER_REFLECT)
        self.assertFalse(compare(ref, query)['mask'].any())

    def test_healthy_bright_border_translation_has_no_padding_artifact(self):
        ref = np.full((600, 800, 3), 220, np.uint8)
        ref[80:520, 120:680] = textured_surface(width=560, height=440)
        query = cv2.warpAffine(ref, np.float32([[1, 0, 40], [0, 1, 0]]),
                               (800, 600), borderMode=cv2.BORDER_REFLECT)
        result = compare(ref, query)
        self.assertFalse(result['mask'].any(),
                         'Warp padding leaked into the unchanged bright border')
        self.assertEqual(result['metadata']['regions'], [])

    def test_healthy_jpeg_recompression(self):
        ref = textured_surface()
        stream = io.BytesIO()
        Image.fromarray(ref).save(stream, format='JPEG', quality=85)
        stream.seek(0)
        query = np.array(Image.open(stream).convert('RGB'))
        self.assertFalse(compare(ref, query)['mask'].any())

    def test_blank_reference_is_rejected(self):
        ref = np.full((289, 385, 3), 110, dtype=np.uint8)
        with self.assertRaises(ReferenceMismatch):
            compare(ref, ref)

    def test_different_dimensions_are_rejected(self):
        ref = textured_surface()
        with self.assertRaises(ReferenceMismatch):
            compare(ref, ref[:-1])

    def test_unrelated_reference_is_rejected(self):
        with self.assertRaises(ReferenceMismatch):
            compare(textured_surface(seed=812), textured_surface(seed=901))

    def test_similar_palette_and_partial_structure_do_not_validate_wrong_surface(self):
        ref = textured_surface(seed=812)
        other = textured_surface(seed=901)
        # Preserve some structure and the same color distribution, but replace
        # texture across the whole surface. This is not a localized defect.
        query = np.rint(.6 * ref + .4 * other).astype(np.uint8)
        with self.assertRaises(ReferenceMismatch):
            compare(ref, query)

    def test_excluded_overlap_has_no_scores_or_detections(self):
        ref = textured_surface()
        query = cv2.warpAffine(ref, np.float32([[1, 0, 9], [0, 1, 5]]),
                               (ref.shape[1], ref.shape[0]))
        query[:, :5] = 245
        result = compare(ref, query)
        valid = result['valid_mask']
        self.assertFalse(valid[:, :5].any())
        self.assertFalse(result['mask'][~valid].any())
        self.assertTrue((result['scores'][~valid] == 0).all())
        self.assertFalse(result['mask'].any())

    def test_odd_large_dimensions_return_inspected_coordinates(self):
        ref = textured_surface(width=1201, height=809)
        query = cv2.warpAffine(ref, np.float32([[1, 0, 5.3], [0, 1, -3.7]]),
                               (1201, 809), borderMode=cv2.BORDER_REFLECT)
        query, truth = add_scratch(query, 873, 541)
        result = compare(ref, query)
        self.assertEqual(result['mask'].shape, (809, 1201))
        self.assert_localizes(result, truth)
        x, y = result['metadata']['maximum_xy']
        nearby = cv2.dilate(truth.astype(np.uint8), np.ones((9, 9), np.uint8))
        self.assertTrue(nearby[y, x], 'Peak coordinates are not on the inspected-image defect')


if __name__ == '__main__':
    unittest.main()
