import unittest

import numpy as np

from evaluate_defect_size import region_coverage, size_bin, threshold_from_normals


class DefectSizeTest(unittest.TestCase):
    def test_calibration_uses_only_provided_healthy_scores(self):
        values = [.1, .2, .3, .4, .5]
        threshold = threshold_from_normals(values, allowed_alarms=1)
        self.assertEqual(sum(value >= threshold for value in values), 1)

    def test_region_coverage_is_confined_to_truth_component(self):
        scores = np.array([[.9, .2], [.8, .1]])
        region = np.array([[False, True], [True, False]])
        result = region_coverage(scores, region, threshold=.5, top_one_percent=.85)
        self.assertEqual(result["pixels"], 2)
        self.assertEqual(result["coverage_at_calibrated_threshold"], .5)
        self.assertEqual(result["coverage_in_image_top_1pct"], 0)
        self.assertEqual([size_bin(area) for area in (99, 100, 999, 1000)],
                         ["under_100", "100_to_999", "100_to_999", "1000_plus"])


if __name__ == "__main__":
    unittest.main()
