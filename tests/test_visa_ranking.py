import unittest

import numpy as np

from analyze_visa_ranking import analyze, image_ap, image_auc


class VisaRankingTest(unittest.TestCase):
    def test_exact_rank_metrics(self):
        self.assertEqual(image_auc(np.array([.9, .8]), np.array([.2, .1])), 1)
        self.assertEqual(image_ap(np.array([.9, .8]), np.array([.2, .1])), 1)
        self.assertEqual(image_auc(np.array([.1]), np.array([.9])), 0)

    def test_threshold_uses_only_calibration_normals(self):
        rows = []
        for category, positives, negatives in (
            ("candle", [.6], [.1, .2, .3, .4]),
            ("capsules", [.5], [.15, .25, .35, .45]),
        ):
            rows.extend({"class": category, "variant": "baseline", "label": "bad",
                         "max_score": value} for value in positives)
            rows.extend({"class": category, "variant": "baseline", "label": "normal",
                         "max_score": value} for value in negatives)
        report = {"per_image": rows, "dataset_manifest_sha256": "test", "head_sha256": {}}
        result = analyze(report, allowed_healthy_alarms=1)["variants"]["baseline"]
        self.assertGreater(result["threshold_from_candle_normals"], .3)
        self.assertLess(result["threshold_from_candle_normals"], .4)
        self.assertEqual(result["by_class"]["capsules"]["healthy_alarm_rate_at_candle_threshold"], .5)


if __name__ == "__main__":
    unittest.main()
