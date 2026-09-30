import unittest

import numpy as np

from evaluate_visa_slice import PixelAccumulator


class VisaMetricsTest(unittest.TestCase):
    def test_perfect_ranking_and_no_healthy_alarm(self):
        metrics = PixelAccumulator()
        scores = np.array([[.9, .8], [.2, .1]], np.float32)
        truth = np.array([[True, True], [False, False]])
        metrics.add(scores, truth, abnormal=True)
        metrics.add(np.zeros((2, 2), np.float32), np.zeros((2, 2), bool), abnormal=False)
        result = metrics.summary()
        self.assertAlmostEqual(result["pixel_auroc_1001_bins"], 1.0)
        self.assertAlmostEqual(result["pixel_ap_1001_bins"], 1.0)
        self.assertAlmostEqual(result["pixel_f1_at_0_5"], 1.0)
        self.assertEqual(result["healthy_image_false_alarm_rate_at_0_5"], 0.0)

    def test_inverted_ranking_and_healthy_alarm(self):
        metrics = PixelAccumulator()
        metrics.add(np.array([[.1, .9]], np.float32),
                    np.array([[True, False]]), abnormal=True)
        metrics.add(np.array([[.9]], np.float32),
                    np.array([[False]]), abnormal=False)
        result = metrics.summary()
        self.assertAlmostEqual(result["pixel_auroc_1001_bins"], 0.0)
        self.assertEqual(result["healthy_image_false_alarm_rate_at_0_5"], 1.0)


if __name__ == "__main__":
    unittest.main()
