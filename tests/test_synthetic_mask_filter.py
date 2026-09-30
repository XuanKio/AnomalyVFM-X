import unittest

import numpy as np

from synthetic_mask_filter import hysteresis_mask


class HysteresisMaskTest(unittest.TestCase):
    def test_keeps_weak_connected_scratch_without_unrelated_changes(self):
        scores = np.zeros((30, 50), np.float32)
        scores[10:13, 10:14] = 4  # accepted high-confidence seed
        scores[11, 14:35] = 1.5  # weak continuation
        scores[22, 22:30] = 2  # disconnected edit-region noise
        scores[11, 45:] = 5  # strong response outside the generator's edit
        support = np.zeros_like(scores, dtype=bool)
        support[5:25, 5:40] = True
        mask, metadata = hysteresis_mask(scores, support)
        self.assertTrue(mask[11, 34])
        self.assertFalse(mask[22, 25])
        self.assertFalse(mask[11, 47])
        self.assertEqual(metadata["reason"], "grown_from_strong_seed")

    def test_rejects_pair_without_reliable_seed(self):
        scores = np.full((30, 30), 1.5, np.float32)
        mask, metadata = hysteresis_mask(scores, np.ones_like(scores, bool))
        self.assertFalse(mask.any())
        self.assertEqual(metadata["reason"], "no_strong_seed")


if __name__ == "__main__":
    unittest.main()
