import unittest

import numpy as np

from synthetic_label_triage import triage_label


class SyntheticLabelTriageTest(unittest.TestCase):
    def test_linked_change_is_candidate_and_distant_change_is_ignored(self):
        official = np.zeros((80, 100), dtype=bool)
        official[20:24, 20:24] = True
        scores = np.zeros(official.shape, dtype=np.float32)
        scores[20:30, 22:34] = 4
        scores[50:60, 70:82] = 4
        candidate, uncertain, info = triage_label(official, scores, np.ones_like(official))
        self.assertTrue(candidate[25, 30])
        self.assertFalse(candidate[55, 75])
        self.assertTrue(uncertain[55, 75])
        self.assertEqual(info["official_pixels"], 16)
        self.assertTrue(info["requires_review"])

    def test_unexplained_mask_is_kept_for_review(self):
        official = np.zeros((80, 100), dtype=bool)
        official[20:30, 20:30] = True
        candidate, uncertain, info = triage_label(
            official, np.zeros(official.shape, dtype=np.float32), np.ones_like(official))
        np.testing.assert_array_equal(candidate, official)
        self.assertFalse(uncertain.any())
        self.assertIn("no_independent_residual_review", info["review_reasons"])


if __name__ == "__main__":
    unittest.main()
