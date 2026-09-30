import unittest

import torch

from evaluate_feature_layers import patch_distance


class FeatureLayerProbeTest(unittest.TestCase):
    def test_nearest_normal_patch_distance(self):
        query = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [-1.0, 0.0], [1.0, 1.0]]])
        bank = torch.tensor([[1.0], [0.0]])
        scores = patch_distance(query, bank, grid=2)
        self.assertEqual(tuple(scores.shape), (1, 1, 2, 2))
        self.assertAlmostEqual(scores[0, 0, 0, 0].item(), 0)
        self.assertAlmostEqual(scores[0, 0, 0, 1].item(), 1)
        self.assertAlmostEqual(scores[0, 0, 1, 0].item(), 1)  # 1-cos(-x,+x) clipped
        self.assertAlmostEqual(scores[0, 0, 1, 1].item(), 1 - 2 ** -.5, places=6)


if __name__ == "__main__":
    unittest.main()
