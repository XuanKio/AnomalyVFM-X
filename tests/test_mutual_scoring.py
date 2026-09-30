import unittest

import torch

from evaluate_mutual_scoring import aggregate, interval_mean, mutual_scores, standardize


class MutualScoringTest(unittest.TestCase):
    def test_interval_mean_ignores_largest_distances(self):
        distances = torch.tensor([[0.1, 0.2, 0.9, 1.0]])
        self.assertTrue(torch.isclose(interval_mean(distances, 0.5), torch.tensor([0.15])).all())

    def test_aggregate_keeps_shape_and_unit_norm(self):
        out = aggregate(torch.randn(16, 8), grid=4, radius=3).float()
        self.assertEqual(out.shape, (16, 8))
        self.assertTrue(torch.allclose(out.norm(dim=-1), torch.ones(16), atol=1e-2))

    def test_patch_missing_from_other_images_scores_highest(self):
        torch.manual_seed(0)
        grid, count = 4, 6
        normal = torch.randn(grid * grid, 8)
        batch = normal.repeat(count, 1, 1) + 0.01 * torch.randn(count, grid * grid, 8)
        batch[0, 5] = torch.randn(8) * 5  # one defect patch in image 0
        scores = mutual_scores({24: batch.half()}, grid, fraction=0.3, chunk=2)
        self.assertEqual(int(scores[0].flatten().argmax()), 5)

    def test_standardize_is_shared_across_batch(self):
        out = standardize(torch.stack([torch.zeros(4, 4), torch.ones(4, 4)]))
        self.assertGreater(float(out[1].min()), float(out[0].max()))


if __name__ == "__main__":
    unittest.main()
