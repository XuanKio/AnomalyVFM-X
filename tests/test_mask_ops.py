import unittest

import torch
import torchvision.transforms as T

from mask_ops import masked_mean, preserve_defects_at_decoder_scale


class MaskOpsTest(unittest.TestCase):
    def test_single_pixel_survives_training_downsample(self):
        mask = torch.zeros(1, 1024, 1536)
        mask[0, 511, 767] = 1
        old = T.Resize((96, 96))(mask)
        self.assertEqual(int((old > 0.5).sum()), 0)
        new = preserve_defects_at_decoder_scale(mask, 96)
        self.assertGreater(int((new > 0.5).sum()), 0)
        self.assertEqual(tuple(new.shape), (1, 96, 96))

    def test_empty_mask_stays_empty(self):
        new = preserve_defects_at_decoder_scale(torch.zeros(1, 672, 672), 96)
        self.assertEqual(int(new.sum()), 0)

    def test_masked_mean_excludes_uncertain_pixels_without_changing_all_valid_case(self):
        loss = torch.tensor([1.0, 100.0], requires_grad=True)
        self.assertEqual(masked_mean(loss, torch.ones(2)).item(), 50.5)
        result = masked_mean(loss, torch.tensor([1.0, 0.0]))
        self.assertEqual(result.item(), 1.0)
        result.backward()
        self.assertEqual(loss.grad.tolist(), [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
