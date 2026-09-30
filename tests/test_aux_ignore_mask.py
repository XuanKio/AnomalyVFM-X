import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torchvision.transforms as T

from aux_dataset import AuxilaryDataset
from mask_ops import preserve_defects_at_decoder_scale


class AuxIgnoreMaskTest(unittest.TestCase):
    def test_reviewed_ignore_map_is_excluded_only_on_bad_image(self):
        with tempfile.TemporaryDirectory() as root:
            dataset = Path(root)
            for folder in ("train/ok", "train/bad", "ground_truth/bad", "ground_truth/ignore"):
                (dataset / folder).mkdir(parents=True)
            rgb = np.zeros((8, 8, 3), dtype=np.uint8)
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[0, 0] = 255
            ignore = np.zeros((8, 8), dtype=np.uint8)
            ignore[6, 6] = 255
            Image.fromarray(rgb).save(dataset / "train/ok/000.png")
            Image.fromarray(rgb).save(dataset / "train/bad/000.png")
            Image.fromarray(mask).save(dataset / "ground_truth/bad/000.png")
            Image.fromarray(ignore).save(dataset / "ground_truth/ignore/000.png")
            samples = AuxilaryDataset(dataset, T.ToTensor(),
                                      lambda tensor: preserve_defects_at_decoder_scale(tensor, 4))
            by_label = {sample["is_anom"]: sample for sample in samples}
            self.assertEqual(tuple(by_label[1.0]["valid_mask"].shape), (1, 4, 4))
            self.assertEqual(int(by_label[1.0]["valid_mask"].sum()), 15)
            self.assertEqual(int(by_label[1.0]["mask"].sum()), 1)
            self.assertEqual(int(by_label[0.0]["valid_mask"].sum()), 16)


if __name__ == "__main__":
    unittest.main()
