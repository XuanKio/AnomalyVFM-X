import io
import unittest
import zipfile
from unittest.mock import patch

import numpy as np
import torchvision.transforms as T
from PIL import Image

from remote_aux_dataset import RemoteAuxilaryDataset
from remote_synthetic_sample import BoundedRangeFile, ROOT


def png(values):
    stream = io.BytesIO()
    Image.fromarray(np.asarray(values, dtype=np.uint8)).save(stream, format="PNG")
    return stream.getvalue()


class RemoteDatasetTest(unittest.TestCase):
    def test_reads_matching_good_bad_and_mask_without_local_files(self):
        archive_bytes = io.BytesIO()
        with zipfile.ZipFile(archive_bytes, "w") as archive:
            archive.writestr(f"{ROOT}/train/ok/00000.png",
                             png(np.full((4, 4, 3), 40)))
            archive.writestr(f"{ROOT}/train/bad/00000.png",
                             png(np.full((4, 4, 3), 180)))
            mask = np.zeros((4, 4), np.uint8)
            mask[1, 2] = 255
            archive.writestr(f"{ROOT}/ground_truth/bad/00000.png", png(mask))
        with patch("remote_aux_dataset.BoundedRangeFile",
                   side_effect=lambda *_: io.BytesIO(archive_bytes.getvalue())):
            dataset = RemoteAuxilaryDataset(T.ToTensor(), lambda x: x,
                                            max_network_bytes=1024)
            self.assertEqual(len(dataset), 2)
            good, bad = dataset[0], dataset[1]
            self.assertEqual(good["is_anom"], 0)
            self.assertEqual(bad["is_anom"], 1)
            self.assertEqual(int(good["mask"].sum()), 0)
            self.assertEqual(int(bad["mask"].sum()), 1)
            self.assertEqual(tuple(bad["image"].shape), (3, 4, 4))
            dataset.close()

    def test_rejects_server_ignoring_byte_ranges(self):
        reader = BoundedRangeFile.__new__(BoundedRangeFile)
        reader.url = "https://example.invalid/data.zip"
        reader.length = 100
        reader.max_network_bytes = 100
        reader.network_bytes = 0
        class Session:
            def get(self, *_args, **_kwargs):
                class Response:
                    status_code = 200
                    headers = {}
                    def close(self):
                        pass
                return Response()
        reader.session = Session()
        with self.assertRaisesRegex(ValueError, "ignored or changed"):
            reader._range(0, 10)


if __name__ == "__main__":
    unittest.main()
