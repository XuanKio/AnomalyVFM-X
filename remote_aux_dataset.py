"""On-demand training pairs from the released Hugging Face ZIP.

This keeps the ZIP remote. HTTP transfers still occur for the directory and
every image/mask read, so training speed depends on network latency.
"""
from __future__ import annotations

import io
import re
import zipfile

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from remote_synthetic_sample import BoundedRangeFile, ROOT, URL


class RemoteAuxilaryDataset(Dataset):
    def __init__(self, transform, mask_transform, *, url: str = URL,
                 max_network_bytes: int):
        self.remote = BoundedRangeFile(url, max_network_bytes)
        try:
            self.archive = zipfile.ZipFile(self.remote)
            names = set(self.archive.namelist())
            pattern = re.compile(rf"^{ROOT}/train/bad/(\d{{5}})\.png$")
            ids = sorted(match.group(1) for name in names if (match := pattern.fullmatch(name)))
            self.ids = [number for number in ids if
                        f"{ROOT}/train/ok/{number}.png" in names and
                        f"{ROOT}/ground_truth/bad/{number}.png" in names]
            if not self.ids:
                raise ValueError("Remote ZIP contains no complete good/bad/mask pairs")
        except Exception:
            self.remote.close()
            raise
        self.transform = transform
        self.mask_transform = mask_transform
        self.url = url

    def __len__(self) -> int:
        return 2 * len(self.ids)

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < len(self):
            raise IndexError(index)
        bad = index >= len(self.ids)
        number = self.ids[index % len(self.ids)]
        part = "bad" if bad else "ok"
        name = f"{ROOT}/train/{part}/{number}.png"
        with Image.open(io.BytesIO(self.archive.read(name))) as opened:
            image = self.transform(opened.convert("RGB"))
        if bad:
            mask_name = f"{ROOT}/ground_truth/bad/{number}.png"
            with Image.open(io.BytesIO(self.archive.read(mask_name))) as opened:
                mask = T.ToTensor()(opened.convert("L"))
        else:
            mask = torch.zeros((1, image.shape[-2], image.shape[-1]))
        mask = self.mask_transform(mask)
        mask = torch.where(mask > 0.5, 1.0, 0.0)
        return {"image": image, "mask": mask, "is_anom": float(bad),
                "path": f"{self.url}#{name}"}

    def close(self) -> None:
        self.archive.close()
        self.remote.close()
