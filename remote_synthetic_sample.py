"""Fetch a bounded number of AnomalyVFM synthetic pairs from its remote ZIP.

ZIP central-directory and selected members are fetched with HTTP byte ranges.
This is a connectivity/sample check, not a replacement training DataLoader.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import zipfile
from collections import OrderedDict
from pathlib import Path

import requests
from PIL import Image


URL = "https://huggingface.co/datasets/MaticFuc/AnomalyVFM_Synthetic_Dataset/resolve/main/dataset.zip"
ROOT = "synthetic_dataset_flux_filter_dinov3"
BLOCK = 1024 * 1024


class BoundedRangeFile(io.BufferedIOBase):
    """Seekable remote file that refuses servers ignoring Range or large reads."""

    def __init__(self, url: str, max_network_bytes: int):
        self.session = requests.Session()
        head = self.session.head(url, allow_redirects=True, timeout=30)
        head.raise_for_status()
        self.length = int(head.headers["Content-Length"])
        if head.headers.get("Accept-Ranges") != "bytes":
            raise ValueError("Dataset server does not advertise byte ranges")
        self.url = url
        self.position = 0
        self.max_network_bytes = max_network_bytes
        self.network_bytes = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        target = offset if whence == 0 else self.position + offset if whence == 1 else self.length + offset
        if target < 0 or target > self.length:
            raise ValueError("Seek outside remote ZIP")
        self.position = target
        return target

    def _range(self, start: int, end: int) -> bytes:
        expected = end - start
        if self.network_bytes + expected > self.max_network_bytes:
            raise ValueError("Network byte cap reached; choose a smaller sample")
        response = self.session.get(self.url, headers={"Range": f"bytes={start}-{end-1}"},
                                    timeout=60, stream=True)
        expected_header = f"bytes {start}-{end-1}/{self.length}"
        try:
            if response.status_code != 206 or response.headers.get("Content-Range") != expected_header:
                raise ValueError("Server ignored or changed the requested byte range")
            chunks = []
            received = 0
            for chunk in response.iter_content(1024 * 1024):
                received += len(chunk)
                if received > expected:
                    raise IOError("Remote server sent more bytes than requested")
                chunks.append(chunk)
            data = b"".join(chunks)
        finally:
            response.close()
        if len(data) != expected:
            raise IOError("Truncated remote ZIP range")
        self.network_bytes += len(data)
        return data

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.length - self.position
            if size > 16 * BLOCK:
                raise ValueError("Unbounded remote ZIP read is disabled")
        size = min(size, self.length - self.position)
        if not size:
            return b""
        if size >= BLOCK:
            data = self._range(self.position, self.position + size)
            self.position += size
            return data
        chunks = []
        left = size
        while left:
            start = self.position // BLOCK * BLOCK
            if start not in self.cache:
                self.cache[start] = self._range(start, min(start + BLOCK, self.length))
                if len(self.cache) > 2:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(start)
            block = self.cache[start]
            offset = self.position - start
            part = block[offset:offset + left]
            chunks.append(part)
            self.position += len(part)
            left -= len(part)
        return b"".join(chunks)

    def close(self) -> None:
        self.session.close()
        super().close()


def fetch_sample(output: Path, count: int, max_network_mib: int = 64,
                 url: str = URL) -> dict:
    if not 1 <= count <= 20 or max_network_mib < 16:
        raise ValueError("count must be 1..20 and max_network_mib must be >=16")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    with BoundedRangeFile(url, max_network_mib * 1024 * 1024) as remote:
        with zipfile.ZipFile(remote) as archive:
            names = set(archive.namelist())
            pattern = re.compile(rf"^{ROOT}/train/bad/(\d{{5}})\.png$")
            ids = sorted(match.group(1) for name in names if (match := pattern.fullmatch(name)))
            complete = [number for number in ids if
                        f"{ROOT}/train/ok/{number}.png" in names and
                        f"{ROOT}/ground_truth/bad/{number}.png" in names]
            if len(complete) < count:
                raise ValueError("Remote archive has too few complete image/mask pairs")
            # Evenly distribute selections by archive ID, without inspecting image content.
            selected = [complete[round(i * (len(complete) - 1) / max(count - 1, 1))]
                        for i in range(count)]
            rows = []
            for number in selected:
                files = {}
                for part in ("train/ok", "train/bad", "ground_truth/bad"):
                    archive_name = f"{ROOT}/{part}/{number}.png"
                    content = archive.read(archive_name)  # ZIP CRC checked by zipfile.
                    target = output / part / f"{number}.png"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
                    with Image.open(target) as image:
                        image.verify()
                    files[part] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                rows.append({"id": number, "files": files})
        report = {"url": url, "archive_bytes": remote.length,
                  "network_bytes": remote.network_bytes, "sample_count": count,
                  "selection": "evenly spaced file IDs, before reading image content",
                  "pairs": rows}
    (output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--max-network-mib", type=int, default=64)
    args = parser.parse_args()
    result = fetch_sample(args.output, args.count, args.max_network_mib)
    print(json.dumps({key: result[key] for key in ("archive_bytes", "network_bytes", "sample_count")},
                     indent=2))


if __name__ == "__main__":
    main()
