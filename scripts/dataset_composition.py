"""Reconstruct where the dataset's images came from (see report/dataset_sources.md).

The merged dataset was renamed (snake_XXXXXX / snake_like_XXXXXX), so the sources are inferred:
1. image-size statistics per class, and ID blocks with constant resolution (one block = one export)
2. contact sheets of images sampled across the ID range, to see what each block contains

Usage: python scripts/dataset_composition.py   -> prints stats, writes results/dataset_composition/*.jpg
"""
from __future__ import annotations

import collections
import os
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from snakeguard import load_config


def main():
    cfg = load_config()
    root = Path(cfg["data_root"])
    out = Path(cfg["results_dir"]) / "dataset_composition"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ["train", "valid", "test"]:
        for f in os.listdir(root / split / "images"):
            kind = "snake_like" if f.startswith("snake_like") else "snake"
            num = int(re.findall(r"\d+", f)[-1])
            with Image.open(root / split / "images" / f) as im:
                w, h = im.size
            rows.append((kind, num, w, h, split, f))

    for kind in ["snake", "snake_like"]:
        r = sorted((x for x in rows if x[0] == kind), key=lambda x: x[1])
        sizes = collections.Counter((x[2], x[3]) for x in r)
        print(f"{kind}: {len(r)} images, ids {r[0][1]}-{r[-1][1]}; top sizes: {sizes.most_common(8)}")
        blocks, cur = [], None  # (size, first_id, last_id, count)
        for x in r:
            key = (x[2], x[3])
            if cur and key == cur[0] and x[1] - cur[2] <= 5:
                cur = (key, cur[1], x[1], cur[3] + 1)
            else:
                if cur:
                    blocks.append(cur)
                cur = (key, x[1], x[1], 1)
        blocks.append(cur)
        for b in blocks:
            if b[3] >= 200:
                print(f"   same-size block {b[0][0]}x{b[0][1]}: ids {b[1]}-{b[2]} ({b[3]} images)")

        pick = [r[i] for i in np.linspace(0, len(r) - 1, 48).astype(int)]
        tiles = []
        for x in pick:
            im = cv2.resize(cv2.imread(str(root / x[4] / "images" / x[5])), (160, 160))
            cv2.putText(im, str(x[1]), (3, 14), 0, 0.45, (0, 255, 255), 1)
            tiles.append(im)
        sheet = np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, 48, 8)])
        cv2.imwrite(str(out / f"contact_sheet_{kind}.jpg"), sheet)
    print("contact sheets in", out)


if __name__ == "__main__":
    main()
