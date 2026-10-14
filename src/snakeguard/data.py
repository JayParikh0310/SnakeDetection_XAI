"""Reading the YOLO-format dataset, including polygon labels (used as fine GT masks)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class GTObject:
    cls: int
    box: np.ndarray                    # xyxy, pixels, original image
    polygon: np.ndarray | None = None  # (K, 2) pixels, original image; None for box-only labels


@dataclass
class Sample:
    image_id: str
    image_path: Path
    objects: list[GTObject] = field(default_factory=list)

    def load_rgb(self) -> np.ndarray:
        img = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"cannot read {self.image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def parse_label_file(path: Path, w: int, h: int) -> list[GTObject]:
    """YOLO label lines: `cls cx cy bw bh` (box) or `cls x1 y1 x2 y2 ...` (polygon), normalized."""
    objs = []
    if not path.exists():
        return objs
    for line in path.read_text().splitlines():
        v = line.split()
        if not v:
            continue
        cls, coords = int(float(v[0])), np.array(v[1:], dtype=np.float64)
        if len(coords) == 4:
            cx, cy, bw, bh = coords * [w, h, w, h]
            objs.append(GTObject(cls, np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])))
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            poly = coords.reshape(-1, 2) * [w, h]
            box = np.array([*poly.min(0), *poly.max(0)])  # same box ultralytics derives from segments
            objs.append(GTObject(cls, box, poly))
        # anything else is malformed and skipped
    return objs


def list_split(data_root: str | Path, split: str) -> list[Path]:
    img_dir = Path(data_root) / split / "images"
    return sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)


def load_sample(img_path: Path, shape_hw: tuple[int, int] | None = None) -> Sample:
    if shape_hw is None:
        img = cv2.imread(str(img_path))
        shape_hw = img.shape[:2]
    h, w = shape_hw
    lbl = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
    return Sample(img_path.stem, img_path, parse_label_file(lbl, w, h))


def polygon_mask(poly: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape_hw, np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    return m.astype(bool)
