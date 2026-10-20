"""Venom dataset: SnakeCLEF 2023 subset listed in `data/snakeclef/manifest.csv` (built by
scripts/build_venom_dataset.py). Labels: `venomous` (SnakeCLEF MIVS) and `class_id` (species)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

from .. import PROJECT_ROOT

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_venom_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else PROJECT_ROOT / "configs" / "venom.yaml"
    cfg = yaml.safe_load(open(path))
    sc = cfg["snakeclef"]
    for k in ("root", "meta_dir", "image_dir", "manifest"):
        sc[k] = str((PROJECT_ROOT / sc[k]).resolve())
    if "train" in cfg:
        cfg["train"]["out_dir"] = str((PROJECT_ROOT / cfg["train"]["out_dir"]).resolve())
    return cfg


def load_manifest(cfg: dict, split: str | None = None) -> pd.DataFrame:
    """Rows with an image on disk. `species_idx` is a dense 0..S-1 index over all species in the manifest
    (same mapping for every split)."""
    m = pd.read_csv(cfg["snakeclef"]["manifest"], dtype={"observation_id": str})
    species = np.sort(m.class_id.unique())
    m["species_idx"] = np.searchsorted(species, m.class_id)
    if "available" in m:
        m = m[m.available.astype(bool)]
    if split is not None:
        m = m[m.split == split]
    return m.reset_index(drop=True)


def train_transform(size: int = 224, crop_scale=(0.35, 1.0)) -> T.Compose:
    return T.Compose([
        T.ToImage(),
        T.RandomResizedCrop(size, scale=tuple(crop_scale), antialias=True),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2, 0.02),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(size: int = 224, normalize: bool = True) -> T.Compose:
    """Shorter side -> size, centre crop. Used for valid/test and for detector crops in the pipeline.
    normalize=False keeps pixels in [0, 1] (the XAI code perturbs in pixel space and normalizes itself)."""
    steps = [T.ToImage(), T.Resize(size, antialias=True), T.CenterCrop(size), T.ToDtype(torch.float32, scale=True)]
    if normalize:
        steps.append(T.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return T.Compose(steps)


class VenomDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, image_dir: str | Path, transform):
        self.paths = [str(Path(image_dir) / p) for p in manifest.image_path]
        self.venom = manifest.venomous.to_numpy(np.float32)
        self.species = manifest.species_idx.to_numpy(np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = cv2.imread(self.paths[i], cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"cannot read {self.paths[i]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img), torch.tensor(self.venom[i]), torch.tensor(self.species[i])
