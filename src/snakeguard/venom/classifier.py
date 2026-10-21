"""Plug-in for the SnakeGuard pipeline: `pipeline.VenomClassifier` backed by a trained VenomNet."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..pipeline import VenomClassifier
from .data import eval_transform
from .model import VenomNet, pick_device


def rescale_score(p: float, threshold: float) -> float:
    """Monotone piecewise-linear map that sends the operating threshold to 0.5: [0, thr] -> [0, 0.5],
    [thr, 1] -> [0.5, 1]. Rankings (and so AUROC / recall / specificity at the threshold) are unchanged."""
    if threshold <= 0 or threshold >= 1:
        return float(p)
    return float(0.5 * p / threshold if p < threshold else 0.5 + 0.5 * (p - threshold) / (1 - threshold))


class TorchVenomClassifier(VenomClassifier):
    """predict(crop_rgb uint8 HxWx3) -> P(venomous) from the deployed score (`score`: venom head or species head).
    `raw_threshold` is the operating point chosen on valid (venomous recall target).

    rescale=False: predict returns the raw probability; use `threshold` (= raw_threshold) as
    `PipelineConfig.venom_threshold`. rescale=True (the app): predict returns `rescale_score(p)`, so the operating
    point is 0.5 (`threshold` = 0.5) and the pipeline's threat score, which multiplies P(venomous), reflects the
    venom decision (raw probabilities sit far below 0.5 at the 0.141 threshold)."""

    def __init__(self, checkpoint: str | Path, device: str = "auto", min_side: int = 16, rescale: bool = False):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.device = pick_device(device)
        self.model = VenomNet(ck["n_species"], ck["backbone"], pretrained=False)
        self.model.load_state_dict(ck["model"])
        self.model.to(self.device).eval()
        self.raw_threshold = float(ck.get("threshold", 0.5))
        self.rescale = rescale
        self.threshold = 0.5 if rescale else self.raw_threshold
        # "venom": venom head; "species": sum of P(species) over venomous species (chosen on valid, see PLAN.md)
        self.score = ck.get("score", "venom")
        self.venomous_species = torch.as_tensor(ck.get("venomous_species", []), dtype=torch.long).to(self.device)
        self.img_size = int(ck.get("img_size", 224))
        self.tf = eval_transform(self.img_size)
        self.min_side = min_side

    @torch.no_grad()
    def predict_raw(self, crop_rgb: np.ndarray) -> float | None:
        if crop_rgb is None or min(crop_rgb.shape[:2]) < self.min_side:
            return None  # too small to judge -> pipeline treats unknown as venomous
        x = self.tf(np.ascontiguousarray(crop_rgb))[None].to(self.device)
        v, s = self.model(x)
        if self.score == "species":
            return float(s.float().softmax(-1)[0, self.venomous_species].sum().clamp(0, 1))
        return float(torch.sigmoid(v)[0])

    def predict(self, crop_rgb: np.ndarray) -> float | None:
        p = self.predict_raw(crop_rgb)
        if p is None or not self.rescale:
            return p
        return rescale_score(p, self.raw_threshold)
