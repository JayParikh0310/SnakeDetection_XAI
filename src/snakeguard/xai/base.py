"""Shared Explainer contract (see CLAUDE.md §5).

Every explainer returns a saliency map normalized to [0, 1]. Detector explainers work in the
letterboxed input space (`explain_lb`, used by the metrics so that no resampling happens) and
`explain` maps the result back to the original image resolution."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import torch

from ..detection.yolo import Detection, Letterbox, YoloDetector


def normalize(sal: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Min-max to [0, 1] over the valid region; outside the valid region the map is 0."""
    sal = np.nan_to_num(np.asarray(sal, np.float32))
    v = sal[valid] if valid is not None else sal
    lo, hi = (float(v.min()), float(v.max())) if v.size else (0.0, 0.0)
    out = (sal - lo) / (hi - lo) if hi > lo else np.zeros_like(sal)
    if valid is not None:
        out[~valid] = 0
    return out.astype(np.float32)


class Explainer(ABC):
    name: str = "base"
    stage: Literal["detector", "classifier"] = "detector"

    @abstractmethod
    def explain(self, image: np.ndarray, target) -> np.ndarray:
        """image: HxWx3 RGB uint8 -> HxW float32 saliency in [0, 1]."""


class DetectorExplainer(Explainer):
    stage = "detector"

    def __init__(self, detector: YoloDetector):
        self.det = detector

    @abstractmethod
    def _explain_lb(self, x: torch.Tensor, target: Detection) -> np.ndarray:
        """x: (1,3,S,S) letterboxed input -> (S,S) raw (unnormalized) saliency."""

    def explain_lb(self, x: torch.Tensor, lb: Letterbox, target: Detection) -> np.ndarray:
        return normalize(self._explain_lb(x, target), lb.valid_mask())

    def explain(self, image: np.ndarray, target: Detection) -> np.ndarray:
        x, lb = self.det.preprocess(image)
        return normalize(lb.map_to_orig(self.explain_lb(x, lb, target)))
