"""Greedy detection <-> ground-truth matching, class-agnostic, so class confusions are visible."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data import GTObject
from ..detection.yolo import Detection


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


@dataclass
class Match:
    det: Detection | None
    gt: GTObject | None
    iou: float
    kind: str  # "TP" | "confusion" (box ok, wrong class) | "FP" (no GT) | "FN" (missed GT)


def match(dets: list[Detection], gts: list[GTObject], iou_thr: float = 0.5) -> list[Match]:
    """Detections in descending score order each take the best unmatched GT with IoU >= thr.
    Class is ignored for matching and then used to label the match TP or confusion."""
    dets = sorted(dets, key=lambda d: -d.score)
    ious = iou_matrix(np.array([d.box for d in dets]).reshape(-1, 4),
                      np.array([g.box for g in gts]).reshape(-1, 4))
    used, out = set(), []
    for i, d in enumerate(dets):
        best, best_iou = None, iou_thr
        for j in range(len(gts)):
            if j not in used and ious[i, j] >= best_iou:
                best, best_iou = j, ious[i, j]
        if best is None:
            out.append(Match(d, None, float(ious[i].max()) if len(gts) else 0.0, "FP"))
        else:
            used.add(best)
            out.append(Match(d, gts[best], float(best_iou), "TP" if gts[best].cls == d.cls else "confusion"))
    out += [Match(None, g, 0.0, "FN") for j, g in enumerate(gts) if j not in used]
    return out
