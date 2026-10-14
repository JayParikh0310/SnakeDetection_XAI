"""Quantitative XAI metrics for detector explanations. Everything runs in the letterboxed input
space (H x W). Only the valid (non-padding) region is ranked or perturbed."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torchvision.transforms.functional as TF
from scipy.stats import spearmanr

from ..detection.yolo import PAD_VALUE, Detection, Letterbox, YoloDetector


# ---------------------------------------------------------------- faithfulness
@dataclass
class CurveResult:
    fractions: np.ndarray  # fraction of valid pixels removed/inserted
    scores: np.ndarray     # target score at each step
    auc: float             # area under the raw-score curve
    auc_norm: float        # AUC of scores / score(original image); comparable across targets


def _rank_map(sal: np.ndarray, valid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rank of each valid pixel (0 = most salient). Ties are broken randomly. Invalid = +inf."""
    v = sal[valid].astype(np.float64)
    order = np.lexsort((rng.random(v.size), -v))  # sort by -saliency, then random tiebreak
    ranks = np.empty(v.size, np.float32)
    ranks[order] = np.arange(v.size, dtype=np.float32)
    out = np.full(sal.shape, np.inf, np.float32)
    out[valid] = ranks
    return out


def blurred(x: torch.Tensor, sigma: float) -> torch.Tensor:
    k = int(2 * round(3 * sigma) + 1)
    return TF.gaussian_blur(x, kernel_size=[k, k], sigma=[sigma, sigma])


def deletion_insertion(det: YoloDetector, x: torch.Tensor, lb: Letterbox, target: Detection,
                       sal: np.ndarray, steps: int = 50, deletion_fill: str = "gray",
                       insertion_start: str = "blur", blur_sigma: float = 25, seed: int = 0,
                       batch: int = 64, mode: str = "similarity") -> dict[str, CurveResult]:
    """Petsiuk et al. 2018. Deletion: remove pixels from most to least salient (lower AUC = better).
    Insertion: reveal pixels on a blurred canvas (higher AUC = better)."""
    valid = lb.valid_mask()
    ranks = torch.from_numpy(_rank_map(sal, valid, np.random.default_rng(seed))).to(x.device)
    n = int(valid.sum())
    ks = torch.linspace(0, n, steps + 1, device=x.device)
    masks = (ranks[None] < ks[:, None, None]).float()[:, None]  # (steps+1, 1, S, S): 1 = selected

    def fill(kind):
        if kind == "gray":
            return torch.full_like(x, PAD_VALUE / 255)
        if kind == "blur":
            return blurred(x, blur_sigma)
        if kind == "black":
            return torch.zeros_like(x)
        raise ValueError(kind)

    with torch.no_grad():
        s0 = float(det.score(x, target, mode=mode)[0])
        out = {}
        for name, canvas, keep_selected in (("deletion", fill(deletion_fill), False),
                                            ("insertion", fill(insertion_start), True)):
            if keep_selected:   # insertion: selected pixels come from x, the rest from the canvas
                xb = masks * x + (1 - masks) * canvas
            else:               # deletion: selected pixels are replaced by the canvas
                xb = masks * canvas + (1 - masks) * x
            # padding stays exactly as in the original input
            pad = torch.from_numpy(~valid).to(x.device)
            xb[:, :, pad] = x[:, :, pad]
            sc = det.score(xb, target, batch=batch, mode=mode).cpu().numpy()
            fr = (ks / n).cpu().numpy()
            auc = float(np.trapezoid(sc, fr))
            out[name] = CurveResult(fr, sc, auc, auc / s0 if s0 > 1e-6 else float("nan"))
    return out


# ---------------------------------------------------------------- localization
def gt_mask_lb(lb: Letterbox, box_orig: np.ndarray, polygon_orig: np.ndarray | None = None) -> np.ndarray:
    """GT region in letterboxed space: the polygon if available, else the box."""
    import cv2
    m = np.zeros(lb.shape, np.uint8)
    if polygon_orig is not None:
        cv2.fillPoly(m, [np.round(lb.points_to_lb(polygon_orig)).astype(np.int32)], 1)
    else:
        x0, y0, x1, y1 = np.round(lb.box_to_lb(box_orig)).astype(int)
        m[max(y0, 0):y1, max(x0, 0):x1] = 1
    return m.astype(bool)


def pointing_game(sal: np.ndarray, gt: np.ndarray, tolerance: int = 15) -> float:
    """1 if the maximum of the map lies within `tolerance` px of the GT region (Zhang et al. 2018)."""
    if sal.max() <= 0:
        return 0.0
    yy, xx = np.unravel_index(np.argmax(sal), sal.shape)
    y0, y1 = max(0, yy - tolerance), yy + tolerance + 1
    x0, x1 = max(0, xx - tolerance), xx + tolerance + 1
    ys, xs = np.ogrid[y0:min(y1, sal.shape[0]), x0:min(x1, sal.shape[1])]
    disk = (ys - yy) ** 2 + (xs - xx) ** 2 <= tolerance ** 2
    return float(gt[y0:y1, x0:x1][disk].any())


def energy_pointing_game(sal: np.ndarray, gt: np.ndarray) -> float:
    """Share of saliency mass inside the GT region (Wang et al. 2020, Score-CAM)."""
    tot = float(sal.sum())
    return float(sal[gt].sum()) / tot if tot > 0 else 0.0


def saliency_iou(sal: np.ndarray, gt: np.ndarray, valid: np.ndarray, top_frac: float | None = None) -> float:
    """IoU between the GT and the top-salient pixels. By default top = as many pixels as the GT has."""
    v = sal[valid]
    k = int(gt[valid].sum()) if top_frac is None else int(top_frac * v.size)
    if k == 0:
        return 0.0
    thr = np.partition(v, -k)[-k]
    top = (sal >= thr) & valid
    inter, union = (top & gt).sum(), (top | gt).sum()
    return float(inter / union) if union else 0.0


# ---------------------------------------------------------------- agreement
def _down(sal: np.ndarray, valid: np.ndarray, size: int = 64) -> np.ndarray:
    import cv2
    ys, xs = np.where(valid)
    crop = sal[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return cv2.resize(crop.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA).ravel()


def rank_correlation(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float:
    da, db = _down(a, valid), _down(b, valid)
    if da.std() == 0 or db.std() == 0:
        return 0.0
    return float(spearmanr(da, db).statistic)


def topk_iou(a: np.ndarray, b: np.ndarray, valid: np.ndarray, frac: float = 0.2) -> float:
    va, vb = a[valid], b[valid]
    k = max(1, int(frac * va.size))
    ta = np.zeros(va.size, bool); ta[np.argsort(-va)[:k]] = True
    tb = np.zeros(vb.size, bool); tb[np.argsort(-vb)[:k]] = True
    return float((ta & tb).sum() / (ta | tb).sum())


# ---------------------------------------------------------------- sanity / robustness
class RandomizedDetector:
    """Context manager: randomizes the detector's weights from the top (head) down to layer
    `down_to` (cascading randomization, Adebayo et al. 2018) and restores them on exit."""

    def __init__(self, det: YoloDetector, down_to: int, seed: int = 0):
        self.det, self.down_to, self.seed = det, down_to, seed

    def __enter__(self):
        self.backup = copy.deepcopy(self.det.model.state_dict())
        g = torch.Generator().manual_seed(self.seed)
        layers = self.det.model.model
        for i in range(len(layers) - 1, self.down_to - 1, -1):
            for m in layers[i].modules():
                if isinstance(m, torch.nn.Conv2d):
                    w = torch.randn(m.weight.shape, generator=g) * m.weight.std().cpu()
                    m.weight.data.copy_(w.to(m.weight.device))
                    if m.bias is not None:
                        m.bias.data.zero_()
        return self.det

    def __exit__(self, *exc):
        self.det.model.load_state_dict(self.backup)
        return False


def noisy(x: torch.Tensor, std: float, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device=x.device).manual_seed(seed)
    return (x + std * torch.randn(x.shape, generator=g, device=x.device)).clamp(0, 1)
