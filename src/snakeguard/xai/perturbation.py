"""Model-agnostic perturbation explainers: occlusion sensitivity and D-RISE, plus trivial
baselines (random, box prior) that tell whether a method beats chance or "just the box"."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..detection.yolo import PAD_VALUE, Detection, YoloDetector
from .base import DetectorExplainer


class Occlusion(DetectorExplainer):
    """Slide a gray patch over the input. Saliency = drop in the target score, averaged over the
    pixels each patch covered and then over several patch sizes (Zeiler & Fergus 2014)."""
    name = "occlusion"

    def __init__(self, detector: YoloDetector, patch_sizes=(48, 96), stride_frac: float = 0.5,
                 batch: int = 64, mode: str = "similarity"):
        super().__init__(detector)
        self.patch_sizes, self.stride_frac, self.batch, self.mode = patch_sizes, stride_frac, batch, mode

    @staticmethod
    def _positions(n: int, p: int, stride: int) -> list[int]:
        if p >= n:
            return [0]
        pos = list(range(0, n - p + 1, stride))
        if pos[-1] != n - p:
            pos.append(n - p)
        return pos

    @torch.no_grad()
    def _explain_lb(self, x, target):
        H, W = x.shape[-2:]
        base = self.det.score(x, target, mode=self.mode)[0]
        total = torch.zeros(H, W, device=x.device)
        for p in self.patch_sizes:
            stride = max(1, int(p * self.stride_frac))
            coords = [(yy, xx) for yy in self._positions(H, p, stride) for xx in self._positions(W, p, stride)]
            acc = torch.zeros(H, W, device=x.device)
            cnt = torch.zeros(H, W, device=x.device)
            for i in range(0, len(coords), self.batch):
                chunk = coords[i:i + self.batch]
                xb = x.repeat(len(chunk), 1, 1, 1)
                for j, (yy, xx) in enumerate(chunk):
                    xb[j, :, yy:yy + p, xx:xx + p] = PAD_VALUE / 255
                drop = base - self.det.score(xb, target, batch=self.batch, mode=self.mode)
                for j, (yy, xx) in enumerate(chunk):
                    acc[yy:yy + p, xx:xx + p] += drop[j]
                    cnt[yy:yy + p, xx:xx + p] += 1
            total += acc / cnt.clamp(min=1)
        return (total / len(self.patch_sizes)).clamp(min=0).cpu().numpy()


class DRISE(DetectorExplainer):
    """Petsiuk et al. 2021: random low-res binary masks, upsampled with random shifts. Saliency =
    sum of masks weighted by the similarity of the masked image's predictions to the target.
    `center=True` subtracts the mean weight, which removes the uniform bias of plain RISE."""
    name = "drise"

    def __init__(self, detector: YoloDetector, n_masks: int = 2000, grid: int = 16, p: float = 0.5,
                 batch: int = 64, seed: int = 42, center: bool = True, mode: str = "similarity"):
        super().__init__(detector)
        self.n, self.grid, self.p, self.batch = n_masks, grid, p, batch
        self.seed, self.center, self.mode = seed, center, mode

    def _masks(self, n: int, H: int, W: int, gen: torch.Generator, device) -> torch.Tensor:
        # square cells on the longer side; the grid is sized so it covers the shorter side too
        cell = int(np.ceil(max(H, W) / self.grid))
        gh, gw = int(np.ceil(H / cell)), int(np.ceil(W / cell))
        grid = (torch.rand(n, 1, gh, gw, generator=gen, device=device) < self.p).float()
        big = F.interpolate(grid, size=((gh + 1) * cell, (gw + 1) * cell), mode="bilinear", align_corners=False)
        sx = torch.randint(0, cell, (n,), generator=gen, device=device).tolist()
        sy = torch.randint(0, cell, (n,), generator=gen, device=device).tolist()
        return torch.stack([big[i, :, sy[i]:sy[i] + H, sx[i]:sx[i] + W] for i in range(n)])

    @torch.no_grad()
    def _explain_lb(self, x, target):
        H, W = x.shape[-2:]
        gen = torch.Generator(device=x.device).manual_seed(self.seed)
        sal = torch.zeros(H, W, device=x.device)
        weights, masks_sum = [], torch.zeros(H, W, device=x.device)
        for i in range(0, self.n, self.batch):
            m = self._masks(min(self.batch, self.n - i), H, W, gen, x.device)
            xb = x * m + (1 - m) * (PAD_VALUE / 255)
            w = self.det.score(xb, target, batch=self.batch, mode=self.mode)
            sal += (w[:, None, None] * m[:, 0]).sum(0)
            masks_sum += m[:, 0].sum(0)
            weights.append(w)
        w_all = torch.cat(weights)
        if self.center:
            sal -= w_all.mean() * masks_sum
        return (sal / self.n).clamp(min=0).cpu().numpy()


class RandomMap(DetectorExplainer):
    """Random baseline: smooth random blobs (upsampled low-res noise), so that its ordering has the
    same spatial coherence as a CAM. The proposal's "random-order baseline"."""
    name = "random"

    def __init__(self, detector: YoloDetector, grid: int = 20, seed: int = 42):
        super().__init__(detector)
        self.grid, self.rng = grid, np.random.default_rng(seed)

    def _explain_lb(self, x, target):
        H, W = x.shape[-2:]
        g = torch.from_numpy(self.rng.random((1, 1, self.grid, self.grid), dtype=np.float32))
        return F.interpolate(g, size=(H, W), mode="bilinear", align_corners=False)[0, 0].numpy()


class BoxPrior(DetectorExplainer):
    """Trivial baseline: a Gaussian centred on the detected box. An XAI method should beat this
    on faithfulness. If it only matches it, the method is basically just highlighting the box."""
    name = "boxprior"

    def _explain_lb(self, x, target):
        H, W = x.shape[-2:]
        x0, y0, x1, y1 = target.box_lb
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        sx, sy = max((x1 - x0) / 4, 1), max((y1 - y0) / 4, 1)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        return np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) / 2)
