"""CAM-family explainers for the YOLO detector: Eigen-CAM, Grad-CAM, Grad-CAM++.

Hooks sit on the neck outputs that feed the Detect head (P3 stride 8, P4 stride 16, P5 stride 32).
Multi-scale mode computes one CAM per scale, upsamples each to the input size, normalizes it,
and fuses the scales. This is what the proposal asks for: no single-scale hook on the coarsest
layer only."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..detection.yolo import Detection, YoloDetector
from .base import DetectorExplainer

SCALE_NAMES = ("P3", "P4", "P5")


class _ActivationHooks:
    def __init__(self, modules, keep_grad: bool):
        self.acts: list[torch.Tensor | None] = [None] * len(modules)
        self.handles = []
        for i, m in enumerate(modules):
            def hook(_m, _inp, out, i=i):
                if keep_grad and out.requires_grad:
                    out.retain_grad()
                self.acts[i] = out
            self.handles.append(m.register_forward_hook(hook))

    def remove(self):
        for h in self.handles:
            h.remove()


def _upsample(cam: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(cam[None, None], size=size, mode="bilinear", align_corners=False)[0, 0]


def _minmax(t: torch.Tensor) -> torch.Tensor:
    lo, hi = t.min(), t.max()
    return (t - lo) / (hi - lo) if hi > lo else torch.zeros_like(t)


class _CAMBase(DetectorExplainer):
    """scales: subset of ("P3", "P4", "P5"). fusion: "mean" of per-scale normalized maps,
    "max", or "sum" (raw maps summed, so magnitudes decide)."""
    needs_grad = False

    def __init__(self, detector: YoloDetector, scales=SCALE_NAMES, fusion: str = "mean",
                 mode: str = "similarity", name: str | None = None):
        super().__init__(detector)
        self.scales = tuple(scales)
        self.fusion = fusion
        self.mode = mode
        idx = [detector.scale_layers[SCALE_NAMES.index(s)] for s in self.scales]
        self.modules = [detector.layer(i) for i in idx]
        if name:
            self.name = name

    def _cam(self, act: torch.Tensor, grad: torch.Tensor | None) -> torch.Tensor:
        raise NotImplementedError

    def _explain_lb(self, x: torch.Tensor, target: Detection) -> np.ndarray:
        hooks = _ActivationHooks(self.modules, keep_grad=self.needs_grad)
        try:
            if self.needs_grad:
                x = x.clone().requires_grad_(True)  # params are frozen, so grads need the input to require them
                with torch.enable_grad():
                    raw = self.det.raw(x)
                    s = self.det.target_score(raw, target.box_lb, target.cls, self.mode)
                    s.sum().backward()
                grads = [a.grad for a in hooks.acts]
            else:
                with torch.no_grad():
                    self.det.raw(x)
                grads = [None] * len(hooks.acts)
            size = tuple(x.shape[-2:])
            maps = [_upsample(self._cam(a[0].detach(), None if g is None else g[0]), size)
                    for a, g in zip(hooks.acts, grads)]
        finally:
            hooks.remove()
        if len(maps) == 1:
            fused = maps[0]
        elif self.fusion == "mean":
            fused = torch.stack([_minmax(m) for m in maps]).mean(0)
        elif self.fusion == "max":
            fused = torch.stack([_minmax(m) for m in maps]).max(0).values
        elif self.fusion == "sum":
            fused = torch.stack(maps).sum(0)
        else:
            raise ValueError(self.fusion)
        return fused.cpu().numpy()


class EigenCAM(_CAMBase):
    """Projection of the activations onto their first principal component (Muhammad & Yeasin 2020).
    Class-agnostic: needs no gradients and no target.
    center=False follows the paper (SVD of raw activations). center=True subtracts the per-channel
    spatial mean first, as pytorch-grad-cam does."""
    name = "eigencam"

    def __init__(self, *args, center: bool = False, **kw):
        super().__init__(*args, **kw)
        self.center = center

    def _cam(self, act, grad):
        c, h, w = act.shape
        a = act.reshape(c, h * w).T.float()          # (hw, C)
        if self.center:
            a = a - a.mean(0, keepdim=True)
        # First right singular vector of `a` = top eigenvector of the small C x C matrix a^T a.
        # Computed on CPU in float64: cuSOLVER's SVD fails intermittently on CUDA under memory pressure.
        _, evecs = torch.linalg.eigh((a.T @ a).double().cpu())
        v = evecs[:, -1].to(a.device, a.dtype)
        proj = a @ v                                  # (hw,)
        # The SVD sign is arbitrary. Orient the component so that it correlates positively with
        # the activation energy at each location (strongly responding locations = salient).
        energy = act.reshape(c, h * w).float().norm(dim=0)
        if torch.corrcoef(torch.stack([proj, energy]))[0, 1] < 0:
            proj = -proj
        return proj.reshape(h, w).clamp(min=0)


class GradCAM(_CAMBase):
    """Selvaraju et al. 2017, target = similarity score of the explained detection."""
    name = "gradcam"
    needs_grad = True

    def _cam(self, act, grad):
        w = grad.mean((1, 2))
        return (w[:, None, None] * act).sum(0).clamp(min=0)


class LayerCAM(_CAMBase):
    """Jiang et al. 2021: each location is weighted by its own positive gradient,
    cam = ReLU(sum_k ReLU(dS/dA_k(i,j)) * A_k(i,j)). Unlike channel-global weights it does not light up
    every place where a channel fires (e.g. image borders). It was designed for fusing shallow,
    high-resolution layers with deep ones."""
    name = "layercam"
    needs_grad = True

    def _cam(self, act, grad):
        return (grad.clamp(min=0) * act).sum(0).clamp(min=0)


class GradCAMPlusPlus(_CAMBase):
    """Chattopadhay et al. 2018 (closed-form alphas, as in pytorch-grad-cam)."""
    name = "gradcampp"
    needs_grad = True

    def _cam(self, act, grad):
        g2, g3 = grad.pow(2), grad.pow(3)
        sum_a = act.sum((1, 2))
        denom = 2 * g2 + sum_a[:, None, None] * g3
        alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha = torch.where(grad != 0, alpha, torch.zeros_like(alpha))
        w = (alpha * grad.clamp(min=0)).sum((1, 2))
        return (w[:, None, None] * act).sum(0).clamp(min=0)
