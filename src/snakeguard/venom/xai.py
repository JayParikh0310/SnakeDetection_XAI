"""B3: explainers for the venom classifier, on the shared `Explainer` contract (CLAUDE.md §5).

Every method explains the score the deployed classifier uses (`TorchVenomClassifier.score`: species-head
P(venomous) or the venom head). Target: 1 = "venomous" (explain s), 0 = "harmless" (explain 1 - s).

Spaces: the model sees a 224x224 centre crop of the shorter-side-resized image (`eval_transform`). Methods work
on that crop in pixel space [0, 1] (`explain_x`, used by the metrics, so A's `eval/metrics.py` applies
unchanged). `explain` maps the map back to the original image; pixels outside the centre crop get 0.

Reuse of Member A's code, so detector and classifier results are comparable:
  - CAM maths: `EigenCAM/GradCAM/GradCAMPlusPlus/LayerCAM._cam`, `_upsample`, `_minmax`, mean fusion.
  - Occlusion, RISE (= A's DRISE class with a classifier score) and the random baseline are A's classes,
    driven through `VenomScorer.score(x, target, batch, mode)`, the same signature as `YoloDetector.score`.
Own implementations: Integrated Gradients, LIME and KernelSHAP on a regular patch grid (no extra dependencies).
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from ..detection.yolo import PAD_VALUE
from ..xai.base import Explainer, normalize
from ..xai.cam import EigenCAM, GradCAM, GradCAMPlusPlus, LayerCAM, _ActivationHooks, _minmax, _upsample
from ..xai.perturbation import DRISE, Occlusion, RandomMap
from .classifier import TorchVenomClassifier
from .data import IMAGENET_MEAN, IMAGENET_STD, eval_transform

GRAY = PAD_VALUE / 255  # same "removed" value as the detector metrics
# EfficientNet-B0 stages used as CAM layers: features[3] 28x28, [5] 14x14, [8] 7x7 (last block, 1280 ch)
MULTI_SCALE = (3, 5, 8)
LAST = (8,)


class VenomScorer:
    """Differentiable classifier score on pixel-space input x in [0, 1], shape (B, 3, S, S)."""

    def __init__(self, clf: TorchVenomClassifier):
        self.clf, self.model, self.device = clf, clf.model.eval(), clf.device
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.mean = torch.tensor(IMAGENET_MEAN, device=self.device)[:, None, None]
        self.std = torch.tensor(IMAGENET_STD, device=self.device)[:, None, None]

    def __call__(self, x: torch.Tensor, target: int = 1) -> torch.Tensor:
        v, s = self.model((x - self.mean) / self.std)
        if self.clf.score == "species":
            p = s.float().softmax(-1)[:, self.clf.venomous_species].sum(-1).clamp(0, 1)
        else:
            p = torch.sigmoid(v.float())
        return p if int(target) == 1 else 1 - p

    @torch.no_grad()
    def score(self, x: torch.Tensor, target: int = 1, batch: int = 64, mode: str | None = None) -> torch.Tensor:
        """Batched no-grad score; signature matches YoloDetector.score so A's perturbation code can use it."""
        return torch.cat([self(x[i:i + batch], target) for i in range(0, len(x), batch)])


@dataclass
class CropView:
    """Maps between the original image and the model's 224 centre crop (torchvision Resize + CenterCrop)."""
    orig_h: int
    orig_w: int
    new_h: int
    new_w: int
    top: int
    left: int
    size: int

    @classmethod
    def of(cls, h: int, w: int, size: int = 224) -> "CropView":
        if h <= w:   # torchvision: shorter side -> size, longer side -> int(size * long / short)
            nh, nw = size, int(size * w / h)
        else:
            nh, nw = int(size * h / w), size
        return cls(h, w, nh, nw, int(round((nh - size) / 2.0)), int(round((nw - size) / 2.0)), size)

    def box_to_crop(self, box) -> np.ndarray:
        """xyxy box in original pixels -> xyxy in crop pixels, clipped to the crop."""
        sx, sy = self.new_w / self.orig_w, self.new_h / self.orig_h
        b = np.asarray(box, np.float64) * [sx, sy, sx, sy] - [self.left, self.top, self.left, self.top]
        return np.clip(b, 0, self.size)

    def box_mask(self, box) -> np.ndarray:
        x0, y0, x1, y1 = np.round(self.box_to_crop(box)).astype(int)
        m = np.zeros((self.size, self.size), bool)
        m[y0:y1, x0:x1] = True
        return m

    def map_to_orig(self, sal: np.ndarray) -> np.ndarray:
        import cv2
        canvas = np.zeros((self.new_h, self.new_w), np.float32)
        canvas[self.top:self.top + self.size, self.left:self.left + self.size] = sal
        return cv2.resize(canvas, (self.orig_w, self.orig_h), interpolation=cv2.INTER_LINEAR)


class ClassifierExplainer(Explainer):
    stage = "classifier"

    def __init__(self, clf: TorchVenomClassifier, name: str | None = None):
        self.clf = clf
        self.scorer = VenomScorer(clf)
        self.size = clf.img_size
        self.tf = eval_transform(self.size, normalize=False)
        if name:
            self.name = name

    def preprocess(self, image: np.ndarray) -> tuple[torch.Tensor, CropView]:
        x = self.tf(np.ascontiguousarray(image))[None].to(self.clf.device)
        return x, CropView.of(*image.shape[:2], size=self.size)

    def _explain_x(self, x: torch.Tensor, target: int) -> np.ndarray:
        raise NotImplementedError

    def explain_x(self, x: torch.Tensor, target: int = 1) -> np.ndarray:
        """x: (1, 3, S, S) pixel space -> (S, S) map in [0, 1]."""
        return normalize(self._explain_x(x, target))

    def explain(self, image: np.ndarray, target: int = 1) -> np.ndarray:
        x, view = self.preprocess(image)
        return normalize(view.map_to_orig(self.explain_x(x, target)))


# ------------------------------------------------------------------ CAM family (A's maths)
_CAM_FNS = {"eigencam": EigenCAM._cam, "gradcam": GradCAM._cam, "gradcampp": GradCAMPlusPlus._cam,
            "layercam": LayerCAM._cam}


class ClassifierCAM(ClassifierExplainer):
    """`kind` in eigencam / gradcam / gradcampp / layercam; `layers` = indices into model.features.
    Several layers: per-layer maps upsampled, min-max normalized and averaged (A's "mean" fusion)."""

    def __init__(self, clf, kind: str, layers=LAST, name: str | None = None):
        super().__init__(clf, name or kind)
        self.kind, self.layers = kind, tuple(layers)
        self.fn = _CAM_FNS[kind]
        self.needs_grad = kind != "eigencam"
        self.self_arg = SimpleNamespace(center=False)  # EigenCAM._cam reads self.center (paper version)

    def _explain_x(self, x, target):
        hooks = _ActivationHooks([self.clf.model.features[i] for i in self.layers], keep_grad=self.needs_grad)
        try:
            if self.needs_grad:
                x = x.clone().requires_grad_(True)
                with torch.enable_grad():
                    self.scorer(x, target).sum().backward()
                grads = [a.grad for a in hooks.acts]
            else:
                with torch.no_grad():
                    self.scorer(x, target)
                grads = [None] * len(hooks.acts)
            size = tuple(x.shape[-2:])
            # EigenCAM._cam casts to float64 before moving to CPU, which MPS can't do -> give it CPU tensors
            dev = torch.device("cpu") if self.kind == "eigencam" else x.device
            maps = [_upsample(self.fn(self.self_arg, a[0].detach().float().to(dev),
                                      None if g is None else g[0].float()), size).to(x.device)
                    for a, g in zip(hooks.acts, grads)]
        finally:
            hooks.remove()
        fused = maps[0] if len(maps) == 1 else torch.stack([_minmax(m) for m in maps]).mean(0)
        return fused.cpu().numpy()


# ------------------------------------------------------------------ gradients
class IntegratedGradients(ClassifierExplainer):
    """Sundararajan et al. 2017: (x - x0) * mean of dS/dx along the straight path from a black baseline.
    The map is the positive part of the channel sum (evidence for the target), lightly blurred (sigma px) so it
    can be ranked region-wise like the other maps."""
    name = "integrated_gradients"

    def __init__(self, clf, steps: int = 32, sigma: float = 4.0, batch: int = 16, name: str | None = None):
        super().__init__(clf, name)
        self.steps, self.sigma, self.batch = steps, sigma, batch

    def _explain_x(self, x, target):
        x0 = torch.zeros_like(x)
        alphas = (torch.arange(self.steps, device=x.device, dtype=x.dtype) + 0.5) / self.steps  # midpoint rule
        grad_sum = torch.zeros_like(x)
        for i in range(0, self.steps, self.batch):
            a = alphas[i:i + self.batch][:, None, None, None]
            xi = (x0 + a * (x - x0)).requires_grad_(True)
            with torch.enable_grad():
                self.scorer(xi, target).sum().backward()
            grad_sum += xi.grad.sum(0, keepdim=True)
        attr = ((x - x0) * grad_sum / self.steps).sum(1)[0].clamp(min=0)
        if self.sigma > 0:
            k = int(2 * round(3 * self.sigma) + 1)
            attr = TF.gaussian_blur(attr[None, None], [k, k], [self.sigma, self.sigma])[0, 0]
        return attr.detach().cpu().numpy()


# ------------------------------------------------------------------ A's perturbation classes, driven by the scorer
class _FromDetectorExplainer(ClassifierExplainer):
    """Runs one of A's DetectorExplainer classes with VenomScorer in place of the detector. Those classes only
    call `self.det.score(x, target, batch=..., mode=...)` and use the input shape."""

    def __init__(self, clf, inner_cls, name: str, **kw):
        super().__init__(clf, name)
        self.inner = inner_cls(self.scorer, **kw)

    def _explain_x(self, x, target):
        return self.inner._explain_lb(x, target)


# ------------------------------------------------------------------ surrogate models on a patch grid
class _GridSurrogate(ClassifierExplainer):
    """Features = cells of a grid x grid patch grid; a cell is "off" when filled with gray. n_samples evaluations."""

    def __init__(self, clf, grid: int = 8, n_samples: int = 1000, batch: int = 64, seed: int = 42, name: str | None = None):
        super().__init__(clf, name)
        self.grid, self.n, self.batch, self.seed = grid, n_samples, batch, seed

    def _coalitions(self, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def _fit(self, z: np.ndarray, f: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @torch.no_grad()
    def _explain_x(self, x, target):
        rng = np.random.default_rng(self.seed)
        z = self._coalitions(rng)                                   # (N, M) in {0, 1}
        H, W = x.shape[-2:]
        f = []
        for i in range(0, len(z), self.batch):
            zb = torch.from_numpy(z[i:i + self.batch]).float().to(x.device).view(-1, 1, self.grid, self.grid)
            m = F.interpolate(zb, size=(H, W), mode="nearest")
            f.append(self.scorer.score(x * m + (1 - m) * GRAY, target, batch=self.batch))
        phi = self._fit(z, torch.cat(f).cpu().numpy().astype(np.float64))
        cell = torch.from_numpy(phi.astype(np.float32)).view(1, 1, self.grid, self.grid)
        return F.interpolate(cell, size=(H, W), mode="nearest")[0, 0].clamp(min=0).numpy()


class LIME(_GridSurrogate):
    """Ribeiro et al. 2016, image settings of the `lime` package: random on/off patches (p = 0.5) plus the full image,
    cosine-distance kernel sqrt(exp(-d^2 / 0.25^2)), weighted ridge regression (alpha = 1)."""
    name = "lime"
    kernel_width, alpha = 0.25, 1.0

    def _coalitions(self, rng):
        M = self.grid ** 2
        z = (rng.random((self.n, M)) < 0.5).astype(np.uint8)
        z[0] = 1
        return z

    def _fit(self, z, f):
        from sklearn.linear_model import Ridge
        zf = z.astype(np.float64)
        cos = zf.sum(1) / (np.sqrt(np.maximum(zf.sum(1), 1e-12)) * np.sqrt(z.shape[1]))
        d = 1 - cos
        w = np.sqrt(np.exp(-(d ** 2) / self.kernel_width ** 2))
        return Ridge(alpha=self.alpha).fit(zf, f, sample_weight=w).coef_


class KernelSHAP(_GridSurrogate):
    """Lundberg & Lee 2017. Coalition sizes k ~ Shapley kernel (M-1)/(k(M-k)), uniform subsets of that size; the
    empty and full coalitions are included and enforced with a large weight (efficiency: sum phi = f(1) - f(0))."""
    name = "kernelshap"

    def _coalitions(self, rng):
        M = self.grid ** 2
        k = np.arange(1, M)
        pk = (M - 1) / (k * (M - k)); pk /= pk.sum()
        z = np.zeros((self.n, M), np.uint8)
        z[1] = 1                                                    # z[0] = empty, z[1] = full
        for i, ki in enumerate(rng.choice(k, self.n - 2, p=pk), start=2):
            z[i, rng.choice(M, ki, replace=False)] = 1
        return z

    def _fit(self, z, f):
        w = np.ones(len(z)); w[:2] = 1e6
        y = f - f[0]
        sw = np.sqrt(w)[:, None]
        phi, *_ = np.linalg.lstsq(z.astype(np.float64) * sw, y * sw[:, 0], rcond=None)
        return phi


# ------------------------------------------------------------------ registry
def build_classifier_explainers(clf: TorchVenomClassifier, cfg: dict | None = None, names=None) -> dict[str, ClassifierExplainer]:
    """Classifier explainers keyed by name (see configs/venom.yaml `xai:`)."""
    cfg = cfg or {}
    occ, rise, sur = cfg.get("occlusion", {}), cfg.get("rise", {}), cfg.get("surrogate", {})
    seed = cfg.get("seed", 42)
    factories = {
        "eigencam": lambda: ClassifierCAM(clf, "eigencam", LAST),
        "gradcam": lambda: ClassifierCAM(clf, "gradcam", LAST),
        "gradcampp": lambda: ClassifierCAM(clf, "gradcampp", LAST),
        "gradcampp_ms": lambda: ClassifierCAM(clf, "gradcampp", MULTI_SCALE, name="gradcampp_ms"),
        "layercam_ms": lambda: ClassifierCAM(clf, "layercam", MULTI_SCALE, name="layercam_ms"),
        "integrated_gradients": lambda: IntegratedGradients(clf, **cfg.get("integrated_gradients", {})),
        "occlusion": lambda: _FromDetectorExplainer(clf, Occlusion, "occlusion",
                                                    **{"patch_sizes": (32, 64), "stride_frac": 0.5, "batch": 64, **occ}),
        "rise": lambda: _FromDetectorExplainer(clf, DRISE, "rise",
                                               **{"n_masks": 1000, "grid": 7, "p": 0.5, "batch": 64, "seed": seed, **rise}),
        "lime": lambda: LIME(clf, seed=seed, **sur),
        "kernelshap": lambda: KernelSHAP(clf, seed=seed, **sur),
        "random": lambda: _FromDetectorExplainer(clf, RandomMap, "random", grid=7, seed=seed),
    }
    names = names or list(factories)
    unknown = set(names) - set(factories)
    if unknown:
        raise KeyError(f"unknown classifier explainers: {unknown}")
    return {n: factories[n]() for n in names}
