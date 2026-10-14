"""Overlay helpers shared by the figure scripts, pipeline and app."""
from __future__ import annotations

import cv2
import numpy as np

from .. import CLASS_NAMES

BOX_COLORS = {0: (220, 40, 40), 1: (40, 120, 230)}


def overlay(img_rgb: np.ndarray, sal: np.ndarray, alpha: float = 0.5, cmap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Blend a [0,1] saliency map (same HxW as the image) onto the RGB image."""
    heat = cv2.applyColorMap(np.uint8(255 * np.clip(sal, 0, 1)), cmap)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.uint8((1 - alpha) * img_rgb + alpha * heat)


def draw_box(img_rgb: np.ndarray, box, cls: int, score: float | None = None, label: str | None = None,
             thickness: int = 3) -> np.ndarray:
    out = img_rgb.copy()
    x0, y0, x1, y1 = np.round(box).astype(int)
    c = BOX_COLORS.get(cls, (255, 255, 0))
    cv2.rectangle(out, (x0, y0), (x1, y1), c, thickness)
    txt = label or (CLASS_NAMES.get(cls, str(cls)) + (f" {score:.2f}" if score is not None else ""))
    scale = max(0.5, out.shape[1] / 1000)
    (tw, th), _ = cv2.getTextSize(txt, 0, scale, 2)
    y = max(th + 4, y0)
    cv2.rectangle(out, (x0, y - th - 6), (x0 + tw + 4, y), c, -1)
    cv2.putText(out, txt, (x0 + 2, y - 4), 0, scale, (255, 255, 255), 2)
    return out
