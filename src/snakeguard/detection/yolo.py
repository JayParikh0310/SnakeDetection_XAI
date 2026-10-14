"""Thin wrapper around the frozen YOLO11n detector that exposes what XAI needs:
letterbox preprocessing with exact inverse mapping, raw (pre-NMS) predictions with gradients,
NMS that keeps anchor indices, and a differentiable per-detection target score."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torchvision

PAD_VALUE = 114  # ultralytics letterbox fill


@dataclass
class Letterbox:
    """Maps between original-image pixels and the letterboxed network input (H x W)."""
    ratio: float
    left: int
    top: int
    new_w: int
    new_h: int
    orig_h: int
    orig_w: int
    H: int
    W: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.H, self.W

    @property
    def valid(self) -> tuple[int, int, int, int]:
        """x0, y0, x1, y1 of the real-image region inside the letterboxed input."""
        return self.left, self.top, self.left + self.new_w, self.top + self.new_h

    def valid_mask(self) -> np.ndarray:
        m = np.zeros((self.H, self.W), bool)
        x0, y0, x1, y1 = self.valid
        m[y0:y1, x0:x1] = True
        return m

    def box_to_lb(self, box: np.ndarray) -> np.ndarray:
        b = np.asarray(box, np.float64) * self.ratio
        return b + [self.left, self.top, self.left, self.top]

    def box_to_orig(self, box: np.ndarray) -> np.ndarray:
        b = (np.asarray(box, np.float64) - [self.left, self.top, self.left, self.top]) / self.ratio
        return np.clip(b, 0, [self.orig_w, self.orig_h, self.orig_w, self.orig_h])

    def points_to_lb(self, pts: np.ndarray) -> np.ndarray:
        return np.asarray(pts, np.float64) * self.ratio + [self.left, self.top]

    def map_to_orig(self, sal: np.ndarray) -> np.ndarray:
        """Crop the valid region from an (H, W) map and resize to the original image."""
        x0, y0, x1, y1 = self.valid
        crop = np.ascontiguousarray(sal[y0:y1, x0:x1], dtype=np.float32)
        return cv2.resize(crop, (self.orig_w, self.orig_h), interpolation=cv2.INTER_LINEAR)


def letterbox(img_rgb: np.ndarray, size: int = 640, rect: bool = True, stride: int = 32) -> tuple[np.ndarray, Letterbox]:
    """Same as ultralytics LetterBox(auto=rect, stride=32). rect=True pads only up to a multiple of
    the stride (ultralytics predict/val default), so explanations see almost no artificial gray border."""
    h, w = img_rgb.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = size - nw, size - nh
    if rect:
        dw, dh = dw % stride, dh % stride
    dw, dh = dw / 2, dh / 2
    resized = img_rgb if (nw, nh) == (w, h) else cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    H, W = nh + top + bottom, nw + left + right
    out = np.full((H, W, 3), PAD_VALUE, np.uint8)
    out[top:top + nh, left:left + nw] = resized
    return out, Letterbox(r, left, top, nw, nh, h, w, H, W)


@dataclass
class Detection:
    box: np.ndarray     # xyxy, original image pixels
    box_lb: np.ndarray  # xyxy, letterboxed pixels
    cls: int
    score: float
    anchor: int         # index into raw predictions


def box_iou_one(boxes: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    """IoU of boxes (..., N, 4) with one box (4,), differentiable w.r.t. `boxes`."""
    x0 = torch.maximum(boxes[..., 0], box[0])
    y0 = torch.maximum(boxes[..., 1], box[1])
    x1 = torch.minimum(boxes[..., 2], box[2])
    y1 = torch.minimum(boxes[..., 3], box[3])
    inter = (x1 - x0).clamp(min=0) * (y1 - y0).clamp(min=0)
    a = (boxes[..., 2] - boxes[..., 0]).clamp(min=0) * (boxes[..., 3] - boxes[..., 1]).clamp(min=0)
    b = (box[2] - box[0]) * (box[3] - box[1])
    return inter / (a + b - inter + 1e-9)


class YoloDetector:
    def __init__(self, weights: str, device: str | None = None, imgsz: int = 640, rect: bool = True):
        from ultralytics import YOLO

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.imgsz, self.rect = imgsz, rect
        self.yolo = YOLO(weights)
        self.model = self.yolo.model.to(self.device).float().eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.names = dict(self.model.names)
        self.nc = len(self.names)
        self.detect_head = self.model.model[-1]
        # indices of the neck outputs (P3, P4, P5) that feed the Detect head
        self.scale_layers = list(self.detect_head.f)

    # ---------- preprocessing ----------
    def preprocess(self, img_rgb: np.ndarray) -> tuple[torch.Tensor, Letterbox]:
        lb_img, lb = letterbox(img_rgb, self.imgsz, self.rect)
        x = torch.from_numpy(lb_img).permute(2, 0, 1).float().div_(255).unsqueeze(0)
        return x.to(self.device), lb

    def layer(self, idx: int) -> torch.nn.Module:
        return self.model.model[idx]

    # ---------- raw predictions ----------
    def raw(self, x: torch.Tensor) -> torch.Tensor:
        """(B,3,S,S) in [0,1] -> (B, N, 4+nc): xyxy boxes (letterbox px) + class probabilities."""
        out = self.model(x)
        if isinstance(out, dict):  # some ultralytics versions return dicts for end2end heads
            out = out.get("one2many", out)
        if isinstance(out, (list, tuple)):
            out = out[0]
        out = out.transpose(1, 2)  # (B, N, 4+nc), boxes are cx,cy,w,h
        cxcy, wh = out[..., :2], out[..., 2:4]
        return torch.cat([cxcy - wh / 2, cxcy + wh / 2, out[..., 4:]], -1)

    @torch.no_grad()
    def raw_batched(self, x: torch.Tensor, batch: int = 64) -> torch.Tensor:
        return torch.cat([self.raw(x[i:i + batch]) for i in range(0, len(x), batch)])

    # ---------- detection ----------
    @torch.no_grad()
    def detect_tensor(self, x: torch.Tensor, lb: Letterbox, conf: float = 0.25, iou: float = 0.7,
                      max_det: int = 300, agnostic: bool = False) -> list[Detection]:
        raw = self.raw(x)[0]
        scores, cls = raw[:, 4:].max(1)
        keep = torch.nonzero(scores > conf).squeeze(1)
        boxes = raw[keep, :4]
        if agnostic:
            k = torchvision.ops.nms(boxes, scores[keep], iou)
        else:
            k = torchvision.ops.batched_nms(boxes, scores[keep], cls[keep], iou)
        k = k[:max_det]
        dets = []
        for i in keep[k].tolist():
            b = raw[i, :4].cpu().numpy().astype(np.float64)
            dets.append(Detection(lb.box_to_orig(b), b, int(cls[i]), float(scores[i]), i))
        return dets

    def detect(self, img_rgb: np.ndarray, **kw) -> list[Detection]:
        x, lb = self.preprocess(img_rgb)
        return self.detect_tensor(x, lb, **kw)

    # ---------- target score ----------
    @staticmethod
    def target_score(raw: torch.Tensor, box_lb, cls: int, mode: str = "similarity") -> torch.Tensor:
        """D-RISE style similarity of the predictions to one target detection:
        s = max_i IoU(box_i, target) * p_i(cls). Differentiable; shape (B,).
        mode="contrastive": IoU * (p_cls - max p_other), explains "snake rather than look-alike"."""
        box = torch.as_tensor(np.asarray(box_lb, np.float32), device=raw.device)
        iou = box_iou_one(raw[..., :4], box)
        p = raw[..., 4:]
        if mode == "similarity":
            s = p[..., cls]
        elif mode == "contrastive":
            others = torch.cat([p[..., :cls], p[..., cls + 1:]], -1)
            s = p[..., cls] - others.max(-1).values
        else:
            raise ValueError(mode)
        return (iou * s).max(-1).values

    def score(self, x: torch.Tensor, det: Detection, batch: int = 64, mode: str = "similarity") -> torch.Tensor:
        """No-grad target score for a batch of (perturbed) inputs."""
        with torch.no_grad():
            out = [self.target_score(self.raw(x[i:i + batch]), det.box_lb, det.cls, mode)
                   for i in range(0, len(x), batch)]
        return torch.cat(out)
