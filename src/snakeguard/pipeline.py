"""SnakeGuard runtime pipeline (proposal §2, stages Detect → Classify → Locate → Alert + XAI layer).

    frame ─► detector ─► conflict resolution ─► small-object filter ─► venom classifier (pluggable)
          ─► proximity (bbox area) ─► 3-of-5 temporal confirmation ─► threat score 0-100 ─► beep interval
          └► optional explanation of the most threatening detection (+ explanation-based trust check)

Everything is configurable via `PipelineConfig`. Member B's classifier plugs in through `VenomClassifier`.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .detection.yolo import Detection, YoloDetector
from .eval.matching import iou_matrix

SNAKE, LOOKALIKE = 0, 1


class VenomClassifier:
    """Interface for Member B's classifier. Return P(venomous) for an RGB crop, or None if unknown."""

    def predict(self, crop_rgb: np.ndarray) -> float | None:
        return None


@dataclass
class PipelineConfig:
    conf: float = 0.25
    iou: float = 0.7
    # conflict resolution: a snake and a look-alike box on the same object (IoU >= conflict_iou).
    # Safety bias: keep the snake unless the look-alike is clearly more confident.
    conflict_iou: float = 0.5
    lookalike_margin: float = 0.15
    # small-object filter: drop boxes smaller than this fraction of the frame (distant/noisy)
    min_area_frac: float = 0.001
    # proximity from bbox area (fraction of the frame): sqrt(area) grows ~ 1/distance
    area_far: float = 0.005
    area_near: float = 0.25
    # temporal confirmation: k of the last n frames must contain a snake
    window: int = 5
    k_confirm: int = 3
    # threat score weights; unknown venom status is treated conservatively
    w_proximity: float = 0.6
    w_venom: float = 0.4
    venom_unknown: float = 1.0
    venom_threshold: float = 0.5
    # beep interval (ms) from threat 0 → 100
    beep_slow_ms: int = 1500
    beep_fast_ms: int = 150


@dataclass
class SnakeObservation:
    det: Detection
    area_frac: float
    proximity: float
    p_venom: float | None
    threat: float


@dataclass
class FrameResult:
    detections: list[Detection]           # after post-processing (snakes + look-alikes)
    snakes: list[SnakeObservation]
    confirmed: bool                       # temporal confirmation satisfied
    threat: float                         # 0-100, 0 unless confirmed
    venomous_alert: bool
    beep_interval_ms: int | None          # None = silent
    explanation: dict = field(default_factory=dict)


def resolve_conflicts(dets: list[Detection], iou_thr: float, margin: float) -> list[Detection]:
    """Cross-class duplicates on the same object: keep one. Ties go to 'snake' (safety bias)."""
    if len(dets) < 2:
        return list(dets)
    ious = iou_matrix(np.array([d.box for d in dets]), np.array([d.box for d in dets]))
    drop = set()
    for i, a in enumerate(dets):
        for j, b in enumerate(dets):
            if i < j and a.cls != b.cls and ious[i, j] >= iou_thr:
                snake, look = (i, j) if a.cls == SNAKE else (j, i)
                drop.add(snake if dets[look].score > dets[snake].score + margin else look)
    return [d for k, d in enumerate(dets) if k not in drop]


def proximity(area_frac: float, cfg: PipelineConfig) -> float:
    lo, hi = np.sqrt(cfg.area_far), np.sqrt(cfg.area_near)
    return float(np.clip((np.sqrt(area_frac) - lo) / (hi - lo), 0, 1))


def threat_score(conf: float, prox: float, p_venom: float | None, cfg: PipelineConfig) -> float:
    v = cfg.venom_unknown if p_venom is None else p_venom
    return float(100 * conf * (cfg.w_proximity * prox + cfg.w_venom * v))


def beep_interval(threat: float, cfg: PipelineConfig) -> int | None:
    if threat <= 0:
        return None
    t = threat / 100
    return int(round(cfg.beep_slow_ms + t * (cfg.beep_fast_ms - cfg.beep_slow_ms)))


class SnakeGuard:
    def __init__(self, detector: YoloDetector, cfg: PipelineConfig | None = None,
                 classifier: VenomClassifier | None = None, explainer=None):
        self.det = detector
        self.cfg = cfg or PipelineConfig()
        self.clf = classifier or VenomClassifier()
        self.explainer = explainer
        self.history: deque[bool] = deque(maxlen=self.cfg.window)

    def reset(self):
        self.history.clear()

    def process(self, frame_rgb: np.ndarray, explain: bool = False) -> FrameResult:
        c = self.cfg
        x, lb = self.det.preprocess(frame_rgb)
        dets = self.det.detect_tensor(x, lb, conf=c.conf, iou=c.iou)
        dets = resolve_conflicts(dets, c.conflict_iou, c.lookalike_margin)
        h, w = frame_rgb.shape[:2]
        area = lambda d: float((d.box[2] - d.box[0]) * (d.box[3] - d.box[1]) / (h * w))
        dets = [d for d in dets if area(d) >= c.min_area_frac]

        snakes = []
        for d in (d for d in dets if d.cls == SNAKE):
            x0, y0, x1, y1 = np.round(d.box).astype(int)
            crop = frame_rgb[max(y0, 0):y1, max(x0, 0):x1]
            pv = self.clf.predict(crop) if crop.size else None
            a = area(d)
            pr = proximity(a, c)
            snakes.append(SnakeObservation(d, a, pr, pv, threat_score(d.score, pr, pv, c)))
        snakes.sort(key=lambda s: -s.threat)

        self.history.append(bool(snakes))
        confirmed = sum(self.history) >= c.k_confirm
        threat = snakes[0].threat if (confirmed and snakes) else 0.0
        venom_alert = confirmed and any(s.p_venom is not None and s.p_venom >= c.venom_threshold for s in snakes)
        res = FrameResult(dets, snakes, confirmed, threat, venom_alert, beep_interval(threat, c))

        if explain and self.explainer is not None and dets:
            target = snakes[0].det if snakes else max(dets, key=lambda d: d.score)
            sal = self.explainer.explain_lb(x, lb, target)
            res.explanation = {"method": self.explainer.name, "target": target,
                               "saliency": lb.map_to_orig(sal),
                               "mass_in_box": explanation_mass_in_box(sal, target)}
        return res


def explanation_mass_in_box(sal_lb: np.ndarray, det: Detection) -> float:
    """Share of saliency inside the detected box (letterboxed space). A low value means the
    detector is 'looking' outside the object, which warns that the detection may be spurious."""
    x0, y0, x1, y1 = np.round(det.box_lb).astype(int)
    tot = float(sal_lb.sum())
    return float(sal_lb[max(y0, 0):y1, max(x0, 0):x1].sum()) / tot if tot > 0 else 0.0
