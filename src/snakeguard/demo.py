"""B5: demo logic behind the Gradio app (scripts/app.py), kept free of Gradio so it can be tested.

Photo mode: one frame through Member A's `SnakeGuard` pipeline with temporal confirmation off (a single photo
can't satisfy 3-of-5), the venom classifier plugged in, and two explanations:
  - detector:   why a box is a snake / look-alike (A's LayerCAM P3-P5, A's recommended default; occlusion = audit)
  - venom:      why the top snake crop is judged venomous / harmless (B's LayerCAM ms; RISE = audit)
Video mode: frames through the same pipeline with the real 3-of-5 confirmation; annotated video + threat timeline.

Safety wording follows CLAUDE.md rule 9: unknown venom status counts as venomous, and every result carries
the "not a substitute for expert or medical advice" notice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import CLASS_NAMES, load_config
from .detection import YoloDetector
from .pipeline import LOOKALIKE, SNAKE, FrameResult, PipelineConfig, SnakeGuard, venom_crop
from .xai import build_explainers
from .xai.viz import draw_box, overlay

DISCLAIMER = ("**Research prototype.** SnakeGuard can miss snakes and mislabel species. It is not a substitute for "
              "expert identification or medical advice. Treat every snake as dangerous, keep your distance, and get "
              "emergency medical help after any bite.")

DET_EXPLAINER, DET_AUDIT = "layercam", "occlusion"
# Venom crop context: detector box grown by 25 % per side (test recall 0.917 tight -> 0.944, crop_eval_test.md)
VENOM_CROP_MARGIN = 0.25
VENOM_EXPLAINER, VENOM_AUDIT = "layercam_ms", "rise"


@dataclass
class SnakeInfo:
    box: np.ndarray
    det_score: float
    p_venom: float | None
    venom_label: str          # "venomous" | "harmless" | "unknown"
    proximity: float
    threat: float


@dataclass
class PhotoResult:
    annotated: np.ndarray                 # frame with boxes + labels
    verdict: str                          # one-line headline
    details_md: str                       # markdown details
    snakes: list[SnakeInfo]
    n_lookalikes: int
    threat: float
    beep_interval_ms: int | None
    detector_map: np.ndarray | None = None    # overlay on the frame
    detector_method: str | None = None
    detector_mass_in_box: float | None = None
    venom_map: np.ndarray | None = None       # overlay on the snake crop
    venom_method: str | None = None
    extra: dict = field(default_factory=dict)


def venom_label(p: float | None, threshold: float) -> str:
    if p is None:
        return "unknown"
    return "venomous" if p >= threshold else "harmless"


def verdict_text(snakes: list[SnakeInfo], n_look: int, confirmed: bool = True) -> str:
    if not snakes:
        if n_look:
            return f"No snake. {n_look} snake-like object(s) (e.g. rope, root, hose) detected: no alert."
        return "No snake detected."
    if not confirmed:
        return "Possible snake: waiting for confirmation over the next frames."
    top = snakes[0]
    if top.venom_label == "venomous":
        return f"SNAKE: possibly venomous (medically important). Threat {top.threat:.0f}/100. Move away."
    if top.venom_label == "unknown":
        return f"SNAKE: venom status unknown, treated as venomous. Threat {top.threat:.0f}/100. Move away."
    return (f"SNAKE: likely a harmless species, but identification can be wrong. Threat {top.threat:.0f}/100. "
            "Keep your distance.")


def annotate(frame: np.ndarray, res: FrameResult, threshold: float) -> np.ndarray:
    out = frame.copy()
    pv = {id(s.det): s.p_venom for s in res.snakes}
    for d in res.detections:
        if d.cls == SNAKE:
            lab = venom_label(pv.get(id(d)), threshold)
            txt = f"snake {d.score:.2f} | {lab}" + (f" {pv[id(d)]:.2f}" if pv.get(id(d)) is not None else "")
        else:
            txt = f"look-alike {d.score:.2f}"
        out = draw_box(out, d.box, d.cls, label=txt, thickness=max(2, out.shape[1] // 300))
    return out


def _open_writer(out_path: str, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """H.264 (plays in browsers) when OpenCV has an encoder for it, else mp4v. Windows OpenCV wheels ship without
    openh264, so avc1 fails to open there and would silently write an empty file."""
    for fourcc in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"cannot open a video writer for {out_path}")


class SnakeGuardDemo:
    def __init__(self, weights: str | None = None, venom_checkpoint: str | None = None, device: str | None = "auto",
                 classifier=None):
        """`classifier`: an already-built VenomClassifier (tests); otherwise loaded from `venom_checkpoint`.
        No checkpoint -> the pipeline's stub (venom status unknown -> treated as venomous)."""
        from .venom.model import pick_device
        cfg = load_config()
        self.cfg = cfg
        dev = str(pick_device(device)) if device == "auto" else device
        self.detector = YoloDetector(weights or cfg["weights"], device=dev)
        if classifier is None and venom_checkpoint and Path(venom_checkpoint).exists():
            from .venom import TorchVenomClassifier
            # rescale: 0.5 = decision line, so the pipeline's threat score reflects the venom decision
            classifier = TorchVenomClassifier(venom_checkpoint, device=dev, rescale=True)
        self.clf = classifier
        self.threshold = float(getattr(classifier, "threshold", 0.5))
        self._det_ex, self._venom_ex = {}, {}

    # explainers are built lazily (the audit ones are slow and rarely used)
    def det_explainer(self, name: str):
        if name not in self._det_ex:
            self._det_ex[name] = build_explainers(self.detector, {**self.cfg["xai"], "seed": self.cfg["seed"]}, [name])[name]
        return self._det_ex[name]

    def venom_explainer(self, name: str):
        from .venom.classifier import TorchVenomClassifier
        if not isinstance(self.clf, TorchVenomClassifier):
            return None
        if name not in self._venom_ex:
            from .venom import load_venom_config
            from .venom.xai import build_classifier_explainers
            vcfg = load_venom_config()
            self._venom_ex[name] = build_classifier_explainers(self.clf, {**vcfg.get("xai", {}), "seed": vcfg["seed"]}, [name])[name]
        return self._venom_ex[name]

    def pipeline_config(self, photo: bool) -> PipelineConfig:
        c = PipelineConfig(venom_threshold=self.threshold, venom_crop_margin=VENOM_CROP_MARGIN)
        if photo:
            c.window, c.k_confirm = 1, 1   # a single photo: no temporal confirmation
        return c

    def guard(self, photo: bool, explainer=None) -> SnakeGuard:
        return SnakeGuard(self.detector, self.pipeline_config(photo), classifier=self.clf, explainer=explainer)

    def _snakes(self, res: FrameResult) -> list[SnakeInfo]:
        return [SnakeInfo(s.det.box, s.det.score, s.p_venom, venom_label(s.p_venom, self.threshold), s.proximity, s.threat)
                for s in res.snakes]

    def analyze_photo(self, image_rgb: np.ndarray, explain: bool = True, audit: bool = False) -> PhotoResult:
        if image_rgb is None:
            raise ValueError("no image")
        image_rgb = np.ascontiguousarray(image_rgb[..., :3])
        det_name = DET_AUDIT if audit else DET_EXPLAINER
        g = self.guard(photo=True, explainer=self.det_explainer(det_name) if explain else None)
        res = g.process(image_rgb, explain=explain)
        snakes = self._snakes(res)
        n_look = sum(d.cls == LOOKALIKE for d in res.detections)
        out = PhotoResult(annotate(image_rgb, res, self.threshold), verdict_text(snakes, n_look), "", snakes, n_look,
                          res.threat, res.beep_interval_ms)
        if explain and res.explanation:
            e = res.explanation
            out.detector_map = draw_box(overlay(image_rgb, e["saliency"]), e["target"].box, e["target"].cls,
                                        label=CLASS_NAMES[e["target"].cls], thickness=2)
            out.detector_method, out.detector_mass_in_box = e["method"], e["mass_in_box"]
        if explain and snakes and snakes[0].p_venom is not None:
            vname = VENOM_AUDIT if audit else VENOM_EXPLAINER
            ex = self.venom_explainer(vname)
            c = venom_crop(image_rgb, snakes[0].box, VENOM_CROP_MARGIN)   # the crop the classifier saw
            if ex is not None and min(c.shape[:2]) >= 16:
                target = int(snakes[0].venom_label == "venomous")
                out.venom_map = overlay(c, ex.explain(c, target))
                out.venom_method = vname
        out.details_md = self.details(out)
        return out

    def details(self, r: PhotoResult) -> str:
        lines = [f"### {r.verdict}", ""]
        if r.snakes:
            lines += ["| # | Detector confidence | Venom score | Venom status | Proximity | Threat |", "|---|---|---|---|---|---|"]
            for i, s in enumerate(r.snakes, 1):
                pv = "–" if s.p_venom is None else f"{s.p_venom:.2f}"
                lines.append(f"| {i} | {s.det_score:.2f} | {pv} | {s.venom_label} | {s.proximity:.2f} | {s.threat:.0f}/100 |")
            beep = "silent" if r.beep_interval_ms is None else f"every {r.beep_interval_ms} ms"
            raw = getattr(self.clf, "raw_threshold", self.threshold)
            lines += ["", f"Beep: **{beep}**. Venom score ≥ {self.threshold:.1f} means *venomous* (model probability "
                      f"≥ {raw:.3f}, tuned to catch 95 % of venomous snakes, so false venom alerts on harmless snakes "
                      "are expected)."]
        if r.detector_method:
            m = r.detector_mass_in_box
            warn = " Low: the detector may be reacting to the background; treat this detection with caution." if m < 0.5 else ""
            lines += ["", f"**Why the detector fired** ({r.detector_method}): {m:.0%} of the explanation lies inside the box.{warn}"]
        if r.venom_method:
            lines += ["", f"**Why this venom call** ({r.venom_method}): the heatmap on the crop shows which parts of the "
                      "snake drove the decision. It shows where the model looked, not proof of the species."]
        if r.snakes and self.clf is None:
            lines += ["", "No venom classifier loaded: venom status is unknown and treated as venomous."]
        lines += ["", DISCLAIMER]
        return "\n".join(lines)

    # ------------------------------------------------------------------ video
    def analyze_video(self, path: str, out_path: str, target_fps: float = 5.0, max_seconds: float = 30.0):
        """Process a video at ~target_fps through the pipeline with 3-of-5 confirmation. Writes an annotated mp4
        (H.264, or mp4v where no H.264 encoder is available) and returns (out_path, timeline rows)."""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"cannot open video {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(fps / target_fps)))
        g = self.guard(photo=False)
        writer, rows, i = None, [], 0
        try:
            while True:
                ok, bgr = cap.read()
                if not ok or i / fps > max_seconds:
                    break
                if i % step == 0:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    res = g.process(rgb)
                    snakes = self._snakes(res)
                    n_look = sum(d.cls == LOOKALIKE for d in res.detections)
                    frame = annotate(rgb, res, self.threshold)
                    banner = verdict_text(snakes, n_look, res.confirmed)
                    frame = self._banner(frame, banner, res.threat, res.confirmed and bool(snakes))
                    if writer is None:
                        h, w = frame.shape[:2]
                        writer = _open_writer(out_path, fps / step, (w, h))
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    rows.append({"time_s": round(i / fps, 2), "snakes": len(snakes), "lookalikes": n_look,
                                 "confirmed": res.confirmed, "threat": round(res.threat, 1),
                                 "beep_ms": res.beep_interval_ms,
                                 "venom": snakes[0].venom_label if snakes else ""})
                i += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()
        if not rows:
            raise ValueError("no frames could be read from the video")
        return out_path, rows

    @staticmethod
    def _banner(frame: np.ndarray, text: str, threat: float, alert: bool) -> np.ndarray:
        out = frame.copy()
        h = max(28, frame.shape[0] // 14)
        color = (200, 30, 30) if alert else (60, 60, 60)
        cv2.rectangle(out, (0, 0), (out.shape[1], h), color, -1)
        scale = max(0.45, out.shape[1] / 1400)
        cv2.putText(out, text[:90], (8, int(h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
        return out
