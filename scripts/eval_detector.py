"""A1: baseline evaluation of the frozen detector.

1. Official ultralytics metrics (mAP50, mAP50-95, P, R per class) on valid + test.
2. Own matching analysis on test at the operating threshold: TP / class confusion / FP / FN,
   a confusion matrix with a background row/column, and a failure gallery.
3. Detector latency.

Usage:  python scripts/eval_detector.py [--splits valid test] [--conf 0.25] [--gallery 24]
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from snakeguard import CLASS_NAMES, load_config
from snakeguard.data import list_split, load_sample
from snakeguard.detection import YoloDetector
from snakeguard.eval.matching import match

COLORS = {0: (220, 40, 40), 1: (40, 120, 230)}  # RGB: snake red, look-alike blue
SHORT = {0: "snake", 1: "look-alike"}


def resolved_data_yaml(cfg: dict, out: Path) -> str:
    """The raw data.yaml has relative paths that ultralytics resolves against its own datasets dir.
    Write a derived copy with an absolute `path` instead of modifying the dataset."""
    import yaml
    d = yaml.safe_load(open(cfg["data_yaml"]))
    d["path"] = cfg["data_root"]
    p = out / "data_resolved.yaml"
    p.write_text(yaml.safe_dump(d))
    return str(p)


def ultralytics_val(det: YoloDetector, cfg: dict, split: str, out: Path) -> dict:
    r = det.yolo.val(data=resolved_data_yaml(cfg, out), split={"valid": "val"}.get(split, split),
                     imgsz=cfg["detector"]["imgsz"], batch=16,
                     conf=0.001, iou=cfg["detector"]["iou"], plots=True, project=str(out),
                     name=f"ultralytics_{split}", exist_ok=True, verbose=False, workers=0)
    res = {"split": split, "mAP50": float(r.box.map50), "mAP50-95": float(r.box.map),
           "precision": float(r.box.mp), "recall": float(r.box.mr)}
    for i, c in enumerate(r.box.ap_class_index):
        name = CLASS_NAMES[int(c)]
        p, rc, ap50, ap = r.box.class_result(i)
        res[name] = {"precision": float(p), "recall": float(rc), "AP50": float(ap50), "AP50-95": float(ap)}
    res["speed_ms"] = {k: float(v) for k, v in r.speed.items()}
    return res


def draw(img, matches):
    vis = img.copy()
    for m in matches:
        if m.gt is not None:  # GT in green (thin)
            x0, y0, x1, y1 = m.gt.box.astype(int)
            cv2.rectangle(vis, (x0, y0), (x1, y1), (40, 200, 40), 2)
            cv2.putText(vis, f"GT {SHORT[m.gt.cls]}", (x0, max(15, y0 - 4)), 0, 0.5, (40, 200, 40), 1)
        if m.det is not None:
            x0, y0, x1, y1 = m.det.box.astype(int)
            cv2.rectangle(vis, (x0, y0), (x1, y1), COLORS[m.det.cls], 3)
            cv2.putText(vis, f"{SHORT[m.det.cls]} {m.det.score:.2f} {m.kind}", (x0, min(vis.shape[0] - 5, y1 + 16)),
                        0, 0.55, COLORS[m.det.cls], 2)
    return vis


def own_analysis(det: YoloDetector, cfg: dict, split: str, out: Path, conf: float, n_gallery: int):
    rows, failures = [], {"FP": [], "FN": [], "confusion": []}
    times = []
    for p in list_split(cfg["data_root"], split):
        s = load_sample(p)
        img = s.load_rgb()
        if det.device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        dets = det.detect(img, conf=conf, iou=cfg["detector"]["iou"])
        if det.device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1000)
        ms = match(dets, s.objects, cfg["detector"]["match_iou"])
        for m in ms:
            rows.append({"image_id": s.image_id, "kind": m.kind,
                         "gt_cls": None if m.gt is None else m.gt.cls,
                         "det_cls": None if m.det is None else m.det.cls,
                         "score": None if m.det is None else m.det.score, "iou": m.iou,
                         "gt_area_frac": None if m.gt is None else
                         float((m.gt.box[2] - m.gt.box[0]) * (m.gt.box[3] - m.gt.box[1]) / (img.shape[0] * img.shape[1])),
                         "has_polygon": None if m.gt is None else m.gt.polygon is not None})
        for kind in failures:
            worst = [m for m in ms if m.kind == kind]
            if worst:
                sev = max((m.det.score if m.det else 1.0) for m in worst)  # most confident mistakes first
                failures[kind].append((sev, p, ms))
    df = pd.DataFrame(rows)
    df.to_csv(out / f"matches_{split}.csv", index=False)

    # confusion matrix: rows = GT (snake, sll, background), cols = predicted (snake, sll, background)
    labels = ["snake", "snake_like_object", "background"]
    cm = pd.DataFrame(0, index=[f"GT {l}" for l in labels], columns=[f"pred {l}" for l in labels])
    for _, r in df.iterrows():
        g = "background" if pd.isna(r.gt_cls) else CLASS_NAMES[int(r.gt_cls)]
        d = "background" if pd.isna(r.det_cls) else CLASS_NAMES[int(r.det_cls)]
        cm.loc[f"GT {g}", f"pred {d}"] += 1
    cm.to_csv(out / f"confusion_{split}.csv")

    gal = out / f"gallery_{split}"
    for kind, items in failures.items():
        d = gal / kind
        d.mkdir(parents=True, exist_ok=True)
        for k, (sev, p, ms) in enumerate(sorted(items, key=lambda t: -t[0])[:n_gallery]):
            img = load_sample(p).load_rgb()
            cv2.imwrite(str(d / f"{k:02d}_{p.stem}.jpg"), cv2.cvtColor(draw(img, ms), cv2.COLOR_RGB2BGR))

    counts = Counter(df.kind)
    per_cls = {}
    for c, name in CLASS_NAMES.items():
        g = df[df.gt_cls == c]
        tp = int(((g.kind == "TP")).sum())
        per_cls[name] = {"gt": len(g), "TP": tp, "recall": tp / max(len(g), 1),
                         "confused_as_other": int((g.kind == "confusion").sum()),
                         "missed": int((g.kind == "FN").sum()),
                         "FP_background": int(((df.kind == "FP") & (df.det_cls == c)).sum())}
        pred = df[df.det_cls == c]
        per_cls[name]["precision"] = tp / max(len(pred), 1)
    return {"split": split, "conf": conf, "counts": dict(counts), "per_class": per_cls,
            "detector_latency_ms": {"mean": float(np.mean(times[5:])), "p50": float(np.median(times[5:]))},
            "confusion_matrix": cm.to_dict()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["valid", "test"])
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--gallery", type=int, default=24)
    ap.add_argument("--skip-ultralytics", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    conf = a.conf if a.conf is not None else cfg["detector"]["conf"]
    out = Path(cfg["results_dir"]) / "detector_eval"
    out.mkdir(parents=True, exist_ok=True)
    det = YoloDetector(cfg["weights"])
    prev = out / "summary.json"
    summary = json.loads(prev.read_text()) if prev.exists() else {}  # merge, so --skip-ultralytics keeps mAPs
    summary.update({"weights": cfg["weights"], "device": str(det.device),
                    "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None})
    for split in a.splits:
        if not a.skip_ultralytics:
            summary[f"ultralytics_{split}"] = ultralytics_val(det, cfg, split, out)
            det = YoloDetector(cfg["weights"])  # val() may modify the wrapped model (fuse/half); reload
        summary[f"analysis_{split}"] = own_analysis(det, cfg, split, out, conf, a.gallery)
        print(json.dumps({k: v for k, v in summary.items() if split in k}, indent=1, default=str))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
