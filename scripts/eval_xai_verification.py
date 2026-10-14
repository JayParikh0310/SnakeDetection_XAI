"""A6 extra: can explanations flag wrong detections? (the proposal's "validation layer", measured)

For every detection (conf >= --conf) we compute:
  score              detector confidence (baseline signal)
  mass_<m>           share of explanation m's saliency inside the detected box
  peak_in_<m>        whether the map's maximum falls inside the box
  agree_eig_gpp      Spearman agreement between Eigen-CAM and Grad-CAM++ (the proposal's cross-check)
Label = 1 if the detection is wrong (background FP or class confusion), else 0.
A logistic regression is fit on the valid split and evaluated on test. We report AUROC for
"detection is wrong" and the false-alarm reduction at a fixed snake recall.

Usage: python scripts/eval_xai_verification.py [--conf 0.1] [--n-valid 1000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from snakeguard import load_config
from snakeguard.data import list_split, load_sample
from snakeguard.detection import YoloDetector
from snakeguard.eval import metrics as M
from snakeguard.eval.matching import match
from snakeguard.pipeline import explanation_mass_in_box
from snakeguard.xai import build_explainers

METHODS = ["eigencam", "gradcampp", "layercam"]


def features(det, explainers, cfg, split, conf, n, seed, csv: Path) -> pd.DataFrame:
    """Resumable: rows are appended to `csv` per image; images listed in `<csv>.done` are skipped."""
    paths = list_split(cfg["data_root"], split)
    if n and n < len(paths):
        rng = np.random.default_rng(seed)
        paths = [paths[i] for i in sorted(rng.choice(len(paths), n, replace=False))]
    done_f = csv.with_suffix(".done")
    done = set(done_f.read_text().split()) if done_f.exists() else set()
    for k, p in enumerate(paths):
        if p.stem in done:
            continue
        rows = []
        s = load_sample(p)
        x, lb = det.preprocess(s.load_rgb())
        valid = lb.valid_mask()
        dets = det.detect_tensor(x, lb, conf=conf, iou=cfg["detector"]["iou"])
        for m in match(dets, s.objects, cfg["detector"]["match_iou"]):
            if m.det is None:
                continue
            d = m.det
            r = {"image_id": s.image_id, "kind": m.kind, "det_cls": d.cls, "score": d.score,
                 "wrong": int(m.kind != "TP"),
                 "area": float((d.box_lb[2] - d.box_lb[0]) * (d.box_lb[3] - d.box_lb[1]) / valid.sum())}
            maps = {}
            for name, ex in explainers.items():
                sal = ex.explain_lb(x, lb, d)
                maps[name] = sal
                r[f"mass_{name}"] = explanation_mass_in_box(sal, d)
                yy, xx = np.unravel_index(np.argmax(sal), sal.shape)
                r[f"peak_in_{name}"] = float(d.box_lb[0] <= xx <= d.box_lb[2] and d.box_lb[1] <= yy <= d.box_lb[3])
            r["agree_eig_gpp"] = M.rank_correlation(maps["eigencam"], maps["gradcampp"], valid)
            rows.append(r)
        if rows:
            pd.DataFrame(rows).to_csv(csv, mode="a", header=not csv.exists(), index=False)
        with open(done_f, "a") as f:
            f.write(p.stem + "\n")
        if k % 200 == 0:
            print(f"{split}: {k}/{len(paths)}", flush=True)
    return pd.read_csv(csv)


def fpr_at_recall(y_wrong, risk, recall=0.95):
    """Keep detections with risk below a threshold chosen so `recall` of correct snakes survive.
    Return the share of wrong detections that still survive (lower = more false alarms removed)."""
    correct = risk[y_wrong == 0]
    thr = np.quantile(correct, recall)
    return float((risk[y_wrong == 1] <= thr).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--n-valid", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config()
    out = Path(cfg["results_dir"]) / "xai_verification"
    out.mkdir(parents=True, exist_ok=True)
    det = YoloDetector(cfg["weights"])
    ex = build_explainers(det, {**cfg["xai"], "seed": cfg["seed"]}, METHODS)
    data = {}
    for split, n in (("valid", a.n_valid), ("test", a.n_test)):
        f = out / f"features_{split}_conf{a.conf}.csv"
        data[split] = features(det, ex, cfg, split, a.conf, n, cfg["seed"], f)
        print(split, len(data[split]), "detections,", int(data[split].wrong.sum()), "wrong")

    xai = [f"{k}_{m}" for m in METHODS for k in ("mass", "peak_in")] + ["agree_eig_gpp"]
    feature_sets = {
        "confidence only": ["score"],
        "confidence + area": ["score", "area"],
        "XAI only": xai,
        "confidence + XAI": ["score"] + xai,
        "confidence + area + XAI": ["score", "area"] + xai,
    }
    tr, te = data["valid"], data["test"]
    res = []
    for name, cols in feature_sets.items():
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        clf.fit(tr[cols], tr.wrong)
        risk = clf.predict_proba(te[cols])[:, 1]
        res.append({"features": name, "AUROC_wrong": roc_auc_score(te.wrong, risk),
                    "wrong_kept_at_95rec": fpr_at_recall(te.wrong.to_numpy(), risk, 0.95),
                    "wrong_kept_at_90rec": fpr_at_recall(te.wrong.to_numpy(), risk, 0.90)})
    single = []
    for c in ["score", "area", "agree_eig_gpp"] + [f"mass_{m}" for m in METHODS]:
        auc = roc_auc_score(te.wrong, -te[c])  # lower value = more suspicious
        single.append({"feature": c, "AUROC_wrong (low=suspicious)": auc,
                       "mean_TP": te[te.wrong == 0][c].mean(), "mean_wrong": te[te.wrong == 1][c].mean()})
    r = pd.DataFrame(res); s = pd.DataFrame(single)
    r.to_csv(out / "verification_results.csv", index=False); s.to_csv(out / "single_features.csv", index=False)
    md = ["# Explanation-based verification (test split, model fit on valid)", "",
          f"Detections at conf >= {a.conf}: train {len(tr)} ({int(tr.wrong.sum())} wrong), "
          f"test {len(te)} ({int(te.wrong.sum())} wrong)", "",
          "wrong_kept_at_95rec = share of wrong detections that still pass when the threshold keeps "
          "95% of correct ones (lower is better).", "", r.to_markdown(index=False, floatfmt=".3f"), "",
          "## Single features", s.to_markdown(index=False, floatfmt=".3f")]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
