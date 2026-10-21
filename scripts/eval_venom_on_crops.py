"""B5 check: does the venom classifier still work on the input the pipeline gives it (the detector's snake crop)?

The classifier was trained/evaluated on whole SnakeCLEF photos; `SnakeGuard.process` feeds it the detector's box.
For a seeded sample of test images where the detector finds a snake, score: the whole image, the tight box crop,
and the box enlarged by a context margin (each side grows by margin x box size, clipped to the image). Recall /
specificity use the deployed threshold (chosen on valid on whole images).

Usage: python scripts/eval_venom_on_crops.py results/venom_clf/b0_reg [--n 600] [--split test]
Writes <run>/crop_eval_<split>.csv (per image) and crop_eval_<split>.md (summary with bootstrap CIs)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from snakeguard import load_config
from snakeguard.detection import YoloDetector
from snakeguard.venom import TorchVenomClassifier, load_venom_config
from snakeguard.venom.model import pick_device

MARGINS = (0.0, 0.25, 0.5, 1.0)


def expand(box, margin, h, w):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    return np.array([max(0, x0 - margin * bw), max(0, y0 - margin * bh), min(w, x1 + margin * bw), min(h, y1 + margin * bh)])


def crop(img, box):
    x0, y0, x1, y1 = np.round(box).astype(int)
    return img[y0:y1, x0:x1]


def stats(y, p, thr):
    return {"recall": float((p[y == 1] >= thr).mean()), "specificity": float((p[y == 0] < thr).mean()),
            "auroc": float(roc_auc_score(y, p))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=600, help="images sampled (half venomous, half harmless)")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    vcfg, dcfg = load_venom_config(), load_config()
    run = Path(a.run)
    dev = str(pick_device("auto"))
    clf = TorchVenomClassifier(run / "best.pt", device=dev)
    det = YoloDetector(dcfg["weights"], device=dev)
    pred = pd.read_csv(run / f"predictions_{a.split}.csv", dtype={"observation_id": str}).drop_duplicates("observation_id")
    sample = pd.concat([g.sample(min(a.n // 2, len(g)), random_state=vcfg["seed"]) for _, g in pred.groupby("venomous")])
    rows = []
    for k, r in enumerate(sample.itertuples()):
        img = cv2.cvtColor(cv2.imread(str(Path(vcfg["snakeclef"]["image_dir"]) / r.image_path)), cv2.COLOR_BGR2RGB)
        snakes = [d for d in det.detect(img, conf=dcfg["detector"]["conf"]) if d.cls == 0]
        row = {"image_path": r.image_path, "observation_id": r.observation_id, "venomous": r.venomous,
               "detected": bool(snakes), "p_full": clf.predict(img)}
        if snakes:
            b = max(snakes, key=lambda d: d.score).box
            h, w = img.shape[:2]
            row["box_area_frac"] = float((b[2] - b[0]) * (b[3] - b[1]) / (h * w))
            for m in MARGINS:
                row[f"p_crop{m}"] = clf.predict(crop(img, expand(b, m, h, w)))
        rows.append(row)
        if k % 100 == 0:
            print(f"{k}/{len(sample)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(run / f"crop_eval_{a.split}.csv", index=False)
    d = df[df.detected & df[[f"p_crop{m}" for m in MARGINS]].notna().all(1)]
    thr = clf.threshold
    cols = ["p_full"] + [f"p_crop{m}" for m in MARGINS]
    rng = np.random.default_rng(0)
    out = []
    for c in cols:
        st = stats(d.venomous.to_numpy(), d[c].to_numpy(), thr)
        boots = []
        for _ in range(a.n_boot):
            s = d.iloc[rng.integers(0, len(d), len(d))]
            boots.append(stats(s.venomous.to_numpy(), s[c].to_numpy(), thr))
        b = pd.DataFrame(boots)
        out.append({"input": c, **{k: v for k, v in st.items()},
                    **{f"{k}_ci": f"[{b[k].quantile(.025):.3f}, {b[k].quantile(.975):.3f}]" for k in st}})
    res = pd.DataFrame(out)
    md = [f"# Venom classifier on detector crops ({a.split})", "",
          f"{len(df)} sampled images ({int(df.venomous.sum())} venomous); detector found a snake in {int(df.detected.sum())}; "
          f"evaluated on those {len(d)} ({int(d.venomous.sum())} venomous). Threshold {thr:.4f} (valid, whole images). "
          "`p_cropM` = detector box grown by M x box size on each side.", "",
          res[["input", "recall", "recall_ci", "specificity", "specificity_ci", "auroc", "auroc_ci"]].to_markdown(index=False, floatfmt=".3f")]
    (run / f"crop_eval_{a.split}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
