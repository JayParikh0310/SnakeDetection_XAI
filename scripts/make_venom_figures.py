"""B3: qualitative grids of the classifier explainers on the valid split (method selection happens on valid).

Rows: images per outcome at the deployed threshold (TP = venomous caught, FN = venomous missed,
TN = harmless cleared, FP = harmless flagged), most confident first. Columns: image + every method.
Each map explains the predicted decision: target 1 (venomous) if score >= threshold, else 0 (harmless).

Usage: python scripts/make_venom_figures.py results/venom_clf/b0_reg [--per-kind 3] [--methods ...]
Writes <run>/figures/grid_<kind>.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from snakeguard.venom import TorchVenomClassifier, load_venom_config
from snakeguard.venom.xai import build_classifier_explainers
from snakeguard.xai.viz import overlay

TITLES = {"eigencam": "Eigen-CAM", "gradcam": "Grad-CAM", "gradcampp": "Grad-CAM++", "gradcampp_ms": "Grad-CAM++ (3 scales)",
          "layercam_ms": "LayerCAM (3 scales)", "integrated_gradients": "Integrated Gradients", "occlusion": "Occlusion",
          "rise": "RISE", "lime": "LIME (8x8)", "kernelshap": "KernelSHAP (8x8)", "random": "Random"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--per-kind", type=int, default=3)
    ap.add_argument("--methods", nargs="+", default=list(TITLES))
    a = ap.parse_args()
    cfg = load_venom_config()
    run = Path(a.run)
    clf = TorchVenomClassifier(run / "best.pt")
    ex = build_classifier_explainers(clf, {**cfg.get("xai", {}), "seed": cfg["seed"]}, a.methods)
    col = {"species": "p_venom_species", "venom": "p_venom"}[clf.score]
    d = pd.read_csv(run / "predictions_valid.csv")
    d["s"] = d[col]
    pred = d.s >= clf.threshold
    kinds = {"TP": d[pred & (d.venomous == 1)].sort_values("s", ascending=False),
             "FN": d[~pred & (d.venomous == 1)].sort_values("s"),
             "TN": d[~pred & (d.venomous == 0)].sort_values("s"),
             "FP": d[pred & (d.venomous == 0)].sort_values("s", ascending=False)}
    out = run / "figures"
    out.mkdir(exist_ok=True)
    for kind, g in kinds.items():
        rows = g.drop_duplicates("class_id").head(a.per_kind)  # different species per row
        fig, axes = plt.subplots(len(rows), len(a.methods) + 1, figsize=(2.4 * (len(a.methods) + 1), 2.6 * len(rows)),
                                 squeeze=False)
        for i, r in enumerate(rows.itertuples()):
            img = cv2.cvtColor(cv2.imread(str(Path(cfg["snakeclef"]["image_dir"]) / r.image_path)), cv2.COLOR_BGR2RGB)
            target = int(r.s >= clf.threshold)
            panels = [img] + [overlay(img, ex[m].explain(img, target)) for m in a.methods]
            for j, im in enumerate(panels):
                ax = axes[i, j]; ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
                if i == 0:
                    ax.set_title("image" if j == 0 else TITLES.get(a.methods[j - 1], a.methods[j - 1]), fontsize=8)
            axes[i, 0].set_ylabel(f"{r.binomial_name}\n{'venomous' if r.venomous else 'harmless'}, "
                                  f"P(ven)={r.s:.2f}\nexplains: {'venomous' if target else 'harmless'}", fontsize=7)
        fig.suptitle(f"{kind} (valid, threshold {clf.threshold:.3f})", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"grid_{kind}.png", dpi=110)
        plt.close(fig)
        print("wrote", out / f"grid_{kind}.png", flush=True)


if __name__ == "__main__":
    main()
