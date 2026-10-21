"""B6: report figure of the demo app's photo analysis (same code path as scripts/app.py).

Picks seeded test images (venomous and harmless, detector finds a snake) and shows: detections + verdict,
"why the detector fired" (detector LayerCAM), "why this venom call" (venom LayerCAM ms on the context crop).

Usage: python scripts/make_app_figure.py [--n 2]   -> results/app/app_examples.png
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from snakeguard import PROJECT_ROOT
from snakeguard.demo import SnakeGuardDemo
from snakeguard.venom import load_venom_config

RUN = PROJECT_ROOT / "results" / "venom_clf" / "b0_reg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2, help="images per class (venomous / harmless)")
    a = ap.parse_args()
    vcfg = load_venom_config()
    demo = SnakeGuardDemo(venom_checkpoint=str(RUN / "best.pt"))
    pred = pd.read_csv(RUN / "predictions_test.csv").drop_duplicates("observation_id")
    picked = []
    for v in (1, 0):
        for r in pred[pred.venomous == v].sample(frac=1, random_state=vcfg["seed"]).itertuples():
            img = cv2.cvtColor(cv2.imread(str(Path(vcfg["snakeclef"]["image_dir"]) / r.image_path)), cv2.COLOR_BGR2RGB)
            res = demo.analyze_photo(img)
            if res.snakes and res.venom_map is not None and min(img.shape[:2]) >= 200:
                picked.append((r, res))
            if sum(p[0].venomous == v for p in picked) == a.n:
                break
    fig, axes = plt.subplots(len(picked), 3, figsize=(12, 3.6 * len(picked)), squeeze=False)
    for i, (r, res) in enumerate(picked):
        truth = "venomous" if r.venomous else "harmless"
        panels = [(res.annotated, f"{r.binomial_name} (truth: {truth})"),
                  (res.detector_map, f"Why the detector fired ({res.detector_method}, "
                                     f"{res.detector_mass_in_box:.0%} in box)"),
                  (res.venom_map, f"Why this venom call ({res.venom_method})")]
        for j, (im, title) in enumerate(panels):
            ax = axes[i, j]; ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, fontsize=8)
        axes[i, 0].set_xlabel("\n".join(textwrap.wrap(res.verdict, 60)), fontsize=8)
    fig.tight_layout()
    out = PROJECT_ROOT / "results" / "app"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "app_examples.png", dpi=120)
    print("wrote", out / "app_examples.png", [(p[0].binomial_name, p[1].verdict) for p in picked])


if __name__ == "__main__":
    main()
