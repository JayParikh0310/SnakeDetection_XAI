"""A5: qualitative figures.

1. Method grids: image+box | Eigen-CAM | Grad-CAM++ (multi-scale) | Grad-CAM++ (P5 only) | LayerCAM | Occlusion | D-RISE
   for a few examples of each kind: TP snake, TP look-alike, confusions, background FPs.
2. Matched pairs (proposal §2): every look-alike (TP sll / confusion / FP) is paired with the real
   snake that the detector itself finds most similar (cosine similarity of RoI-pooled P5 features),
   and the two are shown side by side under all core methods.

Usage: python scripts/make_figures.py --split test --pool 400 --per-kind 4 --pairs 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from snakeguard import CLASS_NAMES, load_config
from snakeguard.data import list_split, load_sample
from snakeguard.detection import YoloDetector
from snakeguard.eval.matching import match
from snakeguard.xai import build_explainers
from snakeguard.xai.viz import draw_box, overlay

METHODS = ["eigencam", "gradcampp", "gradcampp_p5", "layercam", "occlusion", "drise"]
TITLES = {"eigencam": "Eigen-CAM", "gradcampp": "Grad-CAM++ (P3-P5)", "gradcampp_p5": "Grad-CAM++ (P5 only)",
          "layercam": "LayerCAM (P3-P5)", "occlusion": "Occlusion", "drise": "D-RISE"}


def kind_label(m) -> str:
    if m.kind == "TP":
        return f"TP_{CLASS_NAMES[m.det.cls]}"
    if m.kind == "confusion":
        return f"confusion_{CLASS_NAMES[m.gt.cls]}_as_{CLASS_NAMES[m.det.cls]}"
    return f"FP_{CLASS_NAMES[m.det.cls]}"


def collect(det, cfg, split, pool, seed):
    """Run the detector on a seeded pool of images. Return targets + RoI embeddings."""
    paths = list_split(cfg["data_root"], split)
    rng = np.random.default_rng(seed)
    paths = [paths[i] for i in sorted(rng.choice(len(paths), min(pool, len(paths)), replace=False))]
    feats = {}
    h = det.layer(det.scale_layers[2]).register_forward_hook(lambda m, i, o: feats.__setitem__("p5", o))
    items = []
    try:
        for p in paths:
            s = load_sample(p)
            img = s.load_rgb()
            x, lb = det.preprocess(img)
            dets = det.detect_tensor(x, lb, conf=cfg["detector"]["conf"], iou=cfg["detector"]["iou"])
            for m in match(dets, s.objects, cfg["detector"]["match_iou"]):
                if m.det is None:
                    continue
                box = torch.tensor([[0, *m.det.box_lb]], dtype=torch.float32, device=det.device)
                emb = torchvision.ops.roi_align(feats["p5"], box, output_size=3, spatial_scale=1 / 32).mean((2, 3))[0]
                items.append({"path": p, "match": m, "label": kind_label(m),
                              "emb": torch.nn.functional.normalize(emb, dim=0).cpu().numpy()})
    finally:
        h.remove()
    return items


def explain_row(det, explainers, item):
    img = load_sample(item["path"]).load_rgb()
    d = item["match"].det
    x, lb = det.preprocess(img)
    maps = {n: lb.map_to_orig(e.explain_lb(x, lb, d)) for n, e in explainers.items()}
    gt = item["match"].gt
    base = draw_box(img, d.box, d.cls, d.score)
    if gt is not None and gt.cls != d.cls:
        base = draw_box(base, gt.box, gt.cls, label=f"GT {CLASS_NAMES[gt.cls]}", thickness=2)
    return base, [draw_box(overlay(img, maps[n]), d.box, d.cls, d.score, label=" ", thickness=2)
                  for n in explainers]


def grid(rows, row_titles, col_titles, path):
    fig, axes = plt.subplots(len(rows), len(col_titles), figsize=(3 * len(col_titles), 3 * len(rows)), squeeze=False)
    for i, imgs in enumerate(rows):
        for j, im in enumerate(imgs):
            ax = axes[i, j]
            ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(col_titles[j], fontsize=10)
            if j == 0:
                ax.set_ylabel(row_titles[i], fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--pool", type=int, default=400)
    ap.add_argument("--per-kind", type=int, default=4)
    ap.add_argument("--pairs", type=int, default=8)
    a = ap.parse_args()
    cfg = load_config()
    out = Path(cfg["results_dir"]) / "figures" / a.split
    out.mkdir(parents=True, exist_ok=True)
    det = YoloDetector(cfg["weights"])
    explainers = build_explainers(det, {**cfg["xai"], "seed": cfg["seed"]}, METHODS)
    items = collect(det, cfg, a.split, a.pool, cfg["seed"])
    cols = ["detection"] + [TITLES[m] for m in METHODS]
    print(f"{len(items)} targets:", {k: sum(i['label'] == k for i in items) for k in sorted({i['label'] for i in items})})

    # 1. method grids per kind (most confident examples)
    for label in sorted({i["label"] for i in items}):
        sel = sorted([i for i in items if i["label"] == label], key=lambda i: -i["match"].det.score)[:a.per_kind]
        rows, titles = [], []
        for it in sel:
            base, ov = explain_row(det, explainers, it)
            rows.append([base] + ov); titles.append(f"{it['path'].stem}\n{label}")
        grid(rows, titles, cols, out / f"grid_{label}.png")

    # 2. matched pairs: look-alikes vs. their most similar real snake
    snakes = [i for i in items if i["label"] == "TP_snake"]
    lookalikes = [i for i in items if i["label"] != "TP_snake"]
    # hard cases first: confusions and FPs, then the most confident TP look-alikes
    lookalikes.sort(key=lambda i: (i["label"].startswith("TP"), -i["match"].det.score))
    if snakes:
        S = np.stack([s["emb"] for s in snakes])
        rows, titles = [], []
        for it in lookalikes[:a.pairs]:
            j = int(np.argmax(S @ it["emb"]))
            sim = float(S[j] @ it["emb"])
            for member, tag in ((it, it["label"]), (snakes[j], f"matched snake (cos {sim:.2f})")):
                base, ov = explain_row(det, explainers, member)
                rows.append([base] + ov); titles.append(f"{member['path'].stem}\n{tag}")
        for k in range(0, len(rows), 4):  # 2 pairs per figure
            grid(rows[k:k + 4], titles[k:k + 4], cols, out / f"pairs_{k // 4:02d}.png")
    print("figures in", out)


if __name__ == "__main__":
    main()
