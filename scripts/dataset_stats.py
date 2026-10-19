"""Dataset understanding: statistics + charts for the report (results/dataset_stats/).

Per split and class: images, objects, box vs polygon labels, empty (background) images, objects per
image, object size (share of image area), aspect ratio, and position of box centres.

Usage: python scripts/dataset_stats.py
"""
from __future__ import annotations

import collections
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from snakeguard import load_config

NAMES = {0: "snake", 1: "look-alike"}
COL = {0: "#c0392b", 1: "#2e86c1"}


def main():
    cfg = load_config()
    root = Path(cfg["data_root"])
    out = Path(cfg["results_dir"]) / "dataset_stats"
    out.mkdir(parents=True, exist_ok=True)
    stats, objs = {}, []
    for split in ["train", "valid", "test"]:
        s = collections.Counter()
        per_img = collections.Counter()
        for f in os.listdir(root / split / "images"):
            stem = os.path.splitext(f)[0]
            s["images"] += 1
            s["img_" + ("snake_like" if f.startswith("snake_like") else "snake")] += 1
            lbl = root / split / "labels" / (stem + ".txt")
            lines = [l.split() for l in lbl.read_text().splitlines() if l.strip()] if lbl.exists() else []
            per_img[min(len(lines), 5)] += 1
            if not lines:
                s["empty_images"] += 1
            for v in lines:
                c = int(float(v[0])); xy = np.array(v[1:], float)
                if len(xy) == 4:
                    cx, cy, w, h = xy; kind = "box"
                else:
                    p = xy.reshape(-1, 2); (x0, y0), (x1, y1) = p.min(0), p.max(0)
                    cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0; kind = "polygon"
                s[f"obj_{NAMES[c]}"] += 1
                s[f"{kind}_labels"] += 1
                objs.append((split, c, kind, cx, cy, w, h))
        s["objects_per_image"] = {("5+" if k == 5 else str(k)): v for k, v in sorted(per_img.items())}
        stats[split] = dict(s)

    a = np.array([(o[1], o[3], o[4], o[5], o[6]) for o in objs])
    area = a[:, 3] * a[:, 4]
    summ = {"splits": stats, "objects_total": len(objs)}
    for c in (0, 1):
        m = a[:, 0] == c
        summ[f"{NAMES[c]}_area_pct"] = {q: float(np.percentile(area[m] * 100, q)) for q in (10, 25, 50, 75, 90)}
        summ[f"{NAMES[c]}_small_lt_1pct"] = float((area[m] < 0.01).mean())
        summ[f"{NAMES[c]}_large_gt_50pct"] = float((area[m] > 0.5).mean())
    (out / "stats.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))

    # chart 1: class counts per split
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(3); w = 0.38
    for i, c in enumerate((0, 1)):
        vals = [stats[s].get(f"obj_{NAMES[c]}", 0) for s in ("train", "valid", "test")]
        b = ax.bar(x + (i - 0.5) * w, vals, w, color=COL[c], label=NAMES[c])
        ax.bar_label(b, fontsize=8)
    ax.set_xticks(x, ["train", "valid", "test"]); ax.set_ylabel("annotated objects")
    ax.set_title("Objects per class and split"); ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "class_counts.png", dpi=160); plt.close(fig)

    # chart 2: object size distribution
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.logspace(-4, 0, 40)
    for c in (0, 1):
        ax.hist(area[a[:, 0] == c], bins=bins, alpha=0.6, color=COL[c], label=NAMES[c])
    ax.set_xscale("log"); ax.set_xlabel("object area / image area (log scale)"); ax.set_ylabel("objects")
    ax.set_title("Object size distribution"); ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "object_sizes.png", dpi=160); plt.close(fig)

    # chart 3: where objects are (box centres)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))
    for ax, c in zip(axes, (0, 1)):
        m = a[:, 0] == c
        ax.hist2d(a[m, 1], a[m, 2], bins=30, range=[[0, 1], [0, 1]], cmap="magma")
        ax.invert_yaxis(); ax.set_title(f"{NAMES[c]} box centres"); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(out / "box_centres.png", dpi=160); plt.close(fig)

    # chart 4: image resolution mix
    sizes = collections.Counter()
    for split in ["train", "valid", "test"]:
        for f in os.listdir(root / split / "images"):
            with Image.open(root / split / "images" / f) as im:
                sizes[im.size] += 1
    top = sizes.most_common(10)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh([f"{w}x{h}" for (w, h), _ in top][::-1], [n for _, n in top][::-1], color="#7f8c8d")
    ax.set_xlabel("images"); ax.set_title("Most common image resolutions")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "resolutions.png", dpi=160); plt.close(fig)
    print("charts in", out)


if __name__ == "__main__":
    main()
