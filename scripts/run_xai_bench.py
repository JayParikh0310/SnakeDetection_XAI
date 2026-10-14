"""A4/A5: benchmark the detector explainers on a split.

For every image: run the detector, match detections to GT, and explain up to --max-targets
detections per image (TPs, class confusions and background FPs). Every explanation of every
target gets these metrics:
  faithfulness  deletion / insertion AUC (raw + normalized by the original score)
  localization  pointing game, energy pointing game, top-k IoU vs. the GT box and the GT polygon
  cost          runtime in ms
  optional      --stability (map similarity under input noise)
                --sanity (map similarity after cascading weight randomization: head, neck, all)
Pairwise method agreement (Spearman, top-20% IoU) goes in a second CSV.

The run is resumable: rows are appended per image and images already in the CSV are skipped.

Usage:
  python scripts/run_xai_bench.py --split valid --n 100 --tag dev
  python scripts/run_xai_bench.py --split test --tag final --stability --sanity
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from snakeguard import load_config
from snakeguard.data import list_split, load_sample
from snakeguard.detection import YoloDetector
from snakeguard.eval import metrics as M
from snakeguard.eval.matching import match
from snakeguard.xai import build_explainers

CORE = ["eigencam", "gradcampp", "occlusion"]
DEFAULT_METHODS = ["eigencam", "eigencam_p5", "eigencam_centered", "gradcam", "gradcampp", "gradcampp_p5", "gradcampp_p3",
                   "gradcampp_contrast", "occlusion", "drise", "random", "boxprior"]
NO_STABILITY = {"random", "boxprior"}  # these don't depend on the input/model
# Fixed schema: rows are appended per image and differ in which metrics apply (polygon GT, FPs
# without GT, diagnostics subset), so every row is written with exactly these columns in this order.
COLUMNS = ["image_id", "target", "kind", "det_cls", "gt_cls", "score", "match_iou", "box_area_frac",
           "method", "runtime_ms", "empty_map", "del_auc", "del_auc_norm", "ins_auc", "ins_auc_norm",
           "pg_box", "ebpg_box", "iou_box", "pg_poly", "ebpg_poly", "iou_poly", "stability",
           "sanity_head", "sanity_neck", "sanity_all"]
PAIR_COLUMNS = ["image_id", "target", "kind", "a", "b", "spearman", "topk_iou"]


def append_csv(rows: list[dict], path: Path, columns: list[str]):
    extra = set().union(*(r.keys() for r in rows)) - set(columns)
    if extra:
        raise KeyError(f"columns missing from schema: {extra}")
    df = pd.DataFrame(rows).reindex(columns=columns)
    if path.exists():
        with open(path) as f:
            header = f.readline().strip().split(",")
        if header != columns:
            raise RuntimeError(f"{path} was written with a different schema; delete it or use a new --tag")
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def timed(fn, cuda: bool):
    if cuda:
        torch.cuda.synchronize()
    t = time.perf_counter()
    r = fn()
    if cuda:
        torch.cuda.synchronize()
    return r, (time.perf_counter() - t) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid")
    ap.add_argument("--n", type=int, default=None, help="random subset of images (seeded)")
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--expensive", nargs="*", default=["drise"], help="methods run only on --n-expensive images")
    ap.add_argument("--n-expensive", type=int, default=200)
    ap.add_argument("--max-targets", type=int, default=2)
    ap.add_argument("--stability", action="store_true")
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--n-diag", type=int, default=150, help="images that get the --stability/--sanity diagnostics")
    ap.add_argument("--diag-exclude", nargs="*", default=["drise"], help="methods skipped in the diagnostics (cost)")
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    ev, dcfg = cfg["eval"], cfg["detector"]
    out = Path(cfg["results_dir"]) / "xai_bench" / f"{a.split}_{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps({"args": vars(a), "config": cfg,
                                                     "torch": torch.__version__}, indent=1))
    det = YoloDetector(cfg["weights"])
    cuda = det.device.type == "cuda"
    explainers = build_explainers(det, {**cfg["xai"], "seed": cfg["seed"]}, a.methods)

    paths = list_split(cfg["data_root"], a.split)
    rng = np.random.default_rng(cfg["seed"])
    if a.n and a.n < len(paths):
        paths = [paths[i] for i in sorted(rng.choice(len(paths), a.n, replace=False))]
    exp_ids = {p.stem for p in paths[:: max(1, len(paths) // a.n_expensive)][:a.n_expensive]}
    diag_ids = {p.stem for p in paths[:: max(1, len(paths) // a.n_diag)][:a.n_diag]}
    no_diag = NO_STABILITY | set(a.diag_exclude)

    rows_f, pair_f, curve_f = out / "metrics.csv", out / "agreement.csv", out / "curves.npz"
    done = set(pd.read_csv(rows_f, usecols=["image_id"]).image_id.astype(str)) if rows_f.exists() else set()
    curves = dict(np.load(curve_f)) if curve_f.exists() else {}
    print(f"{len(paths)} images, {len(done)} already done -> {out}")

    t_start = time.time()
    for n_img, p in enumerate(paths):
        if p.stem in done:
            continue
        s = load_sample(p)
        img = s.load_rgb()
        x, lb = det.preprocess(img)
        valid = lb.valid_mask()
        dets = det.detect_tensor(x, lb, conf=dcfg["conf"], iou=dcfg["iou"])
        targets = [m for m in match(dets, s.objects, dcfg["match_iou"]) if m.det is not None][:a.max_targets]
        rows, pairs = [], []
        for t_idx, m in enumerate(targets):
            d = m.det
            gt_box = M.gt_mask_lb(lb, m.gt.box) if m.gt is not None else None
            gt_poly = (M.gt_mask_lb(lb, m.gt.box, m.gt.polygon)
                       if m.gt is not None and m.gt.polygon is not None else None)
            maps = {}
            for name, ex in explainers.items():
                if name in a.expensive and p.stem not in exp_ids:
                    continue
                sal, ms = timed(lambda: ex.explain_lb(x, lb, d), cuda)
                maps[name] = sal
                di = M.deletion_insertion(det, x, lb, d, sal, steps=ev["steps"], deletion_fill=ev["deletion_fill"],
                                          insertion_start=ev["insertion_start"], blur_sigma=ev["blur_sigma"],
                                          seed=cfg["seed"])
                for k in ("deletion", "insertion"):
                    key = f"{name}|{m.kind}|{k}"
                    curves[key] = curves.get(key, 0) + di[k].scores / max(d.score, 1e-6)
                    curves[key + "|n"] = curves.get(key + "|n", 0) + 1
                r = {"image_id": s.image_id, "target": t_idx, "kind": m.kind, "det_cls": d.cls,
                     "gt_cls": None if m.gt is None else m.gt.cls, "score": d.score, "match_iou": m.iou,
                     "box_area_frac": float((d.box_lb[2] - d.box_lb[0]) * (d.box_lb[3] - d.box_lb[1]) / valid.sum()),
                     "method": name, "runtime_ms": ms, "empty_map": float(sal.max() <= 0),
                     "del_auc": di["deletion"].auc, "del_auc_norm": di["deletion"].auc_norm,
                     "ins_auc": di["insertion"].auc, "ins_auc_norm": di["insertion"].auc_norm}
                if gt_box is not None:
                    r.update(pg_box=M.pointing_game(sal, gt_box, ev["pointing_tolerance"]),
                             ebpg_box=M.energy_pointing_game(sal, gt_box),
                             iou_box=M.saliency_iou(sal, gt_box, valid))
                if gt_poly is not None:
                    r.update(pg_poly=M.pointing_game(sal, gt_poly, ev["pointing_tolerance"]),
                             ebpg_poly=M.energy_pointing_game(sal, gt_poly),
                             iou_poly=M.saliency_iou(sal, gt_poly, valid))
                if a.stability and p.stem in diag_ids and name not in no_diag:
                    r["stability"] = M.rank_correlation(sal, ex.explain_lb(M.noisy(x, ev["noise_std"], cfg["seed"]), lb, d), valid)
                rows.append(r)
            if a.sanity and p.stem in diag_ids:
                # YOLO11n: layers 0-10 backbone, 11-22 neck, 23 Detect head
                levels = {"head": len(det.model.model) - 1, "neck": 11, "all": 0}
                for lvl, down_to in levels.items():
                    with M.RandomizedDetector(det, down_to, seed=cfg["seed"]):
                        for r in rows:
                            if r["target"] == t_idx and r["method"] in maps and r["method"] not in no_diag:
                                rs = explainers[r["method"]].explain_lb(x, lb, d)
                                r[f"sanity_{lvl}"] = M.rank_correlation(maps[r["method"]], rs, valid)
            for a_name, b_name in itertools.combinations(maps, 2):
                pairs.append({"image_id": s.image_id, "target": t_idx, "kind": m.kind, "a": a_name, "b": b_name,
                              "spearman": M.rank_correlation(maps[a_name], maps[b_name], valid),
                              "topk_iou": M.topk_iou(maps[a_name], maps[b_name], valid, ev["topk_frac"])})
        if not rows:  # still record that the image was processed (no detections)
            rows = [{"image_id": s.image_id, "method": None, "kind": "no_detection"}]
        append_csv(rows, rows_f, COLUMNS)
        if pairs:
            append_csv(pairs, pair_f, PAIR_COLUMNS)
        np.savez(curve_f, **curves)  # saved per image so a resumed run stays consistent with metrics.csv
        if n_img % 20 == 0:
            el = time.time() - t_start
            print(f"[{n_img + 1}/{len(paths)}] {el / 60:.1f} min elapsed", flush=True)
    np.savez(curve_f, **curves)
    print("done")


if __name__ == "__main__":
    main()
