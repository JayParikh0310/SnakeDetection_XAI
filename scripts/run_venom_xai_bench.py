"""B4: benchmark the venom-classifier explainers with Member A's metrics (eval/metrics.py).

Images: a seeded sample of one split (default 80 TP + 80 FP + 20 FN + 20 TN), stratified by outcome at the deployed threshold (TP = venomous caught,
FN = venomous missed, TN = harmless cleared, FP = harmless flagged). Each map explains the model's decision
(target 1 if score >= threshold else 0), in the 224 centre crop the model sees.

Metrics per (image, method), same names/meaning as A's bench where they exist:
  faithfulness  del_auc(_norm), ins_auc(_norm)    A's deletion_insertion (gray deletion, blurred insertion)
                del_auc_range, ins_auc_range      same curves rescaled between the blank-image score (0) and the original (1)
  localization  pg_detbox, ebpg_detbox, iou_detbox  PROXY GT: the frozen YOLO detector's best snake box (conf >= 0.25)
                                                  mapped into the crop; empty when the detector finds no snake
  cost          runtime_ms
  diagnostics   stability (noise), sanity_head / sanity_top / sanity_all (cascading randomization), on --n-diag images
Pairwise agreement (Spearman, top-20% IoU) -> agreement.csv. Resumable per image.

Usage:
  python scripts/run_venom_xai_bench.py results/venom_clf/b0_reg --split valid --n-venomous-decision 3 --n-harmless-decision 3 --tag dev --n-expensive 4 --n-diag 4
  python scripts/run_venom_xai_bench.py results/venom_clf/b0_reg --split test --tag final
Faithfulness is primarily read on venomous decisions (TP, FP): with no evidence the model says "harmless", so
deletion/insertion barely move for harmless decisions; for those use the baseline-corrected del/ins_auc_range.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from snakeguard import load_config
from snakeguard.detection import YoloDetector
from snakeguard.detection.yolo import PAD_VALUE, Letterbox
from snakeguard.eval import metrics as M
from snakeguard.venom import TorchVenomClassifier, load_venom_config
from snakeguard.venom.xai import build_classifier_explainers

METHODS = ["eigencam", "gradcam", "gradcampp", "gradcampp_ms", "layercam_ms", "integrated_gradients",
           "occlusion", "rise", "lime", "kernelshap", "random"]
EXPENSIVE = ["rise", "lime", "kernelshap"]
NO_DIAG = {"random", "rise", "lime", "kernelshap", "occlusion"}   # input/model-independent or too slow
COLUMNS = ["image_id", "image_path", "target", "kind", "det_cls", "gt_cls", "class_id", "binomial_name", "score",
           "blank_score", "method", "runtime_ms", "empty_map", "del_auc", "del_auc_norm", "ins_auc", "ins_auc_norm",
           "del_auc_range", "ins_auc_range",
           "det_box_conf", "pg_detbox", "ebpg_detbox", "iou_detbox", "stability", "sanity_head", "sanity_top", "sanity_all"]
PAIR_COLUMNS = ["image_id", "target", "kind", "a", "b", "spearman", "topk_iou"]


class RandomizedClassifier:
    """Cascading model-parameter randomization (Adebayo et al. 2018) for VenomNet, restored on exit.
    level: "head" = venom + species heads; "top" = heads + features[6:] (7x7 stages); "all" = every layer."""

    def __init__(self, model: torch.nn.Module, level: str, seed: int = 0):
        self.model, self.level, self.seed = model, level, seed

    def _modules(self):
        heads = [self.model.venom_head, self.model.species_head]
        if self.level == "head":
            return heads
        if self.level == "top":
            return heads + list(self.model.features[6:])
        return heads + list(self.model.features)

    def __enter__(self):
        self.backup = copy.deepcopy(self.model.state_dict())
        g = torch.Generator().manual_seed(self.seed)
        for mod in self._modules():
            for m in mod.modules():
                if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                    w = torch.randn(m.weight.shape, generator=g) * m.weight.std().cpu()
                    m.weight.data.copy_(w.to(m.weight.device))
                    if m.bias is not None:
                        m.bias.data.zero_()
        return self.model

    def __exit__(self, *exc):
        self.model.load_state_dict(self.backup)
        return False


def append_csv(rows, path: Path, columns):
    extra = set().union(*(r.keys() for r in rows)) - set(columns)
    if extra:
        raise KeyError(f"columns missing from schema: {extra}")
    if path.exists() and open(path).readline().strip().split(",") != columns:
        raise RuntimeError(f"{path} was written with a different schema; use a new --tag")
    pd.DataFrame(rows).reindex(columns=columns).to_csv(path, mode="a", header=not path.exists(), index=False)


def range_auc(fr: np.ndarray, sc: np.ndarray, s0: float, s_blank: float) -> float:
    """AUC of the curve rescaled so the original image = 1 and a blank (gray) image = 0. Needed because the
    classifier's no-evidence output is "harmless" (P(venomous | gray) ~ 0.14): for harmless decisions the plain
    normalized curve hardly moves. Undefined (NaN) when the original is not above the blank score."""
    d = s0 - s_blank
    return float(np.trapezoid((sc - s_blank) / d, fr)) if d > 1e-3 else float("nan")


def sample_images(pred: pd.DataFrame, thr: float, per_kind: dict, seed: int) -> pd.DataFrame:
    """per_kind[kind] images per outcome, at most one image per observation, seeded."""
    v = pred.venomous == 1
    hit = pred.s >= thr
    pred = pred.assign(kind=np.select([v & hit, v & ~hit, ~v & ~hit, ~v & hit], ["TP", "FN", "TN", "FP"], default="?"))
    pred = pred.drop_duplicates("observation_id")
    parts = [g.sample(min(per_kind[k], len(g)), random_state=seed) for k, g in pred.groupby("kind")]
    return pd.concat(parts).sort_values(["kind", "image_path"]).reset_index(drop=True)


def every_kth(ids: list[str], n: int) -> set[str]:
    return set(ids[:: max(1, len(ids) // max(n, 1))][:n]) if n else set()


def timed(fn, device):
    sync = torch.mps.synchronize if device.type == "mps" else (torch.cuda.synchronize if device.type == "cuda" else None)
    if sync: sync()
    t = time.perf_counter(); r = fn()
    if sync: sync()
    return r, (time.perf_counter() - t) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="trained run with best.pt + predictions_<split>.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-venomous-decision", type=int, default=80, help="images per kind for TP and FP")
    ap.add_argument("--n-harmless-decision", type=int, default=20, help="images per kind for FN and TN")
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--n-expensive", type=int, default=80, help="images that get rise/lime/kernelshap")
    ap.add_argument("--n-diag", type=int, default=80, help="images that get stability + sanity")
    ap.add_argument("--tag", default="final")
    a = ap.parse_args()

    vcfg, dcfg = load_venom_config(), load_config()
    ev, seed = dcfg["eval"], vcfg["seed"]
    run = Path(a.run)
    out = Path(dcfg["results_dir"]) / "venom_xai_bench" / f"{a.split}_{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    clf = TorchVenomClassifier(run / "best.pt")
    explainers = build_classifier_explainers(clf, {**vcfg.get("xai", {}), "seed": seed}, a.methods)
    scorer = next(iter(explainers.values())).scorer
    det = YoloDetector(dcfg["weights"])
    (out / "run_config.json").write_text(json.dumps({"args": vars(a), "venom_config": vcfg, "eval": ev,
                                                     "classifier": {"score": clf.score, "threshold": clf.threshold},
                                                     "torch": torch.__version__}, indent=1, default=str))

    pred = pd.read_csv(run / f"predictions_{a.split}.csv", dtype={"observation_id": str})
    pred["s"] = pred[{"species": "p_venom_species", "venom": "p_venom"}[clf.score]]
    nv, nh = a.n_venomous_decision, a.n_harmless_decision
    imgs = sample_images(pred, clf.threshold, {"TP": nv, "FP": nv, "FN": nh, "TN": nh}, seed)
    ids = imgs.image_path.tolist()
    exp_ids, diag_ids = every_kth(ids, a.n_expensive), every_kth(ids, a.n_diag)
    print(f"{len(imgs)} images {imgs.kind.value_counts().to_dict()} -> {out}", flush=True)

    rows_f, pair_f, curve_f = out / "metrics.csv", out / "agreement.csv", out / "curves.npz"
    done = set(pd.read_csv(rows_f, usecols=["image_path"]).image_path) if rows_f.exists() else set()
    curves = dict(np.load(curve_f)) if curve_f.exists() else {}
    S = clf.img_size
    lb = Letterbox(1.0, 0, 0, S, S, S, S, S, S)  # identity "letterbox": whole crop is valid
    valid = lb.valid_mask()
    t0 = time.time()
    for k, r in enumerate(imgs.itertuples()):
        if r.image_path in done:
            continue
        img = cv2.cvtColor(cv2.imread(str(Path(vcfg["snakeclef"]["image_dir"]) / r.image_path)), cv2.COLOR_BGR2RGB)
        ex0 = next(iter(explainers.values()))
        x, view = ex0.preprocess(img)
        target = int(r.s >= clf.threshold)
        snakes = [d for d in det.detect(img, conf=dcfg["detector"]["conf"]) if d.cls == 0]
        box = max(snakes, key=lambda d: d.score) if snakes else None
        gt = view.box_mask(box.box) if box is not None else None
        if gt is not None and not gt.any():
            gt = None  # detected snake lies outside the centre crop
        base = {"image_id": Path(r.image_path).stem, "image_path": r.image_path, "target": 0, "kind": r.kind,
                "det_cls": target, "gt_cls": int(r.venomous), "class_id": r.class_id, "binomial_name": r.binomial_name,
                "score": float(scorer.score(x, target)[0]), "det_box_conf": None if box is None else box.score,
                "blank_score": float(scorer.score(torch.full_like(x, PAD_VALUE / 255), target)[0])}
        rows, maps = [], {}
        for name, ex in explainers.items():
            if name in EXPENSIVE and r.image_path not in exp_ids:
                continue
            sal, ms = timed(lambda: ex.explain_x(x, target), x.device)
            maps[name] = sal
            di = M.deletion_insertion(scorer, x, lb, target, sal, steps=ev["steps"], deletion_fill=ev["deletion_fill"],
                                      insertion_start=ev["insertion_start"], blur_sigma=ev["blur_sigma"], seed=seed)
            for kk in ("deletion", "insertion"):
                key = f"{name}|{r.kind}|{kk}"
                curves[key] = curves.get(key, 0) + di[kk].scores / max(base["score"], 1e-6)
                curves[key + "|n"] = curves.get(key + "|n", 0) + 1
            row = {**base, "method": name, "runtime_ms": ms, "empty_map": float(sal.max() <= 0),
                   "del_auc": di["deletion"].auc, "del_auc_norm": di["deletion"].auc_norm,
                   "ins_auc": di["insertion"].auc, "ins_auc_norm": di["insertion"].auc_norm,
                   "del_auc_range": range_auc(di["deletion"].fractions, di["deletion"].scores, base["score"], base["blank_score"]),
                   "ins_auc_range": range_auc(di["insertion"].fractions, di["insertion"].scores, base["score"], base["blank_score"])}
            if gt is not None:
                row.update(pg_detbox=M.pointing_game(sal, gt, tolerance=5), ebpg_detbox=M.energy_pointing_game(sal, gt),
                           iou_detbox=M.saliency_iou(sal, gt, valid))
            if r.image_path in diag_ids and name not in NO_DIAG:
                row["stability"] = M.rank_correlation(sal, ex.explain_x(M.noisy(x, ev["noise_std"], seed), target), valid)
            rows.append(row)
        if r.image_path in diag_ids:
            for level in ("head", "top", "all"):
                with RandomizedClassifier(clf.model, level, seed=seed):
                    for row in rows:
                        if row["method"] not in NO_DIAG:
                            rs = explainers[row["method"]].explain_x(x, target)
                            row[f"sanity_{level}"] = M.rank_correlation(maps[row["method"]], rs, valid)
        pairs = [{"image_id": base["image_id"], "target": 0, "kind": r.kind, "a": p, "b": q,
                  "spearman": M.rank_correlation(maps[p], maps[q], valid),
                  "topk_iou": M.topk_iou(maps[p], maps[q], valid, ev["topk_frac"])}
                 for p, q in itertools.combinations(maps, 2)]
        append_csv(rows, rows_f, COLUMNS)
        if pairs:
            append_csv(pairs, pair_f, PAIR_COLUMNS)
        np.savez(curve_f, **curves)
        if k % 10 == 0:
            print(f"[{k + 1}/{len(imgs)}] {(time.time() - t0) / 60:.1f} min", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
