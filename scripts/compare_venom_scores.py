"""B2: compare venom scores and runs on one split (default valid; tune on valid, report the final choice on test).

Scores per run (from predictions_<split>.csv written by train_venom.py):
  venom     P(venomous) from the venom head
  species   sum of P(species) over venomous species (species head)
  mean      average of the two
Metrics: AUROC, AP, and specificity at the highest threshold giving venomous recall >= 0.95 / 0.90 (threshold
chosen on the same split, so this compares rankings, not a deployable threshold). 95% CIs and paired differences
vs the reference (first run, `venom`) come from a bootstrap over observations; the threshold is re-chosen in
every resample.

Usage: python scripts/compare_venom_scores.py results/venom_clf/b0 [results/venom_clf/b0_reg ...] [--split valid]
Writes results/venom_clf/compare_<split>.{csv,md}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from train_venom import pick_threshold

SCORES = ("venom", "species", "mean")


def scores(d: pd.DataFrame) -> dict[str, np.ndarray]:
    v, s = d.p_venom.to_numpy(), d.p_venom_species.to_numpy()
    return {"venom": v, "species": s, "mean": (v + s) / 2}


def spec_at_recall(y, p, recall):
    thr = pick_threshold(y, p, recall)
    return float((p[y == 0] < thr).mean())


def metric_fns():
    return {"auroc": roc_auc_score, "ap": average_precision_score,
            "spec@rec95": lambda y, p: spec_at_recall(y, p, 0.95),
            "spec@rec90": lambda y, p: spec_at_recall(y, p, 0.90)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    runs = [Path(r) for r in a.runs]
    preds = {r.name: pd.read_csv(r / f"predictions_{a.split}.csv", dtype={"observation_id": str}) for r in runs}
    base = next(iter(preds.values()))
    for name, d in preds.items():  # every run must score the same images in the same order
        assert (d.image_path.to_numpy() == base.image_path.to_numpy()).all(), name
    y = base.venomous.to_numpy()
    cand = {f"{r}:{s}": p for r, d in preds.items() for s, p in scores(d).items()}
    ref = f"{runs[0].name}:venom"
    fns = metric_fns()

    point = {c: {m: f(y, p) for m, f in fns.items()} for c, p in cand.items()}
    rng = np.random.default_rng(a.seed)
    groups = base.groupby("observation_id").indices
    keys = np.array(list(groups))
    boot = {c: {m: [] for m in fns} for c in cand}
    for _ in range(a.n_boot):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])
        for c, p in cand.items():
            for m, f in fns.items():
                boot[c][m].append(f(y[idx], p[idx]))
    rows = []
    for c in cand:
        r = {"candidate": c}
        for m in fns:
            b = np.array(boot[c][m]); d = b - np.array(boot[ref][m])
            r[m] = point[c][m]
            r[m + "_ci"] = f"[{np.percentile(b, 2.5):.3f}, {np.percentile(b, 97.5):.3f}]"
            r[m + "_diff"] = point[c][m] - point[ref][m]
            r[m + "_diff_ci"] = f"[{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]"
        rows.append(r)
    df = pd.DataFrame(rows)
    out = runs[0].parent
    df.to_csv(out / f"compare_{a.split}.csv", index=False)
    cols = ["candidate", "auroc", "auroc_diff_ci", "ap", "spec@rec95", "spec@rec95_diff_ci", "spec@rec90"]
    md = [f"# Venom score comparison ({a.split}, n = {len(y)}, {len(keys)} observations)", "",
          f"Reference: `{ref}`. `*_diff_ci` = 95 % bootstrap CI of the paired difference vs the reference "
          f"({a.n_boot} resamples over observations). spec@rec95 = specificity at venomous recall >= 0.95.", "",
          df[cols].to_markdown(index=False, floatfmt=".3f")]
    (out / f"compare_{a.split}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
