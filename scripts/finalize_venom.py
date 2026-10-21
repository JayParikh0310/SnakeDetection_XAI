"""B2: lock in the chosen model + score and report it on test (once).

Takes a trained run, a score (`venom` head or `species`-head sum, chosen on valid with compare_venom_scores.py),
picks the threshold on valid (venomous recall >= target_recall from configs/venom.yaml), evaluates test with
observation-bootstrap CIs, writes <run>/final_metrics.json + <run>/per_species_test_final.csv, and stores
{score, threshold} in <run>/best.pt so TorchVenomClassifier uses them.

Usage: python scripts/finalize_venom.py results/venom_clf/b0_reg --score species
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from snakeguard.venom import load_venom_config
from train_venom import evaluate, pick_threshold

COLUMN = {"venom": "p_venom", "species": "p_venom_species"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--score", choices=list(COLUMN), required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    run = Path(a.run)
    target = load_venom_config()["train"]["target_recall"]
    d = {}
    for s in ("valid", "test"):
        df = pd.read_csv(run / f"predictions_{s}.csv", dtype={"observation_id": str})
        df["p_venom"] = df[COLUMN[a.score]]  # evaluate() scores the `p_venom` column
        d[s] = df
    thr = pick_threshold(d["valid"].venomous.to_numpy(), d["valid"].p_venom.to_numpy(), target)
    res = {"run": run.name, "score": a.score, "threshold": thr, "target_recall": target,
           "valid": evaluate(d["valid"], thr, 0), "test": evaluate(d["test"], thr, a.n_boot)}
    (run / "final_metrics.json").write_text(json.dumps(res, indent=1))
    t = d["test"]
    per = t.assign(correct=(t.p_venom >= thr) == (t.venomous == 1)).groupby(["class_id", "binomial_name"]).agg(
        n=("correct", "size"), venomous=("venomous", "first"), acc=("correct", "mean"),
        mean_p=("p_venom", "mean")).reset_index().sort_values(["venomous", "acc"], ascending=[False, True])
    per.to_csv(run / "per_species_test_final.csv", index=False)
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    ck.update({"score": a.score, "threshold": thr, "target_recall": target})
    torch.save(ck, run / "best.pt")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
