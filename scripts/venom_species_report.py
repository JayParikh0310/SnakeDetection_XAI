"""B6: per-species venom results for a named species list (default: India's "Big Four"), with Wilson 95 % CIs.

Reads <run>/predictions_test.csv (written by train_venom.py) and the deployed score + threshold from <run>/best.pt
(set by finalize_venom.py). Recall = share of that species' test images scored >= threshold (all are venomous).
Training counts come from data/snakeclef/manifest.csv (available images only).

Usage: python scripts/venom_species_report.py results/venom_clf/b0_reg [--species "Naja naja" ...]
Writes <run>/species_report_test.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from snakeguard.venom import load_manifest, load_venom_config

BIG_FOUR = {"Naja naja": "spectacled cobra", "Bungarus caeruleus": "common krait",
            "Daboia russelii": "Russell's viper", "Echis carinatus": "saw-scaled viper"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--species", nargs="*", default=list(BIG_FOUR))
    a = ap.parse_args()
    run = Path(a.run)
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    col = {"species": "p_venom_species", "venom": "p_venom"}[ck.get("score", "venom")]
    thr = float(ck["threshold"])
    pred = pd.read_csv(run / "predictions_test.csv")
    train = load_manifest(load_venom_config(), "train")
    rows = []
    for sp in a.species:
        t = pred[pred.binomial_name == sp]
        k, n = int((t[col] >= thr).sum()), len(t)
        lo, hi = wilson(k, n)
        rows.append({"species": sp, "common name": BIG_FOUR.get(sp, ""),
                     "venomous (MIVS)": int(t.venomous.max()) if n else None,
                     "train images": int((train.binomial_name == sp).sum()), "test images": n,
                     "caught": k, "recall": k / n if n else float("nan"), "95% CI": f"[{lo:.2f}, {hi:.2f}]",
                     "median score": float(t[col].median()) if n else float("nan")})
    df = pd.DataFrame(rows)
    md = [f"# Per-species venom results (test, {run.name}, score `{col}`, threshold {thr:.4f})", "",
          "Recall = share of the species' test images flagged venomous. Wilson 95 % CI; images of one observation "
          "are correlated, so the CI is optimistic.", "", df.to_markdown(index=False, floatfmt=".2f")]
    (run / "species_report_test.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
