"""B4: tables + plots for a classifier XAI bench run (reuses A's analyze_bench.py helpers).

Outputs in <run>/analysis/:
  summary.csv / summary.md   mean ± 95% bootstrap CI per method: all images and per outcome (TP/FN/TN/FP)
  vs_random.csv              per-image win rate over the random baseline (deletion / insertion)
  wilcoxon.csv               paired Wilcoxon tests for the key comparisons
  agreement.csv / .png       mean pairwise Spearman between methods
  curves_<kind>.png, bars_<metric>.png

Usage: python scripts/analyze_venom_bench.py results/venom_xai_bench/test_final
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import analyze_bench as AB

METRICS = ["del_auc_norm", "ins_auc_norm", "del_auc_range", "ins_auc_range", "del_auc", "ins_auc", "pg_detbox", "ebpg_detbox", "iou_detbox",
           "empty_map", "stability", "sanity_head", "sanity_top", "sanity_all", "runtime_ms"]
HIGHER_BETTER = {"del_auc_norm": False, "del_auc_range": False, "del_auc": False, "runtime_ms": False, "empty_map": False,
                 "sanity_head": False, "sanity_top": False, "sanity_all": False}
ORDER = ["eigencam", "gradcam", "gradcampp", "gradcampp_ms", "layercam_ms", "integrated_gradients", "occlusion",
         "rise", "lime", "kernelshap", "random"]
PAIRS = [("occlusion", "gradcampp"), ("occlusion", "eigencam"), ("gradcampp", "eigencam"), ("gradcampp_ms", "gradcampp"),
         ("layercam_ms", "gradcampp"), ("layercam_ms", "occlusion"), ("integrated_gradients", "occlusion"),
         ("kernelshap", "lime"), ("kernelshap", "occlusion"), ("rise", "occlusion")]
KINDS = ["TP", "FN", "TN", "FP"]
MAIN = ["del_auc_norm", "ins_auc_norm", "pg_detbox", "ebpg_detbox", "stability", "sanity_head", "sanity_all", "runtime_ms"]

# A's helpers read these module globals
AB.METRICS, AB.HIGHER_BETTER, AB.ORDER = METRICS, HIGHER_BETTER, ORDER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    a = ap.parse_args()
    run = Path(a.run)
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    df = pd.read_csv(run / "metrics.csv")
    df["key"] = df.image_path
    n_box = int(df.drop_duplicates("key").det_box_conf.notna().sum())
    md = [f"# Classifier XAI benchmark: {run.name}", "",
          f"Images: {df.key.nunique()} ({df.drop_duplicates('key').kind.value_counts().to_dict()}). Each map explains the "
          f"model's decision. Localization (`*_detbox`) is a **proxy**: the detector's snake box, available for "
          f"{n_box} images. Lower is better for del_auc_norm, sanity_* and runtime; higher for the rest. "
          "Bold = best non-baseline method. ± = half the 95% bootstrap CI.", ""]
    s_all = AB.summarize(df)
    s_all.to_csv(out / "summary.csv", index=False)
    ven = df[df.det_cls == 1]                     # venomous decisions (TP + FP): the primary faithfulness set
    s_ven = AB.summarize(ven); s_ven.to_csv(out / "summary_venomous_decisions.csv", index=False)
    md += [f"## Venomous decisions (TP + FP, {ven.key.nunique()} images): primary", AB.to_markdown(s_ven, MAIN), ""]
    harm = df[df.det_cls == 0]
    if harm.key.nunique() >= 5:
        s_h = AB.summarize(harm); s_h.to_csv(out / "summary_harmless_decisions.csv", index=False)
        md += [f"## Harmless decisions (FN + TN, {harm.key.nunique()} images): localization / diagnostics only",
               "Faithfulness is not measurable here: a blank image already scores 'harmless' (P(venomous | gray) is at "
               "the threshold), so deletion/insertion hardly move; the baseline-corrected `*_range` columns are kept in "
               "metrics.csv but are unstable (tiny original-minus-blank gap) and not reported.",
               AB.to_markdown(s_h, ["pg_detbox", "ebpg_detbox", "stability", "sanity_head", "sanity_all"]), ""]
    md += ["## All images", AB.to_markdown(s_all, MAIN), ""]
    for kind in KINDS:
        g = df[df.kind == kind]
        if g.key.nunique() >= 5:
            s = AB.summarize(g); s.to_csv(out / f"summary_{kind}.csv", index=False)
            md += [f"## {kind} only ({g.key.nunique()} images)",
                   AB.to_markdown(s, ["del_auc_norm", "ins_auc_norm", "pg_detbox", "ebpg_detbox"]), ""]

    # TP = confident venomous detections: the cleanest faithfulness set (FP scores sit just above the blank-image
    # score, so their curves are compressed by the same "harmless by default" effect as harmless decisions)
    for label, sub, suffix in (("venomous decisions", ven, ""), ("TP only", df[df.kind == "TP"], "_TP")):
        rnd = sub[sub.method == "random"].set_index("key")
        wins = []
        for method, g in sub[sub.method != "random"].groupby("method"):
            g = g.set_index("key").join(rnd[["del_auc_norm", "ins_auc_norm"]], rsuffix="_rnd", how="inner")
            wins.append({"method": method, "n": len(g),
                         "del_win_rate": float((g.del_auc_norm < g.del_auc_norm_rnd).mean()),
                         "ins_win_rate": float((g.ins_auc_norm > g.ins_auc_norm_rnd).mean()),
                         "del_gain_vs_random": float((g.del_auc_norm_rnd - g.del_auc_norm).mean()),
                         "ins_gain_vs_random": float((g.ins_auc_norm - g.ins_auc_norm_rnd).mean())})
        wins = pd.DataFrame(wins)
        wins["order"] = wins.method.map({m: i for i, m in enumerate(ORDER)}).fillna(99)
        wins = wins.sort_values("order").drop(columns="order")
        wins.to_csv(out / f"vs_random{suffix}.csv", index=False)
        md += [f"## Against the random baseline (same image, {label})", wins.to_markdown(index=False, floatfmt=".3f"), ""]
        tests = []
        for x, y in PAIRS:
            for m in ["del_auc_norm", "ins_auc_norm"]:
                j = pd.concat([sub[sub.method == x].set_index("key")[m], sub[sub.method == y].set_index("key")[m]],
                              axis=1, keys=["x", "y"]).dropna()
                if len(j) >= 10 and (j.x != j.y).any():
                    tests.append({"a": x, "b": y, "metric": m, "n": len(j), "mean_a": j.x.mean(), "mean_b": j.y.mean(),
                                  "p_value": wilcoxon(j.x, j.y).pvalue})
        if tests:
            t = pd.DataFrame(tests); t.to_csv(out / f"wilcoxon{suffix}.csv", index=False)
            md += [f"## Paired Wilcoxon tests ({label})", t.to_markdown(index=False, floatfmt=".4g"), ""]

    if (run / "agreement.csv").exists():
        ag = pd.read_csv(run / "agreement.csv")
        agm = ag.groupby(["a", "b"])[["spearman", "topk_iou"]].mean().reset_index()
        agm.to_csv(out / "agreement.csv", index=False)
        methods = [m for m in ORDER if m in set(ag.a) | set(ag.b)]
        mat = pd.DataFrame(np.eye(len(methods)), index=methods, columns=methods)
        for _, r in agm.iterrows():
            mat.loc[r.a, r.b] = mat.loc[r.b, r.a] = r.spearman
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(mat.values, cmap="viridis", vmin=-0.2, vmax=1)
        ax.set_xticks(range(len(methods)), methods, rotation=60, ha="right"); ax.set_yticks(range(len(methods)), methods)
        for i in range(len(methods)):
            for k in range(len(methods)):
                ax.text(k, i, f"{mat.values[i, k]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if mat.values[i, k] < 0.5 else "black")
        fig.colorbar(im, label="mean Spearman"); ax.set_title("Classifier method agreement")
        fig.tight_layout(); fig.savefig(out / "agreement.png", dpi=150); plt.close(fig)
        md += ["## Agreement (mean Spearman of the maps)", mat.round(2).to_markdown(), ""]

    if (run / "curves.npz").exists():
        cv = dict(np.load(run / "curves.npz"))
        for kind in KINDS:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5)); plotted = False
            for ax, k in zip(axes, ["deletion", "insertion"]):
                for mth in ORDER:
                    key = f"{mth}|{kind}|{k}"
                    if key in cv:
                        y = cv[key] / cv[key + "|n"]
                        ax.plot(np.linspace(0, 1, len(y)), y, label=mth, ls="--" if mth == "random" else "-")
                        plotted = True
                ax.set_title(f"{k} ({kind})"); ax.set_xlabel("fraction of pixels"); ax.set_ylabel("score / original")
                ax.grid(alpha=.3)
            if plotted:
                axes[1].legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(out / f"curves_{kind}.png", dpi=150)
            plt.close(fig)
    for m in ["del_auc_norm", "ins_auc_norm", "ebpg_detbox", "runtime_ms"]:
        if m not in s_all:
            continue
        s = s_all.dropna(subset=[m])
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.bar(s.method, s[m], yerr=np.vstack([s[m] - s[m + "_lo"], s[m + "_hi"] - s[m]]), capsize=3, color="#888")
        ax.set_title(m); ax.tick_params(axis="x", rotation=45)
        if m == "runtime_ms":
            ax.set_yscale("log")
        fig.tight_layout(); fig.savefig(out / f"bars_{m}.png", dpi=150); plt.close(fig)
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
