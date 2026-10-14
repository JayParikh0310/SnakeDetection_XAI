"""A5: aggregate a benchmark run into report tables + plots.

Outputs in <run>/analysis/:
  summary.csv / summary.md   mean ± 95% bootstrap CI per method (all targets, and per kind/class)
  vs_random.csv              per-target win rate of each method over the random baseline (deletion & insertion)
  wilcoxon.csv               paired Wilcoxon tests between core methods + ablations
  agreement.csv              mean pairwise Spearman / top-k IoU between methods
  curves_*.png               mean deletion / insertion curves
  bars_*.png                 metric bar charts

Usage: python scripts/analyze_bench.py results/xai_bench/test_final [more runs on the same images ...] [--out DIR]
Several runs are merged (e.g. a later run that adds methods). For a method present in more than one
run, the first run wins.
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

METRICS = ["del_auc_norm", "ins_auc_norm", "del_auc", "ins_auc", "pg_box", "ebpg_box", "iou_box",
           "pg_poly", "ebpg_poly", "iou_poly", "empty_map", "stability", "sanity_head", "sanity_neck", "sanity_all", "runtime_ms"]
HIGHER_BETTER = {"del_auc_norm": False, "del_auc": False, "runtime_ms": False, "empty_map": False,
                 "sanity_head": False, "sanity_neck": False, "sanity_all": False}
ORDER = ["eigencam", "eigencam_p5", "eigencam_centered", "eigencam_fsum", "gradcam", "gradcampp", "gradcampp_fmax", "gradcampp_fsum", "gradcampp_p5", "gradcampp_p4", "gradcampp_p3",
         "gradcampp_contrast", "layercam", "layercam_fsum", "layercam_p5", "occlusion", "drise", "boxprior", "random"]
PAIRS = [("occlusion", "gradcampp"), ("occlusion", "eigencam"), ("gradcampp", "eigencam"),
         ("gradcampp", "gradcampp_p5"), ("gradcampp", "gradcampp_p3"), ("eigencam", "eigencam_p5"),
         ("gradcampp", "gradcam"), ("drise", "occlusion")]


def boot_ci(v: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, (n, len(v))).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for method, g in df.groupby("method"):
        r = {"method": method, "n": len(g)}
        for m in METRICS:
            if m in g and g[m].notna().any():
                v = g[m].to_numpy(float)
                lo, hi = boot_ci(v)
                r[m] = np.nanmean(v); r[m + "_lo"] = lo; r[m + "_hi"] = hi
        out.append(r)
    s = pd.DataFrame(out)
    s["order"] = s.method.map({m: i for i, m in enumerate(ORDER)}).fillna(99)
    return s.sort_values("order").drop(columns="order")


def to_markdown(s: pd.DataFrame, cols: list[str]) -> str:
    cols = [c for c in cols if c in s]
    best = {}
    for c in cols:
        cand = s[~s.method.isin(["random", "boxprior"])][c]
        if cand.notna().any():
            best[c] = cand.max() if HIGHER_BETTER.get(c, True) else cand.min()
    lines = ["| method | n | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for _, r in s.iterrows():
        cells = []
        for c in cols:
            v = r.get(c)
            if pd.isna(v):
                cells.append("–"); continue
            txt = f"{v:.1f}" if c == "runtime_ms" else f"{v:.3f}"
            if c + "_lo" in r and not pd.isna(r[c + "_lo"]) and c != "runtime_ms":
                txt += f" ±{(r[c + '_hi'] - r[c + '_lo']) / 2:.3f}"
            if c in best and np.isclose(v, best[c]):
                txt = f"**{txt}**"
            cells.append(txt)
        lines.append(f"| {r.method} | {int(r.n)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    runs = [Path(r) for r in a.runs]
    run = runs[0]
    out = Path(a.out) if a.out else run / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    parts, seen = [], set()
    for r in runs:
        d = pd.read_csv(r / "metrics.csv")
        d = d[d.method.notna() & ~d.method.isin(seen)]
        seen |= set(d.method)
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["key"] = df.image_id.astype(str) + "#" + df.target.astype(int).astype(str)

    main_cols = ["del_auc_norm", "ins_auc_norm", "pg_box", "ebpg_box", "ebpg_poly", "iou_poly",
                 "stability", "sanity_head", "sanity_all", "runtime_ms"]
    md = [f"# XAI benchmark: {run.name}", "",
          f"Targets: {df.key.nunique()} detections on {df.image_id.nunique()} images. "
          f"Kinds: {df.drop_duplicates('key').kind.value_counts().to_dict()}", "",
          "Lower is better for del_auc_norm, sanity_* and runtime. Higher is better for the rest. "
          "Bold marks the best non-baseline method. ± is half the 95% bootstrap CI.", ""]
    s_all = summarize(df)
    s_all.to_csv(out / "summary.csv", index=False)
    md += ["## All targets", to_markdown(s_all, main_cols), ""]
    for kind in ["TP", "confusion", "FP"]:
        g = df[df.kind == kind]
        if g.key.nunique() >= 5:
            s = summarize(g); s.to_csv(out / f"summary_{kind}.csv", index=False)
            md += [f"## {kind} only ({g.key.nunique()} targets)",
                   to_markdown(s, ["del_auc_norm", "ins_auc_norm", "pg_box", "ebpg_box", "ebpg_poly"]), ""]
    for c, name in {0: "snake", 1: "snake_like_object"}.items():
        g = df[(df.kind == "TP") & (df.det_cls == c)]
        if g.key.nunique() >= 5:
            s = summarize(g); s.to_csv(out / f"summary_TP_{name}.csv", index=False)
            md += [f"## TP {name} ({g.key.nunique()} targets)",
                   to_markdown(s, ["del_auc_norm", "ins_auc_norm", "pg_box", "ebpg_box", "ebpg_poly"]), ""]

    # win rate vs. random on the same target
    rnd = df[df.method == "random"].set_index("key")
    wins = []
    for method, g in df[df.method != "random"].groupby("method"):
        g = g.set_index("key").join(rnd[["del_auc_norm", "ins_auc_norm"]], rsuffix="_rnd", how="inner")
        wins.append({"method": method, "n": len(g),
                     "del_win_rate": float((g.del_auc_norm < g.del_auc_norm_rnd).mean()),
                     "ins_win_rate": float((g.ins_auc_norm > g.ins_auc_norm_rnd).mean()),
                     "del_gain_vs_random": float((g.del_auc_norm_rnd - g.del_auc_norm).mean()),
                     "ins_gain_vs_random": float((g.ins_auc_norm - g.ins_auc_norm_rnd).mean())})
    wins = pd.DataFrame(wins)
    if len(wins):
        wins["order"] = wins.method.map({m: i for i, m in enumerate(ORDER)}).fillna(99)
        wins = wins.sort_values("order").drop(columns="order")
        wins.to_csv(out / "vs_random.csv", index=False)
        md += ["## Against the random-order baseline (same target)", wins.to_markdown(index=False, floatfmt=".3f"), ""]

    # paired tests
    tests = []
    for x, y in PAIRS:
        for m in ["del_auc_norm", "ins_auc_norm", "ebpg_box"]:
            gx = df[df.method == x].set_index("key")[m]
            gy = df[df.method == y].set_index("key")[m]
            j = pd.concat([gx, gy], axis=1, keys=["x", "y"]).dropna()
            if len(j) >= 10 and (j.x != j.y).any():
                st = wilcoxon(j.x, j.y)
                tests.append({"a": x, "b": y, "metric": m, "n": len(j), "mean_a": j.x.mean(),
                              "mean_b": j.y.mean(), "p_value": st.pvalue})
    if tests:
        t = pd.DataFrame(tests); t.to_csv(out / "wilcoxon.csv", index=False)
        md += ["## Paired Wilcoxon tests", t.to_markdown(index=False, floatfmt=".4g"), ""]

    # agreement
    ag_files = [r / "agreement.csv" for r in runs if (r / "agreement.csv").exists()]
    if ag_files:
        ag = pd.concat([pd.read_csv(f) for f in ag_files]).drop_duplicates(["image_id", "target", "a", "b"])
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
        fig.colorbar(im, label="mean Spearman"); ax.set_title("Method agreement")
        fig.tight_layout(); fig.savefig(out / "agreement.png", dpi=150); plt.close(fig)
        core = agm[agm.a.isin(["eigencam", "gradcampp", "occlusion"]) & agm.b.isin(["eigencam", "gradcampp", "occlusion"])]
        md += ["## Agreement between core methods", core.to_markdown(index=False, floatfmt=".3f"), ""]

    # curves
    cv = {}
    for r in reversed(runs):  # earlier runs win
        if (r / "curves.npz").exists():
            cv.update(dict(np.load(r / "curves.npz")))
    if cv:
        for kind in ["TP", "confusion", "FP"]:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            plotted = False
            for ax, k in zip(axes, ["deletion", "insertion"]):
                for mth in ORDER:
                    key = f"{mth}|{kind}|{k}"
                    if key in cv:
                        y = cv[key] / cv[key + "|n"]
                        ax.plot(np.linspace(0, 1, len(y)), y, label=mth,
                                lw=2.5 if mth in ("eigencam", "gradcampp", "occlusion") else 1.2,
                                ls="--" if mth in ("random", "boxprior") else "-")
                        plotted = True
                ax.set_title(f"{k} ({kind})"); ax.set_xlabel("fraction of pixels"); ax.set_ylabel("score / original")
                ax.grid(alpha=.3)
            if plotted:
                axes[1].legend(fontsize=7, ncol=2)
                fig.tight_layout(); fig.savefig(out / f"curves_{kind}.png", dpi=150)
            plt.close(fig)

    for m in ["del_auc_norm", "ins_auc_norm", "ebpg_box", "runtime_ms"]:
        if m not in s_all:
            continue
        fig, ax = plt.subplots(figsize=(8, 3.5))
        s = s_all.dropna(subset=[m])
        err = np.vstack([s[m] - s[m + "_lo"], s[m + "_hi"] - s[m]]) if m + "_lo" in s else None
        ax.bar(s.method, s[m], yerr=err, capsize=3,
               color=["#c44" if x in ("eigencam", "gradcampp", "occlusion") else "#888" for x in s.method])
        ax.set_title(m); ax.tick_params(axis="x", rotation=45)
        if m == "runtime_ms":
            ax.set_yscale("log")
        fig.tight_layout(); fig.savefig(out / f"bars_{m}.png", dpi=150); plt.close(fig)

    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
