"""Collect every XAI comparison figure into one numbered folder for presenting: results/XAI_COMPARISON/.

Copies the existing plots/grids (in presenting order) and renders the two summary tables (detector, venom classifier)
as PNG images from the benchmark CSVs. Run after analyze_bench.py / analyze_venom_bench.py.
Usage:  .venv/Scripts/python scripts/make_comparison_folder.py
"""
from __future__ import annotations

import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from snakeguard import PROJECT_ROOT

R = PROJECT_ROOT / "results"
OUT = R / "XAI_COMPARISON"
DET = R / "xai_bench" / "test_final" / "analysis"
VEN = R / "venom_xai_bench" / "test_final" / "analysis"
FIG = R / "figures" / "test"
VFIG = R / "venom_clf" / "b0_reg" / "figures"

DET_NAMES = {"layercam": "LayerCAM (P3-P5)", "occlusion": "Occlusion", "drise": "D-RISE",
             "gradcampp_p4": "Grad-CAM++ (P4 only)", "gradcampp": "Grad-CAM++ (P3-P5)", "gradcampp_p3": "Grad-CAM++ (P3 only)",
             "gradcam": "Grad-CAM", "eigencam_p5": "Eigen-CAM (P5 only)", "eigencam": "Eigen-CAM (P3-P5)",
             "layercam_p5": "LayerCAM (P5 only)", "gradcampp_p5": "Grad-CAM++ (P5 only)",
             "boxprior": "Box prior (baseline)", "random": "Random map (baseline)"}
VEN_NAMES = {"gradcampp_ms": "Grad-CAM++ (3 scales)", "layercam_ms": "LayerCAM (3 scales)", "gradcam": "Grad-CAM",
             "gradcampp": "Grad-CAM++ (last stage)", "occlusion": "Occlusion", "rise": "RISE", "lime": "LIME (8x8)",
             "kernelshap": "KernelSHAP (8x8)", "integrated_gradients": "Integrated Gradients", "eigencam": "Eigen-CAM",
             "random": "Random map (baseline)"}

# (source, new name); numbered in presenting order
COPIES = [
    (DET / "bars_del_auc_norm.png", "detector_deletion_AUC_lower_is_better.png"),
    (DET / "bars_ins_auc_norm.png", "detector_insertion_AUC_higher_is_better.png"),
    (DET / "curves_TP.png", "detector_deletion_insertion_curves.png"),
    (DET / "bars_ebpg_box.png", "detector_localization_energy_in_box.png"),
    (DET / "bars_runtime_ms.png", "detector_runtime_per_explanation.png"),
    (DET / "agreement.png", "detector_method_agreement.png"),
    (FIG / "pairs_00.png", "detector_heatmaps_snake_vs_lookalike_1.png"),
    (FIG / "pairs_01.png", "detector_heatmaps_snake_vs_lookalike_2.png"),
    (FIG / "pairs_02.png", "detector_heatmaps_snake_vs_lookalike_3.png"),
    (FIG / "pairs_03.png", "detector_heatmaps_snake_vs_lookalike_4.png"),
    (FIG / "grid_TP_snake.png", "detector_heatmaps_correct_snakes.png"),
    (FIG / "grid_TP_snake_like_object.png", "detector_heatmaps_correct_lookalikes.png"),
    (FIG / "grid_FP_snake.png", "detector_heatmaps_wrong_snake_detections.png"),
    (FIG / "grid_FP_snake_like_object.png", "detector_heatmaps_wrong_lookalike_detections.png"),
    (VEN / "bars_del_auc_norm.png", "venom_deletion_AUC_lower_is_better.png"),
    (VEN / "bars_ins_auc_norm.png", "venom_insertion_AUC_higher_is_better.png"),
    (VEN / "curves_TP.png", "venom_deletion_insertion_curves.png"),
    (VEN / "bars_ebpg_detbox.png", "venom_localization_energy_in_snake_box.png"),
    (VEN / "bars_runtime_ms.png", "venom_runtime_per_explanation.png"),
    (VEN / "agreement.png", "venom_method_agreement.png"),
    (VFIG / "grid_TP.png", "venom_heatmaps_correct_venomous.png"),
    (VFIG / "grid_FN.png", "venom_heatmaps_missed_venomous.png"),
    (VFIG / "grid_FP.png", "venom_heatmaps_false_alarms.png"),
    (VFIG / "grid_TN.png", "venom_heatmaps_correct_harmless.png"),
]


def table_png(df: pd.DataFrame, cols: dict[str, str], names: dict[str, str], title: str, note: str,
              best: str, bad: set[str], path) -> None:
    """Render a summary table; rows sorted by deletion AUC, `best` green, `bad` red, baselines grey."""
    df = df[df.method.isin(names)].sort_values("del_auc_norm")
    rows = []
    for _, r in df.iterrows():
        row = [names[r.method], str(int(r.n))]
        for c in cols:
            v = r.get(c)
            row.append("–" if pd.isna(v) else (f"{v:,.0f} ms" if c == "runtime_ms" else f"{v:.3f}"))
        rows.append(row)
    header = ["Method", "n", *cols.values()]
    fig, ax = plt.subplots(figsize=(13, 0.42 * len(rows) + 1.6), dpi=150)
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(11)
    t.scale(1, 1.55)
    for (i, j), cell in t.get_celld().items():
        cell.set_edgecolor("#c9d6c3")
        if j == 0:
            cell.set_text_props(ha="left")
            cell._loc = "left"
        if i == 0:
            cell.set_facecolor("#2c5f2d")
            cell.set_text_props(color="white", weight="bold")
            continue
        m = df.method.iloc[i - 1]
        if m == best:
            cell.set_facecolor("#d8eed3")
            cell.set_text_props(weight="bold")
        elif m in bad:
            cell.set_facecolor("#f8dcd8")
        elif m in ("random", "boxprior"):
            cell.set_facecolor("#f1f1f1")
            cell.set_text_props(style="italic", color="#555555")
    t.auto_set_column_width(list(range(len(header))))
    ax.set_title(title, fontsize=15, weight="bold", loc="left", pad=12)
    fig.text(0.01, 0.01, note, fontsize=9.5, color="#444444", ha="left", va="bottom", wrap=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    note = ("Green = recommended (real-time default). Red = fails a check. Grey = baselines. "
            "Deletion: lower = more faithful. Insertion, pointing, energy in box: higher = better. "
            "Sanity (head randomized): lower = explanation depends on the model (good).")
    det = pd.read_csv(DET / "summary.csv")
    table_png(det, {"del_auc_norm": "Deletion ↓", "ins_auc_norm": "Insertion ↑", "pg_box": "Pointing game ↑",
                    "ebpg_box": "Energy in box ↑", "sanity_head": "Sanity (head) ↓", "stability": "Stability ↑",
                    "runtime_ms": "Time"}, DET_NAMES,
              "Detector XAI comparison: 868 detections, test set", note,
              best="layercam", bad={"eigencam", "eigencam_p5", "gradcampp_p5"}, path=OUT / "01_detector_comparison_TABLE.png")
    ven = pd.read_csv(VEN / "summary_venomous_decisions.csv")
    table_png(ven, {"del_auc_norm": "Deletion ↓", "ins_auc_norm": "Insertion ↑", "pg_detbox": "Pointing (snake box) ↑",
                    "ebpg_detbox": "Energy in snake box ↑", "sanity_head": "Sanity (head) ↓", "runtime_ms": "Time"},
              VEN_NAMES, "Venom classifier XAI comparison: venomous decisions, test set",
              note + " RISE/LIME/KernelSHAP on a 60-image subset. Insertion > 1: scores normalized by the original probability.",
              best="layercam_ms", bad={"eigencam"}, path=OUT / "16_venom_comparison_TABLE.png")
    n = 2
    for src, name in COPIES:
        if name.startswith("venom_") and n < 17:
            n = 17
        shutil.copy(src, OUT / f"{n:02d}_{name}")
        n += 1
    for md, name in [(DET / "summary.md", "detector_full_numbers.md"), (VEN / "summary.md", "venom_full_numbers.md"),
                     (R / "xai_verification" / "summary.md", "detector_can_xai_flag_wrong_detections.md")]:
        shutil.copy(md, OUT / f"zz_{name}")
    print(f"wrote {len(list(OUT.iterdir()))} files to {OUT}")


if __name__ == "__main__":
    main()
