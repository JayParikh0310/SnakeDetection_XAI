# CLAUDE.md — SnakeGuard XAI

Instructions for every Claude Code agent (and human) working on this project. Two people work
on it, each with their own agent. **Read this file, then `PLAN.md`, then `PROGRESS.md` before
doing anything.**

## 1. Project in one paragraph

SnakeGuard is an ankle-worn camera early-warning system (proposal:
`SnakeGuard_Literature_Review_and_Proposal_revised.pdf`). A frozen YOLO11n detector separates real
snakes from look-alikes (ropes, roots, hoses, sticks). An **Explainability & Validation Layer**
(Eigen-CAM, multi-scale Grad-CAM++, occlusion + extras) explains each detection. The methods are
**compared quantitatively** (deletion/insertion AUC vs. a random baseline, localization, agreement,
sanity, stability, runtime), with matched snake vs. look-alike case studies. A pipeline wraps this:
temporal confirmation, proximity, threat score 0–100, beep rate, and a pluggable venom classifier.
Member A owns the detector/XAI/eval/pipeline. Member B owns the venom classifier + its XAI + the
demo app.

## 2. Ground truth about the assets (verified 2026-10-13)

| Asset | Path | Facts |
|---|---|---|
| Detector weights | `best.pt` | Ultralytics **YOLO11n** `DetectionModel`, trained 2026-06-27 on Colab, 30 epochs, imgsz 640. **Frozen, never modify.** Neck outputs feeding Detect = layers 16/19/22 (P3/P4/P5). Layers 0-10 = backbone, 11-22 = neck, 23 = head. |
| Dataset | `final_dataset_v4/final_dataset_v4/` | YOLO format, `0: snake`, `1: snake_like_object`. train 20,294 / valid 2,172 / test 841 images. Mixed labels: boxes (5 fields) and **polygons** (used as fine GT masks). 1,813 empty train images. Filenames `snake_XXXXXX` / `snake_like_XXXXXX`, no species info. **Never modify.** |
| `data.yaml` | same folder | Relative paths. Ultralytics resolves them against its own datasets dir, so `scripts/eval_detector.py` writes `results/detector_eval/data_resolved.yaml` with an absolute `path`. Ultralytics calls the split `val`, not `valid`. |
| Proposal / lit review | the PDF | 4 pages + matrix. XAI commitments are listed in `PLAN.md` §2. **SHAP/LIME explicitly set aside for the detector.** |

## 3. Environment

- Machine A: Windows 11, RTX 4060 Laptop 8 GB, Python 3.12. Venv at `.venv/`.
- Setup: see the header of `requirements.txt` (CUDA torch first, then `pip install -r requirements.txt`, then `pip install -e .`).
- Run everything with `.venv/Scripts/python` (Windows) from the project root.
- On Windows, bash loops over the 23k dataset files time out. Use Python for file ops.
- Don't `cd` into `results/...` folders in the shell. Windows then locks them and `rm` fails.
- Python output redirected to a log is buffered. Progress prints use `flush=True`.

## 4. Layout (actual)

```
configs/default.yaml           all paths + hyperparameters (detector, XAI, eval)
src/snakeguard/
  __init__.py                  load_config(), CLASS_NAMES, PROJECT_ROOT
  data.py                      dataset reader (boxes + polygons)
  detection/yolo.py            YoloDetector: letterbox (rect, = ultralytics), raw preds, NMS with
                               anchor ids, differentiable target_score (D-RISE similarity)
  xai/base.py                  Explainer / DetectorExplainer contract
  xai/cam.py                   EigenCAM, GradCAM, GradCAMPlusPlus, LayerCAM (multi-scale P3/P4/P5, fusion)
  xai/perturbation.py          Occlusion, DRISE, RandomMap (baseline), BoxPrior (baseline)
  xai/__init__.py              build_explainers(): the registry of method names
  xai/viz.py                   overlay / draw_box
  eval/metrics.py              deletion/insertion, pointing game, EBPG, IoU, agreement, randomization, noise
  eval/matching.py             detection <-> GT matching (TP / confusion / FP / FN)
  pipeline.py                  SnakeGuard runtime pipeline + VenomClassifier interface (for B)
scripts/
  eval_detector.py             A1 baseline eval -> results/detector_eval/
  run_xai_bench.py             A4/A5 benchmark (resumable) -> results/xai_bench/<split>_<tag>/
  analyze_bench.py             tables/plots for a bench run -> <run>/analysis/
  make_figures.py              qualitative grids + matched snake/look-alike pairs -> results/figures/
  eval_xai_verification.py     can explanations flag wrong detections? -> results/xai_verification/
  retry.sh                     restart a resumable GPU script after crashes
  run_rest_offline.sh          unattended chain: test bench -> analysis -> figures -> verification
report/                        A_methods.md, A_results.md (Member A report drafts)
tests/                         pytest. test_model.py needs best.pt + dataset (marked gpu)
results/                       all outputs (large, don't share blindly)
```

## 5. Shared contracts (don't change without updating both members' code + PROGRESS.md)

**Explainer** (`src/snakeguard/xai/base.py`):
- `explain(image_rgb_uint8_HxWx3, target) -> HxW float32 in [0,1]` (original resolution).
- Detector explainers also have `explain_lb(x, lb, det) -> map in letterboxed input space`, which
  the metrics use. Subclasses implement `_explain_lb(x, det)` and return a raw map.
- Detector target = `Detection` (box, box_lb, cls, score, anchor). Classifier target (B) = class index.

**Target score** for every detector method and metric: `YoloDetector.target_score` =
max over raw predictions of IoU(box, target box) × p(target class). `mode="contrastive"` uses
p(cls) − p(other class).

**Venom classifier** (B → pipeline): subclass `pipeline.VenomClassifier` and implement
`predict(crop_rgb) -> P(venomous) or None`.

**Benchmark CSV** (`metrics.csv`): one row per (image, target, method). Columns: kind, det_cls,
gt_cls, score, runtime_ms, empty_map, del_auc(_norm), ins_auc(_norm), pg/ebpg/iou _box/_poly,
stability, sanity_head/neck/all.

## 6. Working rules for agents

1. **Start of session:** read CLAUDE.md → PLAN.md → PROGRESS.md. Work out which member you work
   for (ask if unclear). Only take that member's tasks, and mark them IN PROGRESS.
2. **End of session (mandatory):** update PROGRESS.md (status, dated session log with numbers,
   files touched, blockers, next step). The next agent has no memory of your chat.
3. **Test every step:** `.venv/Scripts/python -m pytest tests -q` must stay green. Add tests for new
   code. When something looks wrong, check it against a reference (e.g. pytorch-grad-cam) before
   calling it a finding.
4. Never modify `best.pt` or the raw dataset. Outputs go to `results/`.
5. Seeds are fixed (config `seed: 42`). Each bench run writes `run_config.json`.
6. Tune on **valid**, report final numbers on **test**.
7. Every reported number must come from a script + config in this project.
8. Scope decisions go in the PLAN.md §8 decision log.
9. Safety-adjacent domain: unknown venom status is treated as venomous (pipeline default), and the
   app must say it isn't a substitute for expert or medical advice.
10. Git: the members commit and push themselves. **Agents never commit or push.** `.gitignore`
    excludes `.venv/`, the dataset, caches, logs and the large regenerable results. The dataset is
    shared separately.

## 7. Commands

```
.venv/Scripts/python -m pytest tests -q                                   # 55 tests, ~1 min
.venv/Scripts/python scripts/eval_detector.py                             # A1 (~10 min)
.venv/Scripts/python scripts/run_xai_bench.py --split valid --n 100 --tag dev --stability --sanity
.venv/Scripts/python scripts/run_xai_bench.py --split test --tag final --stability --sanity
.venv/Scripts/python scripts/analyze_bench.py results/xai_bench/test_final
.venv/Scripts/python scripts/make_figures.py --split test
.venv/Scripts/python scripts/eval_xai_verification.py
```

## 8. Gotchas found so far

- Square letterboxing adds big gray borders that coarse CAMs latch onto (P5 Grad-CAM++ put all
  its saliency in the padding). The wrapper therefore uses **rect letterbox** (pad to a multiple of
  32), like ultralytics predict/val.
- **P5-only Grad-CAM++ is often an empty map** on this model. pytorch-grad-cam gives the same
  result, so it's a property of the method, not a bug. This is evidence for the proposal's
  multi-scale design.
- pytorch-grad-cam's Grad-CAM++ adds eps=1e-6 in the alpha denominator. That dominates with this
  model's tiny gradients (~1e-4), so our paper-exact version differs from the library (r≈0.8–0.95 on P3).
- Eigen-CAM sign: the SVD sign is arbitrary. We orient it by correlation with the activation energy
  (an earlier "mean activation" rule produced empty maps; there's a regression test).
- Eigen-CAM is class- and head-agnostic, so its sanity_head similarity is 1.0 by construction.
- **GPU contention → CUDA faults.** With another GPU-heavy app running (e.g. a game), or two of our
  GPU jobs at once, Windows' driver watchdog kills kernels: `CUDA error: unspecified launch failure`
  or cuSOLVER errors, at random places. Run **one GPU job at a time**, with nothing else on the GPU,
  through `bash scripts/retry.sh <log> <tries> .venv/Scripts/python scripts/<resumable script> ...`.
  Check `nvidia-smi` first.
- Eigen-CAM uses eigh on CPU (a^T a), not CUDA SVD, because of the cuSOLVER failures above.
