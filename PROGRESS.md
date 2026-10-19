# PROGRESS.md — living tracker

**Every agent session must update this file before ending** (see CLAUDE.md §6).
Statuses: `TODO` · `IN PROGRESS (A|B)` · `BLOCKED (reason)` · `DONE (date)`.

## Current state (overwrite this block each session)
- **Workstream A (Member A) is COMPLETE** (2026-10-14). All code, experiments, figures and report
  drafts are done. 55 tests green.
- **Last updated:** 2026-10-14 by Member A's agent
- **For Member B's agent: start here.**
  - Your tasks: B1–B6 below. A's interfaces you can use: `pipeline.VenomClassifier` (implement
    `predict(crop_rgb) -> P(venomous)`), `xai.base.Explainer` (implement `explain`), and
    `eval/metrics.py` (deletion/insertion, pointing game etc. work for any map + score function).
  - For the app (B5): use `pipeline.SnakeGuard(detector, classifier=..., explainer=build_explainers(det, names=["layercam"])["layercam"])`.
    `process(frame, explain=True)` returns boxes, threat score, beep interval and a LayerCAM heatmap.
    **Use LayerCAM as the default explainer** (see the results). Offer occlusion as a slow "audit" option.
  - GPU rule: one GPU job at a time, nothing else on the GPU (see the CLAUDE.md gotchas).
- **Blockers:** none.
- **Review 2 deliverables** (objectives, dataset selection & understanding, methodology; no results, as
  requested): `report/SnakeGuard_Review2.docx` (22 pages, incl. 30 anticipated questions with answers)
  and the slides: **`report/SnakeGuard_Review2_15slides.pptx` (final, 15 slides)**, built by `report/build/make_pptx.js`. The older long deck `report/SnakeGuard_Review2.pptx` (25 slides, edited by hand) is kept; its generator is `make_pptx_full27.js`. Speaker script: `report/SnakeGuard_Review2_Script.docx` (15 slides, ~15 min), text in `report/build/script_slides15.js`, built by `make_script.js`. Generators: `report/build/make_docx.js`
  and `make_pptx.js` (run with `NODE_PATH=report/build/node_modules`). Supporting analysis:
  `scripts/dataset_stats.py`, `scripts/dataset_quality.py`.
- **Dataset provenance** (asked for by the supervisor): `report/dataset_sources.md`. It contains the
  composition reconstructed from the images (belts, fallen trees, fire-hose cabinets, pipes,
  mangrove roots, underwater pipes / nooses, iNaturalist-style snakes + one 1080×1080 Roboflow snake
  set) and the closest public source links. **TODO (dataset creator):** confirm the exact sources
  from his Roboflow workspace and Kaggle download history, and update that file.

## Members & environments
| Member | Machine / GPU | Python | Notes |
|---|---|---|---|
| A | Win 11, RTX 4060 Laptop 8 GB | 3.12.10 | `.venv` with torch 2.14.1+cu126, ultralytics 8.4.177 (see requirements.txt) |
| B | _tbd_ | _tbd_ | |

## Task board
| ID | Task | Owner | Status | Notes / output |
|---|---|---|---|---|
| A0 | Env + package + tests | A | DONE (2026-10-13) | 55 tests green (`pytest tests -q`) |
| A1 | Baseline detector eval + failure gallery | A | DONE (2026-10-13) | `results/detector_eval/` |
| A2 | YOLO wrapper + Explainer interface | A | DONE (2026-10-13) | matches ultralytics predictions exactly (tested) |
| A3 | Explainers | A | DONE (2026-10-13) | eigencam(+p5, centered, fsum), gradcam, gradcampp(+p3/p4/p5, fmax/fsum, contrastive), layercam(+fsum, p5), occlusion, drise, baselines random/boxprior |
| A4 | Eval harness | A | DONE (2026-10-13) | `eval/metrics.py`, `scripts/run_xai_bench.py`, `scripts/analyze_bench.py` |
| A5 | Benchmark + analysis + figures | A | DONE (2026-10-14) | `results/xai_bench/{valid_dev,test_final}/analysis/`, `results/figures/test/` |
| A6 | Pipeline + XAI verification | A | DONE (2026-10-14) | `pipeline.py` (tested); `results/xai_verification/summary.md` |
| A7 | Report material (A) | A | DONE (2026-10-14) | `report/A_methods.md`, `report/A_results.md` |
| B1–B6 | Venom classifier, its XAI, app, report | B | TODO | Interfaces ready: `pipeline.VenomClassifier`, `xai.base.Explainer`, `eval/metrics.py` |
| J1, J2 | Joint discussion, final report | A+B | TODO | |

## Key results
| Item | Value | Source | Date |
|---|---|---|---|
| Detector **test** mAP50 / mAP50-95 | **0.865 / 0.617** (P 0.834, R 0.791) | `results/detector_eval/summary.json` | 2026-10-13 |
| test AP50 snake / look-alike | 0.925 / 0.805 | same | |
| valid mAP50 / mAP50-95 | 0.888 / 0.617 | same | |
| test @conf 0.25: TP / FP / FN / class confusions | 756 / 163 / 193 / **0** (valid: 11 confusions) | `matches_test.csv` | |
| snake recall / precision @0.25 (test) | 0.859 / 0.914 | same | |
| look-alike recall / precision @0.25 (test) | 0.703 / 0.695 | same | |
| Detector latency (RTX 4060, our wrapper, incl. NMS) | ~42 ms/img (ultralytics: 4.4 ms inference) | same | |
| **XAI test: deletion ↓ / insertion ↑** | occlusion 0.077/0.846 · **LayerCAM 0.082/0.765 (27 ms)** · Grad-CAM++ 0.152/0.577 · Eigen-CAM 0.244/0.438 · random 0.204/0.295 | `results/xai_bench/test_final/analysis/summary.md` | 2026-10-14 |
| Multi-scale vs P5-only Grad-CAM++ (deletion) | 0.152 vs 0.364, p = 2e-103; P5-only empty for 53 % of detections | same | 2026-10-14 |
| Wrong-detection AUROC: conf vs conf+XAI | 0.900 → 0.906 (+0.006, 95 % CI [0.002, 0.010]) | `results/xai_verification/summary.md` | 2026-10-14 |
| Lit-review comparison (mAP50) | Ref 4: 90.2 %, Ref 5: 94.5 %, Ref 8: 93.4 % (different datasets, snake-only) | proposal | |

## Findings so far (for the report)
1. **Errors are mostly misses and box granularity, not snake↔look-alike confusion.** 0 confusions on
   test, 11 on valid. 102/163 test FPs overlap a GT object (IoU>0.3); e.g. one box around a cluster of
   mangrove roots that were annotated individually. Missed objects are small (median GT area 10.7 %
   of the image vs. 36.6 % for TPs). Gallery: `results/detector_eval/gallery_test/`.
2. **Padding artifact:** square letterboxing → coarse CAMs put saliency on the gray border. Fixed by
   using rect letterbox (same as ultralytics predict).
3. **P5-only Grad-CAM++ is often empty** (also in pytorch-grad-cam), which supports the proposal's
   multi-scale fusion.
4. **Border artifact (qualitative):** Eigen-CAM and channel-weighted Grad-CAM(++) highlight image
   borders (conv zero-padding). Occlusion / D-RISE focus on the object. LayerCAM (per-location
   gradient weighting, built for multi-layer fusion) was added as a candidate fix.
5. Eigen-CAM is head/class-agnostic → sanity_head similarity = 1.0 by construction.
6. **Dev benchmark (valid, 100 imgs, 110 detections)**, `results/xai_bench/valid_dev/analysis/summary.md`:
   - Occlusion is the most faithful (deletion 0.088, insertion 0.834) but costs ~1.4 s per explanation.
   - **LayerCAM (multi-scale) almost matches occlusion** (0.097 / 0.802) at ~35 ms (40x faster). It has
     the best localization (pointing game 0.97, 86 % of saliency inside the box) and beats the box-prior
     baseline on deletion (0.097 vs 0.192), so it isn't just highlighting the box.
   - Grad-CAM++ (multi-scale) is only about as good as random on deletion (win rate 0.50), but clearly
     better on insertion. **Eigen-CAM is worse than random on deletion** (win rate 0.32).
   - Multi-scale Grad-CAM++ beats P5-only significantly (deletion 0.184 vs 0.291, p = 1e-5), which
     supports the proposal's multi-scale claim.
   - Eigen-CAM and Grad-CAM++ agree with each other (Spearman 0.51) but **not with occlusion (≈ 0)**.
     So the proposal's "activation vs gradient agreement" check would give false confidence. The
     quantitative test is what settles it.

## Session log (newest first)
### 2026-10-14 — Member A's agent
- Overnight chain (`scripts/run_rest_offline.sh`) finished with no crashes: final test benchmark (868
  detections), analysis, figures, XAI verification.
- Test confirms dev: occlusion is the most faithful; LayerCAM is nearly as faithful at 45x the speed and
  is the recommended default; Eigen-CAM and Grad-CAM++ fail the head-randomization sanity check
  (1.00 / 0.86) and are weak on deletion; multi-scale >> P5-only. Two agreement clusters:
  {Eigen-CAM, Grad-CAM++} vs {LayerCAM, occlusion, D-RISE}.
- Wrote `report/A_results.md` (all tables + interpretation + recommendation + limitations).
- Fixed earlier: CSV schema drift in the bench (fixed schema + end-to-end test), cuSOLVER SVD failures
  (→ eigh on CPU), GPU contention with a game (→ one job at a time + `scripts/retry.sh`).
### 2026-10-13 (b) — Member A's agent
- Read the proposal PDF. Aligned PLAN.md §2 with it (Eigen-CAM, multi-scale Grad-CAM++, occlusion,
  deletion/insertion vs random, matched pairs; no SHAP/LIME for the detector).
- User decisions: no git repo, no deadline, the classifier is B's business. Removed from A's plan.
- Built the env + whole A codebase (see CLAUDE.md §4). Found and fixed: Eigen-CAM sign bug
  (regression test), square-padding artifact (→ rect letterbox), dataset yaml path/split-name issue,
  gallery label bug. Verified the CAMs against pytorch-grad-cam.
- Ran A1 (numbers above). Started the dev bench (`results/xai_bench/valid_dev`) and XAI verification.
### 2026-10-13 (a) — Member A's agent
- Inspected the repo and wrote the initial CLAUDE.md / PLAN.md / PROGRESS.md.
