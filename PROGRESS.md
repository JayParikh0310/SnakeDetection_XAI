# PROGRESS.md — living tracker

**Every agent session must update this file before ending** (see CLAUDE.md §6).
Statuses: `TODO` · `IN PROGRESS (A|B)` · `BLOCKED (reason)` · `DONE (date)`.

## Current state (overwrite this block each session)
- **Workstream A (Member A) is COMPLETE** (2026-10-14). 55 A tests (+3 B tests) green.
- **Workstream B (Member B) started 2026-10-14.** B1 (venom dataset) DONE 2026-10-15: SnakeCLEF 2023 subset in
  `data/snakeclef/` (`data` is a symlink to `~/snakeguard-data`, outside iCloud). **B2 DONE 2026-10-20.** Chosen model:
  `results/venom_clf/b0_reg/best.pt` with score = species head (stored in the checkpoint with its threshold 0.1413).
  Test: AUROC 0.888, venomous recall 0.953, specificity 0.565. Use `TorchVenomClassifier(...)` and
  `PipelineConfig(venom_threshold=clf.threshold)`. **B3 DONE 2026-10-20** (`src/snakeguard/venom/xai.py`, 11 methods,
  qualitative grids in `results/venom_clf/b0_reg/figures/`). **B4 DONE 2026-10-20** (`results/venom_xai_bench/test_final/`,
  write-up `report/B_results.md`). Recommended app explainer for venom alerts: LayerCAM ms (same as A's detector choice).
  **B5 DONE 2026-10-20.** App: `.venv/bin/python scripts/app.py` → http://127.0.0.1:7860 (Photo + Video tabs; logic in
  `src/snakeguard/demo.py`). The app uses `PipelineConfig.venom_crop_margin = 0.25` and `TorchVenomClassifier(rescale=True)`
  (decision line at 0.5, so the threat score reflects venom status). **B6 draft DONE 2026-10-20:** `report/B_report.md`
  (1 item left marked [check]: official MoHFW protocol citation). Next: J1/J2 with Jay; PR #1 review + PROGRESS.md conflict.
  **For Jay:** `pipeline.py` gained an optional `venom_crop_margin` (default 0.0 = your original crop; tested equal) and a
  `venom_crop()` helper. Jay's commit `6555ff7` (Review-2 reports) is on origin and not yet pulled into B's checkout:
  B's uncommitted `PROGRESS.md` also rewrote the "Current state" block → expect a small merge conflict there.
- **Notes for B's agent:**
  - A's interfaces: `pipeline.VenomClassifier` (implement `predict(crop_rgb) -> P(venomous)`),
    `xai.base.Explainer` (implement `explain`), `eval/metrics.py`.
  - App (B5): `pipeline.SnakeGuard(detector, classifier=..., explainer=build_explainers(det, names=["layercam"])["layercam"])`.
    **LayerCAM is the default explainer**, occlusion is the slow "audit" option.
  - B's machine is macOS (Apple M4, no CUDA): run with `.venv/bin/python`, not `.venv/Scripts/python`.
    On B's machine `.venv` is a **symlink to `~/venvs/snakeguard-xai`** (outside iCloud, see CLAUDE.md §8).
  - Load the venom data from `data/snakeclef/manifest.csv` with `available == True` (199 rows have no image).
    `YoloDetector` picks cuda or cpu only (no MPS), so the detector runs on CPU here.
  - Only ~20 GB disk free on B's machine.
- **Blockers:** the 22 `gpu`-marked tests need `final_dataset_v4` (Jay is sending it to B).

## Members & environments
| Member | Machine / GPU | Python | Notes |
|---|---|---|---|
| A | Win 11, RTX 4060 Laptop 8 GB | 3.12.10 | `.venv` with torch 2.14.1+cu126, ultralytics 8.4.177 (see requirements.txt) |
| B | macOS 27, Apple M4, 16 GB (MPS, no CUDA) | 3.12.15 | `~/venvs/snakeguard-xai` (conda-created), linked as `.venv`; torch 2.14.1 (PyPI, MPS available), ultralytics 8.4.177, same pins as requirements.txt. 36 non-gpu tests green |

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
| B1 | Venom dataset (species → venomous, leakage-free splits) | B | DONE (2026-10-15) | SnakeCLEF 2023, 240 px; `scripts/build_venom_dataset.py`, `configs/venom.yaml` → `data/snakeclef/manifest.csv`: 50,441 usable imgs (train 34,838 / valid 7,841 / test 7,762), 1,784 species, 285 venomous |
| B2 | Train/evaluate classifier (venomous recall) | B | DONE (2026-10-20) | Chosen on valid: `results/venom_clf/b0_reg` + species-head score. Test AUROC 0.888, recall 0.953 / specificity 0.565 at thr 0.1413 (`final_metrics.json`). Baseline b0 kept for comparison |
| B3 | Classifier explainers (shared `Explainer`) | B | DONE (2026-10-20) | `venom/xai.py`: eigencam, gradcam, gradcampp (last stage), gradcampp_ms, layercam_ms (stages 3/5/8), integrated_gradients, occlusion, rise, lime, kernelshap, random. `build_classifier_explainers()`. Figures: `scripts/make_venom_figures.py` |
| B4 | Classifier XAI benchmark (A's harness) | B | DONE (2026-10-20) | `results/venom_xai_bench/test_final/analysis/summary.md`; write-up `report/B_results.md` §4–5 |
| B5 | Demo app (Gradio) on A's pipeline | B | DONE (2026-10-20) | `scripts/app.py`, `src/snakeguard/demo.py`, `tests/test_demo.py` (10 tests). Photo tab verified in the browser; Video tab tested by unit test only |
| B6 | Report: intro, lit review, classifier, app, ethics | B | DRAFT DONE (2026-10-20) | `report/B_report.md` (~3,800 words; complements A's Review-2 doc); figure `results/app/app_examples.png`; Big-Four table `results/venom_clf/b0_reg/species_report_test.md`. 6 [check] items open |
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
| **Venom clf (B2, b0) test AUROC** | **0.870** [0.859, 0.881] (species-head score 0.880); AP 0.791 | `results/venom_clf/b0/metrics.json` | 2026-10-20 |
| Venom clf test @ thr 0.0169 (chosen on valid for recall ≥ 0.95) | recall **0.952** [0.942, 0.962], specificity **0.496** [0.478, 0.515], precision 0.516 | same | 2026-10-20 |
| Venom clf test @ 0.5 | recall 0.715, specificity 0.843 | same | 2026-10-20 |
| **Venom clf FINAL (b0_reg, species score) test AUROC** | **0.888** [0.878, 0.898]; AP 0.813 | `results/venom_clf/b0_reg/final_metrics.json` | 2026-10-20 |
| Venom clf FINAL test @ thr 0.1413 (valid, recall ≥ 0.95) | recall **0.953** [0.944, 0.962], specificity **0.565** [0.547, 0.583], precision 0.553 | same | 2026-10-20 |
| Venom model selection (valid): AUROC / spec@rec95 | b0 venom 0.869 / 0.502 · b0 species 0.882 / 0.541 · b0_reg venom 0.875 / 0.518 · **b0_reg species 0.891 / 0.564** | `results/venom_clf/compare_valid.md` | 2026-10-20 |
| **Venom XAI test, TP (80): deletion ↓ / insertion ↑** | Grad-CAM++ ms 0.692/1.000 · **LayerCAM ms 0.712/1.093 (67 ms)** · Grad-CAM 0.758/1.178 · occlusion 0.779/1.071 · Grad-CAM++ 0.800/1.001 · Eigen-CAM 0.855/0.986 · random 0.934/0.920; RISE/LIME/SHAP (n = 20) 0.76–0.82 / 1.19–1.26 at ~8 s | `results/venom_xai_bench/test_final/analysis/summary_TP.csv` | 2026-10-20 |
| Venom XAI sanity (head randomization ↓) | Grad-CAM 0.02 (passes) · LayerCAM ms 0.81 · IG 0.85 · Grad-CAM++ 0.96 · Eigen-CAM 1.00 | `.../analysis/summary.csv` | 2026-10-20 |
| Classifier "no-evidence" output | P(venomous | all-gray image) = 0.143 ≈ threshold 0.141 → deletion/insertion uninformative for harmless decisions | `scripts/run_venom_xai_bench.py` (blank_score column) | 2026-10-20 |
| Venom clf on detector crops (test, 522 imgs with a detected snake) | venomous recall: whole image 0.952 · tight crop 0.917 · +25 % context 0.944 · +50 % 0.948 · +100 % 0.952 | `results/venom_clf/b0_reg/crop_eval_test.md` | 2026-10-20 |
| India's Big Four, test recall at the deployed threshold | cobra 0.96 (n 25) · saw-scaled viper 0.93 (15) · Russell's viper 0.80 (20) · **common krait 0.75 (16)** | `results/venom_clf/b0_reg/species_report_test.md` | 2026-10-20 |
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
### 2026-10-21 — Member B's agent
- B6 committed by the user (ab0ef9c) and pushed to PR #1.
- Closed 5 of 6 [check] items in `report/B_report.md`: installed `poppler` (Homebrew) and read the proposal PDF
  (`pdftotext`); §2.1 now has a table of the proposal's identification systems (Refs 1, 2, 3, 5 from its matrix:
  data, method, results, limitations; VenoScan's inconsistent accuracy noted; Serpent-ID also uses SnakeCLEF 2023).
  Verified Bolon et al. 2020 and Suraweera et al. 2020 (eLife, ~58,000 deaths/yr in India) with DOIs; SnakeCLEF
  states no dataset licence (wording updated). Open: official MoHFW protocol citation for the Big Four statement.
- Machine B note: `pdftotext`/`pdftoppm` now available (`brew install poppler`), so agents can read the proposal PDF.
### 2026-10-20 (i) — Member B's agent
- B1–B5 committed by the user on branch `member-b-venom` (0f2a06e), pushed, PR JayParikh0310/SnakeDetection_XAI#1
  opened with Jay as reviewer. It conflicts with Jay's `6555ff7` in PROGRESS.md; the user left the conflict for Jay.
- B6: read Jay's Review-2 docx (from origin/master) so B's sections complement it. Its §3.8 promised a focus on
  India's Big Four → new `scripts/venom_species_report.py` (Wilson CIs): cobra 0.96, saw-scaled viper 0.93,
  Russell's viper 0.80, **common krait 0.75 [0.51, 0.90]**. New `scripts/make_app_figure.py` → `results/app/app_examples.png`.
- `report/B_report.md`: venom-stage intro, literature review (species/venom ID, classifier XAI and its evaluation),
  dataset, classifier, classifier XAI, app, ethics & safety, limitations, references. 6 [check] markers (proposal's
  reviewed systems, krait/India burden sources, SnakeCLEF licence, one author list). 85 non-gpu tests green.
- Not done: nothing committed since 0f2a06e (B6 files, test additions, PROGRESS/CLAUDE/PLAN updates are uncommitted).
### 2026-10-20 (h) — Member B's agent (resumed)
- User decisions: (1) crop context as an **optional pipeline setting**; (2) **rescale the venom score in B's classifier**.
- `pipeline.py` (A's file, additive): `PipelineConfig.venom_crop_margin` (default 0.0) + `venom_crop()`; with 0.0 it is
  identical to the old crop (test over 200 random boxes incl. out-of-frame). `venom/classifier.py`: `rescale_score()`
  (monotone, threshold → 0.5) and `TorchVenomClassifier(rescale=False|True)`, `predict_raw`, `raw_threshold`; default
  unchanged, so all reported numbers stand. Demo: margin 0.25, rescale=True, explains the same enlarged crop.
- `tests/test_demo.py` +5 tests (crop equality, margin, pipeline passes the margin crop, rescale monotone/decisions,
  rescaled plug-in + threat effect). **84 non-gpu tests green** (incl. A's pipeline tests).
- Browser check, same venomous *Agkistrodon bilineatus* photo: before "likely harmless", P 0.07, threat 50 → after
  "possibly venomous", venom score 0.57, threat 66, beep 614 ms.
- Jay pushed `6555ff7` (Review-2 docx/pptx, dataset stats scripts; touches PROGRESS.md and .gitignore only). Not pulled.
### 2026-10-20 (g) — Member B's agent (paused by the user mid-B5)
- Installed `gradio==6.30.0` (pinned in requirements.txt). `src/snakeguard/demo.py` (`SnakeGuardDemo`: photo mode with
  confirmation off, venom classifier plugged in, detector LayerCAM + venom LayerCAM-ms on the snake crop, audit =
  occlusion + RISE; video mode with 3-of-5 confirmation → H.264 mp4 + timeline; rule-9 wording + disclaimer) and
  `scripts/app.py` (Gradio UI). Detector on MPS via `YoloDetector(device="mps")`: 33 ms vs 84 ms CPU. `tests/test_demo.py`
  (5 tests). Verified in the browser: upload → boxes, verdict, table, both heatmaps.
- Preview pane can't launch the app itself (macOS blocks its helper from reading ~/Documents: "Operation not permitted");
  run the app from a terminal and attach the preview by URL (`.claude/launch.json` has only the url).
- **Found a train/deploy mismatch:** the pipeline classifies the detector's tight box crop, the classifier was trained on whole
  photos. `scripts/eval_venom_on_crops.py` (test, 600 sampled, 522 with a detected snake, 252 venomous):
  venomous recall whole image 0.952 → tight crop **0.917** [0.880, 0.948]; crop + 25 % context 0.944; +50 % 0.948; +100 % 0.952.
  AUROC 0.900 / 0.881 / 0.902 / 0.900 / 0.903 (`results/venom_clf/b0_reg/crop_eval_test.md`).
- Next: (1) add a context margin (~0.25–0.5) to the venom crop — `SnakeGuard.process` is A's code, so either a
  `PipelineConfig` option agreed with Jay or do it inside `TorchVenomClassifier`; (2) decide with Jay how the threat
  score should use the calibrated venom threshold; (3) re-verify the app, finish docs, then B6.
### 2026-10-20 (f) — Member B's agent
- Final B4 run finished (200 test images, 78 min; detector snake box found for 177). Analysis + `report/B_results.md`.
- FP decisions have the same compression as harmless ones (scores just above the blank-image score; random deletion
  1.245), so the analysis now also reports **TP-only** win rates and Wilcoxon tests (`vs_random_TP.csv`, `wilcoxon_TP.csv`).
- TP results: all methods beat random but margins are modest (confident classifier). Multi-scale > last-stage Grad-CAM++
  (deletion 0.692 vs 0.800, p < 1e-4); LayerCAM ms beats occlusion (deletion 0.712 vs 0.779, p = 0.0009; insertion
  1.093 vs 1.071, p = 0.0055). Eigen-CAM weakest. Localization proxy (venomous decisions): CAMs 0.80–0.90 of saliency
  in the snake box, occlusion 0.60 ≈ random 0.58. Sanity: only Grad-CAM passes head randomization (0.02); LayerCAM ms
  0.81, IG 0.85, Grad-CAM++ 0.96, Eigen-CAM 1.00. Agreement: CAM cluster vs perturbation cluster (as on the detector).
- Recommendation: LayerCAM ms for the app's venom-alert explanation; Grad-CAM + RISE for audits.
### 2026-10-20 (e) — Member B's agent
- B4 code: `scripts/run_venom_xai_bench.py` (resumable; A's `deletion_insertion`, `pointing_game`, `energy_pointing_game`,
  `saliency_iou`, `rank_correlation`, `topk_iou`, `noisy`; `RandomizedClassifier` = cascading randomization head /
  top (features[6:] + heads) / all) and `scripts/analyze_venom_bench.py` (reuses A's `analyze_bench` helpers).
  Localization uses a **proxy GT**: the frozen detector's best snake box (conf ≥ 0.25) mapped into the 224 crop.
  `CropView.box_to_crop/box_mask` added. `tests/test_venom_bench.py` (6 tests).
- Dev run (valid, 12 images) found a methodological issue: **with no evidence the classifier says "harmless"**
  (P(venomous | all-gray) = 0.143, threshold 0.141). For harmless decisions (FN/TN) deleting the whole image keeps
  91–96 % of the score, so deletion/insertion can't measure faithfulness there. Decision: faithfulness is read on
  venomous decisions (TP + FP); harmless decisions only feed localization / sanity / stability / agreement. A
  baseline-corrected AUC (`*_range`) was tried and is unstable (tiny original-minus-blank gap); kept in the CSV only.
- Final run started: test, 80 TP + 80 FP + 20 FN + 20 TN (one image per observation, seed 42); RISE/LIME/KernelSHAP
  and diagnostics on an 80-image subset.
### 2026-10-20 (d) — Member B's agent
- B3: `src/snakeguard/venom/xai.py` on the shared `Explainer` contract (`stage = "classifier"`, target 1 = venomous,
  0 = harmless = 1 − score). Explains the deployed score (species head). `explain_x` = 224 centre crop in pixel space
  (for A's metrics), `explain` maps back to the original via `CropView` (outside the crop = 0).
  - Reuses A's code unchanged: CAM maths (`*._cam`, `_upsample`, `_minmax`, mean fusion), `Occlusion`, `DRISE`
    (= RISE with a classifier score) and `RandomMap`, driven by `VenomScorer.score(x, target, batch, mode)`
    (same signature as `YoloDetector.score`).
  - Own: Integrated Gradients (32 steps, black baseline, positive part, blur σ 4), LIME and KernelSHAP on an 8×8
    patch grid, 1,000 samples (no lime/shap/scikit-image dependency).
  - MPS gotcha: A's `EigenCAM._cam` casts to float64 before `.cpu()`, which MPS rejects; B's wrapper passes CPU
    tensors (A's file untouched; A's detector never runs on MPS).
- Cost per explanation on the M4: CAMs 50–250 ms, IG/occlusion ~2 s, RISE/LIME/KernelSHAP 9–12 s.
- `tests/test_venom_xai.py` (20 tests: crop mapping vs torchvision, scorer = plug-in, all methods' shape/range,
  KernelSHAP exact on an additive model + efficiency, LIME rank). 68 non-gpu tests green.
- Qualitative (valid, `figures/grid_{TP,FN,TN,FP}.png`): last-stage CAMs are near-identical coarse blobs;
  LayerCAM-ms, IG, occlusion and RISE follow the body/pattern; Grad-CAM++-ms is noisy (28×28 stage). FN examples:
  barely visible snake (Trimeresurus), boomslang judged harmless while looking at its body. Centre crop drops the ends
  of tall/wide photos.
### 2026-10-20 (c) — Member B's agent
- `b0_reg` finished (10 epochs, ~11 min/epoch on the charger at 55 img/s; best epoch 8, valid AUROC venom 0.875 /
  species 0.891). Less overfitting: train venom loss 0.36 at the end (b0: 0.13), valid still rising at epoch 8.
- Comparison on valid (`compare_valid.md`, paired observation bootstrap vs b0:venom): b0_reg:species best on every
  metric: AUROC +0.022 [+0.017, +0.029], spec@rec95 +0.062 [+0.032, +0.089]. The mean ensemble is not better.
- Chosen: b0_reg + species score. New `scripts/finalize_venom.py` re-picks the threshold on valid for that score
  (0.1413), evaluates test once, writes `final_metrics.json` + `per_species_test_final.csv`, stores {score, threshold}
  in `best.pt`. `TorchVenomClassifier` supports `score="species"` (plug-in matches the saved predictions to <1e-6).
- Test: AUROC 0.888 [0.878, 0.898], recall 0.953 [0.944, 0.962], specificity 0.565 [0.547, 0.583] (b0 venom head:
  0.870 / 0.952 / 0.496). 77 of 285 venomous species have ≥ 1 miss; worst (n ≥ 5): Dendroaspis jamesoni 0.40,
  Atractaspis bibronii 0.50, Micrurus frontalis 0.50, Bungarus niger 0.63. 48 non-gpu tests green.
### 2026-10-20 (b) — Member B's agent
- New `scripts/compare_venom_scores.py`: venom-head vs species-head vs mean score per run, on one split, with paired
  bootstrap CIs over observations. Run b0 on **valid**: AUROC venom 0.869 / species **0.882** (diff +0.013
  [+0.009, +0.017]) / mean 0.880; specificity at recall ≥ 0.95: 0.502 / **0.541** (+0.039 [+0.006, +0.060]) / 0.534.
- `train_venom.py`: new train settings `dropout` (default 0.2) and `venom_label_smoothing` (default 0.0) — defaults
  reproduce b0 — and `--set key=value` overrides (saved in run_config.json). 47 non-gpu tests green.
- Started `b0_reg`: dropout 0.5, weight_decay 0.1, venom_label_smoothing 0.1, crop_scale [0.25, 1.0] (one combined
  variant, so a gain can't be attributed to a single change). Mac on the charger.
### 2026-10-20 — Member B's agent
- Run `b0` finished (10 epochs, ~3 h 15 on battery, 17–24 min/epoch). Best epoch 8 (valid AUROC 0.8687).
- Test (7,762 imgs, 36 % venomous): AUROC 0.870 [0.859, 0.881], AP 0.791. Threshold 0.0169 from valid →
  test recall 0.952 [0.942, 0.962], specificity 0.496 [0.478, 0.515]. At 0.5: recall 0.715, specificity 0.843.
  CIs = bootstrap over observations (`scripts/train_venom.py`).
- Observations: (1) the venom head overfits (train venom loss 0.55 → 0.13, valid AUROC flat after epoch 6) and is
  over-confident, hence the very low threshold; (2) the species-head venom score (sum of P(species) over venomous
  species) is better on valid every epoch from 1 on (test 0.880 vs 0.870); (3) species top-1 is only 17 % at
  240 px with ≤ 25–100 imgs/species. Worst venomous species on test (n ≥ 5): Micrurus frontalis 0.50,
  Austrelaps labialis 0.57, Dendroaspis jamesoni 0.60, Oxyuranus scutellatus 0.69 (`per_species_test.csv`).
- Candidate B2 improvements (not done): use/ensemble the species-head score, stronger regularisation or fewer
  epochs, more data per species / 500 px images (disk-limited), larger backbone.
### 2026-10-19 — Member B's agent
- Finished moving the dataset out of iCloud (13,295 evicted files re-downloaded, 16 in parallel, ~20 min; 0 failures).
  All 50,441 images decode in `~/snakeguard-data/snakeclef`; `data` → symlink; iCloud copy removed.
- Run `b0` crashed at the end of epoch 0 in `predict()`: venom mask on CPU, logits on MPS. Fixed (mask moved to the
  device) + regression test `test_predict_on_device` (cpu + mps). Added `--limit N` smoke mode; a 256-image
  1-epoch smoke run exercised the full path (threshold, test metrics, bootstrap, per-species, `best.pt` loads via
  `TorchVenomClassifier`). 46 non-gpu tests green.
- Training speed on battery: 29 img/s (vs ~60 measured earlier, likely on the charger); the loader gives 540+ img/s
  with 6 workers, so the GPU power state is the limit. User chose to train on battery.
### 2026-10-15 (c) — Member B's agent
- First training run `b0` crashed at step 0: iCloud ("Optimize Mac Storage", disk 94 % full) had evicted
  9,182 of the dataset images in `~/Documents` (dataless placeholders) and DataLoader workers could not read them.
- Fix in progress: copy the dataset to `~/snakeguard-data/snakeclef` (outside iCloud) and symlink `data/`.
  Evicted files take ~1.3 s each to re-download, so `scripts/copy_data_out_of_icloud.py` fetches them 16 in
  parallel. Paused by the user at 23,614 / 50,450 files. `.gitignore`: added the `data` symlink and
  `results/venom_clf/*/last.pt`.
- Training-speed profile (M4): GPU-only 59–62 img/s at batch 64, bf16 no gain, channels_last 28 img/s.
### 2026-10-15 (b) — Member B's agent
- B2 code: `src/snakeguard/venom/` — `data.py` (manifest loader, `available` rows only; train/eval transforms),
  `model.py` (`VenomNet`: torchvision EfficientNet-B0 ImageNet + venom head + 1,784-way species head;
  `features` = CAM layer for B3), `classifier.py` (`TorchVenomClassifier(VenomClassifier)`: `predict(crop)`,
  `.threshold`; crops < 16 px → None = unknown). `scripts/train_venom.py` (resumable per epoch; best epoch by
  valid AUROC; threshold on valid for venomous recall ≥ 0.95; test metrics with observation-bootstrap CIs,
  per-species table). `tests/test_venom_model.py` (8 tests). 44 non-gpu tests green.
- Speed on the M4: ~60 img/s training (GPU-bound; bf16 no gain, channels_last 2x slower) → 10 epochs ≈ 1 h 50.
- Training run `b0` started (caffeinate). Next: read `metrics.json`, then B3.
### 2026-10-15 — Member B's agent
- Extraction finished with no network retries: 50,441 / 50,640 images (1.6 GB). The 199 others (0.4 %, 71
  venomous, all from the train archive) are listed in the metadata but absent from the 240 px archive
  (checked: the archive was read to the end; no file with the same stem). No species lost; 57 observations
  lost all photos. Manifest keeps them with `available = False` instead of resampling.
- All 50,441 images decode (OpenCV). Max side 75–240 px (median 240), min side median 180; 212 images
  have min side < 100 px.
- Usable split sizes: train 34,838 (38.0 % venomous, 1,784 species) / valid 7,841 (37.2 %, 1,761) /
  test 7,762 (36.1 %, 1,761).
- Script: `--mark` (refresh `available` only), `archives_done.json` so finished archives are not
  re-streamed (a re-run had started re-reading 4.8 GB looking for the 199 absent files), `--force`.
- Env problem: `~/Documents` is in iCloud Drive, which sets the `hidden` flag on files inside `.venv`;
  Python ≥ 3.12.13 skips hidden `.pth` files, so `import snakeguard` failed. Fix: venv recreated at
  `~/venvs/snakeguard-xai` (outside iCloud), `.venv` is a symlink to it; `.gitignore` now also ignores the
  `.venv` symlink. 36 non-gpu tests green.
- Next: B2 — train the venom classifier on MPS (choose backbone; venomous recall is the key metric).
### 2026-10-14 — Member B's agent
- Explored the repo (read-only report to B). Set up B's env on macOS: `.venv` (Python 3.12.13) with the
  pinned requirements (PyPI torch with MPS instead of cu126). 33 non-gpu tests green; the 22 gpu tests
  need `final_dataset_v4`, not on B's machine yet.
- B1 decision (user): SnakeCLEF 2023 subset. Checked sizes first: train-small 4.84 GB, val-small 0.45 GB,
  metadata 16 MB. Label = `MIVS` from `venomous_status_list.csv` (medically important venomous; 285 of
  1,784 species). The official server's HTTPS cert is self-signed → config uses plain HTTP.
- Added `configs/venom.yaml`, `scripts/build_venom_dataset.py` (plan → manifest; extract → streams the
  archives and keeps only manifest images, archives never stored), `tests/test_venom_data.py` (3 tests).
- Subset: cap 100 images/venomous species, 25/harmless species, whole observations, seed 42; split by
  observation within species 70/15/15. Result: 50,640 images, 29,025 observations, all 1,784 species.
  train 34,982 (38.0 % venomous) / valid 7,868 (37.2 %) / test 7,790 (36.1 %). 22 species (1–2
  observations) are train-only. The iNat train metadata had 2 exact duplicate rows (dropped).
- Next: finish the extraction (log `data/snakeclef/extract.log`), verify images, then B2.
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
