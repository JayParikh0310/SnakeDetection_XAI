# PLAN.md — SnakeGuard XAI: full project plan

This is the stable plan. Day-to-day status lives in `PROGRESS.md`. Scope changes go in §8.

Members: **A** = Member A (this machine, user `jaypa`) · **B** = Member B (partner).

---

## 1. Goal

SnakeGuard is an ankle-worn camera early-warning system (see the proposal PDF). The pipeline
is: Sense → **Detect** (snake vs. look-alike) → Classify (venomous?) → Locate (bbox area as a
distance proxy) → Alert (beep rate). On top of it sits an **Explainability & Validation Layer**
that checks *why* the detector fires. That layer is the main focus of this project.

```
frame ─► YOLO11n detector (best.pt, frozen) ─► boxes {snake, snake_like_object}
             │                                      │
             │                       post-processing: conflict resolution, small-object
             │                       filter, 3-of-5 temporal confirmation, proximity,
             │                       threat score 0–100  (+ venom classifier from B)
             ▼
   XAI layer: Eigen-CAM · multi-scale Grad-CAM++ · Occlusion  (+ D-RISE, Grad-CAM extras)
             ▼
   Quantitative validation: deletion/insertion AUC vs. random baseline, localization,
   cross-method agreement, sanity checks; matched snake vs. look-alike analysis
```

Research questions:
- **RQ1** Which XAI method gives the most *faithful* explanation of the YOLO11n detector?
  Measured by deletion/insertion AUC against a random-order baseline.
- **RQ2** Do the explanations land on the snake (GT box / polygon), or on background cues?
- **RQ3** Matched snake vs. look-alike pairs: what drives correct detections and confusions?
  (coiled body / head shape vs. thin dark curved texture)
- **RQ4** Does multi-scale fusion beat a single-scale hook for Grad-CAM++ (the proposal's claim)?
  Do activation-based and gradient-based methods agree?
- **RQ5** Faithfulness vs. localization vs. stability vs. runtime: which method should the
  device/app use by default?

## 2. Literature-review alignment (from `SnakeGuard_Literature_Review_and_Proposal_revised.pdf`)

Commitments in the proposal, and how we cover them:

| Proposal commitment | Where |
|---|---|
| Eigen-CAM: activation-based, no gradients, fast first-pass check | A3 |
| Grad-CAM++: class-targeted, **fused across the multiple YOLO feature scales** (not only the coarsest layer) | A3 (+ single-scale ablation, RQ4) |
| Occlusion sensitivity: model-agnostic "ground-truth" cross-check | A3 |
| Deletion/insertion AUC compared with a **random-order baseline**, used to pick the most reliable method objectively | A4, A5 |
| Agreement between the activation-based and gradient-based methods counts as stronger evidence | A4 (agreement metrics), A5 |
| Hard-case testing on **matched pairs** of real snakes vs. look-alikes, side by side | A5 |
| SHAP/LIME **set aside** (poor fit for detection) | Not used for the detector. Mention the rationale in the report. |
| Pipeline extras named in the gap analysis: box conflict resolution, small-object area filter, **3-of-5 sliding-window temporal confirmation**, bbox-area proximity, **graded 0–100 threat score**, venom status | A6 (venom classifier is B's) |
| Lit-review baselines: Ref 4 mAP@0.5 90.2 %, Ref 5 94.5 %, Ref 8 93.4 % | A1 compares our detector against these |

Additions beyond the proposal (extras that strengthen the report): plain Grad-CAM baseline,
D-RISE (a black-box saliency method designed for detectors), a contrastive "snake vs.
look-alike" target, a model-randomization sanity check, and stability under noise.

## 3. Methods: details

### 3.1 Detector XAI (owner A)
All explainers implement the shared interface in `src/snakeguard/xai/base.py` and explain
**one detection** (box + class). The target score is the D-RISE similarity:
`s(img) = max_i IoU(box_i, target_box) · p_i(target_class)` over all raw (pre-NMS) predictions.
The same score is used for gradients, occlusion and the deletion/insertion metric, so the
comparison is consistent.

| Method | Notes |
|---|---|
| Eigen-CAM | 1st principal component of activations. Single-scale and multi-scale (P3/P4/P5) variants. |
| Grad-CAM++ | Hooks the P3/P4/P5 neck outputs feeding `Detect`. Per-scale maps are normalized, then fused. Single-scale ablation. Optional contrastive target `p_snake − p_sll`. |
| Grad-CAM | Baseline. |
| Occlusion | Sliding gray patch, batched on GPU, multi-size average. |
| D-RISE | Random masks weighted by the similarity score. Expensive, so run on a subset. |

### 3.2 Evaluation (shared harness `src/snakeguard/eval/`)
| Axis | Metric |
|---|---|
| Faithfulness | Deletion AUC ↓, insertion AUC ↑, both also as the gap vs. the **random-order baseline** |
| Localization | Pointing game, energy-based pointing game (box and, where available, polygon mask) |
| Agreement | Spearman rank correlation and top-k% IoU between methods |
| Sanity | Model-parameter randomization (Adebayo 2018) |
| Stability | Map similarity under small Gaussian noise / brightness changes |
| Cost | ms per explanation on the RTX 4060 |

Benchmark: tune on valid and report on test. Targets are detections matched to GT (IoU ≥ 0.5)
plus all confusions/FPs. Expensive methods run on a fixed seeded subset.

### 3.3 Venom classifier + its XAI (owner B)
B decides the dataset, model and methods. Constraint: plug into the same `Explainer` interface
and `eval` harness so the results are comparable. Here SHAP/LIME are fine (a classifier is
single-label).

## 4. Work division

### Workstream A (Member A): detector XAI + evaluation + pipeline
| ID | Task |
|---|---|
| A0 | Environment (venv, CUDA torch, ultralytics), package skeleton, tests |
| A1 | Baseline detector evaluation (mAP, P/R, confusion matrix, failure gallery). Compare with the lit-review numbers. |
| A2 | YOLO wrapper (letterbox, raw outputs, own NMS with anchor indices, target score) + `Explainer` interface |
| A3 | Explainers: Eigen-CAM, Grad-CAM, Grad-CAM++ (multi + single scale), Occlusion, D-RISE |
| A4 | Evaluation harness: deletion/insertion (+ random baseline), pointing game, EBPG, agreement, sanity, stability, runtime |
| A5 | Benchmark on test + matched snake/look-alike analysis + figures/tables |
| A6 | SnakeGuard pipeline: post-processing, temporal confirmation, proximity, threat score, pluggable venom classifier (stub until B delivers) |
| A7 | Report material for A's sections (methods, results, discussion) |

### Workstream B (Member B): venom classification + classifier XAI + demo app
| ID | Task |
|---|---|
| B1 | Venom dataset (species → venomous mapping, leakage-free splits) |
| B2 | Train/evaluate the classifier (venomous recall is the key metric) |
| B3 | Classifier explainers via the shared interface (Grad-CAM++, IG, LIME, SHAP, …) |
| B4 | Classifier XAI benchmark using A's harness |
| B5 | Demo app (Gradio) on top of A's pipeline API |
| B6 | Report: intro, literature review, classifier, app, ethics |

Joint: J1 cross-stage discussion and final recommendation; J2 report and slides.

## 5. Timeline
No fixed deadline. Order of work: A0 → A1 → A2 → A3 → A4 → A5 → A6 → A7.
B can start B1/B2 any time. B3/B4 need A2/A4 (already delivered once they're marked DONE).

## 6. Deliverables
Reproducible code (`src/`, `scripts/`, `configs/`, `tests/`), CSV result tables, figures,
the pipeline + app, and the report.

## 7. Open questions
- None blocking for A. Venom classifier, repo/sharing and deadline are handled by the team.

## 8. Decision log
| Date | Decision | Reason |
|---|---|---|
| 2026-10-13 | Keep `best.pt` frozen | XAI explains the existing model |
| 2026-10-13 | No git repo for now; the user handles sharing | User instruction |
| 2026-10-13 | Detector XAI = Eigen-CAM, multi-scale Grad-CAM++, Occlusion (core, from the proposal) + Grad-CAM, D-RISE (extras); no SHAP/LIME for the detector | Literature-review proposal §2 |
| 2026-10-13 | Target score = D-RISE similarity (IoU × class prob, max over raw preds) for every method/metric | One consistent definition of "the detection's confidence", robust to perturbation shifting the anchor |
| 2026-10-13 | Own implementation of the CAM methods (not pytorch-grad-cam) | Full control over multi-scale hooks/targets on YOLO11; fewer version issues |
| 2026-10-13 | Use rect letterbox (pad to a multiple of 32) instead of a square 640x640 input | Square padding produced CAM artifacts on the gray border. Same as ultralytics predict/val |
| 2026-10-13 | Add LayerCAM (per-location gradient weighting, multi-scale) as a candidate | Channel-weighted CAMs highlighted image borders in qualitative checks |
| 2026-10-13 | Dev results (valid, 100 imgs): keep raw Eigen-CAM (beats centered), mean fusion (best for Grad-CAM++, ties for LayerCAM). Drop the contrastive / max / sum / centered variants from the final run (no gain) | `results/xai_bench/valid_dev/analysis/summary.md` |
| 2026-10-13 | Final test run: eigencam(+p5), gradcam, gradcampp(+p3/p4/p5), layercam(+p5), occlusion, drise (200-img subset), random, boxprior. Stability + sanity on 150 images (no D-RISE) | Cost |
| 2026-10-13 | Inference stays fp32 (fp16 would be 1.6x faster but changes scores by up to 0.005) | Exactness; fast enough on an idle GPU |
| 2026-10-14 | Default explainer for pipeline/app = LayerCAM P3–P5 (mean fusion); occlusion = offline audit | Test results: near-occlusion faithfulness, best localization, passes sanity, 27 ms (`report/A_results.md` §7) |
| 2026-10-14 | B1 data = SnakeCLEF 2023 (240 px "small" images), label = SnakeCLEF `MIVS` (medically important venomous) | Species-level labels allow leakage-free splits and per-species analysis; fits B's 20 GB disk. Mildly venomous rear-fanged species count as 0, which matches a bite-warning use |
| 2026-10-14 | B1 subset: ≤100 images per venomous species, ≤25 per harmless species, whole observations, split by observation 70/15/15 (seed 42) | Keeps all 1,784 species, ~1.6 GB, venomous share ~37 % instead of 19 %. Note: the cap changes the class prior, so calibrate/threshold on valid |
| 2026-10-15 | B2 model = EfficientNet-B0 (ImageNet) + venom head + auxiliary species head (loss weight 0.5), 224 px, RandomResizedCrop(0.35–1) | Small/fast on M4 MPS and for the app; plain CNN for CAM XAI (B3); species head exploits that venom status is a species property and gives a second venom estimate |
| 2026-10-15 | B2 operating point: threshold on valid for venomous recall ≥ 0.95; report test recall/specificity with observation-bootstrap CIs | A missed venomous snake costs more than a false alarm. Pipeline should use `TorchVenomClassifier.threshold` as `venom_threshold` |
| 2026-10-20 | B2 model selection on valid only: compare venom-head / species-head / mean scores and runs with `scripts/compare_venom_scores.py` (paired observation bootstrap); report only the chosen model on test | Rule 6 (tune on valid, report on test) |
| 2026-10-20 | B2 final = run `b0_reg` (dropout 0.5, wd 0.1, venom label smoothing 0.1, crop 0.25–1) scored with the **species head** (sum of P(species) over venomous species), threshold 0.1413 on valid | Best on valid on AUROC and specificity at 95 % recall with paired-bootstrap CIs excluding 0 (`results/venom_clf/compare_valid.md`) |
| 2026-10-20 | B3 methods: Eigen-CAM, Grad-CAM, Grad-CAM++ (last stage), Grad-CAM++ and LayerCAM over 3 stages, Integrated Gradients, occlusion, RISE, LIME, KernelSHAP, random; explain the deployed species-head score | Mirrors A's detector set (same CAM maths / occlusion / D-RISE code) so stages are comparable; adds the classifier-appropriate gradient (IG) and surrogate (LIME/SHAP) methods the proposal set aside only for the detector. LIME/SHAP on an 8×8 grid instead of superpixels to avoid new dependencies |
| 2026-10-20 | B4: faithfulness (deletion/insertion) reported on **venomous decisions only** (TP + FP); harmless decisions feed localization/diagnostics. Localization GT = detector's snake box (proxy, SnakeCLEF has no masks). Sample 80 TP + 80 FP + 20 FN + 20 TN test images; expensive methods + diagnostics on 80 | A blank image already scores "harmless" (P(venomous) 0.143 ≈ threshold), so deletion can't move a harmless decision. Venomous decisions are also the alerts the app explains |
| 2026-10-20 | B4 faithfulness headline = **TP** decisions (FP curves are compressed like harmless ones); app explainer for venom alerts = **LayerCAM ms** | `results/venom_xai_bench/test_final/analysis/`; `report/B_results.md` §4–5. Same method as A's detector choice → one consistent map type in the app |
| 2026-10-20 | Venom crop for the classifier = detector box + 25 % context per side (`PipelineConfig.venom_crop_margin`, default 0.0 keeps A's behaviour; the app sets 0.25) | Classifier trained on whole photos: venomous recall 0.917 on tight crops vs 0.944 with 25 % context (`crop_eval_test.md`) |
| 2026-10-20 | App uses `TorchVenomClassifier(rescale=True)`: monotone rescale with the 0.141 threshold at 0.5 | A's threat score multiplies P(venomous); raw probabilities near the 0.141 threshold barely moved the threat. Ranking/metrics unchanged |
