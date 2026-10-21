# Member B: Results (draft for the report)

Venom classification (B1–B2) and its explainability (B3–B4). All numbers come from scripts in this project;
model selection used **valid**, final numbers are on **test**. ± = half of the 95 % bootstrap CI.
Hardware: Apple M4 (MPS).

## 1. Data (B1) — `scripts/build_venom_dataset.py`, `data/snakeclef/manifest_summary.json`

SnakeCLEF 2023 (iNaturalist photos, 240 px "small" release). Label = SnakeCLEF's **MIVS** flag (medically important
venomous snake): 285 of 1,784 species. Mildly venomous rear-fanged species count as harmless, which matches a
bite-warning use.

| | Images | Venomous | Species | Observations |
|---|---|---|---|---|
| train | 34,838 | 38.0 % | 1,784 | 20,108 |
| valid | 7,841 | 37.2 % | 1,761 | 4,430 |
| test | 7,762 | 36.1 % | 1,761 | 4,430 |

Sampling: ≤ 100 images per venomous species, ≤ 25 per harmless species, whole observations, seed 42; split by
observation within each species (an observation never spans two splits). The caps raise the venomous share from 19 %
to ~37 %, so the threshold is calibrated on valid. 199 listed images are missing from the 240 px archive (0.4 %).

## 2. Classifier (B2) — `scripts/train_venom.py`, `compare_venom_scores.py`, `finalize_venom.py`

EfficientNet-B0 (ImageNet) with a venom head and an auxiliary 1,784-way species head, 224 px, 10 epochs.

**Model selection on valid** (`results/venom_clf/compare_valid.md`, paired bootstrap over observations vs b0 venom head):

| Run : score | AUROC | Δ AUROC [95 % CI] | Specificity @ recall 0.95 | Δ [95 % CI] |
|---|---|---|---|---|
| b0 : venom head | 0.869 | – | 0.502 | – |
| b0 : species head | 0.882 | +0.013 [+0.009, +0.018] | 0.541 | +0.039 [+0.008, +0.060] |
| b0_reg : venom head | 0.875 | +0.006 [+0.001, +0.011] | 0.518 | +0.016 [−0.016, +0.046] |
| **b0_reg : species head** | **0.891** | **+0.022 [+0.017, +0.029]** | **0.564** | **+0.062 [+0.032, +0.089]** |

`b0_reg` = dropout 0.5, weight decay 0.1, venom label smoothing 0.1, crops 0.25–1 (one combined variant). The
**species-head score** = Σ P(species) over venomous species; it beats a dedicated venom head because venom status is a
species property.

**Final test** (`results/venom_clf/b0_reg/final_metrics.json`), threshold 0.1413 chosen on valid for venomous recall ≥ 0.95:

| AUROC | AP | Venomous recall | Specificity | Precision |
|---|---|---|---|---|
| **0.888** [0.878, 0.898] | 0.813 | **0.953** [0.944, 0.962] | **0.565** [0.547, 0.583] | 0.553 |

Catching 95 % of venomous snakes costs a false "venomous" on 43.5 % of harmless snakes. That trade-off is deliberate
for a safety device, but it means the venom alert is a precaution, not a verdict. Worst venomous species on test
(n ≥ 5): *Dendroaspis jamesoni* 0.40, *Atractaspis bibronii* 0.50, *Micrurus frontalis* 0.50, *Bungarus niger* 0.63
(`per_species_test_final.csv`). Species top-1 accuracy is only 17 %: fine-grained ID at 240 px with ≤ 25–100 images
per species is hard; venom status is easier than species.

## 3. Explanation methods (B3) — `src/snakeguard/venom/xai.py`

All on the shared `Explainer` contract; they explain the deployed score (species head). Same CAM maths, occlusion and
(D-)RISE code as Member A's detector methods.

| Method | Family | Setting |
|---|---|---|
| Eigen-CAM, Grad-CAM, Grad-CAM++ | activation / gradient CAM | last stage (7×7) |
| Grad-CAM++ ms, LayerCAM ms | gradient CAM, multi-scale | stages at 28 / 14 / 7 px, mean fusion |
| Integrated Gradients | path gradients | 32 steps, black baseline |
| Occlusion, RISE | perturbation | 32/64 px patches; 1,000 masks |
| LIME, KernelSHAP | surrogate | 8×8 patch grid, 1,000 samples |
| Random | baseline | smooth blobs |

## 4. Benchmark (B4) — `scripts/run_venom_xai_bench.py`, `analyze_venom_bench.py`

Test, 200 images: 80 TP, 80 FP, 20 FN, 20 TN (one per observation). RISE/LIME/KernelSHAP and the diagnostics on an
80-image subset. Every map explains the model's decision. Full tables: `results/venom_xai_bench/test_final/analysis/summary.md`.

**A methodological finding first.** With no evidence the classifier says "harmless": an all-gray image gets
P(venomous) = 0.143, right at the threshold. So deletion/insertion **cannot measure faithfulness for harmless decisions**
(deleting the whole image keeps 91–96 % of the score), and FP scores sit just above the blank-image score, so their
curves are compressed too (even random gets 1.25 there). Faithfulness is therefore read on **TP** (confident venomous
detections, the alerts the app explains); harmless decisions only feed localization and diagnostics.

### 4.1 Faithfulness on TP (80 images; `summary_TP.csv`, `vs_random_TP.csv`, `wilcoxon_TP.csv`)

| Method | Deletion ↓ | Insertion ↑ | Beats random (del / ins) | ms |
|---|---|---|---|---|
| Eigen-CAM | 0.855 ±0.123 | 0.986 ±0.056 | 69 % / 73 % | 208 |
| Grad-CAM | 0.758 ±0.110 | 1.178 ±0.140 | 78 % / 85 % | 67 |
| Grad-CAM++ (last stage) | 0.800 ±0.110 | 1.001 ±0.056 | 75 % / 75 % | 67 |
| **Grad-CAM++ ms** | **0.692 ±0.097** | 1.000 ±0.072 | **91 %** / 66 % | 70 |
| **LayerCAM ms** | 0.712 ±0.105 | 1.093 ±0.074 | 85 % / 84 % | 67 |
| Integrated Gradients | 0.826 ±0.132 | 0.953 ±0.063 | 75 % / 66 % | 696 |
| Occlusion | 0.779 ±0.098 | 1.071 ±0.129 | 78 % / 63 % | 1,668 |
| RISE (n = 20) | 0.758 ±0.248 | **1.258 ±0.280** | 85 % / 85 % | 8,008 |
| LIME (n = 20) | 0.789 ±0.275 | 1.250 ±0.268 | 80 % / 85 % | 7,910 |
| KernelSHAP (n = 20) | 0.823 ±0.225 | 1.186 ±0.251 | 70 % / 70 % | 7,951 |
| *Random* | 0.934 ±0.100 | 0.920 ±0.090 | – | 0.4 |

1. **Every method beats random, but the margins are modest**: the classifier is very confident (many TP at P ≈ 1),
   so a lot must be deleted before the score falls. Differences between methods are smaller than on the detector.
2. **Multi-scale fusion helps here too** (mirrors A's detector result): Grad-CAM++ ms vs last-stage Grad-CAM++ deletion
   0.692 vs 0.800 (Wilcoxon p < 1e-4); LayerCAM ms vs Grad-CAM++ deletion 0.712 vs 0.800 and insertion 1.093 vs 1.001
   (both p < 1e-4).
3. **LayerCAM ms beats occlusion** on deletion (0.712 vs 0.779, p = 0.0009) and insertion (1.093 vs 1.071, p = 0.0055)
   at 1/25 of the cost. **Eigen-CAM is weakest** (closest to random). The perturbation/surrogate methods are best on
   insertion but cost ~8 s and were measured on only 20 TP images (wide CIs).

### 4.2 Localization (proxy) — venomous decisions, 160 images

SnakeCLEF has no masks, so the GT is the **frozen detector's snake box** (found in 177 / 200 images). Share of saliency
inside the box: Grad-CAM++ 0.90, Eigen-CAM 0.88, LayerCAM ms 0.85, Grad-CAM 0.80, IG 0.80, LIME 0.74, RISE 0.72,
Grad-CAM++ ms 0.74, KernelSHAP 0.64, occlusion 0.60, random 0.58. The CAMs stay on the snake; occlusion's positive
map is barely better than random by this measure (its maps are blocky and partly off the box).

### 4.3 Sanity and stability (80-image subset, all decisions)

| Method | Stability ↑ | Similarity after randomizing heads ↓ | … top stages + heads ↓ | … all ↓ |
|---|---|---|---|---|
| Eigen-CAM | 0.97 | **1.00 (fails, by construction)** | 0.68 | 0.59 |
| Grad-CAM | 0.77 | **0.02 (passes)** | −0.01 | 0.00 |
| Grad-CAM++ | 0.95 | **0.96 (fails)** | 0.65 | 0.00 |
| Grad-CAM++ ms | 0.86 | 0.79 | 0.65 | 0.00 |
| LayerCAM ms | 0.91 | 0.81 | 0.59 | 0.00 |
| Integrated Gradients | 0.95 | 0.85 | 0.61 | 0.01 |

Same pattern as A's detector: Eigen-CAM cannot depend on the decision head, and last-stage Grad-CAM++ barely changes
when the heads are randomized (its channel weights are dominated by activations). Unlike on the detector, LayerCAM ms and
IG also stay fairly similar (0.81 / 0.85) after head randomization: on this classifier their maps reflect *where the
snake is* (features) more than *why it is venomous* (head). Only plain Grad-CAM passes the head test, at the price of the
lowest stability (0.77).

### 4.4 Agreement (Spearman of the maps, all decisions; `agreement.png`)

Two clusters again: the CAM/gradient methods agree strongly with each other (Eigen-CAM ↔ Grad-CAM++ 0.95, Grad-CAM++ ↔
LayerCAM ms 0.90) but weakly with the perturbation/surrogate methods (occlusion ↔ CAMs 0.10–0.30; KernelSHAP ↔ anything
≤ 0.29). As on the detector, agreement inside the CAM family is no evidence of faithfulness.

## 5. Deployment check: the classifier on detector crops (B5) — `scripts/eval_venom_on_crops.py`

The pipeline classifies the detector's crop, not the whole photo the classifier was trained on. Test, 600 sampled
images, 522 with a detected snake (252 venomous), deployed threshold (`results/venom_clf/b0_reg/crop_eval_test.md`):

| Input | Venomous recall [95 % CI] | Specificity | AUROC |
|---|---|---|---|
| whole image | 0.952 [0.927, 0.979] | 0.596 | 0.900 |
| tight detector crop | **0.917** [0.880, 0.948] | 0.593 | 0.881 |
| crop + 25 % context | 0.944 [0.918, 0.970] | 0.615 | 0.902 |
| crop + 50 % context | 0.948 [0.921, 0.973] | 0.607 | 0.900 |
| crop + 100 % context | 0.952 [0.924, 0.977] | 0.604 | 0.903 |

A tight crop loses about 3.5 points of venomous recall; 25 % context per side recovers most of it, so the app uses
`venom_crop_margin = 0.25`. The app also rescales the venom score so the decision line is at 0.5, because the
pipeline's threat score multiplies P(venomous) and raw values near the 0.141 threshold barely changed the threat.

## 6. Recommendation

| Use | Method | Why |
|---|---|---|
| App explanation of a venom alert | **LayerCAM ms** | Top-2 deletion and the best insertion among cheap methods on TP, 85 % of saliency on the snake, stable (0.91), 67 ms, and the same method A recommends for the detector, so the app shows one consistent kind of map |
| Sanity-critical audit | Grad-CAM + RISE | Grad-CAM is the only method passing the head-randomization test; RISE is best on insertion (offline, ~8 s) |
| Not recommended | Eigen-CAM, last-stage Grad-CAM++ | Weakest faithfulness; fail the head sanity check |

Caveat for the app text: LayerCAM's map shows *where the model looked*, largely the snake itself, not a proof of *why*
it judged the snake venomous (sanity_head 0.81).

## 7. Limitations
- Faithfulness is only measurable for venomous decisions; "why harmless" cannot be tested with deletion/insertion
  because harmless is the model's no-evidence default.
- Localization uses the detector's box as a proxy GT (snake region, not venom-relevant parts such as head shape).
- 240 px images, centre crop: the ends of long snakes in elongated photos are outside the model's view.
- Expensive methods on 20 TP images only; one classifier, one dataset.
- Labels are species-level (MIVS); geographic/species shift to the device's deployment region is not evaluated.
