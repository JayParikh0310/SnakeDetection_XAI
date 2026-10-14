# Member A: Results (draft for the report)

All numbers come from scripts in this project. Settings were tuned on 100 validation images
(`results/xai_bench/valid_dev`). Final numbers are on the **test split**
(`results/xai_bench/test_final`): 868 explained detections (742 TP, 126 FP) on 773 images.
± = half of the 95 % bootstrap CI. Hardware: RTX 4060 Laptop GPU.

## 1. Detector baseline (`results/detector_eval/summary.json`)

| Split | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | AP50 snake | AP50 look-alike |
|---|---|---|---|---|---|---|
| valid | 0.888 | 0.617 | 0.884 | 0.811 | 0.917 | 0.859 |
| **test** | **0.865** | **0.617** | 0.834 | 0.791 | **0.925** | 0.805 |

At the operating threshold (conf 0.25) on test: 756 TP, 163 FP, 193 FN and **0 snake↔look-alike
confusions** (11 on valid). Snake recall / precision: 0.859 / 0.914. Look-alike: 0.703 / 0.695.
Errors are mostly (a) small objects missed (median GT area 10.7 % of the image for misses vs.
36.6 % for hits), e.g. thin, banded, camouflaged snakes at night, and (b) box-granularity
disagreements: 102 of the 163 FPs overlap a real object, e.g. one box around a cluster of roots that
the annotators boxed one by one. The literature reports mAP@0.5 of 90.2–94.5 % (Refs 4, 5, 8) on
other, mostly snake-only data. Our number includes the harder look-alike class.

## 2. Faithfulness and localization (test)

| Method | Deletion ↓ | Insertion ↑ | Beats random (del / ins) | Pointing game | Mass in box | ms |
|---|---|---|---|---|---|---|
| Eigen-CAM (proposal) | 0.244 ±0.010 | 0.438 ±0.016 | 42 % / 74 % | 0.26 | 0.40 | 43 |
| Grad-CAM | 0.216 ±0.010 | 0.449 ±0.015 | 49 % / 76 % | 0.45 | 0.44 | 38 |
| Grad-CAM++ P3–P5 (proposal) | 0.152 ±0.007 | 0.577 ±0.014 | 68 % / 94 % | 0.40 | 0.44 | 28 |
| Grad-CAM++ P5 only | 0.364 ±0.015 | 0.485 ±0.016 | 26 % / 79 % | 0.25 | 0.24 | 27 |
| **LayerCAM P3–P5 (added)** | 0.082 ±0.005 | 0.765 ±0.012 | **90 % / 99 %** | 0.97 | **0.90** | **27** |
| **Occlusion (proposal)** | **0.077 ±0.006** | **0.846 ±0.013** | **91 % / 99 %** | 0.97 | 0.71 | 1,238 |
| D-RISE (200-img subset) | 0.080 ±0.009 | 0.816 ±0.029 | 93 % / 100 % | 0.98 | 0.74 | 4,233 |
| *Box prior (baseline)* | 0.162 ±0.010 | 0.782 ±0.012 | 68 % / 100 % | 1.00 | 0.91 | 13 |
| *Random (baseline)* | 0.204 ±0.007 | 0.295 ±0.009 | – | 0.40 | 0.39 | 5 |

Deletion/insertion are AUCs of the target score normalized by the original score. "Beats random" =
share of detections where the method's curve is better than the random map's curve on the **same**
detection. Full table: `results/xai_bench/test_final/analysis/summary.md`. Curves: `curves_TP.png`, `curves_FP.png`.

**Findings**
1. **Occlusion, the proposal's model-agnostic cross-check, is the most faithful method** (best
   deletion and insertion; Wilcoxon p < 1e-100 vs. Grad-CAM++ and Eigen-CAM). D-RISE matches it on
   deletion but costs 3.4× more.
2. **Eigen-CAM is not faithful on this detector.** On deletion it beats random for only 42 % of
   detections, and its maps mostly highlight image borders and background textures (figures). It
   should not serve as the "first-pass check" without the quantitative test.
3. **Grad-CAM++ beats Eigen-CAM and Grad-CAM, but only moderately beats random** (68 % deletion wins).
   Its maps also contain border/background artefacts, because it weights whole channels.
4. **Multi-scale fusion matters (the proposal's claim is confirmed).** Grad-CAM++ on P5 only is worse
   than random on deletion (26 % wins, 0.364 vs. 0.152, p = 2e-103) and is an *empty map* for 53 % of
   detections. The same holds for LayerCAM: P5-only 0.301 vs. fused 0.082.
5. **LayerCAM, multi-scale (our addition), is the best practical method.** It is within 0.005 of
   occlusion on deletion, has the best localization (97 % pointing game, 90 % of saliency inside the
   box), and is ~45× faster than occlusion. It is not just "highlighting the box": the box-prior
   baseline gets a comparable insertion score but a 2× worse deletion score (0.162 vs. 0.082).
6. Polygon-level localization (pixels of the actual object) is lower for every method (LayerCAM 0.46
   of saliency inside the polygon, the same as the box prior). Explanations resolve "which box
   region", not the exact snake silhouette.

## 3. Cross-checks between methods (agreement, Spearman ρ of the maps)

|  | Grad-CAM++ | LayerCAM | Occlusion | D-RISE |
|---|---|---|---|---|
| Eigen-CAM | **0.56** | −0.01 | −0.07 | 0.02 |
| Grad-CAM++ | | 0.07 | 0.07 | 0.13 |
| LayerCAM | | | **0.62** | **0.43** |
| Occlusion | | | | **0.47** |

There are two clusters: {Eigen-CAM, Grad-CAM++} and {LayerCAM, occlusion, D-RISE}. The proposal
treats agreement between the activation method (Eigen-CAM) and the gradient method (Grad-CAM++) as
strong evidence. Here those two agree with **each other** but **not with the perturbation methods**,
which actually measure the model's response. Agreement alone would give false confidence. The
deletion/insertion test, as the proposal intended, is what settles the disagreement, and it settles
it in favour of the perturbation cluster (and LayerCAM).

## 4. Sanity and stability (150-image subset, 146 detections)

| Method | Stability under noise ↑ | Similarity after randomizing head ↓ | … neck+head ↓ | … all layers ↓ |
|---|---|---|---|---|
| Eigen-CAM | 0.96 | **1.00 (fails)** | −0.24 | −0.11 |
| Grad-CAM | 0.62 | 0.07 | 0.03 | 0.01 |
| Grad-CAM++ | 0.87 | **0.86 (fails)** | −0.14 | −0.01 |
| LayerCAM | 0.96 | 0.27 | 0.25 | 0.10 |
| Occlusion | 0.88 | 0.10 | 0.03 | 0.17 |

The model-parameter randomization test (Adebayo et al. 2018): a faithful explanation should change
when the layers that produce the prediction are randomized. **Eigen-CAM cannot depend on the Detect
head at all** (it is computed from neck activations, so similarity is 1.0 by construction).
**Grad-CAM++'s map barely changes (0.86) when the head is randomized**, so it reflects mostly the
activations, not the decision. That is consistent with its weak deletion scores. LayerCAM and
occlusion change substantially (pass), and LayerCAM is also the most stable under input noise.

## 5. Matched snake / look-alike pairs (`results/figures/test/pairs_*.png`)

Each look-alike detection is shown next to the real snake the detector finds most similar (cosine
of RoI-pooled P5 features, 0.80–0.90). Observations:
- For real snakes, the faithful methods (LayerCAM, occlusion, D-RISE) concentrate on the **body
  curve and head/neck**, e.g. a snake held by hand: saliency runs along the body to the head, not on
  the hand.
- Mangrove-root false positives are driven by **clusters of thin, dark, curved strands** inside the
  box: exactly the "locally snake-like texture" failure mode the proposal hypothesised. The model is
  responding to the right *kind* of evidence; the error is in grouping (one box around many roots).
- Eigen-CAM / Grad-CAM++ on the same images highlight frame borders, sky and water, which would
  lead a human reviewer to the wrong conclusion.

## 6. Can explanations flag wrong detections? (`results/xai_verification/summary.md`)

Detections at conf ≥ 0.1. Logistic regression fit on valid (3,168 detections, 969 wrong), tested on
test (1,242, 418 wrong).

| Features | AUROC (wrong) | Wrong detections still passing at 90 % recall of correct ones |
|---|---|---|
| Confidence only | 0.900 | 31.1 % |
| XAI only | 0.790 | 51.2 % |
| Confidence + box area + XAI | **0.906** | **26.8 %** |

XAI features alone carry real signal (AUROC 0.79), but most of it overlaps with the confidence
score. Adding them gives a **small but statistically reliable gain**: +0.006 AUROC (95 % CI
[0.002, 0.010], bootstrap over images), i.e. about 14 % fewer false alarms passing at 90 % recall.
The explanation layer is mainly a **diagnostic and validation tool** (sections 2–5), not a large
accuracy booster.

## 7. Recommendation (RQ5)

| Use | Method | Why |
|---|---|---|
| On-device / app explanation | **LayerCAM, P3–P5 fused** | Near-occlusion faithfulness, best localization, passes the sanity check, stable, ~27 ms |
| Offline validation / audit | **Occlusion** (+ D-RISE) | Most faithful, model-agnostic reference |
| Not recommended alone | Eigen-CAM, single-scale CAMs | Below or near random on deletion; fail the head-randomization sanity check (Eigen-CAM, Grad-CAM++) |

## 8. Limitations
- Only correctly localized detections and background FPs could be explained. There were no
  snake↔look-alike confusions on test to study.
- Deletion/insertion depend on the replacement value (gray / blur). We used one fixed setting for
  all methods.
- D-RISE was evaluated on a 200-image subset and the diagnostics on 150 images (cost).
- One detector (YOLO11n) and one dataset.
