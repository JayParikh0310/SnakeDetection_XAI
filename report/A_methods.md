# Member A: Methods (draft for the report)

*Draft text. Numbers and figures come from `results/`. See `report/A_results.md` once the final
benchmark has run.*

## 1. Detector under study

The Stage-2 ("Detect") model is a YOLO11n detector (2.58 M parameters, 6.4 GFLOPs) trained for
30 epochs at 640 px on `final_dataset_v4`. The dataset has two classes: `snake` and
`snake_like_object` (ropes, roots, hoses, sticks, belts…). It has 20,294 / 2,172 / 841
train / valid / test images. Most objects are annotated with boxes; a subset is annotated with
polygons. The model is **frozen**: this work explains the model, it does not retrain it.

Inference follows Ultralytics exactly: aspect-preserving resize of the long side to 640 px and
minimal padding to a multiple of 32 ("rect" letterbox). The model gives 4+2 outputs per anchor
(box, two sigmoid class scores). NMS is per class (IoU 0.7). The operating threshold is conf 0.25.
Our wrapper reproduces Ultralytics' predictions exactly (verified by an automated test).

## 2. Explaining one detection: the target score

A detector outputs many boxes, so "the confidence" has to be defined per detection. For a detection
with box *b* and class *c*, we use the D-RISE similarity (Petsiuk et al., 2021):

  s(x) = max_i IoU(b_i(x), b) · p_i(c | x)

over all raw (pre-NMS) predictions *i* of the (possibly perturbed) image *x*. On the unperturbed
image, s equals the detection's confidence (tested). Under perturbation it follows the object even
if a neighbouring anchor takes over. The **same** score drives the gradients of the CAM methods, the
occlusion and D-RISE perturbations, and the deletion/insertion metric, so every method is judged by
the quantity it explains. A contrastive variant, s = IoU · (p(c) − p(other class)), explains
"snake *rather than* look-alike".

## 3. Explanation methods

| Method | Family | Implementation notes |
|---|---|---|
| **Eigen-CAM** (Muhammad & Yeasin 2020) | activation, gradient-free, class-agnostic | First principal component of the activations at P3/P4/P5 (the three neck outputs feeding the Detect head), upsampled, normalized per scale and averaged. The sign is oriented by correlation with activation energy. A centered variant and a single-scale P5 ablation are also included. |
| **Grad-CAM++** (Chattopadhay et al. 2018) | gradient, class-targeted | Gradients of s w.r.t. P3/P4/P5, closed-form α weights (exact formula, no ε), fused across scales as the proposal requires. Ablations: P3-, P4-, P5-only. Fusion rules: mean / max / sum. |
| Grad-CAM (Selvaraju et al. 2017) | gradient | Baseline for Grad-CAM++. |
| **LayerCAM** (Jiang et al. 2021) | gradient, per-location | Weights each location by its own positive gradient instead of one weight per channel. Designed for fusing shallow and deep layers. Added after we saw a border artifact in channel-weighted CAMs. |
| **Occlusion sensitivity** (Zeiler & Fergus 2014) | perturbation, model-agnostic | Gray (114) patches of 48 and 96 px at 50 % stride. Saliency = mean drop of s over the patches covering a pixel, averaged over both sizes. |
| D-RISE (Petsiuk et al. 2021) | perturbation, model-agnostic | 2,000 random 16×16-cell masks (p = 0.5), bilinear upsampling with random shifts, weighted by s, mean-weight debiased. Expensive, so evaluated on a subset. |
| Random | baseline | Smooth random blobs: the proposal's "random-order" reference. |
| Box prior | baseline | Gaussian centred on the detected box: does a method beat "just highlight the box"? |

SHAP and LIME were not used for the detector, following the proposal: they assume single-label
classification.

## 4. Evaluation protocol

All maps are evaluated in the network-input space. Only real-image pixels (not padding) are ranked
or perturbed.

* **Faithfulness: deletion / insertion** (Petsiuk et al. 2018). Pixels are removed (set to gray)
  or revealed (on a σ = 25 blurred canvas) in the order the map ranks them, in 50 steps. We report
  the AUC of s / s(original): deletion ↓, insertion ↑. We also report each method's win rate and
  mean gain against the random baseline on the **same** detection.
* **Localization:** pointing game (map maximum within 15 px of the GT), energy-based pointing game
  (share of saliency inside the GT), and IoU of the top-salient pixels with the GT. All are computed
  against the GT box and, where available, the stricter GT **polygon**.
* **Agreement:** Spearman rank correlation and top-20 % IoU between methods (the proposal's
  activation-vs-gradient cross-check).
* **Sanity:** cascading model-parameter randomization (Adebayo et al. 2018) of head / neck+head /
  all layers. A faithful method's map should change (low similarity).
* **Stability:** similarity of maps under small Gaussian input noise (σ = 0.02).
* **Cost:** wall-clock ms per explanation on an RTX 4060 Laptop GPU.

Explained detections are those at conf ≥ 0.25, matched class-agnostically to GT at IoU ≥ 0.5:
true positives, class confusions and background false positives (≤ 2 per image). Settings were
chosen on a 100-image validation subset. Final numbers are on the test split. Confidence
intervals are 95 % bootstrap. Methods are compared with paired Wilcoxon tests.

## 5. Matched snake / look-alike pairs

As the proposal asks, every look-alike detection (and every confusion or FP) is paired with the
real-snake detection that the **detector itself** finds most similar: the cosine similarity of
RoI-aligned P5 features. Both are shown side by side under all core methods. Seeing what differs
in the explanations of near-identical inputs makes errors diagnosable.

## 6. Explanation-based verification (validation layer)

To test whether explanations add information beyond the confidence score, we compute per detection
(conf ≥ 0.1): the share of saliency inside the box and whether the map's peak is inside the box (for
Eigen-CAM, Grad-CAM++, LayerCAM), plus Eigen-CAM–Grad-CAM++ agreement. A logistic regression fit on
valid predicts "detection is wrong" on test. It is compared with confidence alone (AUROC, and the
share of wrong detections that survive at 95 % / 90 % recall of correct ones).

## 7. Runtime pipeline

`src/snakeguard/pipeline.py` implements the proposal's stages around the detector:
cross-class conflict resolution with a safety bias (keep "snake" unless the look-alike is ≥ 0.15
more confident), a small-object filter, a 3-of-5 sliding-window temporal confirmation, proximity
from √(box area) (0 at 0.5 % of the frame, 1 at 25 %), and a threat score
100 · conf · (0.6 · proximity + 0.4 · P(venomous)). Unknown venom status counts as venomous until
Member B's classifier is plugged in. The threat score maps to a beep interval (1,500 → 150 ms). An
optional explanation of the most threatening detection carries an "explanation mass in box" check.
