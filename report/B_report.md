# Member B: report sections (draft) — venom classification, its explainability, and the demo application

*Draft text for the final report (B6). It complements Member A's Review-2 document (`report/SnakeGuard_Review2.docx`:
problem context, objectives, detection dataset, detector methodology) and `report/A_methods.md` / `A_results.md`.
All numbers come from scripts in this project; the full result tables are in `report/B_results.md`. The one item
marked **[check]** needs a primary source before submission (references verified 2026-10-21).*

---

## 1. Introduction: why the device needs a venom stage

A detector that answers "snake or rope?" is necessary but not sufficient for an early-warning device. Most snakes a
farmer meets are harmless, and an alarm that sounds the same for every snake is soon ignored. The clinically relevant
question is whether the snake belongs to a **medically important venomous** species: one whose bite can cause death
or lasting disability and may require antivenom (WHO, 2019; Gutiérrez et al., 2017). Lay people are poor at telling
venomous from harmless snakes, and this is a recurring obstacle in snakebite care (Bolon et al., 2020).

SnakeGuard therefore adds a **Classify** stage after Detect: every confirmed snake crop is passed to a venom classifier
whose output raises or lowers the threat score and the beep rate. This section of the work (Member B) has four parts:

1. a species-labelled dataset with a venom label and leakage-free splits (B1);
2. a venom classifier tuned for the asymmetric cost of errors, i.e. for **venomous recall** (B2);
3. explanation methods for the classifier, evaluated with the *same* quantitative protocol as the detector (B3, B4);
4. a demonstration application that runs the whole pipeline and presents the results responsibly (B5).

Because a wrong "harmless" verdict is the most dangerous failure of the whole system, the design choices below are
driven by safety first: missing a venomous snake is treated as far worse than a false alarm, unknown status is treated
as venomous, and the application never presents its output as an identification or as medical advice.

## 2. Literature review (venom stage)

### 2.1 Snake identification and venom status from photographs

Image-based snake identification has moved from small, hand-curated datasets to large citizen-science collections.
Durso et al. (2021) released a benchmark of species-labelled photographs from online sources and argued that
automated identification matters both for herpetology and for global health, because treatment and antivenom choice
depend on knowing what bit the patient. This line of work became the **SnakeCLEF** challenge in the LifeCLEF lab.
The 2023 edition (Picek et al., 2023) provides photographs of 1,784 species from iNaturalist and HerpMapper, a
country-to-species mapping, and a per-species flag for **medically important venomous snakes (MIVS)**; its scoring
penalises the confusion of a venomous species with a harmless one more than other errors. The 2024 edition kept the
development data and extended the test set with new regions.

Four of the eight systems reviewed in the project proposal (its literature matrix, References 1–8) address the
identification side of the problem:

| System | Data | Method | Reported result | Limitation relevant here |
|---|---|---|---|---|
| Shivaraj et al. (2024), Ref 1 | ~1,000 snake images | YOLOv8 on video, WhatsApp alerts | all-class F1 ≈ 0.65; separates venomous / non-venomous | snake-only data, single-frame, no threat scoring |
| VenoScan, Prasad et al. (2025), Ref 2 | Kaggle "Snake Dataset-India", 2,043 images, 4 snake types | fine-tuned DenseNet121, web app | accuracy reported inconsistently: 98.73 % (abstract) vs 82.73–90 % (results); recall 0.86 / 0.93 | static images only, 4 types, no detection stage |
| Dube & Bhuru (2022), Ref 3 | 1,500 images, 8 Southern African species | YOLOv5 → TFLite Android app | 71 % accuracy, mAP@0.5 0.713; confuses similar species (mole snake vs night adder) | small dataset (~150 images per class) |
| Serpent-ID, Bharath et al. (2026), Ref 5 | 12,000 images, 35 medically significant Indian species (SnakeCLEF 2023 + public sources) | quantized YOLOv8-nano, Flutter app with first aid and hospital locator | mAP@0.5 94.5 %, 94 % species ID (n = 200) | accuracy drops below ~100 training images per species; no independent venom stage |

These results are not directly comparable with ours: they cover 4 to 35 species and report accuracy or mAP, while we
classify venom status over 1,784 species and report recall at a fixed safety operating point. Serpent-ID's observation
that accuracy falls for species with fewer than ~100 images matches our setting (at most 25 images per harmless species,
100 per venomous one) and our per-species errors (Section 4). Three gaps recur and motivate the design of this stage:

- **Accuracy is reported, the cost asymmetry is not.** Accuracy, F1 or mAP treat a missed cobra like a misnamed rat
  snake. For a warning device the operating point must be chosen for venomous recall, and the price paid in false
  alarms must be reported. None of the four systems does this; VenoScan's inconsistent figures show how fragile a
  single headline accuracy can be.
- **Narrow species coverage.** Four to 35 species is far from the diversity a device meets in the field; species not
  in training are silently forced into one of the known classes.
- **No validation of what the model attends to.** As the proposal notes for all eight systems, none examines whether
  the model looks at the snake (head shape, pattern) or at background cues, so its errors remain unexplained.

### 2.2 Explainability for image classifiers

The project's detector stage set SHAP and LIME aside because they assume single-label classification. The venom
classifier *is* a single-label classifier, so the full range of methods applies:

- **Class activation maps.** Grad-CAM weights feature channels by their mean gradient (Selvaraju et al., 2017);
  Grad-CAM++ uses higher-order weights for multiple object parts (Chattopadhay et al., 2018); LayerCAM weights each
  location by its own positive gradient, which suits fusing shallow and deep layers (Jiang et al., 2021); Eigen-CAM
  uses the first principal component of the activations and needs no gradients (Muhammad & Yeasin, 2020).
- **Path gradients.** Integrated Gradients accumulates gradients along a path from a baseline image to the input and
  satisfies completeness and sensitivity axioms (Sundararajan et al., 2017). Its result depends on the baseline
  (Sturmfels et al., 2020).
- **Perturbation and surrogate models.** Occlusion (Zeiler & Fergus, 2014) and RISE (Petsiuk et al., 2018) measure
  the score change when regions are hidden. LIME fits a local weighted linear surrogate on random perturbations
  (Ribeiro et al., 2016); KernelSHAP estimates Shapley values with a weighted linear regression (Lundberg & Lee, 2017).

**Evaluating explanations.** Visual plausibility is not evidence of faithfulness. Deletion and insertion curves
(Petsiuk et al., 2018) test whether the pixels ranked important actually drive the score; the model-parameter
randomisation test (Adebayo et al., 2018) checks that a map changes when the model's decision layers are destroyed;
and removal-based metrics are known to depend on how "removed" pixels are filled (Hooker et al., 2019). Section 5
shows that this last point is not academic for a venom classifier: a blank image is already classified "harmless",
which makes deletion uninformative for harmless decisions.

### 2.3 What this work adds

1. A venom classifier evaluated **at a safety operating point** (venomous recall ≥ 95 %, chosen on validation), with
   confidence intervals and per-species results, including India's "Big Four".
2. **The same quantitative XAI protocol for both stages** (detector and classifier), with the same code for the CAMs,
   occlusion and RISE, so the methods can be compared across stages.
3. A **deployment check** that the classifier still works on what the pipeline actually feeds it (detector crops),
   which uncovered and fixed a train/deploy mismatch.

## 3. Venom dataset (B1)

**Source.** SnakeCLEF 2023 development data (Picek et al., 2023), 240-pixel ("small") release, with the MIVS flag as the
label: 285 of 1,784 species are medically important venomous. The label is deliberately MIVS rather than "has venom":
mildly venomous rear-fanged species count as harmless, which matches the question a bite-warning device must answer.
The full release did not fit the 20 GB of free disk on Member B's machine, so the images were streamed from the
archives and only a sampled subset was stored (`scripts/build_venom_dataset.py`).

**Sampling.** Per species, whole observations (all photos of one encounter) are drawn in a seeded random order until
a cap is reached: 100 images for venomous species and 25 for harmless ones. Every species is kept, and the venomous
share rises from 18.9 % (SnakeCLEF training metadata) to about 37 %, which gives the minority class more weight. Because this changes
the class prior, the decision threshold is calibrated on validation rather than fixed at 0.5.

**Splits.** Within each species, observations are split 70 / 15 / 15 into train / validation / test, so photographs
of the same encounter never appear in two splits. 199 images listed in the metadata are absent from the 240-pixel
archive (0.4 %); they are kept in the manifest as unavailable rather than resampled.

| Split | Images | Venomous | Species | Observations |
|---|---|---|---|---|
| train | 34,838 | 38.0 % | 1,784 | 20,108 |
| validation | 7,841 | 37.2 % | 1,761 | 4,430 |
| test | 7,762 | 36.1 % | 1,761 | 4,430 |

**Known biases.** The images are citizen-science photographs, mostly taken from standing height by people who
approached the snake, and they are geographically skewed: the largest training sources are Mexico, Australia, Brazil
and the USA, with India fifth among known countries (1,468 training images, 4.2 %; a further 1,806 have no country). Species with few observations have few images.

## 4. Venom classifier (B2)

**Model.** EfficientNet-B0 (Tan & Le, 2019) pretrained on ImageNet, with two heads on the pooled features: a binary
venom head and an auxiliary 1,784-way species head (loss weight 0.5). EfficientNet-B0 is small enough for the
wearable or a paired phone, trains in about two hours on an Apple M4 (114 minutes for 10 epochs), and is a plain CNN, so the CAM methods apply
directly. Input 224 pixels; training uses random resized crops covering 25–100 % of the image (which also imitates
detector crops), flips and mild colour jitter.

**Venom score.** Because venom status is a property of the species, P(venomous) can also be read off the species head
as the summed probability of all venomous species. On validation this species-head score beat the dedicated venom head
(AUROC 0.882 vs 0.869; paired-bootstrap difference +0.013 [+0.009, +0.018]).

**Regularisation.** The first model overfit (training venom loss 0.13 against a flat validation AUROC after epoch 6).
A regularised run (dropout 0.5, weight decay 0.1, label smoothing 0.1 on the venom target, stronger crops) improved
every validation metric. Model selection compared six candidates (two runs × venom head / species head / their mean)
on validation only; the regularised run with the species-head score was best (AUROC 0.891, specificity 0.564 at 95 %
recall).

**Operating point.** The threshold is the highest score at which **at least 95 % of venomous validation images** are
flagged (0.1413). On the test split, evaluated once:

| AUROC | Venomous recall | Specificity | Precision |
|---|---|---|---|
| 0.888 [0.878, 0.898] | 0.953 [0.944, 0.962] | 0.565 [0.547, 0.583] | 0.553 |

The cost of the safety operating point is explicit: **43.5 % of harmless snakes are flagged as venomous.** Section 7
discusses what this means for users.

**India's "Big Four"** (`scripts/venom_species_report.py`, test, deployed threshold; small samples, Wilson 95 % CIs):

| Species | Train images | Test images | Flagged venomous |
|---|---|---|---|
| Spectacled cobra (*Naja naja*) | 61 | 25 | 0.96 [0.80, 0.99] |
| Saw-scaled viper (*Echis carinatus*) | 74 | 15 | 0.93 [0.70, 0.99] |
| Russell's viper (*Daboia russelii*) | 61 | 20 | 0.80 [0.58, 0.92] |
| Common krait (*Bungarus caeruleus*) | 68 | 16 | **0.75 [0.51, 0.90]** |

The common krait is one of the four species held responsible for most venomous bites in India, and it typically
bites at night, often people sleeping on the floor **[check: cite the official MoHFW National Snakebite Management
Protocol for the Big Four statement; only secondary summaries were reachable on 2026-10-21]**. It is the weakest of the four here. A regional deployment would need more krait images and
a region-specific threshold (Section 8).

**From photographs to detector crops.** In the pipeline the classifier receives the detector's snake box, not a
whole photograph. On 522 test images with a detected snake, venomous recall falls from 0.952 on the whole image to
**0.917** on the tight box; enlarging the box by 25 % on each side restores 0.944 (`scripts/eval_venom_on_crops.py`).
The pipeline therefore gained an optional crop margin (`PipelineConfig.venom_crop_margin`), which the application sets
to 0.25.

## 5. Explaining the venom classifier (B3, B4)

**Methods.** Eleven methods behind the project's shared `Explainer` interface, all explaining the deployed score:
Eigen-CAM, Grad-CAM and Grad-CAM++ on the last stage (7×7); Grad-CAM++ and LayerCAM fused over three stages (28, 14
and 7 pixels, the classifier analogue of the detector's P3–P5 fusion); Integrated Gradients; occlusion; RISE; LIME and
KernelSHAP on an 8×8 patch grid; and a random baseline. The CAM arithmetic, occlusion and RISE reuse Member A's code
unchanged, so differences between the stages are differences between the models, not between implementations.

**Protocol.** Member A's metric code, applied to the 224-pixel crop the classifier sees: deletion and insertion AUC
(with win rates against the random map on the same image), cascading weight randomisation (heads → top stages → all),
stability under input noise, pairwise agreement, and runtime. SnakeCLEF has no object masks, so localisation uses a
**proxy ground truth**: the frozen detector's snake box mapped into the crop. Test sample: 80 confident venomous
detections (TP), 80 harmless snakes flagged venomous (FP), 20 missed venomous (FN), 20 cleared harmless (TN).

**A methodological finding.** With no visual evidence, the classifier says "harmless": an all-grey image scores
P(venomous) = 0.143, right at the threshold. Deleting the evidence for a *harmless* decision therefore does not lower
its score, and deletion/insertion cannot measure faithfulness for harmless decisions; flagged harmless snakes (FP) sit
close to the same point. Faithfulness is therefore reported on confident venomous detections, which are also the
alerts the application explains. This is a concrete instance of the baseline dependence discussed by Sturmfels et al.
(2020) and Hooker et al. (2019).

**Summary of results** (details in `report/B_results.md` §4):

- Every method beats the random baseline on venomous detections, but by smaller margins than on the detector,
  because the classifier is often near-certain.
- **Multi-scale fusion helps again**: fused Grad-CAM++ and LayerCAM beat last-stage Grad-CAM++ on deletion
  (p < 1e-4), mirroring the detector result.
- **LayerCAM (multi-scale) beats occlusion** on both deletion and insertion at 1/25 of the cost, and keeps 85 % of
  its saliency on the snake.
- **Sanity:** only plain Grad-CAM passes the head-randomisation test; Eigen-CAM (by construction) and last-stage
  Grad-CAM++ fail it, and LayerCAM and Integrated Gradients change only partly. On this classifier the maps mostly show
  *where the snake is*, less *why it was judged venomous*.
- **Agreement** inside the CAM family is high and agreement with perturbation methods is low, as on the detector, so
  agreement between CAMs is no evidence of correctness.

**Choice for the application:** LayerCAM fused over three stages, the same method Member A recommends for the
detector, so the app shows one consistent kind of map. Its limitation (partial sanity) is stated in the app's text.

## 6. Demonstration application (B5)

**Architecture.** A Gradio web application (`scripts/app.py`) on top of a Gradio-free logic module
(`src/snakeguard/demo.py`, unit-tested). Both run Member A's `SnakeGuard` pipeline with the venom classifier plugged in;
the application adds no decision logic of its own. It runs locally (`127.0.0.1`); a public link is off by default.

| Tab | What it does |
|---|---|
| **Photo** | Runs one frame with temporal confirmation off (a single photo cannot satisfy 3-of-5). Shows the boxes (red = snake, blue = look-alike), a plain-language verdict, a table of detector confidence, venom score, venom status, proximity and threat, the beep interval, and two explanations: *why the detector fired* (detector LayerCAM, with the share of the map inside the box) and *why this venom call* (classifier LayerCAM on the same enlarged crop the classifier saw). *Audit mode* swaps in occlusion and RISE. |
| **Video** | Processes about five frames per second (first 30 s) with the real 3-of-5 confirmation, and returns an annotated video with a status banner plus a timeline of detections, confirmation and threat. |

**Fitting the classifier to the pipeline.** Two integration problems were found by running the application and
fixed with measurements, not by tuning on test:

1. *Crop context* (Section 4): the classifier sees the detector box enlarged by 25 % per side.
2. *Threat scale*: the pipeline's threat score multiplies P(venomous), but the venom threshold is 0.141, so a snake
   just above the threshold added only a few points of threat. The application uses a monotone rescaling of the venom
   score that maps the threshold to 0.5. Rankings, and therefore every reported metric, are unchanged. After both
   fixes, a test-set cantil (*Agkistrodon bilineatus*) that the application had called "likely harmless" (threat 50) is
   reported as "possibly venomous" (threat 66).

**Wording.** Each verdict leads with the action, not the identification: "possibly venomous … Move away", "venom status
unknown, treated as venomous", or "likely a harmless species, but identification can be wrong … Keep your distance".
Every result ends with a fixed notice that the system is a research prototype and not a substitute for expert
identification or medical advice. Figure `results/app/app_examples.png` shows four test images, including a correctly
cleared coral-snake mimic (*Rhinobothryum lentiginosum*) and a harmless snail-eater (*Dipsas turgida*) wrongly flagged
venomous.

## 7. Ethics and safety

**Asymmetric errors and the chosen operating point.** A missed venomous snake can kill; a false alarm costs attention.
The threshold therefore targets 95 % venomous recall, and the price is reported, not hidden: 43.5 % of harmless snakes
are flagged venomous. The residual risk remains real: about 1 in 20 venomous snakes in the test set, and 1 in 4 common
kraits, are not flagged. The device must never imply that "no venom alert" means "safe". The application keeps a
non-zero threat and the advice "keep your distance" for every snake, and unknown venom status (e.g. a crop too small to
judge) is treated as venomous, as required by the project rules.

**Harm to snakes.** A false "venomous" label can encourage people to kill a harmless snake, many of which are
ecologically valuable and some of which are protected. The application therefore advises moving away, never
confronting or killing the animal, and flags its own uncertainty.

**Not a medical device.** Snakebite treatment is decided by clinicians from the patient's signs and local protocols;
a wrong species label could delay or misdirect care. The application states that it is not a substitute for expert
identification or medical advice, and that every bite needs emergency care regardless of the system's output. The
system has not been clinically or field validated.

**Explanations can mislead.** A heatmap looks authoritative. Our own sanity results show that the chosen explanation
mostly reflects where the snake is, not why it was judged venomous. The application says so ("shows where the model
looked, not proof of the species"), and the audit mode exposes the slower, model-agnostic methods for a cross-check.

**Bias and fairness across regions.** The training data are dominated by the Americas and Australia; India, where
an estimated 58,000 people die of snakebite every year (Suraweera et al., 2020), provides 4.2 % of training images. Performance varies by
species, and the weakest of the Big Four is the krait. Results should be reported per region and per medically
important species before any deployment, and the threshold set for the target region.

**Data provenance and licences.** SnakeCLEF images come from iNaturalist and HerpMapper contributors, who choose their
own licences. The SnakeCLEF 2023 task pages state no separate dataset licence (checked 2026-10-21), so the images
remain under their contributors' terms. The images are used for research only and are not
redistributed: the repository contains only the sampling manifest, metrics and derived figures; the images stay on the
member's machine. Only the country code from the metadata is used; no precise locations are processed.

**Privacy.** An ankle camera records whoever is in front of the wearer. The application processes frames locally and
does not keep them beyond the session (uploaded videos are written to the operating system's temporary folder for
processing). A deployed device should process on the device, store no frames by default, and not transmit images.

**Accessibility and over-reliance.** The beep is glanceless by design. The heatmaps use a rainbow colour map that is
hard to read with some forms of colour-vision deficiency; a perceptually uniform map would be better. Any warning
system invites over-reliance: users may relax when it is silent. The prototype's text and the "keep your distance"
advice for every snake are a first mitigation; field studies would be needed to measure the effect.

## 8. Limitations and future work

- **Image resolution and viewpoint.** 240-pixel photographs taken from standing height, unlike low-angle, close-range,
  sometimes blurred frames from an ankle camera. Future work: collect or simulate ankle-view data; use the 500-pixel
  release (needs about 19 GB of disk).
- **Regional calibration.** One global threshold; a deployment should restrict the species list to the region (the
  SnakeCLEF country mapping supports this) and recalibrate the threshold on regional validation data.
- **Rare species.** Harmless species have at most 25 training images; species top-1 accuracy is 17 %. Larger backbones
  and more images per species would help both scores.
- **Harmless decisions cannot be explained faithfully** with deletion/insertion, because "harmless" is the model's
  no-evidence default. A contrastive target (venomous vs. harmless evidence) or a different baseline image could make
  them testable.
- **Proxy localisation.** The detector box marks the snake, not the venom-relevant features (head shape, pupils,
  banding); part-level annotations would allow a sharper test.
- **No field or user evaluation.** Latency on the target hardware, battery use and the effect of the alerts on user
  behaviour remain untested.

## References (venom stage; detector references are in Member A's document)

- Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M. & Kim, B. (2018). Sanity checks for saliency maps. *NeurIPS*.
- Bharath et al. (2026). Serpent-ID: A deep learning-based mobile application for rapid snake identification and bite management guidance. *ICSFT* (reviewed in the proposal, Ref 5).
- Bolon, I., Durso, A. M., Botero Mesa, S., Ray, N., Alcoba, G., Chappuis, F. & Ruiz de Castañeda, R. (2020). Identifying the snake: First scoping review on practices of communities and healthcare providers confronted with snakebite across the world. *PLoS ONE* 15(3): e0229989. https://doi.org/10.1371/journal.pone.0229989
- Chattopadhay, A., Sarkar, A., Howlader, P. & Balasubramanian, V. N. (2018). Grad-CAM++: Generalized gradient-based visual explanations for deep convolutional networks. *WACV*.
- Dube & Bhuru (2022). Snake identification system using convolutional neural networks. *ZCICT* (reviewed in the proposal, Ref 3).
- Durso, A. M., Moorthy, G. K., Mohanty, S. P., Bolon, I., Salathé, M. & Ruiz de Castañeda, R. (2021). Supervised learning computer vision benchmark for snake species identification from photographs: Implications for herpetology and global health. *Frontiers in Artificial Intelligence* 4: 582110.
- Gutiérrez, J. M., Calvete, J. J., Habib, A. G., Harrison, R. A., Williams, D. J. & Warrell, D. A. (2017). Snakebite envenoming. *Nature Reviews Disease Primers* 3: 17063.
- Hooker, S., Erhan, D., Kindermans, P.-J. & Kim, B. (2019). A benchmark for interpretability methods in deep neural networks. *NeurIPS*.
- Jiang, P.-T., Zhang, C.-B., Hou, Q., Cheng, M.-M. & Wei, Y. (2021). LayerCAM: Exploring hierarchical class activation maps for localization. *IEEE Transactions on Image Processing* 30.
- Lundberg, S. M. & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
- Muhammad, M. B. & Yeasin, M. (2020). Eigen-CAM: Class activation map using principal components. *IJCNN*.
- Petsiuk, V., Das, A. & Saenko, K. (2018). RISE: Randomized input sampling for explanation of black-box models. *BMVC*.
- Picek, L., Hrúz, M., Durso, A. M. & Bolon, I. (2023). Overview of SnakeCLEF 2023: Snake identification in medically important scenarios. *CLEF 2023 Working Notes*, CEUR-WS.
- Prasad et al. (2025). VenoScan: AI-powered classification of venomous and non-venomous snakes. *ISACC* (reviewed in the proposal, Ref 2).
- Ribeiro, M. T., Singh, S. & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *KDD*.
- Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D. & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV*.
- Suraweera, W., Warrell, D., Whitaker, R., Menon, G., Rodrigues, R., Fu, S. H., Begum, R., Sati, P., Piyasena, K., Bhatia, M., Brown, P. & Jha, P. (2020). Trends in snakebite deaths in India from 2000 to 2019 in a nationally representative mortality study. *eLife* 9: e54076. https://doi.org/10.7554/eLife.54076
- Shivaraj et al. (2024). Real-time snake detection and alert system. *IJERT* 13(5) (reviewed in the proposal, Ref 1).
- Sturmfels, P., Lundberg, S. & Lee, S.-I. (2020). Visualizing the impact of feature attribution baselines. *Distill*.
- Sundararajan, M., Taly, A. & Yan, Q. (2017). Axiomatic attribution for deep networks. *ICML*.
- Tan, M. & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *ICML*.
- World Health Organization (2019). *Snakebite envenoming: a strategy for prevention and control.* Geneva: WHO.
- Zeiler, M. D. & Fergus, R. (2014). Visualizing and understanding convolutional networks. *ECCV*.
