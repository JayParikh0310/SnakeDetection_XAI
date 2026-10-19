# Dataset collection: final_dataset_v4 (23,307 images)

The dataset was assembled from several public Roboflow Universe and Kaggle datasets, then merged,
relabelled into two classes (`snake`, `snake_like_object`) and renamed (`snake_XXXXXX`,
`snake_like_XXXXXX`). The renaming removed the original source names, so the composition below
was **reconstructed from the images themselves** (image-ID blocks with constant resolution, plus
visual inspection). The candidate sources are the closest public matches. Only the person who
built the dataset can confirm the exact ones: Roboflow keeps a record of every Universe project
cloned into the workspace.

## Composition

| Class | ID block (approx.) | ≈ Images | Content | Typical size | Likely source type |
|---|---|---|---|---|---|
| snake_like_object | 1 – 2,900 | ~2,900 | Belts (product photos, worn belts) | mixed | Roboflow belt dataset(s) |
| snake_like_object | 2,900 – 3,820 | ~820 | Fallen and uprooted trees, roots, branches | 416×416 | Roboflow fallen-tree dataset |
| snake_like_object | 3,970 – 5,120 | ~1,150 | Fire-hose cabinets, hose reels | 512×512 | Roboflow fire-hose / fire-cabinet dataset |
| snake_like_object | 5,120 – 6,290 | ~1,170 | Pipes, plumbing | 416×416 | Roboflow pipe-detection dataset |
| snake_like_object | 6,290 – 7,500 | ~1,220 | Mangrove roots in water (video frames) | 640×360 | Roboflow mangrove-roots dataset |
| snake_like_object | 7,500 – 8,100 | ~600 | Underwater pipelines; ropes / nooses (hanging) | mixed | Roboflow underwater-pipes + a rope/hanging dataset |
| snake | 8,098 – 21,690 | ~13,700 | Wild snakes, many species, natural scenes | max side 500 px (500×375, 375×500, 500×333) | iNaturalist-derived snake data (Kaggle / SnakeCLEF / AIcrowd style) |
| snake | 21,694 – 23,205 | 1,512 | Snakes (incl. ball pythons) | 1080×1080 | One Roboflow snake dataset |

Totals: 15,210 snake images and 8,097 snake-like images (train 20,294 / valid 2,172 / test 841).
Black borders, rotations and salt-and-pepper noise in many images are Roboflow augmentations.

## How the composition was worked out
- Image numbers form continuous blocks, and inside each block every image has the same
  resolution (416×416, 512×512, 640×360, 1080×1080). One block = one source dataset export.
- Contact sheets of images sampled across the ID range show one object type per block (belts →
  fallen trees → fire-hose cabinets → pipes → mangrove roots → underwater pipes / nooses).
- Snake images with a 500-px long side in natural scenes match iNaturalist-derived collections
  (the Kaggle part). The 1080×1080 block is a separate Roboflow snake export.
- Reproduce: `.venv/Scripts/python scripts/dataset_composition.py` prints the size blocks and writes
  contact sheets to `results/dataset_composition/`.

## What to tell the supervisor
- The **categories, counts and composition are solid**: they come straight from the data.
- The **source links below are the closest public matches**, not confirmed originals.
- **To get the exact sources:** the dataset's creator opens his Roboflow workspace (every cloned
  Universe project is listed there) and his Kaggle download history. Then replace the candidate
  links below with the confirmed ones.
- Check each licence before publishing.

## Closest public candidate sources (to verify)

**Snakes**
- SnakeCLEF 2023, iNaturalist + HerpMapper photos (the 500-px images match this style): https://www.imageclef.org/SnakeCLEF2023
- AIcrowd SNAKE challenge: https://www.aicrowd.com/challenges/snake
- Roboflow: snake-detection (2,478 imgs) https://universe.roboflow.com/aisnake/snake-detection-e37wm ·
  YOLO_SNAKE (1,483) https://universe.roboflow.com/yolosnake/yolo_snake ·
  Snake detection (1,371) https://universe.roboflow.com/earth-wls8u/snake-detection-rm3be ·
  Snake (4,618) https://universe.roboflow.com/krit-bm0ue/snake-pazqf ·
  Snake Detection https://universe.roboflow.com/snake-detection-supce/snake-detection-xpnuj

**Look-alikes**
- Belts: https://universe.roboflow.com/vit-c/belt-bqxya (732 imgs) · https://universe.roboflow.com/vit-c/belt-od0wv
- Fallen trees / roots: https://universe.roboflow.com/steve-chun/fallen-trees-1dfmg (classes: roots, treefall) ·
  https://universe.roboflow.com/visual-deformity/fallen-trees-pka1c
- Fire hose / cabinets: https://universe.roboflow.com/astoria-irbko/firehose-detection ·
  https://universe.roboflow.com/video-analytics-nyp1g/fire-hose · https://universe.roboflow.com/fire-eetbm/fire-cabinet-b8afv
- Pipes: https://universe.roboflow.com/archery-ocifk/pipe-detection-rtxrk (4,828) ·
  https://universe.roboflow.com/ritesh-pfyd9/pipe-detection-4crdp · https://universe.roboflow.com/nans/hose_pipe
- Mangrove roots: https://universe.roboflow.com/practice-hquy7/mangrove-roots ·
  https://universe.roboflow.com/lab-6dprv/mangrove-detector-2-0
- Underwater pipes (part of Roboflow's RF100 benchmark): https://universe.roboflow.com/object-detection/underwater-pipes-4ng4t
- Rope / hanging: https://universe.roboflow.com/abnormalbehaviordetect/hang-detection (544) ·
  https://universe.roboflow.com/ningbo-university/rope_real · https://universe.roboflow.com/data-ibvpv/rope-qfqwn

Most of these pages show CC BY 4.0 (fallen-trees by steve-chun: Public Domain; pipe-detection-rtxrk:
MIT). Not every page was checked, so confirm each licence before publishing.
