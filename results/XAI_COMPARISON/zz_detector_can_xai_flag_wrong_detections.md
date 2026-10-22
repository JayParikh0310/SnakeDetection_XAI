# Explanation-based verification (test split, model fit on valid)

Detections at conf >= 0.1: train 3168 (969 wrong), test 1242 (418 wrong)

wrong_kept_at_95rec = share of wrong detections that still pass when the threshold keeps 95% of correct ones (lower is better).

| features                |   AUROC_wrong |   wrong_kept_at_95rec |   wrong_kept_at_90rec |
|:------------------------|--------------:|----------------------:|----------------------:|
| confidence only         |         0.900 |                 0.526 |                 0.311 |
| confidence + area       |         0.900 |                 0.536 |                 0.311 |
| XAI only                |         0.790 |                 0.689 |                 0.512 |
| confidence + XAI        |         0.902 |                 0.574 |                 0.297 |
| confidence + area + XAI |         0.906 |                 0.519 |                 0.268 |

## Single features
| feature        |   AUROC_wrong (low=suspicious) |   mean_TP |   mean_wrong |
|:---------------|-------------------------------:|----------:|-------------:|
| score          |                          0.900 |     0.661 |        0.265 |
| area           |                          0.705 |     0.384 |        0.229 |
| agree_eig_gpp  |                          0.601 |     0.564 |        0.479 |
| mass_eigencam  |                          0.712 |     0.390 |        0.227 |
| mass_gradcampp |                          0.737 |     0.427 |        0.246 |
| mass_layercam  |                          0.613 |     0.927 |        0.865 |