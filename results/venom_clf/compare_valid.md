# Venom score comparison (valid, n = 7841, 4430 observations)

Reference: `b0:venom`. `*_diff_ci` = 95 % bootstrap CI of the paired difference vs the reference (1000 resamples over observations). spec@rec95 = specificity at venomous recall >= 0.95.

| candidate      |   auroc | auroc_diff_ci    |    ap |   spec@rec95 | spec@rec95_diff_ci   |   spec@rec90 |
|:---------------|--------:|:-----------------|------:|-------------:|:---------------------|-------------:|
| b0:venom       |   0.869 | [+0.000, +0.000] | 0.798 |        0.502 | [+0.000, +0.000]     |        0.634 |
| b0:species     |   0.882 | [+0.009, +0.018] | 0.820 |        0.541 | [+0.008, +0.060]     |        0.653 |
| b0:mean        |   0.880 | [+0.009, +0.013] | 0.816 |        0.534 | [+0.010, +0.052]     |        0.651 |
| b0_reg:venom   |   0.875 | [+0.001, +0.011] | 0.802 |        0.518 | [-0.016, +0.046]     |        0.661 |
| b0_reg:species |   0.891 | [+0.017, +0.029] | 0.829 |        0.564 | [+0.032, +0.089]     |        0.698 |
| b0_reg:mean    |   0.888 | [+0.014, +0.024] | 0.823 |        0.549 | [+0.018, +0.082]     |        0.696 |