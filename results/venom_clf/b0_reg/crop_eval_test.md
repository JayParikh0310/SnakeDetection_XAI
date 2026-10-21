# Venom classifier on detector crops (test)

600 sampled images (300 venomous); detector found a snake in 522; evaluated on those 522 (252 venomous). Threshold 0.1413 (valid, whole images). `p_cropM` = detector box grown by M x box size on each side.

| input      |   recall | recall_ci      |   specificity | specificity_ci   |   auroc | auroc_ci       |
|:-----------|---------:|:---------------|--------------:|:-----------------|--------:|:---------------|
| p_full     |    0.952 | [0.927, 0.979] |         0.596 | [0.538, 0.654]   |   0.900 | [0.873, 0.925] |
| p_crop0.0  |    0.917 | [0.880, 0.948] |         0.593 | [0.528, 0.653]   |   0.881 | [0.851, 0.909] |
| p_crop0.25 |    0.944 | [0.918, 0.970] |         0.615 | [0.555, 0.669]   |   0.902 | [0.876, 0.926] |
| p_crop0.5  |    0.948 | [0.921, 0.973] |         0.607 | [0.555, 0.663]   |   0.900 | [0.873, 0.924] |
| p_crop1.0  |    0.952 | [0.924, 0.977] |         0.604 | [0.546, 0.662]   |   0.903 | [0.876, 0.929] |