# Per-species venom results (test, b0_reg, score `p_venom_species`, threshold 0.1413)

Recall = share of the species' test images flagged venomous. Wilson 95 % CI; images of one observation are correlated, so the CI is optimistic.

| species            | common name      |   venomous (MIVS) |   train images |   test images |   caught |   recall | 95% CI       |   median score |
|:-------------------|:-----------------|------------------:|---------------:|--------------:|---------:|---------:|:-------------|---------------:|
| Naja naja          | spectacled cobra |                 1 |             61 |            25 |       24 |     0.96 | [0.80, 0.99] |           0.74 |
| Bungarus caeruleus | common krait     |                 1 |             68 |            16 |       12 |     0.75 | [0.51, 0.90] |           0.38 |
| Daboia russelii    | Russell's viper  |                 1 |             61 |            20 |       16 |     0.80 | [0.58, 0.92] |           0.48 |
| Echis carinatus    | saw-scaled viper |                 1 |             74 |            15 |       14 |     0.93 | [0.70, 0.99] |           0.83 |