# my_merge Hyperparameter Sensitivity

This report is intentionally separate from the main result table. It is used to inspect whether diagnostic client weights and candidate selection are sensitive to validation batch counts and BN recalibration.

## Overall

| setting | stats | eval | bn | mean_acc | rows |
| --- | --- | --- | --- | --- | --- |
| stats1_eval1_bn0 | 1 | 1 | 0 | 0.4418 | 36 |
| stats2_eval1_bn0 | 2 | 1 | 0 | 0.4418 | 36 |
| stats4_eval2_bn1 | 4 | 2 | 1 | 0.4418 | 36 |
| stats4_eval4_bn2 | 4 | 4 | 2 | 0.4418 | 36 |

![overall](../figures/overall_hparam_sensitivity.png)

## Dataset Plots

- `bloodmnist_224`: ![](../figures/bloodmnist_224_hparam_sensitivity.png)
- `chaoshengmnist_224`: ![](../figures/chaoshengmnist_224_hparam_sensitivity.png)
- `dermamnist_224`: ![](../figures/dermamnist_224_hparam_sensitivity.png)
- `organcmnist_224`: ![](../figures/organcmnist_224_hparam_sensitivity.png)
