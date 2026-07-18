# LAMP-Merge Full-Scale Internal Ablation Summary

Experiment root: `/data2/liyapeng_grp/program/MedMNISTMerge/outputs/lamp_merge_internal_ablation_full_20260718_formal_gamma055_s18p75_tau2p5_lambda4p25`.

Coverage: five medical image datasets, four small-model backbones, three client counts, and three Dirichlet beta values. Each ablation therefore contains 180 raw cells and 60 client-average cells when complete. The client-average metric first averages the three beta values for each fixed dataset, backbone, and client count.

This table is a module-internal ablation. Therefore, each replacement is evaluated against the final LAMP-Merge implementation rather than against `avg`. The `avg` baseline is used in the main comparison table, not as the decision criterion for internal component validity.

## Overall Summary

| Setting | Group | Raw cells | Raw mean Acc | Raw >= LAMP | Client-average cells | Client-average mean Acc | Client-average >= LAMP | Mean margin vs LAMP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LAMP-Merge | Formal method | 180 | 0.6221 | 180/180 | 60 | 0.6221 | 60/60 | 0.0000 |
| avg+M2 | Module-level | 180 | 0.2918 | 27/180 | 60 | 0.2918 | 9/60 | -0.3303 |
| M1 only | Module-level | 180 | 0.5891 | 90/180 | 60 | 0.5891 | 30/60 | -0.0329 |
| Global-feature mean | Prototype information | 180 | 0.2650 | 27/180 | 60 | 0.2650 | 9/60 | -0.3570 |
| Classifier-head aggregation | Prototype information | 180 | 0.2563 | 11/180 | 60 | 0.2563 | 2/60 | -0.3658 |
| Shuffled-label prototype | Prototype information | 180 | 0.1957 | 0/180 | 60 | 0.1957 | 0/60 | -0.4263 |
| Support-only synthetic head | Prototype information | 180 | 0.1089 | 0/180 | 60 | 0.1089 | 0/60 | -0.5132 |
| Binary support only | Statistical information | 180 | 0.5518 | 45/180 | 60 | 0.5518 | 0/60 | -0.0703 |
| Global client-size weight | Statistical information | 180 | 0.5939 | 86/180 | 60 | 0.5939 | 4/60 | -0.0281 |
| No prevalence calibration | Statistical information | 180 | 0.5891 | 90/180 | 60 | 0.5891 | 30/60 | -0.0329 |
| Smoothed prevalence prior | Statistical information | 180 | 0.6220 | 135/180 | 60 | 0.6220 | 41/60 | -0.0001 |
| Uniform client weight | Statistical information | 180 | 0.5833 | 82/180 | 60 | 0.5833 | 3/60 | -0.0388 |
| Uniform prevalence prior | Statistical information | 180 | 0.5891 | 90/180 | 60 | 0.5891 | 30/60 | -0.0329 |

## Dataset-Level Client Average

| Dataset | Cells | LAMP-Merge | Classifier-head aggregation margin | Shuffled-label prototype margin | Global-feature mean margin | Support-only synthetic head margin | Uniform client weight margin | Global client-size weight margin | No prevalence calibration margin | Uniform prevalence prior margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bloodmnist_224 | 12 | 0.8181 | -0.6286 | -0.5735 | -0.7293 | -0.7019 | -0.0378 | -0.0336 | 0.0000 | 0.0000 |
| chaoshengmnist_224 | 12 | 0.4611 | -0.2968 | -0.3186 | -0.3524 | -0.3541 | -0.0432 | -0.0297 | 0.0000 | 0.0000 |
| dermamnist_224 | 12 | 0.6240 | -0.0626 | -0.3367 | 0.0448 | -0.4511 | -0.0279 | 0.0008 | -0.1582 | -0.1582 |
| organcmnist_224 | 12 | 0.6305 | -0.4531 | -0.4887 | -0.4071 | -0.5596 | -0.0469 | -0.0394 | -0.0048 | -0.0048 |
| organsmnist_224 | 12 | 0.5767 | -0.3879 | -0.4142 | -0.3413 | -0.4993 | -0.0381 | -0.0388 | -0.0017 | -0.0017 |

## Client-Average Cell File

The complete per-cell table is written to `lamp_merge_internal_ablation_full_client_average.csv`. It includes all methods, the best method set for each `(dataset, backbone, K)` cell, and can be used directly for module-internal ablation tables in the paper.
