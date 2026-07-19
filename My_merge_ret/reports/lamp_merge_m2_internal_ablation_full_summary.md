# LAMP-Merge Full-Scale Internal Ablation Summary

Experiment root: `/data2/liyapeng_grp/program/MedMNISTMerge/outputs/lamp_merge_m2_internal_ablation_full_20260719_aoc_cbp`.

Coverage: five medical image datasets, four small-model backbones, three client counts, and three Dirichlet beta values. Each ablation therefore contains 180 raw cells and 60 client-average cells when complete. The client-average metric first averages the three beta values for each fixed dataset, backbone, and client count.

This table is a module-internal ablation. Therefore, each replacement is evaluated against the final LAMP-Merge implementation rather than against `avg`. The `avg` baseline is used in the main comparison table, not as the decision criterion for internal component validity.

## Overall Summary

| Setting | Group | Raw cells | Raw mean Acc | Raw >= LAMP | Client-average cells | Client-average mean Acc | Client-average >= LAMP | Mean margin vs LAMP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LAMP-Merge | Formal method | 180 | 0.6221 | 180/180 | 60 | 0.6221 | 60/60 | 0.0000 |
| Always-on calibration | M2 internal | 180 | 0.6190 | 141/180 | 60 | 0.6190 | 47/60 | -0.0031 |
| Client-balanced prior | M2 internal | 180 | 0.5897 | 86/180 | 60 | 0.5897 | 17/60 | -0.0324 |

## Dataset-Level Client Average

| Dataset | Cells | LAMP-Merge | Classifier-head aggregation margin | Shuffled-label prototype margin | Global-feature mean margin | Support-only synthetic head margin | Uniform client weight margin | Global client-size weight margin | No prevalence calibration margin | Uniform prevalence prior margin | Always-on calibration margin | Client-balanced prior margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bloodmnist_224 | 12 | 0.8181 | - | - | - | - | - | - | - | - | -0.0124 | -0.0056 |
| chaoshengmnist_224 | 12 | 0.4611 | - | - | - | - | - | - | - | - | -0.0031 | -0.0134 |
| dermamnist_224 | 12 | 0.6240 | - | - | - | - | - | - | - | - | 0.0000 | -0.1279 |
| organcmnist_224 | 12 | 0.6305 | - | - | - | - | - | - | - | - | 0.0000 | -0.0094 |
| organsmnist_224 | 12 | 0.5767 | - | - | - | - | - | - | - | - | 0.0000 | -0.0055 |

## Client-Average Cell File

The complete per-cell table is written to `lamp_merge_internal_ablation_full_client_average.csv`. It includes all methods, the best method set for each `(dataset, backbone, K)` cell, and can be used directly for module-internal ablation tables in the paper.
