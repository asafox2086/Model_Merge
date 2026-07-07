# LAMP-Merge Full-Scale Internal Ablation Summary

Experiment root: `outputs/lamp_merge_internal_ablation_full_20260707_full_internal_ablation_v5_tmux_fullscope`.

Coverage: five medical image datasets, four small-model backbones, three client counts, and three Dirichlet beta values. Each ablation therefore contains 180 raw cells and 60 client-average cells when complete. The client-average metric first averages the three beta values for each fixed dataset, backbone, and client count.

## Overall Summary

| Setting | Group | Raw cells | Raw mean Acc | Raw >= avg | Client-average cells | Client-average mean Acc | Client-average >= avg | Client-average >= LAMP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LAMP-Merge | Formal method | 180 | 0.5884 | 154/180 | 60 | 0.5884 | 53/60 | 60/60 |
| avg+M2 | Module-level | 25 | 0.1927 | 12/25 | 8 | 0.1931 | 3/8 | 0/8 |
| M1 only | Module-level | 180 | 0.5884 | 154/180 | 60 | 0.5884 | 53/60 | 60/60 |
| Global-feature mean | Prototype information | 180 | 0.0833 | 46/180 | 60 | 0.0833 | 7/60 | 0/60 |
| Classifier-head aggregation | Prototype information | 180 | 0.2378 | 98/180 | 60 | 0.2378 | 37/60 | 7/60 |
| Shuffled-label prototype | Prototype information | 180 | 0.1440 | 82/180 | 60 | 0.1440 | 24/60 | 0/60 |
| Support-only synthetic head | Prototype information | 180 | 0.0860 | 38/180 | 60 | 0.0860 | 6/60 | 0/60 |
| Binary support only | Statistical information | 180 | 0.5884 | 154/180 | 60 | 0.5884 | 53/60 | 47/60 |
| Global client-size weight | Statistical information | 180 | 0.5885 | 154/180 | 60 | 0.5885 | 53/60 | 43/60 |
| No prevalence calibration | Statistical information | 180 | 0.5884 | 154/180 | 60 | 0.5884 | 53/60 | 60/60 |
| Smoothed prevalence prior | Statistical information | 180 | 0.5884 | 154/180 | 60 | 0.5884 | 53/60 | 60/60 |
| Uniform client weight | Statistical information | 180 | 0.5884 | 154/180 | 60 | 0.5884 | 53/60 | 47/60 |
| Uniform prevalence prior | Statistical information | 180 | 0.5884 | 154/180 | 60 | 0.5884 | 53/60 | 60/60 |

## Dataset-Level Client Average

| Dataset | Cells | avg | LAMP-Merge | Classifier-head aggregation | Shuffled-label prototype | Uniform client weight | Global client-size weight | No prevalence calibration | Uniform prevalence prior |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bloodmnist_224 | 12 | 0.1714 | 0.8188 | 0.1849 | 0.2450 | 0.8188 | 0.8188 | 0.8188 | 0.8188 |
| chaoshengmnist_224 | 12 | 0.1564 | 0.4618 | 0.1598 | 0.1442 | 0.4618 | 0.4618 | 0.4618 | 0.4618 |
| dermamnist_224 | 12 | 0.5042 | 0.4619 | 0.5167 | 0.0976 | 0.4619 | 0.4623 | 0.4619 | 0.4619 |
| organcmnist_224 | 12 | 0.1286 | 0.6249 | 0.1674 | 0.1116 | 0.6249 | 0.6250 | 0.6249 | 0.6249 |
| organsmnist_224 | 12 | 0.1414 | 0.5744 | 0.1600 | 0.1217 | 0.5744 | 0.5745 | 0.5744 | 0.5744 |

## Client-Average Cell File

The complete per-cell table is written to `lamp_merge_internal_ablation_full_client_average.csv`. It includes all methods, the best method set for each `(dataset, backbone, K)` cell, and can be used directly for module-internal ablation tables in the paper.
