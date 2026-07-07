# LAMP-Merge Full-Scale Internal Ablation Summary

Experiment root: `/data2/liyapeng_grp/program/MedMNISTMerge/outputs/lamp_merge_internal_ablation_full_20260707_full_internal_ablation_v5_tmux_fullscope`.

Coverage: five medical image datasets, four small-model backbones, three client counts, and three Dirichlet beta values. Each ablation therefore contains 180 raw cells and 60 client-average cells when complete. The client-average metric first averages the three beta values for each fixed dataset, backbone, and client count.

This table is a module-internal ablation. Therefore, each replacement is evaluated against the final LAMP-Merge implementation rather than against `avg`. The `avg` baseline is used in the main comparison table, not as the decision criterion for internal component validity.

## Overall Summary

| Setting | Group | Raw cells | Raw mean Acc | Raw >= LAMP | Client-average cells | Client-average mean Acc | Client-average >= LAMP | Mean margin vs LAMP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LAMP-Merge | Formal method | 180 | 0.5884 | 180/180 | 60 | 0.5884 | 60/60 | 0.0000 |
| avg+M2 | Module-level | 180 | 0.2204 | 26/180 | 60 | 0.2204 | 7/60 | -0.3680 |
| M1 only | Module-level | 180 | 0.5884 | 180/180 | 60 | 0.5884 | 60/60 | 0.0000 |
| Global-feature mean | Prototype information | 180 | 0.0833 | 0/180 | 60 | 0.0833 | 0/60 | -0.5051 |
| Classifier-head aggregation | Prototype information | 180 | 0.2378 | 25/180 | 60 | 0.2378 | 7/60 | -0.3506 |
| Shuffled-label prototype | Prototype information | 180 | 0.1440 | 0/180 | 60 | 0.1440 | 0/60 | -0.4443 |
| Support-only synthetic head | Prototype information | 180 | 0.0860 | 0/180 | 60 | 0.0860 | 0/60 | -0.5024 |
| Binary support only | Statistical information | 180 | 0.5884 | 165/180 | 60 | 0.5884 | 47/60 | 0.0000 |
| Global client-size weight | Statistical information | 180 | 0.5885 | 144/180 | 60 | 0.5885 | 43/60 | 0.0001 |
| No prevalence calibration | Statistical information | 180 | 0.5884 | 180/180 | 60 | 0.5884 | 60/60 | 0.0000 |
| Smoothed prevalence prior | Statistical information | 180 | 0.5884 | 180/180 | 60 | 0.5884 | 60/60 | 0.0000 |
| Uniform client weight | Statistical information | 180 | 0.5884 | 165/180 | 60 | 0.5884 | 47/60 | 0.0000 |
| Uniform prevalence prior | Statistical information | 180 | 0.5884 | 180/180 | 60 | 0.5884 | 60/60 | 0.0000 |

## Dataset-Level Client Average

| Dataset | Cells | LAMP-Merge | Classifier-head aggregation margin | Shuffled-label prototype margin | Global-feature mean margin | Support-only synthetic head margin | Uniform client weight margin | Global client-size weight margin | No prevalence calibration margin | Uniform prevalence prior margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bloodmnist_224 | 12 | 0.8188 | -0.6340 | -0.5738 | -0.7367 | -0.7026 | 0.0000 | -0.0001 | 0.0000 | 0.0000 |
| chaoshengmnist_224 | 12 | 0.4618 | -0.3020 | -0.3176 | -0.3531 | -0.3549 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dermamnist_224 | 12 | 0.4619 | 0.0548 | -0.3643 | -0.4290 | -0.3882 | 0.0000 | 0.0003 | 0.0000 | 0.0000 |
| organcmnist_224 | 12 | 0.6249 | -0.4574 | -0.5133 | -0.5241 | -0.5610 | 0.0000 | 0.0001 | 0.0000 | 0.0000 |
| organsmnist_224 | 12 | 0.5744 | -0.4144 | -0.4527 | -0.4825 | -0.5051 | 0.0000 | 0.0001 | 0.0000 | 0.0000 |

## Client-Average Cell File

The complete per-cell table is written to `lamp_merge_internal_ablation_full_client_average.csv`. It includes all methods, the best method set for each `(dataset, backbone, K)` cell, and can be used directly for module-internal ablation tables in the paper.
