# LAMP-Merge Full-Scale Internal Ablation Summary

Experiment root: `outputs/lamp_merge_internal_ablation_full_20260708_force_all_analysis_after_client_stats_internal_ablation`.

Coverage: five medical image datasets, four small-model backbones, three client counts, and three Dirichlet beta values. Each ablation therefore contains 180 raw cells and 60 client-average cells when complete. The client-average metric first averages the three beta values for each fixed dataset, backbone, and client count.

This table is a module-internal ablation. Therefore, each replacement is evaluated against the final LAMP-Merge implementation rather than against `avg`. The `avg` baseline is used in the main comparison table, not as the decision criterion for internal component validity.

## Overall Summary

| Setting | Group | Raw cells | Raw mean Acc | Raw >= LAMP | Client-average cells | Client-average mean Acc | Client-average >= LAMP | Mean margin vs LAMP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LAMP-Merge | Formal method | 180 | 0.6209 | 180/180 | 60 | 0.6209 | 60/60 | 0.0000 |
| M1 only | Module-level | 180 | 0.5880 | 90/180 | 60 | 0.5880 | 30/60 | -0.0329 |
| Global-feature mean | Prototype information | 180 | 0.2645 | 27/180 | 60 | 0.2645 | 9/60 | -0.3564 |
| Classifier-head aggregation | Prototype information | 180 | 0.2579 | 11/180 | 60 | 0.2579 | 2/60 | -0.3630 |
| Shuffled-label prototype | Prototype information | 180 | 0.2002 | 0/180 | 60 | 0.2002 | 0/60 | -0.4207 |
| Support-only synthetic head | Prototype information | 180 | 0.1137 | 0/180 | 60 | 0.1137 | 0/60 | -0.5073 |
| Binary support only | Statistical information | 180 | 0.5517 | 47/180 | 60 | 0.5517 | 2/60 | -0.0692 |
| Uniform client weight | Statistical information | 180 | 0.5839 | 85/180 | 60 | 0.5839 | 3/60 | -0.0370 |

## Dataset-Level Client Average

| Dataset | Cells | LAMP-Merge | Classifier-head aggregation margin | Shuffled-label prototype margin | Global-feature mean margin | Support-only synthetic head margin | Uniform client weight margin | Global client-size weight margin | No prevalence calibration margin | Uniform prevalence prior margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bloodmnist_224 | 12 | 0.8174 | -0.6277 | -0.5728 | -0.7310 | -0.7012 | -0.0371 | - | - | - |
| chaoshengmnist_224 | 12 | 0.4574 | -0.2932 | -0.3159 | -0.3487 | -0.3505 | -0.0401 | - | - | - |
| dermamnist_224 | 12 | 0.6286 | -0.0622 | -0.3271 | 0.0402 | -0.4334 | -0.0262 | - | - | - |
| organcmnist_224 | 12 | 0.6270 | -0.4480 | -0.4823 | -0.4037 | -0.5555 | -0.0446 | - | - | - |
| organsmnist_224 | 12 | 0.5741 | -0.3839 | -0.4056 | -0.3387 | -0.4958 | -0.0371 | - | - | - |

## Client-Average Cell File

The complete per-cell table is written to `lamp_merge_internal_ablation_full_client_average.csv`. It includes all methods, the best method set for each `(dataset, backbone, K)` cell, and can be used directly for module-internal ablation tables in the paper.
