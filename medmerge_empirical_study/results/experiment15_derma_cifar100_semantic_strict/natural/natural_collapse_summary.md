# Natural-Image Partial-Label Collapse Probe

- Dataset: `cifar100_semantic7_derma_strict`
- Model: `resnet`
- Clients: `3`
- Class groups: `3,4,0;2,1;6,5`
- Image size: `224`
- Epochs: `50`
- Max train per class: `all`
- Group class counts: `80,779,228;769,359;99,4693`
- Output hub: `medmerge_empirical_study/results/experiment15_derma_cifar100_semantic_strict/natural/natural_probe_hub/small/cifar100_semantic7_derma_strict/resnet/clients_3/partial_label/seed_42`

## Client Sampling and Single-Client Behavior

| client_id | classes | sampled_num_samples | sampled_class_counts | best_val_acc | test_acc | test_bacc | test_collapse | effective_pred_classes | top_pred_class | test_pred_counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | [3, 4, 0] | 1087 | {"3": 80, "4": 779, "0": 228} | 0.151545 | 0.147132 | 0.383208 | 0.515711 | 3 | 4 | [742, 0, 0, 229, 1034, 0, 0] |
| 1 | [2, 1] | 1128 | {"2": 769, "1": 359} | 0.161515 | 0.161097 | 0.285714 | 0.505237 | 2 | 2 | [0, 992, 1013, 0, 0, 0, 0] |
| 2 | [6, 5] | 4792 | {"6": 99, "5": 4693} | 0.681954 | 0.681297 | 0.270829 | 0.985037 | 2 | 5 | [0, 0, 0, 0, 0, 1975, 30] |

## AVG Behavior

| model | split | accuracy | balanced_accuracy | macro_f1 | collapse_ratio | effective_pred_classes | top_pred_class | mean_confidence | mean_top1_top2_margin | pred_counts | support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| avg | test | 0.212469 | 0.219481 | 0.140299 | 0.792020 | 4 | 4 | 0.290049 | 0.215925 | [60, 0, 79, 0, 1588, 278, 0] | [66, 103, 220, 23, 223, 1341, 29] |

## All Rows

| model | accuracy | balanced_accuracy | macro_f1 | collapse_ratio | effective_pred_classes | top_pred_class | pred_counts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| client_0 | 0.147132 | 0.383208 | 0.091602 | 0.515711 | 3 | 4 | [742, 0, 0, 229, 1034, 0, 0] |
| client_1 | 0.161097 | 0.285714 | 0.077854 | 0.505237 | 2 | 2 | [0, 992, 1013, 0, 0, 0, 0] |
| client_2 | 0.681297 | 0.270829 | 0.241366 | 0.985037 | 2 | 5 | [0, 0, 0, 0, 0, 1975, 30] |
| avg | 0.212469 | 0.219481 | 0.140299 | 0.792020 | 4 | 4 | [60, 0, 79, 0, 1588, 278, 0] |

## Reading

- `collapse_ratio` is the fraction of test samples assigned to the most frequent predicted class.
- `effective_pred_classes` counts how many classes receive at least one prediction.
- A medical-style single-class collapse would have `collapse_ratio=1.0` and `effective_pred_classes=1`.
