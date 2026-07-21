# Natural-Image Partial-Label Collapse Probe

- Dataset: `dermamnist_224_strict_full_noreplace`
- Model: `resnet`
- Clients: `3`
- Class groups: `3,4,0;2,1;6,5`
- Image size: `224`
- Epochs: `50`
- Max train per class: `all`
- Group class counts: `80,779,228;769,359;99,4693`
- Output hub: `medmerge_empirical_study/results/experiment14_strict_domain_control/full_noreplace_svhn/medical/natural_probe_hub/small/dermamnist_224_strict_full_noreplace/resnet/clients_3/partial_label/seed_42`

## Client Sampling and Single-Client Behavior

| client_id | classes | sampled_num_samples | sampled_class_counts | best_val_acc | test_acc | test_bacc | test_collapse | effective_pred_classes | top_pred_class | test_pred_counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | [3, 4, 0] | 1087 | {"3": 80, "4": 779, "0": 228} | 0.141575 | 0.139651 | 0.323412 | 0.776060 | 3 | 4 | [178, 0, 0, 271, 1556, 0, 0] |
| 1 | [2, 1] | 1128 | {"2": 769, "1": 359} | 0.158524 | 0.156110 | 0.274795 | 0.871820 | 2 | 2 | [0, 257, 1748, 0, 0, 0, 0] |
| 2 | [6, 5] | 4792 | {"6": 99, "5": 4693} | 0.679960 | 0.679302 | 0.255945 | 0.980549 | 2 | 5 | [0, 0, 0, 0, 0, 1966, 39] |

## AVG Behavior

| model | split | accuracy | balanced_accuracy | macro_f1 | collapse_ratio | effective_pred_classes | top_pred_class | mean_confidence | mean_top1_top2_margin | pred_counts | support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| avg | test | 0.428429 | 0.290988 | 0.255552 | 0.452868 | 7 | 5 | 0.255785 | 0.156836 | [128, 94, 831, 11, 30, 908, 3] | [66, 103, 220, 23, 223, 1341, 29] |

## All Rows

| model | accuracy | balanced_accuracy | macro_f1 | collapse_ratio | effective_pred_classes | top_pred_class | pred_counts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| client_0 | 0.139651 | 0.323412 | 0.103002 | 0.776060 | 3 | 4 | [178, 0, 0, 271, 1556, 0, 0] |
| client_1 | 0.156110 | 0.274795 | 0.108343 | 0.871820 | 2 | 2 | [0, 257, 1748, 0, 0, 0, 0] |
| client_2 | 0.679302 | 0.255945 | 0.212324 | 0.980549 | 2 | 5 | [0, 0, 0, 0, 0, 1966, 39] |
| avg | 0.428429 | 0.290988 | 0.255552 | 0.452868 | 7 | 5 | [128, 94, 831, 11, 30, 908, 3] |

## Reading

- `collapse_ratio` is the fraction of test samples assigned to the most frequent predicted class.
- `effective_pred_classes` counts how many classes receive at least one prediction.
- A medical-style single-class collapse would have `collapse_ratio=1.0` and `effective_pred_classes=1`.
