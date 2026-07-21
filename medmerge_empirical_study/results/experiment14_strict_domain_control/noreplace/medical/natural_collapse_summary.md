# Natural-Image Partial-Label Collapse Probe

- Dataset: `dermamnist_224_strict_noreplace`
- Model: `resnet`
- Clients: `3`
- Class groups: `3,4,0;2,1;6,5`
- Image size: `224`
- Epochs: `50`
- Max train per class: `all`
- Group class counts: `56,549,160;542,253;69,3308`
- Output hub: `medmerge_empirical_study/results/experiment14_strict_domain_control/noreplace/medical/natural_probe_hub/small/dermamnist_224_strict_noreplace/resnet/clients_3/partial_label/seed_42`

## Client Sampling and Single-Client Behavior

| client_id | classes | sampled_num_samples | sampled_class_counts | best_val_acc | test_acc | test_bacc | test_collapse | effective_pred_classes | top_pred_class | test_pred_counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | [3, 4, 0] | 765 | {"3": 56, "4": 549, "0": 160} | 0.139205 | 0.141035 | 0.336677 | 0.769667 | 3 | 4 | [238, 0, 0, 87, 1086, 0, 0] |
| 1 | [2, 1] | 795 | {"2": 542, "1": 253} | 0.156250 | 0.152374 | 0.267217 | 0.881644 | 2 | 2 | [0, 167, 1244, 0, 0, 0, 0] |
| 2 | [6, 5] | 3377 | {"6": 69, "5": 3308} | 0.681818 | 0.676825 | 0.221277 | 0.990078 | 2 | 5 | [0, 0, 0, 0, 0, 1397, 14] |

## AVG Behavior

| model | split | accuracy | balanced_accuracy | macro_f1 | collapse_ratio | effective_pred_classes | top_pred_class | mean_confidence | mean_top1_top2_margin | pred_counts | support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| avg | test | 0.669738 | 0.142857 | 0.114601 | 1.000000 | 1 | 5 | 0.376116 | 0.855169 | [0, 0, 0, 0, 0, 1411, 0] | [46, 72, 155, 16, 157, 945, 20] |

## All Rows

| model | accuracy | balanced_accuracy | macro_f1 | collapse_ratio | effective_pred_classes | top_pred_class | pred_counts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| client_0 | 0.141035 | 0.336677 | 0.098133 | 0.769667 | 3 | 4 | [238, 0, 0, 87, 1086, 0, 0] |
| client_1 | 0.152374 | 0.267217 | 0.108339 | 0.881644 | 2 | 2 | [0, 167, 1244, 0, 0, 0, 0] |
| client_2 | 0.676825 | 0.221277 | 0.207601 | 0.990078 | 2 | 5 | [0, 0, 0, 0, 0, 1397, 14] |
| avg | 0.669738 | 0.142857 | 0.114601 | 1.000000 | 1 | 5 | [0, 0, 0, 0, 0, 1411, 0] |

## Reading

- `collapse_ratio` is the fraction of test samples assigned to the most frequent predicted class.
- `effective_pred_classes` counts how many classes receive at least one prediction.
- A medical-style single-class collapse would have `collapse_ratio=1.0` and `effective_pred_classes=1`.
