# Natural-Image Partial-Label Collapse Probe

- Dataset: `svhn8_blood_strict_full_noreplace`
- Model: `resnet`
- Clients: `3`
- Class groups: `3,4,0;2,1,7;6,5`
- Image size: `224`
- Epochs: `50`
- Max train per class: `all`
- Group class counts: `2026,849,852;1085,2181,1643;2330,993`
- Output hub: `medmerge_empirical_study/results/experiment14_strict_domain_control/full_blood_svhn/natural/natural_probe_hub/small/svhn8_blood_strict_full_noreplace/resnet/clients_3/partial_label/seed_42`

## Client Sampling and Single-Client Behavior

| client_id | classes | sampled_num_samples | sampled_class_counts | best_val_acc | test_acc | test_bacc | test_collapse | effective_pred_classes | top_pred_class | test_pred_counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | [3, 4, 0] | 3727 | {"3": 2026, "4": 849, "0": 852} | 0.304907 | 0.306343 | 0.367544 | 0.600409 | 3 | 3 | [773, 0, 0, 2054, 594, 0, 0, 0] |
| 1 | [2, 1, 7] | 4909 | {"2": 1085, "1": 2181, "7": 1643} | 0.397780 | 0.387898 | 0.354748 | 0.417129 | 3 | 2 | [0, 1249, 1427, 0, 0, 0, 0, 745] |
| 2 | [6, 5] | 3323 | {"6": 2330, "5": 993} | 0.269276 | 0.270681 | 0.240951 | 0.674949 | 2 | 6 | [0, 0, 0, 0, 0, 1112, 2309, 0] |

## AVG Behavior

| model | split | accuracy | balanced_accuracy | macro_f1 | collapse_ratio | effective_pred_classes | top_pred_class | mean_confidence | mean_top1_top2_margin | pred_counts | support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| avg | test | 0.252265 | 0.211645 | 0.114972 | 0.755919 | 4 | 6 | 0.148710 | 0.066391 | [24, 0, 0, 9, 0, 802, 2586, 0] | [244, 624, 311, 579, 243, 284, 666, 470] |

## All Rows

| model | accuracy | balanced_accuracy | macro_f1 | collapse_ratio | effective_pred_classes | top_pred_class | pred_counts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| client_0 | 0.306343 | 0.367544 | 0.183700 | 0.600409 | 3 | 3 | [773, 0, 0, 2054, 594, 0, 0, 0] |
| client_1 | 0.387898 | 0.354748 | 0.210706 | 0.417129 | 3 | 2 | [0, 1249, 1427, 0, 0, 0, 0, 745] |
| client_2 | 0.270681 | 0.240951 | 0.103098 | 0.674949 | 2 | 6 | [0, 0, 0, 0, 0, 1112, 2309, 0] |
| avg | 0.252265 | 0.211645 | 0.114972 | 0.755919 | 4 | 6 | [24, 0, 0, 9, 0, 802, 2586, 0] |

## Reading

- `collapse_ratio` is the fraction of test samples assigned to the most frequent predicted class.
- `effective_pred_classes` counts how many classes receive at least one prediction.
- A medical-style single-class collapse would have `collapse_ratio=1.0` and `effective_pred_classes=1`.
