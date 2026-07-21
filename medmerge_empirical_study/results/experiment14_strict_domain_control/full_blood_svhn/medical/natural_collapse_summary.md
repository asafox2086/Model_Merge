# Natural-Image Partial-Label Collapse Probe

- Dataset: `bloodmnist_224`
- Model: `resnet`
- Clients: `3`
- Class groups: `3,4,0;2,1,7;6,5`
- Image size: `224`
- Epochs: `50`
- Max train per class: `all`
- Group class counts: `2026,849,852;1085,2181,1643;2330,993`
- Output hub: `medmerge_empirical_study/results/experiment14_strict_domain_control/full_blood_svhn/medical/natural_probe_hub/small/bloodmnist_224/resnet/clients_3/partial_label/seed_42`

## Client Sampling and Single-Client Behavior

| client_id | classes | sampled_num_samples | sampled_class_counts | best_val_acc | test_acc | test_bacc | test_collapse | effective_pred_classes | top_pred_class | test_pred_counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | [3, 4, 0] | 3727 | {"3": 2026, "4": 849, "0": 852} | 0.311916 | 0.310436 | 0.373247 | 0.653318 | 3 | 3 | [691, 0, 0, 2235, 495, 0, 0, 0] |
| 1 | [2, 1, 7] | 4909 | {"2": 1085, "1": 2181, "7": 1643} | 0.410047 | 0.410699 | 0.375000 | 0.488746 | 3 | 2 | [0, 1269, 1672, 0, 0, 0, 0, 480] |
| 2 | [6, 5] | 3323 | {"6": 2330, "5": 993} | 0.277453 | 0.277404 | 0.249812 | 0.622625 | 2 | 6 | [0, 0, 0, 0, 0, 1291, 2130, 0] |

## AVG Behavior

| model | split | accuracy | balanced_accuracy | macro_f1 | collapse_ratio | effective_pred_classes | top_pred_class | mean_confidence | mean_top1_top2_margin | pred_counts | support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| avg | test | 0.200234 | 0.179303 | 0.102241 | 0.951184 | 2 | 3 | 0.269510 | 0.564693 | [167, 0, 0, 3254, 0, 0, 0, 0] | [244, 624, 311, 579, 243, 284, 666, 470] |

## All Rows

| model | accuracy | balanced_accuracy | macro_f1 | collapse_ratio | effective_pred_classes | top_pred_class | pred_counts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| client_0 | 0.310436 | 0.373247 | 0.198106 | 0.653318 | 3 | 3 | [691, 0, 0, 2235, 495, 0, 0, 0] |
| client_1 | 0.410699 | 0.375000 | 0.245301 | 0.488746 | 3 | 2 | [0, 1269, 1672, 0, 0, 0, 0, 480] |
| client_2 | 0.277404 | 0.249812 | 0.104539 | 0.622625 | 2 | 6 | [0, 0, 0, 0, 0, 1291, 2130, 0] |
| avg | 0.200234 | 0.179303 | 0.102241 | 0.951184 | 2 | 3 | [167, 0, 0, 3254, 0, 0, 0, 0] |

## Reading

- `collapse_ratio` is the fraction of test samples assigned to the most frequent predicted class.
- `effective_pred_classes` counts how many classes receive at least one prediction.
- A medical-style single-class collapse would have `collapse_ratio=1.0` and `effective_pred_classes=1`.
