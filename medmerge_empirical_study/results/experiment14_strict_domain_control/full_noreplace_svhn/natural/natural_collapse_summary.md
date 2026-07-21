# Natural-Image Partial-Label Collapse Probe

- Dataset: `svhn7_derma_strict_full_noreplace`
- Model: `resnet`
- Clients: `3`
- Class groups: `3,4,0;2,1;6,5`
- Image size: `224`
- Epochs: `50`
- Max train per class: `all`
- Group class counts: `80,779,228;769,359;99,4693`
- Output hub: `medmerge_empirical_study/results/experiment14_strict_domain_control/full_noreplace_svhn/natural/natural_probe_hub/small/svhn7_derma_strict_full_noreplace/resnet/clients_3/partial_label/seed_42`

## Client Sampling and Single-Client Behavior

| client_id | classes | sampled_num_samples | sampled_class_counts | best_val_acc | test_acc | test_bacc | test_collapse | effective_pred_classes | top_pred_class | test_pred_counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | [3, 4, 0] | 1087 | {"3": 80, "4": 779, "0": 228} | 0.153539 | 0.152120 | 0.404328 | 0.529676 | 3 | 4 | [559, 0, 0, 384, 1062, 0, 0] |
| 1 | [2, 1] | 1128 | {"2": 769, "1": 359} | 0.158524 | 0.151122 | 0.260925 | 0.897257 | 2 | 2 | [0, 206, 1799, 0, 0, 0, 0] |
| 2 | [6, 5] | 4792 | {"6": 99, "5": 4693} | 0.681954 | 0.676808 | 0.245773 | 0.936160 | 2 | 5 | [0, 0, 0, 0, 0, 1877, 128] |

## AVG Behavior

| model | split | accuracy | balanced_accuracy | macro_f1 | collapse_ratio | effective_pred_classes | top_pred_class | mean_confidence | mean_top1_top2_margin | pred_counts | support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| avg | test | 0.442893 | 0.314797 | 0.190051 | 0.502244 | 3 | 2 | 0.193251 | 0.045755 | [0, 337, 1007, 0, 0, 661, 0] | [66, 103, 220, 23, 223, 1341, 29] |

## All Rows

| model | accuracy | balanced_accuracy | macro_f1 | collapse_ratio | effective_pred_classes | top_pred_class | pred_counts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| client_0 | 0.152120 | 0.404328 | 0.092436 | 0.529676 | 3 | 4 | [559, 0, 0, 384, 1062, 0, 0] |
| client_1 | 0.151122 | 0.260925 | 0.111011 | 0.897257 | 2 | 2 | [0, 206, 1799, 0, 0, 0, 0] |
| client_2 | 0.676808 | 0.245773 | 0.156835 | 0.936160 | 2 | 5 | [0, 0, 0, 0, 0, 1877, 128] |
| avg | 0.442893 | 0.314797 | 0.190051 | 0.502244 | 3 | 2 | [0, 337, 1007, 0, 0, 661, 0] |

## Reading

- `collapse_ratio` is the fraction of test samples assigned to the most frequent predicted class.
- `effective_pred_classes` counts how many classes receive at least one prediction.
- A medical-style single-class collapse would have `collapse_ratio=1.0` and `effective_pred_classes=1`.
