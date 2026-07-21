# Natural-Image Partial-Label Collapse Probe

- Dataset: `dermamnist_224`
- Model: `resnet`
- Clients: `3`
- Class groups: `3,4,0;2,1;6,5`
- Image size: `224`
- Epochs: `50`
- Max train per class: `all`
- Group class counts: `80,779,228;769,359;99,4693`
- Output hub: `medmerge_empirical_study/results/experiment15_derma_cifar100_semantic_strict/medical/natural_probe_hub/small/dermamnist_224/resnet/clients_3/partial_label/seed_42`

## Client Sampling and Single-Client Behavior

| client_id | classes | sampled_num_samples | sampled_class_counts | best_val_acc | test_acc | test_bacc | test_collapse | effective_pred_classes | top_pred_class | test_pred_counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | [3, 4, 0] | 1087 | {"3": 80, "4": 779, "0": 228} | 0.142572 | 0.140150 | 0.365621 | 0.680798 | 3 | 4 | [273, 0, 0, 367, 1365, 0, 0] |
| 1 | [2, 1] | 1128 | {"2": 769, "1": 359} | 0.157527 | 0.153616 | 0.271548 | 0.873815 | 2 | 2 | [0, 253, 1752, 0, 0, 0, 0] |
| 2 | [6, 5] | 4792 | {"6": 99, "5": 4693} | 0.681954 | 0.679801 | 0.265690 | 0.981546 | 2 | 5 | [0, 0, 0, 0, 0, 1968, 37] |

## AVG Behavior

| model | split | accuracy | balanced_accuracy | macro_f1 | collapse_ratio | effective_pred_classes | top_pred_class | mean_confidence | mean_top1_top2_margin | pred_counts | support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| avg | test | 0.668828 | 0.142857 | 0.114508 | 1.000000 | 1 | 5 | 0.275434 | 0.567162 | [0, 0, 0, 0, 0, 2005, 0] | [66, 103, 220, 23, 223, 1341, 29] |

## All Rows

| model | accuracy | balanced_accuracy | macro_f1 | collapse_ratio | effective_pred_classes | top_pred_class | pred_counts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| client_0 | 0.140150 | 0.365621 | 0.097628 | 0.680798 | 3 | 4 | [273, 0, 0, 367, 1365, 0, 0] |
| client_1 | 0.153616 | 0.271548 | 0.108420 | 0.873815 | 2 | 2 | [0, 253, 1752, 0, 0, 0, 0] |
| client_2 | 0.679801 | 0.265690 | 0.223754 | 0.981546 | 2 | 5 | [0, 0, 0, 0, 0, 1968, 37] |
| avg | 0.668828 | 0.142857 | 0.114508 | 1.000000 | 1 | 5 | [0, 0, 0, 0, 0, 2005, 0] |

## Reading

- `collapse_ratio` is the fraction of test samples assigned to the most frequent predicted class.
- `effective_pred_classes` counts how many classes receive at least one prediction.
- A medical-style single-class collapse would have `collapse_ratio=1.0` and `effective_pred_classes=1`.
