# Natural-Image Partial-Label Collapse Probe

- Dataset: `cifar10_32`
- Model: `resnet`
- Clients: `5`
- Class groups: `0,1;2,3;4,5;6,7;8,9`
- Image size: `32`
- Epochs: `8`
- Max train per class: `all`
- Output hub: `medmerge_empirical_study/results/experiment9_natural_image_collapse_probe/natural_probe_hub/small/cifar10_32/resnet/clients_5/partial_label/seed_42`

## Client Sampling and Single-Client Behavior

| client_id | classes | sampled_num_samples | sampled_class_counts | best_val_acc | test_acc | test_bacc | test_collapse | effective_pred_classes | top_pred_class | test_pred_counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | [0, 1] | 8985 | {"0": 4527, "1": 4458} | 0.196400 | 0.192500 | 0.192500 | 0.630400 | 2 | 0 | [6304, 3696, 0, 0, 0, 0, 0, 0, 0, 0] |
| 1 | [2, 3] | 8983 | {"2": 4465, "3": 4518} | 0.180200 | 0.174700 | 0.174700 | 0.527500 | 2 | 2 | [0, 0, 5275, 4725, 0, 0, 0, 0, 0, 0] |
| 2 | [4, 5] | 9004 | {"4": 4477, "5": 4527} | 0.183000 | 0.182200 | 0.182200 | 0.578100 | 2 | 4 | [0, 0, 0, 0, 5781, 4219, 0, 0, 0, 0] |
| 3 | [6, 7] | 9029 | {"6": 4498, "7": 4531} | 0.188400 | 0.193500 | 0.193500 | 0.552000 | 2 | 7 | [0, 0, 0, 0, 0, 0, 4480, 5520, 0, 0] |
| 4 | [8, 9] | 8999 | {"8": 4482, "9": 4517} | 0.190600 | 0.188400 | 0.188400 | 0.594100 | 2 | 9 | [0, 0, 0, 0, 0, 0, 0, 0, 4059, 5941] |

## AVG Behavior

| model | split | accuracy | balanced_accuracy | macro_f1 | collapse_ratio | effective_pred_classes | top_pred_class | mean_confidence | mean_top1_top2_margin | pred_counts | support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| avg | test | 0.162100 | 0.162100 | 0.156868 | 0.242500 | 10 | 3 | 0.223046 | 0.324767 | [705, 682, 1105, 2425, 540, 630, 964, 578, 1197, 1174] | [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000] |

## All Rows

| model | accuracy | balanced_accuracy | macro_f1 | collapse_ratio | effective_pred_classes | top_pred_class | pred_counts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| client_0 | 0.192500 | 0.192500 | 0.067431 | 0.630400 | 2 | 0 | [6304, 3696, 0, 0, 0, 0, 0, 0, 0, 0] |
| client_1 | 0.174700 | 0.174700 | 0.058312 | 0.527500 | 2 | 2 | [0, 0, 5275, 4725, 0, 0, 0, 0, 0, 0] |
| client_2 | 0.182200 | 0.182200 | 0.061665 | 0.578100 | 2 | 4 | [0, 0, 0, 0, 5781, 4219, 0, 0, 0, 0] |
| client_3 | 0.193500 | 0.193500 | 0.064974 | 0.552000 | 2 | 7 | [0, 0, 0, 0, 0, 0, 4480, 5520, 0, 0] |
| client_4 | 0.188400 | 0.188400 | 0.064191 | 0.594100 | 2 | 9 | [0, 0, 0, 0, 0, 0, 0, 0, 4059, 5941] |
| avg | 0.162100 | 0.162100 | 0.156868 | 0.242500 | 10 | 3 | [705, 682, 1105, 2425, 540, 630, 964, 578, 1197, 1174] |

## Reading

- `collapse_ratio` is the fraction of test samples assigned to the most frequent predicted class.
- `effective_pred_classes` counts how many classes receive at least one prediction.
- A medical-style single-class collapse would have `collapse_ratio=1.0` and `effective_pred_classes=1`.

## Medical Contrast

| setting | dataset | model | client split | merge | accuracy | balanced_accuracy | collapse_ratio | effective_pred_classes | top_pred_class | pred_counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| medical | dermamnist_224 | resnet | 3 partial-label clients, beta=0 | avg | 0.6688 | 0.1429 | 1.0000 | 1 | 5 | [0, 0, 0, 0, 0, 2005, 0] |
| natural image | cifar10_32 | resnet | 5 partial-label clients, 2 classes/client | avg | 0.1621 | 0.1621 | 0.2425 | 10 | 3 | [705, 682, 1105, 2425, 540, 630, 964, 578, 1197, 1174] |

The natural-image control does not reproduce the medical single-class collapse under balanced class support.
The AVG model is still weak, but it assigns samples to all 10 classes.
