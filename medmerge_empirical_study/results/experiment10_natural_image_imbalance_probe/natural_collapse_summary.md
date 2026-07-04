# Natural-Image Partial-Label Collapse Probe

- Dataset: `cifar10_32`
- Model: `resnet`
- Clients: `5`
- Class groups: `0,1;2,3;4,5;6,7;8,9`
- Image size: `32`
- Epochs: `8`
- Max train per class: `all`
- Group class counts: `300,300;300,300;300,300;300,300;4000,80`
- Output hub: `medmerge_empirical_study/results/experiment10_natural_image_imbalance_probe/natural_probe_hub/small/cifar10_32/resnet/clients_5/partial_label/seed_42`

## Client Sampling and Single-Client Behavior

| client_id | classes | sampled_num_samples | sampled_class_counts | best_val_acc | test_acc | test_bacc | test_collapse | effective_pred_classes | top_pred_class | test_pred_counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | [0, 1] | 600 | {"0": 300, "1": 300} | 0.170000 | 0.163600 | 0.163600 | 0.522300 | 2 | 1 | [4777, 5223, 0, 0, 0, 0, 0, 0, 0, 0] |
| 1 | [2, 3] | 600 | {"2": 300, "3": 300} | 0.149400 | 0.140500 | 0.140500 | 0.661500 | 2 | 3 | [0, 0, 3385, 6615, 0, 0, 0, 0, 0, 0] |
| 2 | [4, 5] | 600 | {"4": 300, "5": 300} | 0.143800 | 0.147600 | 0.147600 | 0.644600 | 2 | 5 | [0, 0, 0, 0, 3554, 6446, 0, 0, 0, 0] |
| 3 | [6, 7] | 600 | {"6": 300, "7": 300} | 0.164000 | 0.167100 | 0.167100 | 0.622500 | 3 | 7 | [0, 1, 0, 0, 0, 0, 3774, 6225, 0, 0] |
| 4 | [8, 9] | 4080 | {"8": 4000, "9": 80} | 0.113600 | 0.111600 | 0.111600 | 0.977400 | 2 | 8 | [0, 0, 0, 0, 0, 0, 0, 0, 9774, 226] |

## AVG Behavior

| model | split | accuracy | balanced_accuracy | macro_f1 | collapse_ratio | effective_pred_classes | top_pred_class | mean_confidence | mean_top1_top2_margin | pred_counts | support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| avg | test | 0.083200 | 0.083200 | 0.063149 | 0.430700 | 10 | 8 | 0.194464 | 0.269787 | [290, 987, 636, 2073, 142, 382, 872, 194, 4307, 117] | [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000] |

## All Rows

| model | accuracy | balanced_accuracy | macro_f1 | collapse_ratio | effective_pred_classes | top_pred_class | pred_counts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| client_0 | 0.163600 | 0.163600 | 0.054619 | 0.522300 | 2 | 1 | [4777, 5223, 0, 0, 0, 0, 0, 0, 0, 0] |
| client_1 | 0.140500 | 0.140500 | 0.047889 | 0.661500 | 2 | 3 | [0, 0, 3385, 6615, 0, 0, 0, 0, 0, 0] |
| client_2 | 0.147600 | 0.147600 | 0.050357 | 0.644600 | 2 | 5 | [0, 0, 0, 0, 3554, 6446, 0, 0, 0, 0] |
| client_3 | 0.167100 | 0.167100 | 0.058294 | 0.622500 | 3 | 7 | [0, 1, 0, 0, 0, 0, 3774, 6225, 0, 0] |
| client_4 | 0.111600 | 0.111600 | 0.037776 | 0.977400 | 2 | 8 | [0, 0, 0, 0, 0, 0, 0, 0, 9774, 226] |
| avg | 0.083200 | 0.083200 | 0.063149 | 0.430700 | 10 | 8 | [290, 987, 636, 2073, 142, 382, 872, 194, 4307, 117] |

## Reading

- `collapse_ratio` is the fraction of test samples assigned to the most frequent predicted class.
- `effective_pred_classes` counts how many classes receive at least one prediction.
- A medical-style single-class collapse would have `collapse_ratio=1.0` and `effective_pred_classes=1`.

## Contrast

| setting | dataset | imbalance | merge | accuracy | balanced_accuracy | collapse_ratio | effective_pred_classes | top_pred_class | pred_counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| medical original | dermamnist_224 | natural medical imbalance | avg | 0.6688 | 0.1429 | 1.0000 | 1 | 5 | [0, 0, 0, 0, 0, 2005, 0] |
| natural balanced | cifar10_32 | balanced 2-class clients | avg | 0.1621 | 0.1621 | 0.2425 | 10 | 3 | [705, 682, 1105, 2425, 540, 630, 964, 578, 1197, 1174] |
| natural imbalanced | cifar10_32 | one large class-8 client | avg | 0.0832 | 0.0832 | 0.4307 | 10 | 8 | [290, 987, 636, 2073, 142, 382, 872, 194, 4307, 117] |

The imbalanced natural-image client `client_4` nearly collapses by itself, with `collapse_ratio=0.9774`.
However, the averaged CIFAR-10 model still predicts all 10 classes.
This supports the narrower claim that imbalance creates a dominant-class drift, while the full medical single-class collapse needs additional medical-domain fragility.
