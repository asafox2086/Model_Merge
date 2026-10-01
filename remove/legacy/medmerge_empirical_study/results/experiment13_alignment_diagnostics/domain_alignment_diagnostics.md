# Domain Alignment Diagnostics

- Medical case: `model_hub/small/dermamnist_224/resnet/clients_3/beta_0/seed_42`
- Natural case: `medmerge_empirical_study/results/experiment12_natural_image_3client_imbalance_probe/natural_probe_hub/small/cifar10_32/resnet/clients_3/partial_label/seed_42`
- Device: `cuda:0`
- CKA max samples: `2000`

## Feature CKA Summary

| domain | stage | mean_cka | num_pairs |
| --- | --- | --- | --- |
| medical_derma | bn1 | 0.994642 | 3 |
| medical_derma | classifier_input | 0.316591 | 3 |
| medical_derma | layer1 | 0.842579 | 3 |
| medical_derma | layer2 | 0.800724 | 3 |
| medical_derma | layer3 | 0.624328 | 3 |
| medical_derma | layer4 | 0.316591 | 3 |
| natural_cifar_3client | bn1 | 0.992404 | 3 |
| natural_cifar_3client | classifier_input | 0.239942 | 3 |
| natural_cifar_3client | layer1 | 0.837278 | 3 |
| natural_cifar_3client | layer2 | 0.233668 | 3 |
| natural_cifar_3client | layer3 | 0.364615 | 3 |
| natural_cifar_3client | layer4 | 0.239942 | 3 |

## Task-Vector Cosine Summary

| domain | group | mean_cosine | num_pairs |
| --- | --- | --- | --- |
| medical_derma | bn_running_stats | 0.607558 | 3 |
| medical_derma | classifier | 0.107055 | 3 |
| medical_derma | layer3 | 0.030653 | 3 |
| medical_derma | layer4 | 0.030976 | 3 |
| natural_cifar_3client | bn_running_stats | 0.985049 | 3 |
| natural_cifar_3client | classifier | 0.219918 | 3 |
| natural_cifar_3client | layer3 | 0.599215 | 3 |
| natural_cifar_3client | layer4 | 0.960982 | 3 |

## Head-Feature Swap Key Rows

| domain | feature_source | head_source | is_matched_client | is_avg_avg | accuracy | balanced_accuracy | collapse_ratio | effective_pred_classes | top_class | pred_counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| medical_derma | client_0 | client_0 | 1 | 0 | 0.021446 | 0.159242 | 0.895262 | 3 | 3 | [148, 0, 0, 1795, 62, 0, 0] |
| medical_derma | client_1 | client_1 | 1 | 0 | 0.066334 | 0.160125 | 0.634414 | 2 | 1 | [0, 1272, 733, 0, 0, 0, 0] |
| medical_derma | client_2 | client_2 | 1 | 0 | 0.577057 | 0.210008 | 0.788529 | 2 | 5 | [0, 0, 0, 0, 0, 1581, 424] |
| medical_derma | avg | client_0 | 0 | 0 | 0.110224 | 0.174999 | 0.943641 | 3 | 4 | [6, 0, 0, 107, 1892, 0, 0] |
| medical_derma | avg | client_1 | 0 | 0 | 0.113217 | 0.152566 | 0.993516 | 2 | 2 | [0, 13, 1992, 0, 0, 0, 0] |
| medical_derma | avg | client_2 | 0 | 0 | 0.668828 | 0.142857 | 1.000000 | 1 | 5 | [0, 0, 0, 0, 0, 2005, 0] |
| medical_derma | avg | avg | 0 | 1 | 0.668828 | 0.142857 | 1.000000 | 1 | 5 | [0, 0, 0, 0, 0, 2005, 0] |
| natural_cifar_3client | client_0 | client_0 | 1 | 0 | 0.230200 | 0.230200 | 0.397900 | 3 | 2 | [2267, 3754, 3979, 0, 0, 0, 0, 0, 0, 0] |
| natural_cifar_3client | client_1 | client_1 | 1 | 0 | 0.177300 | 0.177300 | 0.418900 | 3 | 4 | [0, 0, 0, 4045, 4189, 1766, 0, 0, 0, 0] |
| natural_cifar_3client | client_2 | client_2 | 1 | 0 | 0.115500 | 0.115500 | 0.941700 | 2 | 8 | [0, 0, 0, 0, 0, 0, 0, 0, 9417, 583] |
| natural_cifar_3client | avg | client_0 | 0 | 0 | 0.191900 | 0.191900 | 0.438600 | 3 | 1 | [2385, 4386, 3229, 0, 0, 0, 0, 0, 0, 0] |
| natural_cifar_3client | avg | client_1 | 0 | 0 | 0.144400 | 0.144400 | 0.485500 | 3 | 3 | [0, 0, 0, 4855, 3039, 2106, 0, 0, 0, 0] |
| natural_cifar_3client | avg | client_2 | 0 | 0 | 0.099800 | 0.099800 | 0.998800 | 2 | 8 | [0, 0, 0, 0, 0, 0, 0, 0, 9988, 12] |
| natural_cifar_3client | avg | avg | 0 | 1 | 0.110500 | 0.110500 | 0.781000 | 8 | 8 | [254, 421, 329, 593, 210, 235, 0, 0, 7810, 148] |

## Linear Probe

| domain | model | train_samples | best_epoch | accuracy | balanced_accuracy | macro_f1 | collapse_ratio | effective_pred_classes | top_class | pred_counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| medical_derma | avg_original_head | 7007 |  | 0.668828 | 0.142857 | 0.114508 | 1.000000 | 1 | 5 | [0, 0, 0, 0, 0, 2005, 0] |
| medical_derma | avg_feature_linear_probe | 7007 | 74 | 0.743142 | 0.573081 | 0.560969 | 0.667830 | 7 | 5 | [52, 142, 207, 35, 205, 1339, 25] |
| natural_cifar_3client | avg_original_head | 45000 |  | 0.110500 | 0.110500 | 0.070777 | 0.781000 | 8 | 8 | [254, 421, 329, 593, 210, 235, 0, 0, 7810, 148] |
| natural_cifar_3client | avg_feature_linear_probe | 45000 | 62 | 0.431400 | 0.431400 | 0.427458 | 0.120200 | 10 | 6 | [1080, 1029, 856, 748, 1001, 1139, 1202, 1005, 934, 1006] |

## Reading

- CKA tests whether clients use aligned hidden representations at the same layer.
- Task-vector cosine tests whether clients move away from the common reference in compatible parameter directions.
- Head-feature swaps test whether one client's classifier can interpret another feature extractor or the averaged feature extractor.
- Linear probe freezes the averaged feature extractor and retrains only a linear classifier; recovery means the averaged feature still contains class information, while failure means the feature itself is badly damaged.
