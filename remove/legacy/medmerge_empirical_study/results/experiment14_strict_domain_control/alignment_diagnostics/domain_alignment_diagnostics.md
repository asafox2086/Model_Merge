# Domain Alignment Diagnostics

- Medical case: `model_hub/small/dermamnist_224/resnet/clients_3/beta_0/seed_42`
- Natural case: `medmerge_empirical_study/results/experiment14_strict_domain_control/natural_probe_hub/small/cifar7_derma_strict/resnet/clients_3/partial_label/seed_42`
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
| natural_cifar_3client | bn1 | 0.982167 | 3 |
| natural_cifar_3client | classifier_input | 0.143687 | 3 |
| natural_cifar_3client | layer1 | 0.786361 | 3 |
| natural_cifar_3client | layer2 | 0.757033 | 3 |
| natural_cifar_3client | layer3 | 0.650460 | 3 |
| natural_cifar_3client | layer4 | 0.143687 | 3 |

## Task-Vector Cosine Summary

| domain | group | mean_cosine | num_pairs |
| --- | --- | --- | --- |
| medical_derma | bn_running_stats | 0.607558 | 3 |
| medical_derma | classifier | 0.107055 | 3 |
| medical_derma | layer3 | 0.030653 | 3 |
| medical_derma | layer4 | 0.030976 | 3 |
| natural_cifar_3client | bn_running_stats | 0.932364 | 3 |
| natural_cifar_3client | classifier | 0.163867 | 3 |
| natural_cifar_3client | layer3 | 0.958356 | 3 |
| natural_cifar_3client | layer4 | 0.960622 | 3 |

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
| natural_cifar_3client | client_0 | client_0 | 1 | 0 | 0.144140 | 0.358081 | 0.477805 | 3 | 4 | [199, 0, 0, 848, 958, 0, 0] |
| natural_cifar_3client | client_1 | client_1 | 1 | 0 | 0.161097 | 0.285714 | 0.922195 | 2 | 2 | [0, 156, 1849, 0, 0, 0, 0] |
| natural_cifar_3client | client_2 | client_2 | 1 | 0 | 0.674314 | 0.250060 | 0.949127 | 2 | 5 | [0, 0, 0, 0, 0, 1903, 102] |
| natural_cifar_3client | avg | client_0 | 0 | 0 | 0.111721 | 0.145022 | 0.998504 | 2 | 4 | [3, 0, 0, 0, 2002, 0, 0] |
| natural_cifar_3client | avg | client_1 | 0 | 0 | 0.139651 | 0.226075 | 0.964090 | 2 | 2 | [0, 72, 1933, 0, 0, 0, 0] |
| natural_cifar_3client | avg | client_2 | 0 | 0 | 0.668828 | 0.142857 | 1.000000 | 1 | 5 | [0, 0, 0, 0, 0, 2005, 0] |
| natural_cifar_3client | avg | avg | 0 | 1 | 0.699252 | 0.191548 | 0.906733 | 2 | 5 | [0, 0, 0, 0, 187, 1818, 0] |

## Linear Probe

| domain | model | train_samples | best_epoch | accuracy | balanced_accuracy | macro_f1 | collapse_ratio | effective_pred_classes | top_class | pred_counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| medical_derma | avg_original_head | 7007 |  | 0.668828 | 0.142857 | 0.114508 | 1.000000 | 1 | 5 | [0, 0, 0, 0, 0, 2005, 0] |
| medical_derma | avg_feature_linear_probe | 7007 | 31 | 0.755112 | 0.551806 | 0.556322 | 0.691272 | 7 | 5 | [92, 99, 153, 14, 234, 1386, 27] |
| natural_cifar_3client | avg_original_head | 7007 |  | 0.699252 | 0.191548 | 0.174710 | 0.906733 | 2 | 5 | [0, 0, 0, 0, 187, 1818, 0] |
| natural_cifar_3client | avg_feature_linear_probe | 7007 | 46 | 0.901746 | 0.774773 | 0.769985 | 0.655362 | 7 | 5 | [70, 112, 240, 17, 224, 1314, 28] |

## Reading

- CKA tests whether clients use aligned hidden representations at the same layer.
- Task-vector cosine tests whether clients move away from the common reference in compatible parameter directions.
- Head-feature swaps test whether one client's classifier can interpret another feature extractor or the averaged feature extractor.
- Linear probe freezes the averaged feature extractor and retrains only a linear classifier; recovery means the averaged feature still contains class information, while failure means the feature itself is badly damaged.
