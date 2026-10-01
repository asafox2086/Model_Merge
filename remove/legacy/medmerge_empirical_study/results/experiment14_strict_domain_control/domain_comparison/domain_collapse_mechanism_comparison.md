# Medical vs Natural Collapse Mechanism Comparison

- Medical case: `model_hub/small/dermamnist_224/resnet/clients_3/beta_0/seed_42`
- Natural-image case: `medmerge_empirical_study/results/experiment14_strict_domain_control/natural_probe_hub/small/cifar7_derma_strict/resnet/clients_3/partial_label/seed_42`
- Data root: `Med_data`
- Device: `cuda:0`

## Model Outputs

| domain | model | accuracy | balanced_accuracy | collapse_ratio | effective_pred_classes | top_class | prediction_entropy_norm | mean_top1_top2_margin | pred_counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| medical_derma | avg | 0.668828 | 0.142857 | 1.000000 | 1 | 5 | 0.000000 | 0.539030 | [0, 0, 0, 0, 0, 2005, 0] |
| natural_cifar_imbalanced | avg | 0.699252 | 0.191548 | 0.906733 | 2 | 5 | 0.159325 | 0.357434 | [0, 0, 0, 0, 187, 1818, 0] |

## Final Feature Geometry

| domain | stage | feature_norm_mean | feature_norm_cv | within_class_mse | between_centroid_mse | fisher_ratio | centroid_dist_mean | nearest_centroid_acc | own_centroid_margin_p05 | effective_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| medical_derma | classifier_input | 9.660561 | 0.141869 | 22.830999 | 4.521045 | 0.198022 | 2.711864 | 0.377556 | -1.631223 | 31.610453 |
| natural_cifar_imbalanced | classifier_input | 8.078012 | 0.105424 | 10.242745 | 7.198135 | 0.702754 | 3.168516 | 0.706733 | -0.970713 | 28.118540 |

## Dominant-Class Margin

| domain | dominant_class | compare | mean_delta | std_delta | mean_over_std | min_delta | p05_delta | pct_positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| medical_derma | 5 | 5-2 | 0.570603 | 0.203555 | 2.803186 | 0.133353 | 0.268922 | 1.000000 |
| medical_derma | 5 | 5-4 | 0.727216 | 0.171161 | 4.248731 | 0.071431 | 0.463657 | 1.000000 |
| medical_derma | 5 | 5-6 | 1.869700 | 0.723917 | 2.582754 | 0.283340 | 0.688538 | 1.000000 |

## Class Logit Decomposition

| domain | class | is_dominant | weight_norm | bias | projection_mean | projection_std | logit_mean | logit_std | logit_p05 | logit_p95 | pred_count | true_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| medical_derma | 2 | 0 | 0.429503 | -0.035517 | -1.560491 | 0.468140 | -1.596008 | 0.468140 | -2.329915 | -0.825342 | 0 | 220 |
| medical_derma | 4 | 0 | 0.437861 | 0.002527 | -1.755148 | 0.382530 | -1.752621 | 0.382530 | -2.283649 | -1.032871 | 0 | 223 |
| medical_derma | 5 | 1 | 0.378644 | -0.031668 | -0.993736 | 0.299953 | -1.025404 | 0.299952 | -1.492594 | -0.521886 | 2005 | 1341 |
| medical_derma | 6 | 0 | 0.609482 | -0.033785 | -2.861320 | 1.009981 | -2.895105 | 1.009981 | -4.484212 | -1.214752 | 0 | 29 |
| natural_cifar_imbalanced | 1 | 0 | 0.752616 | -0.030331 | -2.313020 | 0.496512 | -2.343351 | 0.496512 | -3.025644 | -1.350409 | 0 | 103 |
| natural_cifar_imbalanced | 3 | 0 | 0.825442 | -0.034454 | -2.446108 | 0.479445 | -2.480562 | 0.479445 | -3.297809 | -1.700547 | 0 | 23 |

## Pairwise Logit Decomposition

| domain | dominant_class | compare | projection_mean | projection_std | projection_p05 | projection_min | bias_delta | total_mean | total_std | total_p05 | total_min | pct_total_positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| medical_derma | 5 | 5-2 | 0.566755 | 0.203555 | 0.265074 | 0.129505 | 0.003848 | 0.570603 | 0.203555 | 0.268922 | 0.133353 | 1.000000 |
| medical_derma | 5 | 5-4 | 0.761412 | 0.171161 | 0.497853 | 0.105627 | -0.034195 | 0.727216 | 0.171161 | 0.463657 | 0.071431 | 1.000000 |
| medical_derma | 5 | 5-6 | 1.867584 | 0.723917 | 0.686422 | 0.281224 | 0.002116 | 1.869700 | 0.723917 | 0.688538 | 0.283340 | 1.000000 |

## Heads on AVG Features

| domain | head | top_class | collapse_ratio | effective_pred_classes | balanced_accuracy | dominant_pred_fraction | pred_counts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| medical_derma | client_0 | 4 | 0.943641 | 3 | 0.174999 | 0.000000 | [6, 0, 0, 107, 1892, 0, 0] |
| medical_derma | client_1 | 2 | 0.993516 | 2 | 0.152566 | 0.000000 | [0, 13, 1992, 0, 0, 0, 0] |
| medical_derma | client_2 | 5 | 1.000000 | 1 | 0.142857 | 1.000000 | [0, 0, 0, 0, 0, 2005, 0] |
| medical_derma | avg_head | 5 | 1.000000 | 1 | 0.142857 | 1.000000 | [0, 0, 0, 0, 0, 2005, 0] |
| natural_cifar_imbalanced | client_0 | 4 | 0.998504 | 2 | 0.145022 | 0.000000 | [3, 0, 0, 0, 2002, 0, 0] |
| natural_cifar_imbalanced | client_1 | 2 | 0.964090 | 2 | 0.226075 | 0.000000 | [0, 72, 1933, 0, 0, 0, 0] |
| natural_cifar_imbalanced | client_2 | 5 | 1.000000 | 1 | 0.142857 | 1.000000 | [0, 0, 0, 0, 0, 2005, 0] |
| natural_cifar_imbalanced | avg_head | 5 | 0.906733 | 2 | 0.191548 | 0.906733 | [0, 0, 0, 0, 187, 1818, 0] |

## Parameter Dispersion

| domain | group | weighted_relative_l2_to_avg | mean_delta_norm_per_tensor | param_count | tensor_count |
| --- | --- | --- | --- | --- | --- |
| medical_derma | bn_running_stats | 0.207082 | 452.724888 | 9600 | 40 |
| medical_derma | classifier | 0.787614 | 0.884577 | 3591 | 2 |
| medical_derma | layer4 | 0.064024 | 11.530469 | 8389632 | 7 |
| natural_cifar_imbalanced | bn_running_stats | 0.585210 | 369.865483 | 9600 | 40 |
| natural_cifar_imbalanced | classifier | 0.525665 | 0.630694 | 3591 | 2 |
| natural_cifar_imbalanced | layer4 | 0.696055 | 43.358937 | 8389632 | 7 |

## Reading

- The decisive diagnostic is the dominant-class margin distribution, not the existence of non-IID itself.
- In medical Derma, the dominant class has positive margin against close competitors on every test sample; in imbalanced CIFAR, the dominant class has negative low-tail margins against several competitors, so other classes can still win.
- The final-feature table checks whether the averaged representation preserves class-dependent variation. The margin table checks whether that variation is large enough to overcome the dominant-class offset.
