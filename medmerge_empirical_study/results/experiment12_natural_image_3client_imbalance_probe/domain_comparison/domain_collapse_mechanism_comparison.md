# Medical vs Natural Collapse Mechanism Comparison

- Medical case: `model_hub/small/dermamnist_224/resnet/clients_3/beta_0/seed_42`
- Natural-image case: `medmerge_empirical_study/results/experiment12_natural_image_3client_imbalance_probe/natural_probe_hub/small/cifar10_32/resnet/clients_3/partial_label/seed_42`
- Data root: `Med_data`
- Device: `cuda:0`

## Model Outputs

| domain | model | accuracy | balanced_accuracy | collapse_ratio | effective_pred_classes | top_class | prediction_entropy_norm | mean_top1_top2_margin | pred_counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| medical_derma | avg | 0.668828 | 0.142857 | 1.000000 | 1 | 5 | 0.000000 | 0.539030 | [0, 0, 0, 0, 0, 2005, 0] |
| natural_cifar_imbalanced | avg | 0.110500 | 0.110500 | 0.781000 | 8 | 8 | 0.404410 | 0.517689 | [254, 421, 329, 593, 210, 235, 0, 0, 7810, 148] |

## Final Feature Geometry

| domain | stage | feature_norm_mean | feature_norm_cv | within_class_mse | between_centroid_mse | fisher_ratio | centroid_dist_mean | nearest_centroid_acc | own_centroid_margin_p05 | effective_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| medical_derma | classifier_input | 9.660561 | 0.141869 | 22.830999 | 4.521045 | 0.198022 | 2.711864 | 0.377556 | -1.631223 | 31.610453 |
| natural_cifar_imbalanced | classifier_input | 17.123440 | 0.343431 | 287.752075 | 7.801459 | 0.027112 | 4.090577 | 0.398200 | -1.061636 | 237.550751 |

## Dominant-Class Margin

| domain | dominant_class | compare | mean_delta | std_delta | mean_over_std | min_delta | p05_delta | pct_positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| medical_derma | 5 | 5-2 | 0.570603 | 0.203555 | 2.803186 | 0.133353 | 0.268922 | 1.000000 |
| medical_derma | 5 | 5-4 | 0.727216 | 0.171161 | 4.248731 | 0.071431 | 0.463657 | 1.000000 |
| medical_derma | 5 | 5-6 | 1.869700 | 0.723917 | 2.582754 | 0.283340 | 0.688538 | 1.000000 |
| natural_cifar_imbalanced | 8 | 8-1 | 0.915266 | 0.711141 | 1.287038 | -1.550948 | -0.049698 | 0.937600 |
| natural_cifar_imbalanced | 8 | 8-3 | 0.780737 | 0.696047 | 1.121672 | -3.122090 | -0.119339 | 0.917300 |
| natural_cifar_imbalanced | 8 | 8-9 | 1.535806 | 1.024280 | 1.499401 | -1.132782 | 0.169226 | 0.974100 |

## Class Logit Decomposition

| domain | class | is_dominant | weight_norm | bias | projection_mean | projection_std | logit_mean | logit_std | logit_p05 | logit_p95 | pred_count | true_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| medical_derma | 2 | 0 | 0.429503 | -0.035517 | -1.560491 | 0.468140 | -1.596008 | 0.468140 | -2.329915 | -0.825342 | 0 | 220 |
| medical_derma | 4 | 0 | 0.437861 | 0.002527 | -1.755148 | 0.382530 | -1.752621 | 0.382530 | -2.283649 | -1.032871 | 0 | 223 |
| medical_derma | 5 | 1 | 0.378644 | -0.031668 | -0.993736 | 0.299953 | -1.025404 | 0.299952 | -1.492594 | -0.521886 | 2005 | 1341 |
| medical_derma | 6 | 0 | 0.609482 | -0.033785 | -2.861320 | 1.009981 | -2.895105 | 1.009981 | -4.484212 | -1.214752 | 0 | 29 |
| natural_cifar_imbalanced | 1 | 0 | 0.670566 | 0.025692 | -1.376477 | 0.705105 | -1.350785 | 0.705105 | -2.637314 | -0.457778 | 421 | 1000 |
| natural_cifar_imbalanced | 3 | 0 | 0.668103 | 0.006254 | -1.222509 | 0.716210 | -1.216256 | 0.716210 | -2.557786 | -0.353302 | 593 | 1000 |
| natural_cifar_imbalanced | 8 | 1 | 0.602471 | -0.039031 | -0.396488 | 0.462225 | -0.435519 | 0.462225 | -1.203315 | 0.241860 | 7810 | 1000 |
| natural_cifar_imbalanced | 9 | 0 | 0.739501 | -0.043689 | -1.927637 | 1.000977 | -1.971325 | 1.000977 | -3.860021 | -0.641142 | 148 | 1000 |

## Pairwise Logit Decomposition

| domain | dominant_class | compare | projection_mean | projection_std | projection_p05 | projection_min | bias_delta | total_mean | total_std | total_p05 | total_min | pct_total_positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| medical_derma | 5 | 5-2 | 0.566755 | 0.203555 | 0.265074 | 0.129505 | 0.003848 | 0.570603 | 0.203555 | 0.268922 | 0.133353 | 1.000000 |
| medical_derma | 5 | 5-4 | 0.761412 | 0.171161 | 0.497853 | 0.105627 | -0.034195 | 0.727216 | 0.171161 | 0.463657 | 0.071431 | 1.000000 |
| medical_derma | 5 | 5-6 | 1.867584 | 0.723917 | 0.686422 | 0.281224 | 0.002116 | 1.869700 | 0.723917 | 0.688538 | 0.283340 | 1.000000 |
| natural_cifar_imbalanced | 8 | 8-1 | 0.979989 | 0.711141 | 0.015025 | -1.486224 | -0.064724 | 0.915266 | 0.711141 | -0.049698 | -1.550948 | 0.937600 |
| natural_cifar_imbalanced | 8 | 8-3 | 0.826021 | 0.696047 | -0.074054 | -3.076805 | -0.045285 | 0.780737 | 0.696047 | -0.119339 | -3.122090 | 0.917300 |
| natural_cifar_imbalanced | 8 | 8-9 | 1.531149 | 1.024280 | 0.164569 | -1.137439 | 0.004657 | 1.535806 | 1.024279 | 0.169226 | -1.132782 | 0.974100 |

## Heads on AVG Features

| domain | head | top_class | collapse_ratio | effective_pred_classes | balanced_accuracy | dominant_pred_fraction | pred_counts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| medical_derma | client_0 | 4 | 0.943641 | 3 | 0.174999 | 0.000000 | [6, 0, 0, 107, 1892, 0, 0] |
| medical_derma | client_1 | 2 | 0.993516 | 2 | 0.152566 | 0.000000 | [0, 13, 1992, 0, 0, 0, 0] |
| medical_derma | client_2 | 5 | 1.000000 | 1 | 0.142857 | 1.000000 | [0, 0, 0, 0, 0, 2005, 0] |
| medical_derma | avg_head | 5 | 1.000000 | 1 | 0.142857 | 1.000000 | [0, 0, 0, 0, 0, 2005, 0] |
| natural_cifar_imbalanced | client_0 | 1 | 0.438600 | 3 | 0.191900 | 0.000000 | [2385, 4386, 3229, 0, 0, 0, 0, 0, 0, 0] |
| natural_cifar_imbalanced | client_1 | 3 | 0.485500 | 3 | 0.144400 | 0.000000 | [0, 0, 0, 4855, 3039, 2106, 0, 0, 0, 0] |
| natural_cifar_imbalanced | client_2 | 8 | 0.998800 | 2 | 0.099800 | 0.998800 | [0, 0, 0, 0, 0, 0, 0, 0, 9988, 12] |
| natural_cifar_imbalanced | avg_head | 8 | 0.781000 | 8 | 0.110500 | 0.781000 | [254, 421, 329, 593, 210, 235, 0, 0, 7810, 148] |

## Parameter Dispersion

| domain | group | weighted_relative_l2_to_avg | mean_delta_norm_per_tensor | param_count | tensor_count |
| --- | --- | --- | --- | --- | --- |
| medical_derma | bn_running_stats | 0.207082 | 452.724888 | 9600 | 40 |
| medical_derma | classifier | 0.787614 | 0.884577 | 3591 | 2 |
| medical_derma | layer4 | 0.064024 | 11.530469 | 8389632 | 7 |
| natural_cifar_imbalanced | bn_running_stats | 0.129962 | 125.711910 | 9600 | 40 |
| natural_cifar_imbalanced | classifier | 0.477090 | 0.636303 | 5130 | 2 |
| natural_cifar_imbalanced | layer4 | 0.149379 | 21.203116 | 8389632 | 7 |

## Reading

- The decisive diagnostic is the dominant-class margin distribution, not the existence of non-IID itself.
- In medical Derma, the dominant class has positive margin against close competitors on every test sample; in imbalanced CIFAR, the dominant class has negative low-tail margins against several competitors, so other classes can still win.
- The final-feature table checks whether the averaged representation preserves class-dependent variation. The margin table checks whether that variation is large enough to overcome the dominant-class offset.
