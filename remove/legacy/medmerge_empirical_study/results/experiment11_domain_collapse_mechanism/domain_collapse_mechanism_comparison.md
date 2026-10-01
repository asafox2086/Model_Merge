# Medical vs Natural Collapse Mechanism Comparison

- Medical case: `model_hub/small/dermamnist_224/resnet/clients_3/beta_0/seed_42`
- Natural-image case: `medmerge_empirical_study/results/experiment10_natural_image_imbalance_probe/natural_probe_hub/small/cifar10_32/resnet/clients_5/partial_label/seed_42`
- Data root: `Med_data`
- Device: `cuda:0`

## Model Outputs

| domain | model | accuracy | balanced_accuracy | collapse_ratio | effective_pred_classes | top_class | prediction_entropy_norm | mean_top1_top2_margin | pred_counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| medical_derma | avg | 0.668828 | 0.142857 | 1.000000 | 1 | 5 | 0.000000 | 0.539030 | [0, 0, 0, 0, 0, 2005, 0] |
| natural_cifar_imbalanced | avg | 0.083100 | 0.083100 | 0.431500 | 10 | 8 | 0.747598 | 0.269798 | [293, 983, 637, 2069, 143, 379, 869, 195, 4315, 117] |

## Final Feature Geometry

| domain | stage | feature_norm_mean | feature_norm_cv | within_class_mse | between_centroid_mse | fisher_ratio | centroid_dist_mean | nearest_centroid_acc | own_centroid_margin_p05 | effective_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| medical_derma | classifier_input | 9.660561 | 0.141869 | 22.830999 | 4.521045 | 0.198022 | 2.711864 | 0.377556 | -1.631223 | 31.610453 |
| natural_cifar_imbalanced | classifier_input | 15.605836 | 0.359692 | 242.116608 | 6.853444 | 0.028306 | 3.811618 | 0.394000 | -0.934651 | 216.896500 |

## Dominant-Class Margin

| domain | dominant_class | compare | mean_delta | std_delta | mean_over_std | min_delta | p05_delta | pct_positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| medical_derma | 5 | 5-2 | 0.570603 | 0.203555 | 2.803186 | 0.133353 | 0.268922 | 1.000000 |
| medical_derma | 5 | 5-4 | 0.727216 | 0.171161 | 4.248731 | 0.071431 | 0.463657 | 1.000000 |
| medical_derma | 5 | 5-6 | 1.869700 | 0.723917 | 2.582754 | 0.283340 | 0.688538 | 1.000000 |
| natural_cifar_imbalanced | 8 | 8-1 | 0.390480 | 0.551283 | 0.708311 | -1.583900 | -0.391201 | 0.763000 |
| natural_cifar_imbalanced | 8 | 8-3 | 0.246583 | 0.587772 | 0.419521 | -3.143305 | -0.601115 | 0.656900 |
| natural_cifar_imbalanced | 8 | 8-9 | 0.775293 | 0.665347 | 1.165246 | -1.844042 | -0.108207 | 0.917600 |

## Heads on AVG Features

| domain | head | top_class | collapse_ratio | effective_pred_classes | balanced_accuracy | dominant_pred_fraction | pred_counts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| medical_derma | client_0 | 4 | 0.943641 | 3 | 0.174999 | 0.000000 | [6, 0, 0, 107, 1892, 0, 0] |
| medical_derma | client_1 | 2 | 0.993516 | 2 | 0.152566 | 0.000000 | [0, 13, 1992, 0, 0, 0, 0] |
| medical_derma | client_2 | 5 | 1.000000 | 1 | 0.142857 | 1.000000 | [0, 0, 0, 0, 0, 2005, 0] |
| medical_derma | avg_head | 5 | 1.000000 | 1 | 0.142857 | 1.000000 | [0, 0, 0, 0, 0, 2005, 0] |
| natural_cifar_imbalanced | client_0 | 1 | 0.697600 | 2 | 0.129500 | 0.000000 | [3024, 6976, 0, 0, 0, 0, 0, 0, 0, 0] |
| natural_cifar_imbalanced | client_1 | 3 | 0.624200 | 2 | 0.125800 | 0.000000 | [0, 0, 3758, 6242, 0, 0, 0, 0, 0, 0] |
| natural_cifar_imbalanced | client_2 | 5 | 0.611600 | 2 | 0.108900 | 0.000000 | [0, 0, 0, 0, 3884, 6116, 0, 0, 0, 0] |
| natural_cifar_imbalanced | client_3 | 6 | 0.573400 | 2 | 0.129100 | 0.000000 | [0, 0, 0, 0, 0, 0, 5734, 4266, 0, 0] |
| natural_cifar_imbalanced | client_4 | 8 | 0.999500 | 2 | 0.100000 | 0.999500 | [0, 0, 0, 0, 0, 0, 0, 0, 9995, 5] |
| natural_cifar_imbalanced | avg_head | 8 | 0.431500 | 10 | 0.083100 | 0.431500 | [293, 983, 637, 2069, 143, 379, 869, 195, 4315, 117] |

## Parameter Dispersion

| domain | group | weighted_relative_l2_to_avg | mean_delta_norm_per_tensor | param_count | tensor_count |
| --- | --- | --- | --- | --- | --- |
| medical_derma | bn_running_stats | 0.207082 | 452.724888 | 9600 | 40 |
| medical_derma | classifier | 0.787614 | 0.884577 | 3591 | 2 |
| medical_derma | layer4 | 0.064024 | 11.530469 | 8389632 | 7 |
| natural_cifar_imbalanced | bn_running_stats | 0.124661 | 128.293867 | 9600 | 40 |
| natural_cifar_imbalanced | classifier | 0.444480 | 0.551882 | 5130 | 2 |
| natural_cifar_imbalanced | layer4 | 0.114771 | 17.306285 | 8389632 | 7 |

## Reading

- The decisive diagnostic is the dominant-class margin distribution, not the existence of non-IID itself.
- In medical Derma, the dominant class has positive margin against close competitors on every test sample; in imbalanced CIFAR, the dominant class has negative low-tail margins against several competitors, so other classes can still win.
- The final-feature table checks whether the averaged representation preserves class-dependent variation. The margin table checks whether that variation is large enough to overcome the dominant-class offset.
