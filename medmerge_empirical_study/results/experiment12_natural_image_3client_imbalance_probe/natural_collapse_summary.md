# Natural-Image Partial-Label Collapse Probe

- Dataset: `cifar10_32`
- Model: `resnet`
- Clients: `3`
- Class groups: `0,1,2;3,4,5;8,9`
- Image size: `32`
- Epochs: `8`
- Max train per class: `all`
- Group class counts: `300,300,300;300,300,300;4000,80`
- Output hub: `medmerge_empirical_study/results/experiment12_natural_image_3client_imbalance_probe/natural_probe_hub/small/cifar10_32/resnet/clients_3/partial_label/seed_42`

## Client Sampling and Single-Client Behavior

| client_id | classes | sampled_num_samples | sampled_class_counts | best_val_acc | test_acc | test_bacc | test_collapse | effective_pred_classes | top_pred_class | test_pred_counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | [0, 1, 2] | 900 | {"0": 300, "1": 300, "2": 300} | 0.239000 | 0.230400 | 0.230400 | 0.398300 | 3 | 2 | [2265, 3752, 3983, 0, 0, 0, 0, 0, 0, 0] |
| 1 | [3, 4, 5] | 900 | {"3": 300, "4": 300, "5": 300} | 0.177400 | 0.177300 | 0.177300 | 0.419100 | 3 | 4 | [0, 0, 0, 4044, 4191, 1765, 0, 0, 0, 0] |
| 2 | [8, 9] | 4080 | {"8": 4000, "9": 80} | 0.120800 | 0.115600 | 0.115600 | 0.941700 | 2 | 8 | [0, 0, 0, 0, 0, 0, 0, 0, 9417, 583] |

## AVG Behavior

| model | split | accuracy | balanced_accuracy | macro_f1 | collapse_ratio | effective_pred_classes | top_pred_class | mean_confidence | mean_top1_top2_margin | pred_counts | support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| avg | test | 0.111100 | 0.111100 | 0.071345 | 0.780400 | 8 | 8 | 0.290077 | 0.517584 | [252, 425, 329, 594, 209, 235, 0, 0, 7804, 152] | [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000] |

## All Rows

| model | accuracy | balanced_accuracy | macro_f1 | collapse_ratio | effective_pred_classes | top_pred_class | pred_counts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| client_0 | 0.230400 | 0.230400 | 0.108943 | 0.398300 | 3 | 2 | [2265, 3752, 3983, 0, 0, 0, 0, 0, 0, 0] |
| client_1 | 0.177300 | 0.177300 | 0.082673 | 0.419100 | 3 | 4 | [0, 0, 0, 4044, 4191, 1765, 0, 0, 0, 0] |
| client_2 | 0.115600 | 0.115600 | 0.039016 | 0.941700 | 2 | 8 | [0, 0, 0, 0, 0, 0, 0, 0, 9417, 583] |
| avg | 0.111100 | 0.111100 | 0.071345 | 0.780400 | 8 | 8 | [252, 425, 329, 594, 209, 235, 0, 0, 7804, 152] |

## Reading

- `collapse_ratio` is the fraction of test samples assigned to the most frequent predicted class.
- `effective_pred_classes` counts how many classes receive at least one prediction.
- A medical-style single-class collapse would have `collapse_ratio=1.0` and `effective_pred_classes=1`.
