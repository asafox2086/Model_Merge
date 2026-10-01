# Equal-Count Client Control

- Source case: `model_hub/small/dermamnist_224/resnet/clients_3/beta_0/seed_42`
- Output hub: `medmerge_empirical_study/results/experiment8_balanced_local_clients/equal_count_hub/small/dermamnist_224/resnet/clients_3/balanced_local/seed_42`
- Control: every seen class inside every client is downsampled to `80` train samples.
- Client totals differ when clients have different numbers of seen classes.
- Client class groups are unchanged.

## Client Sampling and Single-Client Evaluation

| client_id | classes | available_num_samples | sampled_num_samples | sampled_class_counts | best_val_acc | source_test_acc | source_test_bacc | source_test_pred_counts | public_test_acc | public_test_bacc | public_test_pred_counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | [3, 4, 0] | 1087 | 240 | {"3": 80, "4": 80, "0": 80} | 0.130608 | 0.128678 | 0.350887 | [301, 0, 0, 643, 1061, 0, 0] | 0.192701 | 0.236549 | [85, 0, 0, 367, 233, 0, 0] |
| 1 | [2, 1] | 1128 | 160 | {"2": 80, "1": 80} | 0.142572 | 0.143142 | 0.250536 | [0, 209, 1796, 0, 0, 0, 0] | 0.173723 | 0.157734 | [0, 81, 604, 0, 0, 0, 0] |
| 2 | [6, 5] | 4792 | 160 | {"6": 80, "5": 80} | 0.677966 | 0.676808 | 0.279510 | [0, 0, 0, 0, 0, 1929, 76] | 0.214599 | 0.237759 | [0, 0, 0, 0, 0, 570, 115] |

## Equal-Count AVG Evaluation

| model | split | accuracy | balanced_accuracy | macro_f1 | collapse_ratio | effective_pred_classes | top_pred_class | mean_confidence | pred_counts | support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| avg | source_test | 0.449875 | 0.403608 | 0.325453 | 0.430424 | 7 | 5 | 0.225150 | [284, 146, 410, 34, 230, 863, 38] | [66, 103, 220, 23, 223, 1341, 29] |
| avg | public_test | 0.188321 | 0.182572 | 0.179877 | 0.392701 | 7 | 2 | 0.204216 | [157, 51, 269, 61, 9, 110, 28] | [128, 128, 103, 71, 81, 101, 73] |

## Reading

- This isolates local class imbalance within each client. It still does not make the data IID because label support remains disjoint.
- If collapse persists or class-wise metrics remain low, equal total sample count is insufficient; label-support mismatch is the core issue.
