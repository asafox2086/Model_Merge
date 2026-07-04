# Equal-Count Client Control

- Source case: `model_hub/small/dermamnist_224/resnet/clients_3/beta_0/seed_42`
- Output hub: `medmerge_empirical_study/results/experiment7_equal_count_clients/equal_count_hub/small/dermamnist_224/resnet/clients_3/equal_total_proportional/seed_42`
- Control: every client is downsampled to `1087` train samples.
- Client class groups are unchanged; within-client class proportions are preserved.

## Client Sampling and Single-Client Evaluation

| client_id | classes | available_num_samples | sampled_num_samples | sampled_class_counts | best_val_acc | source_test_acc | source_test_bacc | source_test_pred_counts | public_test_acc | public_test_bacc | public_test_pred_counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | [3, 4, 0] | 1087 | 1087 | {"3": 80, "4": 779, "0": 228} | 0.144566 | 0.143641 | 0.366058 | [240, 0, 0, 225, 1540, 0, 0] | 0.208759 | 0.242713 | [134, 0, 0, 116, 435, 0, 0] |
| 1 | [2, 1] | 1128 | 1087 | {"2": 741, "1": 346} | 0.157527 | 0.151122 | 0.264614 | [0, 248, 1757, 0, 0, 0, 0] | 0.191241 | 0.172211 | [0, 114, 571, 0, 0, 0, 0] |
| 2 | [6, 5] | 4792 | 1087 | {"6": 22, "5": 1065} | 0.678963 | 0.675810 | 0.235921 | [0, 0, 0, 0, 0, 1963, 42] | 0.189781 | 0.200694 | [0, 0, 0, 0, 0, 631, 54] |

## Equal-Count AVG Evaluation

| model | split | accuracy | balanced_accuracy | macro_f1 | collapse_ratio | effective_pred_classes | top_pred_class | mean_confidence | pred_counts | support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| avg | source_test | 0.160100 | 0.286083 | 0.164275 | 0.811970 | 6 | 4 | 0.317443 | [16, 240, 118, 2, 1628, 0, 1] | [66, 103, 220, 23, 223, 1341, 29] |
| avg | public_test | 0.189781 | 0.185372 | 0.119944 | 0.575182 | 4 | 4 | 0.258095 | [10, 159, 122, 0, 394, 0, 0] | [128, 128, 103, 71, 81, 101, 73] |

## Reading

- This isolates total client sample count. It does not make the data IID because each client still sees only its original class subset.
- If collapse persists or class-wise metrics remain low, equal total sample count is insufficient; label-support mismatch is the core issue.
