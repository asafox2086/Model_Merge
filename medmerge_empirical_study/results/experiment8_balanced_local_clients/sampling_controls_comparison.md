# Sampling Controls Comparison

## AVG Results

| case | split | acc | bacc | f1 | collapse | eff | top | pred_counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original_unequal_local_imbalanced | source_test | 0.668828 | 0.142857 | 0.114508 | 1.000000 | 1 | 5 | [0, 0, 0, 0, 0, 2005, 0] |
| original_unequal_local_imbalanced | public_test | 0.147445 | 0.142857 | 0.036714 | 1.000000 | 1 | 5 | [0, 0, 0, 0, 0, 685, 0] |
| equal_total_preserve_local_ratio | source_test | 0.160100 | 0.286083 | 0.164275 | 0.811970 | 6 | 4 | [16, 240, 118, 2, 1628, 0, 1] |
| equal_total_preserve_local_ratio | public_test | 0.189781 | 0.185372 | 0.119944 | 0.575182 | 4 | 4 | [10, 159, 122, 0, 394, 0, 0] |
| balanced_local_equal_per_seen_class | source_test | 0.449875 | 0.403608 | 0.325453 | 0.430424 | 7 | 5 | [284, 146, 410, 34, 230, 863, 38] |
| balanced_local_equal_per_seen_class | public_test | 0.188321 | 0.182572 | 0.179877 | 0.392701 | 7 | 2 | [157, 51, 269, 61, 9, 110, 28] |

## Interpretation

- Equal total client size alone changes the dominant class but does not restore balanced prediction.
- Balancing local seen classes substantially reduces merge collapse and activates all 7 output classes, but individual clients remain partial-label experts.
- Client-level non-half predictions on global test are expected because most global-test samples are outside that client's label support; they must be forced into one of the seen-class regions.
- Therefore, local class balance helps, but disjoint label support still makes the setup non-IID and not equivalent to normal IID training.
