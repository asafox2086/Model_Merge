# Prediction Collapse Diagnostics by Dataset

Each table averages over the evaluated model/client/beta cases within one dataset.

`collapse_ratio` is the fraction of test samples assigned to the most frequent predicted class.

## organsmnist_224

| method | cases | acc | balanced acc | macro F1 | collapse ratio | effective classes | pred-true TV |
|---|---:|---:|---:|---:|---:|---:|---:|
| client_mean | 4 | 0.1486 | 0.1349 | 0.0563 | 0.7784 | 2.0299 | 0.7586 |
| client_best | 4 | 0.2249 | 0.1553 | 0.0852 | 0.7929 | 2.1428 | 0.7199 |
| avg | 4 | 0.1358 | 0.1402 | 0.0635 | 0.6770 | 2.3065 | 0.7739 |
| ties | 4 | 0.0849 | 0.1415 | 0.0520 | 0.6860 | 2.5606 | 0.7823 |
| dare_linear | 4 | 0.1224 | 0.1232 | 0.0535 | 0.6392 | 2.7491 | 0.7173 |
| dare_ties | 4 | 0.1101 | 0.1603 | 0.0631 | 0.6274 | 2.9335 | 0.7590 |
| regmean | 4 | 0.1165 | 0.1416 | 0.0630 | 0.7495 | 2.4814 | 0.7798 |
| fisher | 4 | 0.0993 | 0.1409 | 0.0563 | 0.7415 | 2.4974 | 0.7503 |
| breadcrumbs | 4 | 0.1589 | 0.0982 | 0.0355 | 0.8458 | 1.6706 | 0.7810 |
| model_stock | 4 | 0.1438 | 0.1449 | 0.0659 | 0.6996 | 2.2675 | 0.7588 |
| from | 4 | 0.1070 | 0.1489 | 0.0589 | 0.6332 | 2.5170 | 0.8031 |
| iso_c | 4 | 0.1568 | 0.1482 | 0.0836 | 0.6742 | 2.6297 | 0.7139 |
| free_merge | 4 | 0.1372 | 0.1410 | 0.0656 | 0.6519 | 2.4085 | 0.7691 |
| robustmerge | 4 | 0.2586 | 0.1438 | 0.0881 | 0.6939 | 2.3693 | 0.6226 |
| LAMP-Merge | 4 | 0.5734 | 0.5460 | 0.5397 | 0.2021 | 9.5033 | 0.0993 |

