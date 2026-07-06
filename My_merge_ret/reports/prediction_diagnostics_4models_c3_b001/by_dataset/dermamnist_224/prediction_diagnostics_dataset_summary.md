# Prediction Collapse Diagnostics by Dataset

Each table averages over the evaluated model/client/beta cases within one dataset.

`collapse_ratio` is the fraction of test samples assigned to the most frequent predicted class.

## dermamnist_224

| method | cases | acc | balanced acc | macro F1 | collapse ratio | effective classes | pred-true TV |
|---|---:|---:|---:|---:|---:|---:|---:|
| client_mean | 4 | 0.2853 | 0.1584 | 0.0714 | 0.9045 | 1.3942 | 0.6875 |
| client_best | 4 | 0.6726 | 0.1890 | 0.1562 | 0.9670 | 1.1366 | 0.3193 |
| avg | 4 | 0.4984 | 0.1614 | 0.1015 | 0.9259 | 1.3118 | 0.4406 |
| ties | 4 | 0.3648 | 0.1506 | 0.0819 | 0.9360 | 1.2061 | 0.5818 |
| dare_linear | 4 | 0.3717 | 0.1395 | 0.0800 | 0.8818 | 1.4224 | 0.5395 |
| dare_ties | 4 | 0.2374 | 0.1451 | 0.0543 | 0.9802 | 1.0942 | 0.7499 |
| regmean | 4 | 0.3414 | 0.1533 | 0.0798 | 0.8884 | 1.3371 | 0.6072 |
| fisher | 4 | 0.6683 | 0.1507 | 0.1286 | 0.9873 | 1.0598 | 0.3185 |
| breadcrumbs | 4 | 0.0322 | 0.1429 | 0.0089 | 0.9840 | 1.0715 | 0.9589 |
| model_stock | 4 | 0.4360 | 0.1520 | 0.0927 | 0.8549 | 1.4847 | 0.4444 |
| from | 4 | 0.5299 | 0.1478 | 0.1019 | 0.9970 | 1.0191 | 0.4680 |
| iso_c | 4 | 0.2885 | 0.1604 | 0.0723 | 0.8178 | 1.6154 | 0.6665 |
| free_merge | 4 | 0.4988 | 0.1614 | 0.1016 | 0.9259 | 1.3118 | 0.4403 |
| robustmerge | 4 | 0.0461 | 0.1555 | 0.0178 | 0.7978 | 1.5972 | 0.9247 |
| my_merge | 4 | 0.4627 | 0.4492 | 0.2934 | 0.3697 | 5.6501 | 0.3309 |

