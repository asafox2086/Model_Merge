# Prediction Collapse Diagnostics by Dataset

Each table averages over the evaluated model/client/beta cases within one dataset.

`collapse_ratio` is the fraction of test samples assigned to the most frequent predicted class.

## chaoshengmnist_224

| method | cases | acc | balanced acc | macro F1 | collapse ratio | effective classes | pred-true TV |
|---|---:|---:|---:|---:|---:|---:|---:|
| client_mean | 4 | 0.1761 | 0.1614 | 0.0710 | 0.8707 | 1.4686 | 0.7959 |
| client_best | 4 | 0.2511 | 0.2153 | 0.1339 | 0.6858 | 2.1465 | 0.6876 |
| avg | 4 | 0.2055 | 0.1624 | 0.0691 | 0.8491 | 1.5095 | 0.7666 |
| ties | 4 | 0.2682 | 0.2335 | 0.1514 | 0.6047 | 2.7298 | 0.6094 |
| dare_linear | 4 | 0.1959 | 0.1503 | 0.0613 | 0.8756 | 1.4115 | 0.7666 |
| dare_ties | 4 | 0.2253 | 0.1952 | 0.1176 | 0.7538 | 2.2628 | 0.6595 |
| regmean | 4 | 0.2075 | 0.1731 | 0.1031 | 0.7513 | 1.9191 | 0.7145 |
| fisher | 4 | 0.1664 | 0.1577 | 0.0637 | 0.8924 | 1.3952 | 0.8109 |
| breadcrumbs | 4 | 0.1271 | 0.1255 | 0.0291 | 0.9969 | 1.0175 | 0.8702 |
| model_stock | 4 | 0.1774 | 0.1389 | 0.0547 | 0.9364 | 1.2700 | 0.7895 |
| from | 4 | 0.2219 | 0.1740 | 0.0875 | 0.8535 | 1.5321 | 0.7469 |
| iso_c | 4 | 0.1972 | 0.1626 | 0.0790 | 0.7653 | 1.7663 | 0.7318 |
| free_merge | 4 | 0.2053 | 0.1622 | 0.0688 | 0.8493 | 1.5077 | 0.7668 |
| robustmerge | 4 | 0.1224 | 0.1316 | 0.0364 | 0.9863 | 1.0681 | 0.8688 |
| my_merge | 4 | 0.4625 | 0.4536 | 0.4378 | 0.2035 | 7.4446 | 0.1575 |

