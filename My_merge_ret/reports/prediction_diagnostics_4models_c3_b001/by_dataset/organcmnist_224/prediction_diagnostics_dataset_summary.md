# Prediction Collapse Diagnostics by Dataset

Each table averages over the evaluated model/client/beta cases within one dataset.

`collapse_ratio` is the fraction of test samples assigned to the most frequent predicted class.

## organcmnist_224

| method | cases | acc | balanced acc | macro F1 | collapse ratio | effective classes | pred-true TV |
|---|---:|---:|---:|---:|---:|---:|---:|
| client_mean | 4 | 0.1412 | 0.1359 | 0.0610 | 0.6614 | 2.4316 | 0.7497 |
| client_best | 4 | 0.2297 | 0.1632 | 0.1067 | 0.5474 | 3.1079 | 0.6055 |
| avg | 4 | 0.1462 | 0.1329 | 0.0618 | 0.6549 | 2.1461 | 0.7609 |
| ties | 4 | 0.1609 | 0.1408 | 0.0701 | 0.6589 | 2.6615 | 0.6811 |
| dare_linear | 4 | 0.1538 | 0.1367 | 0.0642 | 0.6070 | 2.6419 | 0.7253 |
| dare_ties | 4 | 0.1386 | 0.1012 | 0.0464 | 0.7355 | 2.0930 | 0.7310 |
| regmean | 4 | 0.1745 | 0.1365 | 0.0678 | 0.6538 | 2.4287 | 0.6953 |
| fisher | 4 | 0.1717 | 0.1708 | 0.1054 | 0.6081 | 3.1308 | 0.6693 |
| breadcrumbs | 4 | 0.1142 | 0.1036 | 0.0311 | 0.9009 | 1.4261 | 0.8171 |
| model_stock | 4 | 0.1380 | 0.1237 | 0.0526 | 0.7743 | 1.8678 | 0.7656 |
| from | 4 | 0.1904 | 0.1570 | 0.0845 | 0.5921 | 2.5737 | 0.6998 |
| iso_c | 4 | 0.1403 | 0.1398 | 0.0757 | 0.7107 | 2.5383 | 0.7409 |
| free_merge | 4 | 0.1186 | 0.1142 | 0.0396 | 0.7945 | 1.6177 | 0.8286 |
| robustmerge | 4 | 0.1425 | 0.1258 | 0.0534 | 0.7085 | 2.0226 | 0.7771 |
| my_merge | 4 | 0.6249 | 0.6175 | 0.6030 | 0.1772 | 10.1092 | 0.1280 |

