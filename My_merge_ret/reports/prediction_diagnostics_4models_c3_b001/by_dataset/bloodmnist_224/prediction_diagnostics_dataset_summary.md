# Prediction Collapse Diagnostics by Dataset

Each table averages over the evaluated model/client/beta cases within one dataset.

`collapse_ratio` is the fraction of test samples assigned to the most frequent predicted class.

## bloodmnist_224

| method | cases | acc | balanced acc | macro F1 | collapse ratio | effective classes | pred-true TV |
|---|---:|---:|---:|---:|---:|---:|---:|
| client_mean | 4 | 0.1508 | 0.1668 | 0.0594 | 0.8887 | 1.3608 | 0.8303 |
| client_best | 4 | 0.1997 | 0.1820 | 0.0796 | 0.8164 | 1.5669 | 0.7518 |
| avg | 4 | 0.1663 | 0.1746 | 0.0797 | 0.9072 | 1.5593 | 0.7988 |
| ties | 4 | 0.1659 | 0.1748 | 0.0738 | 0.8532 | 1.5218 | 0.7990 |
| dare_linear | 4 | 0.2286 | 0.1908 | 0.1133 | 0.7851 | 2.0533 | 0.6759 |
| dare_ties | 4 | 0.1700 | 0.1667 | 0.0633 | 0.8162 | 1.5993 | 0.7843 |
| regmean | 4 | 0.2014 | 0.2082 | 0.1069 | 0.7846 | 1.9760 | 0.7507 |
| fisher | 4 | 0.1866 | 0.1822 | 0.0892 | 0.8845 | 1.5790 | 0.7865 |
| breadcrumbs | 4 | 0.1366 | 0.1201 | 0.0353 | 0.9324 | 1.2647 | 0.8094 |
| model_stock | 4 | 0.2081 | 0.1843 | 0.0897 | 0.8123 | 1.8387 | 0.7265 |
| from | 4 | 0.1978 | 0.1985 | 0.0911 | 0.7768 | 1.8028 | 0.7570 |
| iso_c | 4 | 0.2051 | 0.2129 | 0.1115 | 0.8009 | 2.0307 | 0.7329 |
| free_merge | 4 | 0.1900 | 0.2001 | 0.1055 | 0.8691 | 1.6915 | 0.7743 |
| robustmerge | 4 | 0.1733 | 0.1557 | 0.0552 | 0.9263 | 1.2224 | 0.8241 |
| LAMP-Merge | 4 | 0.8190 | 0.8071 | 0.8022 | 0.1943 | 7.4876 | 0.0394 |

