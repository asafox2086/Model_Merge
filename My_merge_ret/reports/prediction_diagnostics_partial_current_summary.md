### 3.1 已落盘预测诊断结果（阶段性）

截至本次写入，后台 full-scope prediction diagnostics 已产生 4640 条去重记录，其中 `status=OK` 的记录为 4577 条；按实验 cell 聚合后得到 3811 个可统计 cell。完整 full-scope 目标为 5400 条模型评测记录；当前表仅用于记录已落盘证据，最终论文表格将在所有 shard 完成后替换。

| Dataset | LAMP cells | Client cells | Observed settings |
|---|---:|---:|---:|
| bloodmnist_224 | 33/36 | 34/36 | 25 |
| chaoshengmnist_224 | 27/36 | 27/36 | 25 |
| dermamnist_224 | 36/36 | 36/36 | 25 |
| organcmnist_224 | 29/36 | 30/36 | 25 |
| organsmnist_224 | 29/36 | 30/36 | 25 |

下表先在同一 `(dataset, backbone, K, beta)` 内聚合客户端模型，再对已落盘 cell 取均值。因此，客户端行不会因为客户端数量较大而被额外加权。Collapse Ratio 越低表示预测越不集中于单一类别；Effective Classes 越高表示预测类别覆盖越充分；Pred-True TV 越低表示预测类别分布越接近测试集真实类别分布。

| Setting | Cells | Acc | BA | Macro-F1 | Collapse Ratio | Effective Classes | Pred-True TV |
|---|---:|---:|---:|---:|---:|---:|---:|
| Client mean | 157 | 0.1722 | 0.1478 | 0.0605 | 0.8363 | 1.7088 | 0.7718 |
| avg | 156 | 0.2349 | 0.1430 | 0.0674 | 0.8599 | 1.5771 | 0.7110 |
| ties | 156 | 0.2006 | 0.1439 | 0.0635 | 0.8411 | 1.6518 | 0.7380 |
| dare_linear | 156 | 0.2091 | 0.1379 | 0.0607 | 0.8613 | 1.5583 | 0.7373 |
| dare_ties | 156 | 0.1932 | 0.1381 | 0.0579 | 0.8427 | 1.6599 | 0.7387 |
| regmean | 156 | 0.2130 | 0.1528 | 0.0804 | 0.7739 | 2.1801 | 0.6799 |
| fisher | 93 | 0.2841 | 0.1765 | 0.1063 | 0.7677 | 2.1012 | 0.6319 |
| breadcrumbs | 156 | 0.1417 | 0.1169 | 0.0367 | 0.8858 | 1.4524 | 0.8045 |
| model_stock | 156 | 0.2174 | 0.1351 | 0.0596 | 0.8633 | 1.5412 | 0.7214 |
| from | 156 | 0.2353 | 0.1459 | 0.0686 | 0.8562 | 1.5721 | 0.7186 |
| iso_c | 155 | 0.2316 | 0.1528 | 0.0813 | 0.7862 | 1.9528 | 0.6763 |
| free_merge | 155 | 0.2374 | 0.1434 | 0.0687 | 0.8720 | 1.5423 | 0.7134 |
| robustmerge | 155 | 0.1892 | 0.1443 | 0.0621 | 0.8429 | 1.6702 | 0.7550 |
| LAMP-Merge | 154 | 0.6288 | 0.5336 | 0.5191 | 0.3293 | 6.8915 | 0.1274 |
| M1 only | 154 | 0.5914 | 0.5736 | 0.5311 | 0.2362 | 7.9046 | 0.1528 |
| Classifier-head aggregation | 154 | 0.2700 | 0.1408 | 0.0924 | 0.6933 | 2.6302 | 0.5374 |
| Global-feature mean | 154 | 0.2771 | 0.1163 | 0.0474 | 1.0000 | 1.0000 | 0.7229 |
| Support-only synthetic head | 154 | 0.1170 | 0.1170 | 0.0606 | 0.6014 | 3.3587 | 0.6278 |
| Shuffled-label prototype | 154 | 0.2095 | 0.1457 | 0.1342 | 0.2670 | 7.2066 | 0.2673 |
| Uniform client weight | 154 | 0.5932 | 0.4988 | 0.4797 | 0.3386 | 6.7082 | 0.1622 |
| Binary support only | 154 | 0.5565 | 0.5327 | 0.4895 | 0.2544 | 7.6091 | 0.1859 |
| Global client-size weight | 154 | 0.6043 | 0.5009 | 0.4824 | 0.3456 | 6.6526 | 0.1523 |
| No prevalence calibration | 154 | 0.5914 | 0.5736 | 0.5311 | 0.2362 | 7.9046 | 0.1528 |
| Uniform prevalence prior | 154 | 0.5914 | 0.5736 | 0.5311 | 0.2362 | 7.9046 | 0.1528 |
| Smoothed prevalence prior | 154 | 0.6287 | 0.5337 | 0.5191 | 0.3290 | 6.8953 | 0.1273 |

数据集级阶段性结果如下。`Generic baseline mean` 对已落盘的通用融合基线取均值；`Best baseline` 表示当前已落盘 cell 中该数据集 Accuracy 最高的固定通用基线。该表用于诊断趋势，不替代最终 full-scope 表。

| Dataset | Setting | Cells | Acc | BA | Macro-F1 | Collapse Ratio | Effective Classes | Pred-True TV |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bloodmnist_224 | Client mean | 34 | 0.1598 | 0.1528 | 0.0568 | 0.9227 | 1.2852 | 0.8210 |
| bloodmnist_224 | Generic baseline mean | 378 | 0.1727 | 0.1578 | 0.0673 | 0.8792 | 1.5019 | 0.7816 |
| bloodmnist_224 | Best baseline (fisher) | 15 | 0.2063 | 0.1943 | 0.1052 | 0.7933 | 1.8443 | 0.7353 |
| bloodmnist_224 | LAMP-Merge | 33 | 0.8154 | 0.8036 | 0.7988 | 0.1966 | 7.4761 | 0.0402 |
| bloodmnist_224 | M1 only | 33 | 0.8154 | 0.8036 | 0.7988 | 0.1966 | 7.4761 | 0.0402 |
| bloodmnist_224 | Binary support only | 33 | 0.7785 | 0.7671 | 0.7571 | 0.2134 | 7.3890 | 0.0741 |
| bloodmnist_224 | Uniform client weight | 33 | 0.7785 | 0.7671 | 0.7571 | 0.2134 | 7.3890 | 0.0741 |
| bloodmnist_224 | Shuffled-label prototype | 33 | 0.2438 | 0.2116 | 0.2029 | 0.1965 | 7.4665 | 0.1998 |
| chaoshengmnist_224 | Client mean | 27 | 0.1656 | 0.1618 | 0.0698 | 0.8568 | 1.5434 | 0.7871 |
| chaoshengmnist_224 | Generic baseline mean | 315 | 0.1677 | 0.1543 | 0.0685 | 0.8491 | 1.6058 | 0.7683 |
| chaoshengmnist_224 | Best baseline (fisher) | 18 | 0.2053 | 0.1882 | 0.1104 | 0.7595 | 1.9625 | 0.7078 |
| chaoshengmnist_224 | LAMP-Merge | 27 | 0.4684 | 0.4595 | 0.4419 | 0.1978 | 7.5047 | 0.1447 |
| chaoshengmnist_224 | M1 only | 27 | 0.4684 | 0.4595 | 0.4419 | 0.1978 | 7.5047 | 0.1447 |
| chaoshengmnist_224 | Binary support only | 27 | 0.4274 | 0.4186 | 0.3973 | 0.2262 | 7.1711 | 0.1889 |
| chaoshengmnist_224 | Uniform client weight | 27 | 0.4274 | 0.4186 | 0.3973 | 0.2262 | 7.1711 | 0.1889 |
| chaoshengmnist_224 | Shuffled-label prototype | 27 | 0.1408 | 0.1279 | 0.1350 | 0.1960 | 7.5295 | 0.1975 |
| dermamnist_224 | Client mean | 36 | 0.2511 | 0.1553 | 0.0630 | 0.8805 | 1.5289 | 0.7039 |
| dermamnist_224 | Generic baseline mean | 423 | 0.4026 | 0.1515 | 0.0853 | 0.9205 | 1.3021 | 0.5574 |
| dermamnist_224 | Best baseline (free_merge) | 36 | 0.5089 | 0.1504 | 0.0997 | 0.9522 | 1.1641 | 0.4606 |
| dermamnist_224 | LAMP-Merge | 36 | 0.6286 | 0.3242 | 0.2773 | 0.6995 | 2.9585 | 0.1737 |
| dermamnist_224 | M1 only | 36 | 0.4666 | 0.4413 | 0.2922 | 0.3754 | 5.6120 | 0.3244 |
| dermamnist_224 | Binary support only | 36 | 0.4499 | 0.4028 | 0.2706 | 0.4001 | 5.2745 | 0.3335 |
| dermamnist_224 | Uniform client weight | 36 | 0.6025 | 0.3036 | 0.2582 | 0.6850 | 3.0261 | 0.1990 |
| dermamnist_224 | Shuffled-label prototype | 36 | 0.3015 | 0.1366 | 0.1162 | 0.4750 | 4.2346 | 0.3898 |
| organcmnist_224 | Client mean | 30 | 0.1302 | 0.1356 | 0.0593 | 0.7313 | 2.2168 | 0.7780 |
| organcmnist_224 | Generic baseline mean | 351 | 0.1342 | 0.1268 | 0.0543 | 0.7505 | 2.1077 | 0.7727 |
| organcmnist_224 | Best baseline (fisher) | 21 | 0.1883 | 0.1762 | 0.1072 | 0.6248 | 2.9344 | 0.6775 |
| organcmnist_224 | LAMP-Merge | 29 | 0.6139 | 0.5803 | 0.5769 | 0.2122 | 9.3869 | 0.1332 |
| organcmnist_224 | M1 only | 29 | 0.6147 | 0.6048 | 0.5924 | 0.1777 | 10.0925 | 0.1261 |
| organcmnist_224 | Binary support only | 29 | 0.5740 | 0.5593 | 0.5437 | 0.1909 | 9.7407 | 0.1558 |
| organcmnist_224 | Uniform client weight | 29 | 0.5758 | 0.5379 | 0.5303 | 0.2261 | 9.0366 | 0.1619 |
| organcmnist_224 | Shuffled-label prototype | 29 | 0.1476 | 0.1295 | 0.1100 | 0.1895 | 9.2505 | 0.2499 |
| organsmnist_224 | Client mean | 30 | 0.1393 | 0.1327 | 0.0544 | 0.7717 | 2.0456 | 0.7774 |
| organsmnist_224 | Generic baseline mean | 339 | 0.1463 | 0.1225 | 0.0523 | 0.7811 | 2.0354 | 0.7648 |
| organsmnist_224 | Best baseline (fisher) | 12 | 0.2129 | 0.1942 | 0.1194 | 0.5871 | 3.2250 | 0.6375 |
| organsmnist_224 | LAMP-Merge | 29 | 0.5809 | 0.5084 | 0.5148 | 0.2602 | 8.0424 | 0.1472 |
| organsmnist_224 | M1 only | 29 | 0.5829 | 0.5514 | 0.5448 | 0.2026 | 9.4228 | 0.1019 |
| organsmnist_224 | Binary support only | 29 | 0.5388 | 0.5069 | 0.4883 | 0.2099 | 9.0336 | 0.1572 |
| organsmnist_224 | Uniform client weight | 29 | 0.5428 | 0.4713 | 0.4651 | 0.2680 | 7.7452 | 0.1921 |
| organsmnist_224 | Shuffled-label prototype | 29 | 0.1819 | 0.1148 | 0.1017 | 0.2325 | 8.2557 | 0.2744 |

阶段性结果已经体现出两个趋势。第一，通用融合基线的 Collapse Ratio 普遍较高、Effective Classes 较低，说明其预测分布更容易集中到少数类别。第二，LAMP-Merge 在已落盘 cell 上同时取得更高的 Accuracy、Balanced Accuracy 和 Macro-F1，并降低 Pred-True TV，说明类别原型与患病率先验不仅提高总体正确率，也改善了预测类别分布与真实医学类别分布之间的一致性。
