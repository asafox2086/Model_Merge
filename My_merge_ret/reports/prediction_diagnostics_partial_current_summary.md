### 3.1 已落盘预测诊断结果（阶段性）

截至本次写入，后台 full-scope prediction diagnostics 已产生 4672 条去重记录，其中 `status=OK` 的记录为 4614 条；按实验 cell 聚合后得到 3841 个可统计 cell。完整 full-scope 目标为 5400 条模型评测记录；当前表仅用于记录已落盘证据，最终论文表格将在所有 shard 完成后替换。

| Dataset | LAMP cells | Client cells | Observed settings |
|---|---:|---:|---:|
| bloodmnist_224 | 34/36 | 35/36 | 25 |
| chaoshengmnist_224 | 27/36 | 27/36 | 25 |
| dermamnist_224 | 36/36 | 36/36 | 25 |
| organcmnist_224 | 29/36 | 30/36 | 25 |
| organsmnist_224 | 29/36 | 30/36 | 25 |

下表先在同一 `(dataset, backbone, K, beta)` 内聚合客户端模型，再对已落盘 cell 取均值。因此，客户端行不会因为客户端数量较大而被额外加权。Collapse Ratio 越低表示预测越不集中于单一类别；Effective Classes 越高表示预测类别覆盖越充分；Pred-True TV 越低表示预测类别分布越接近测试集真实类别分布。

| Setting | Cells | Acc | BA | Macro-F1 | Collapse Ratio | Effective Classes | Pred-True TV |
|---|---:|---:|---:|---:|---:|---:|---:|
| Client mean | 158 | 0.1719 | 0.1476 | 0.0603 | 0.8371 | 1.7052 | 0.7722 |
| avg | 157 | 0.2345 | 0.1429 | 0.0672 | 0.8607 | 1.5740 | 0.7116 |
| ties | 157 | 0.2002 | 0.1437 | 0.0633 | 0.8421 | 1.6476 | 0.7388 |
| dare_linear | 157 | 0.2088 | 0.1378 | 0.0605 | 0.8622 | 1.5548 | 0.7379 |
| dare_ties | 157 | 0.1932 | 0.1380 | 0.0578 | 0.8437 | 1.6558 | 0.7392 |
| regmean | 157 | 0.2127 | 0.1526 | 0.0801 | 0.7754 | 2.1725 | 0.6809 |
| fisher | 99 | 0.2735 | 0.1719 | 0.1014 | 0.7741 | 2.0576 | 0.6452 |
| breadcrumbs | 157 | 0.1419 | 0.1170 | 0.0367 | 0.8865 | 1.4495 | 0.8047 |
| model_stock | 157 | 0.2171 | 0.1352 | 0.0595 | 0.8625 | 1.5427 | 0.7217 |
| from | 157 | 0.2349 | 0.1458 | 0.0684 | 0.8572 | 1.5684 | 0.7193 |
| iso_c | 156 | 0.2310 | 0.1526 | 0.0812 | 0.7845 | 1.9567 | 0.6763 |
| free_merge | 156 | 0.2370 | 0.1433 | 0.0686 | 0.8703 | 1.5449 | 0.7137 |
| robustmerge | 156 | 0.1891 | 0.1441 | 0.0619 | 0.8439 | 1.6659 | 0.7555 |
| LAMP-Merge | 155 | 0.6302 | 0.5355 | 0.5211 | 0.3284 | 6.8957 | 0.1268 |
| M1 only | 155 | 0.5931 | 0.5753 | 0.5330 | 0.2359 | 7.9023 | 0.1520 |
| Classifier-head aggregation | 155 | 0.2688 | 0.1407 | 0.0920 | 0.6942 | 2.6250 | 0.5388 |
| Global-feature mean | 155 | 0.2758 | 0.1164 | 0.0472 | 1.0000 | 1.0000 | 0.7242 |
| Support-only synthetic head | 155 | 0.1170 | 0.1168 | 0.0605 | 0.6019 | 3.3529 | 0.6276 |
| Shuffled-label prototype | 155 | 0.2097 | 0.1462 | 0.1347 | 0.2664 | 7.2087 | 0.2668 |
| Uniform client weight | 155 | 0.5949 | 0.5009 | 0.4819 | 0.3376 | 6.7136 | 0.1613 |
| Binary support only | 155 | 0.5584 | 0.5346 | 0.4916 | 0.2539 | 7.6086 | 0.1849 |
| Global client-size weight | 155 | 0.6059 | 0.5030 | 0.4846 | 0.3446 | 6.6583 | 0.1515 |
| No prevalence calibration | 155 | 0.5931 | 0.5753 | 0.5330 | 0.2359 | 7.9023 | 0.1520 |
| Uniform prevalence prior | 155 | 0.5931 | 0.5753 | 0.5330 | 0.2359 | 7.9023 | 0.1520 |
| Smoothed prevalence prior | 155 | 0.6302 | 0.5356 | 0.5211 | 0.3281 | 6.8995 | 0.1267 |

数据集级阶段性结果如下。`Generic baseline mean` 对已落盘的通用融合基线取均值；`Best baseline` 表示当前已落盘 cell 中该数据集 Accuracy 最高的固定通用基线。该表用于诊断趋势，不替代最终 full-scope 表。

| Dataset | Setting | Cells | Acc | BA | Macro-F1 | Collapse Ratio | Effective Classes | Pred-True TV |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bloodmnist_224 | Client mean | 35 | 0.1590 | 0.1520 | 0.0561 | 0.9241 | 1.2812 | 0.8215 |
| bloodmnist_224 | Generic baseline mean | 390 | 0.1725 | 0.1569 | 0.0665 | 0.8800 | 1.4951 | 0.7823 |
| bloodmnist_224 | Best baseline (fisher) | 16 | 0.2055 | 0.1899 | 0.1012 | 0.8062 | 1.7915 | 0.7397 |
| bloodmnist_224 | LAMP-Merge | 34 | 0.8164 | 0.8045 | 0.7997 | 0.1963 | 7.4781 | 0.0400 |
| bloodmnist_224 | M1 only | 34 | 0.8164 | 0.8045 | 0.7997 | 0.1963 | 7.4781 | 0.0400 |
| bloodmnist_224 | Binary support only | 34 | 0.7806 | 0.7691 | 0.7592 | 0.2126 | 7.3935 | 0.0729 |
| bloodmnist_224 | Uniform client weight | 34 | 0.7806 | 0.7691 | 0.7592 | 0.2126 | 7.3935 | 0.0729 |
| bloodmnist_224 | Shuffled-label prototype | 34 | 0.2439 | 0.2118 | 0.2031 | 0.1962 | 7.4685 | 0.1995 |
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
| organcmnist_224 | Generic baseline mean | 356 | 0.1336 | 0.1264 | 0.0538 | 0.7518 | 2.0985 | 0.7739 |
| organcmnist_224 | Best baseline (fisher) | 26 | 0.1701 | 0.1609 | 0.0907 | 0.6675 | 2.6504 | 0.7128 |
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

