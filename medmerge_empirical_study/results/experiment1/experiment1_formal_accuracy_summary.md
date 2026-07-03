# 实验一：已有融合方法正式结果诊断

## 工作流核对

- 已使用已有训练好客户端对应的正式结果表。
- 已覆盖 accuracy、average rank、win count、delta vs Avg、类别先验相关低效诊断。
- 现有正式表不含逐样本预测，因此 balanced accuracy、macro F1、真实 collapse rate 需要通过 `collect_prediction_metrics.py` 重新评估生成。

## 覆盖范围

- 数据集数：7，数据集：bloodmnist, chaoshengmnist, dermamnist, organamnist, organcmnist, organsmnist, pathmnist
- 方法数：12，方法：avg, breadcrumbs, dare_linear, dare_ties, fisher, free_merge, from, iso_c, model_stock, regmean, robustmerge, ties
- setting 数：567
- accuracy 记录数：6804

## 类别先验

| dataset | classes | test n | majority prior | majority class |
|---|---:|---:|---:|---:|
| bloodmnist | 8 | 3421 | 0.1947 | 6 |
| chaoshengmnist | 8 | 1113 | 0.1734 | 6 |
| dermamnist | 7 | 2005 | 0.6688 | 5 |
| organamnist | 11 | 17778 | 0.1848 | 6 |
| organcmnist | 11 | 8216 | 0.2233 | 6 |
| organsmnist | 11 | 8827 | 0.2354 | 6 |
| pathmnist | 9 | 7180 | 0.1864 | 0 |

## 方法排名摘要

| method | mean acc | median acc | wins | top3 | avg rank | low-performance rate | near-majority-prior rate | mean delta vs avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| from | 0.2154 | 0.1547 | 111 | 158 | 5.80 | 39.3% | 14.1% | +0.0143 |
| ties | 0.2050 | 0.1528 | 115 | 231 | 5.52 | 40.6% | 12.3% | +0.0038 |
| fisher | 0.2048 | 0.1374 | 111 | 191 | 6.10 | 44.8% | 15.2% | +0.0037 |
| avg | 0.2011 | 0.1392 | 63 | 182 | 5.02 | 43.7% | 14.3% | +0.0000 |
| free_merge | 0.2007 | 0.1391 | 74 | 58 | 7.43 | 43.0% | 14.3% | -0.0005 |
| iso_c | 0.1933 | 0.1394 | 79 | 116 | 6.75 | 45.1% | 9.0% | -0.0078 |
| regmean | 0.1878 | 0.1374 | 78 | 144 | 6.20 | 47.3% | 9.5% | -0.0134 |
| dare_linear | 0.1852 | 0.1352 | 54 | 132 | 6.21 | 46.9% | 13.4% | -0.0160 |
| dare_ties | 0.1829 | 0.1284 | 71 | 169 | 6.59 | 48.3% | 11.8% | -0.0182 |
| model_stock | 0.1794 | 0.1366 | 69 | 90 | 7.02 | 47.4% | 13.2% | -0.0217 |
| robustmerge | 0.1738 | 0.1374 | 79 | 134 | 7.15 | 47.1% | 10.1% | -0.0273 |
| breadcrumbs | 0.1299 | 0.1087 | 50 | 96 | 8.22 | 67.4% | 11.8% | -0.0713 |

## 方法组摘要

| group | cases | mean acc | median acc | low-performance rate | near-majority-prior rate |
|---|---:|---:|---:|---:|---:|
| weight_geometry | 5670 | 0.1867 | 0.1356 | 46.9% | 12.4% |
| image_statistics | 1134 | 0.1963 | 0.1374 | 46.0% | 12.3% |

## 初步结论

1. 从正式 accuracy 表看，没有单一方法跨全部 setting 稳定领先。
2. 多个方法的 median accuracy 接近数据集随机或多数类先验，提示需要逐样本预测诊断。
3. DermaMNIST 等类别不均衡数据集的多数类先验很高，单看 accuracy 容易把多数类预测误判为有效融合。
4. 下一步必须根据工作流运行逐样本评估，补充 balanced accuracy、macro F1、per-class recall 和真实 collapse rate。

## Top methods by mean accuracy

1. from: mean accuracy 0.2154
2. ties: mean accuracy 0.2050
3. fisher: mean accuracy 0.2048
4. avg: mean accuracy 0.2011
5. free_merge: mean accuracy 0.2007
