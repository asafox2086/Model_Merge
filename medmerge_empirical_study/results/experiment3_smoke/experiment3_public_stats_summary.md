# 实验三：同类型公开数据替代源域验证集

## 工作流核对

对应 `experiment_workflow.md` 第 6 和第 7 节：固定客户端 checkpoint 与源域 test set，只替换 Fisher/RegMean/AdaMerging/BN recalibration 的统计或校准图像来源。

## 结果摘要

| method | calibration source | cases | mean acc | mean balanced acc | mean macro F1 | mean collapse ratio | median effective classes |
|---|---|---:|---:|---:|---:|---:|---:|
| avg | no_image_stats | 1 | 0.6688 | 0.1429 | 0.1145 | 1.0000 | 1.00 |
| bn_recalibration | no_recalibration | 1 | 0.6688 | 0.1429 | 0.1145 | 1.0000 | 1.00 |
| bn_recalibration | public_labeled_reweighted_512 | 1 | 0.6688 | 0.1434 | 0.1158 | 0.9980 | 1.02 |
| bn_recalibration | public_labeled_val_512 | 1 | 0.6673 | 0.1436 | 0.1169 | 0.9955 | 1.03 |
| bn_recalibration | source_val_512 | 1 | 0.6599 | 0.1447 | 0.1218 | 0.9796 | 1.11 |
| fisher | public_labeled_reweighted_512 | 1 | 0.4314 | 0.2038 | 0.1637 | 0.4698 | 2.51 |
| fisher | public_labeled_val_512 | 1 | 0.6539 | 0.1589 | 0.1427 | 0.9576 | 1.24 |
| fisher | source_val_512 | 1 | 0.6688 | 0.1429 | 0.1145 | 0.9995 | 1.00 |
| regmean | public_labeled_reweighted_512 | 1 | 0.6035 | 0.1473 | 0.1320 | 0.8873 | 1.60 |
| regmean | public_labeled_val_512 | 1 | 0.5840 | 0.1924 | 0.1679 | 0.8354 | 1.90 |
| regmean | source_val_512 | 1 | 0.6514 | 0.1598 | 0.1524 | 0.9287 | 1.42 |

## 结论读取方式

重点比较同一 method 下 `source_val_512`、`public_labeled_val_512`、`public_labeled_reweighted_512` 与 `no_image_stats`。如果 source validation 明显高于 public，说明同类型公开医学数据不能稳定替代源域图像统计。
