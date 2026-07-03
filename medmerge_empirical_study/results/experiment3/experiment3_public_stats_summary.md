# 实验三：同类型公开数据替代源域验证集

## 工作流核对

对应 `experiment_workflow.md` 第 6 和第 7 节：固定客户端 checkpoint 与源域 test set，只替换 Fisher/RegMean/AdaMerging/BN recalibration 的统计或校准图像来源。

## 结果摘要

| method | calibration source | cases | mean acc | mean balanced acc | mean macro F1 | mean collapse ratio | median effective classes |
|---|---|---:|---:|---:|---:|---:|---:|
| adamerging | public_labeled_reweighted_512 | 6 | 0.4899 | 0.2322 | 0.1650 | 0.8411 | 2.10 |
| adamerging | public_labeled_val_512 | 6 | 0.4650 | 0.2097 | 0.1481 | 0.8076 | 2.15 |
| adamerging | source_val_512 | 6 | 0.4869 | 0.2299 | 0.1653 | 0.8455 | 1.88 |
| avg | no_image_stats | 6 | 0.4818 | 0.2496 | 0.1878 | 0.7339 | 2.25 |
| bn_recalibration | no_recalibration | 6 | 0.4818 | 0.2496 | 0.1878 | 0.7339 | 2.25 |
| bn_recalibration | public_labeled_reweighted_512 | 6 | 0.5114 | 0.2480 | 0.1941 | 0.8449 | 1.97 |
| bn_recalibration | public_labeled_val_512 | 6 | 0.5061 | 0.2366 | 0.1872 | 0.8621 | 1.85 |
| bn_recalibration | source_val_512 | 6 | 0.6330 | 0.3719 | 0.3481 | 0.7141 | 2.98 |
| fisher | public_labeled_reweighted_512 | 6 | 0.4945 | 0.2897 | 0.2353 | 0.6452 | 3.03 |
| fisher | public_labeled_val_512 | 6 | 0.4944 | 0.2871 | 0.2290 | 0.6681 | 2.72 |
| fisher | source_val_512 | 6 | 0.5015 | 0.2421 | 0.1878 | 0.7713 | 2.36 |
| regmean | public_labeled_reweighted_512 | 6 | 0.4496 | 0.2784 | 0.2178 | 0.5970 | 3.29 |
| regmean | public_labeled_val_512 | 6 | 0.4468 | 0.2861 | 0.2234 | 0.5887 | 3.35 |
| regmean | source_val_512 | 6 | 0.4518 | 0.2586 | 0.2008 | 0.6712 | 2.95 |

## 结论读取方式

重点比较同一 method 下 `source_val_512`、`public_labeled_val_512`、`public_labeled_reweighted_512` 与 `no_image_stats`。如果 source validation 明显高于 public，说明同类型公开医学数据不能稳定替代源域图像统计。
