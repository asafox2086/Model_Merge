# 实验三：同类型公开数据替代源域验证集

## 工作流核对

对应 `experiment_workflow.md` 第 6 和第 7 节：固定客户端 checkpoint 与源域 test set，只替换 Fisher/RegMean/AdaMerging/BN recalibration 的统计或校准图像来源。

## 结果摘要

| method | calibration source | cases | mean acc | mean balanced acc | mean macro F1 | mean collapse ratio | median effective classes |
|---|---|---:|---:|---:|---:|---:|---:|
| avg | no_image_stats | 1 | 0.3017 | 0.2692 | 0.1775 | 0.6764 | 2.21 |
| bn_recalibration | no_recalibration | 1 | 0.3017 | 0.2692 | 0.1775 | 0.6764 | 2.21 |
| bn_recalibration | public_labeled_reweighted_512 | 1 | 0.3332 | 0.2924 | 0.2094 | 0.7749 | 1.97 |
| bn_recalibration | public_labeled_val_512 | 1 | 0.3207 | 0.2745 | 0.1959 | 0.8006 | 1.85 |
| bn_recalibration | source_val_512 | 1 | 0.4598 | 0.4067 | 0.3275 | 0.5773 | 2.98 |
| fisher | public_labeled_reweighted_512 | 1 | 0.4046 | 0.3632 | 0.2725 | 0.5212 | 3.03 |
| fisher | public_labeled_val_512 | 1 | 0.3768 | 0.3575 | 0.2195 | 0.5016 | 2.72 |
| fisher | source_val_512 | 1 | 0.4227 | 0.3821 | 0.3339 | 0.4712 | 4.28 |
| regmean | public_labeled_reweighted_512 | 1 | 0.3180 | 0.2915 | 0.2224 | 0.5522 | 3.29 |
| regmean | public_labeled_val_512 | 1 | 0.3306 | 0.3029 | 0.2331 | 0.5276 | 3.35 |
| regmean | source_val_512 | 1 | 0.3049 | 0.2950 | 0.2340 | 0.6545 | 2.95 |

## 结论读取方式

重点比较同一 method 下 `source_val_512`、`public_labeled_val_512`、`public_labeled_reweighted_512` 与 `no_image_stats`。如果 source validation 明显高于 public，说明同类型公开医学数据不能稳定替代源域图像统计。
