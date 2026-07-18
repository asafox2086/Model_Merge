# LAMP-Merge 已完成实验结果

本文档汇总固定正式参数 $\gamma=0.55$、$s=18.75$、$\tau=2.5$、$\lambda=4.25$ 下已经完成并通过一致性检查的结果。正式范围为 5 个医学图像数据集、4 个视觉 backbone（ResNet、ConvNeXt、ViT-Tiny、Swin-Tiny）、$K\in\{3,5,7\}$ 与 $\beta\in\{0,0.01,0.1\}$，共 180 个 raw cells 和 60 个 client-average cells。CLIP-ViT-B/32 与全部 VLM 结果不参与实验或论文。

超参数分析保留原扫描数据，不随本次固定参数更新而重跑；主表、模块间消融、模块内消融、预测诊断、原型几何和联合 t-SNE 均已按新固定参数重新生成。

## 当前状态

| 实验项 | 覆盖范围 | 状态 |
|---|---:|---|
| 正式主表 | 180 raw / 60 client-average | 已完成并校验 |
| 模块间与模块内消融 | 13 settings × 180 raw | 已完成并校验 |
| 预测坍缩诊断 | 5400 rows，其中 13 个 LAMP 设置各 180 rows | 已完成并校验 |
| Prototype geometry | 11 settings × 180 cases | 已完成并校验 |
| 联合 full-scope t-SNE | 5 datasets × 4 backbones | 20 PNG + 20 PDF 已重建 |

## 一、正式主结果

正式输出根目录为：

`outputs/lamp_merge_internal_ablation_full_20260718_formal_gamma055_s18p75_tau2p5_lambda4p25/full`

正式主表为 `My_merge_ret/汇总表.md`。在保留全部通用 baseline 的条件下，LAMP-Merge 不低于最强非 LAMP baseline 的覆盖为：

| 统计范围 | 胜出或并列胜出 | 比例 |
|---|---:|---:|
| 全表 | 211/240 | 87.9% |
| Raw | 154/180 | 85.6% |
| Client Average | 57/60 | 95.0% |

数据集级正式 Accuracy 如下：

| 数据集 | Cells | 平均 Acc | 最小 Acc | 最大 Acc |
|---|---:|---:|---:|---:|
| bloodmnist_224 | 36 | 0.818084 | 0.767904 | 0.852967 |
| chaoshengmnist_224 | 36 | 0.461066 | 0.415993 | 0.491465 |
| dermamnist_224 | 36 | 0.623996 | 0.570574 | 0.676808 |
| organcmnist_224 | 36 | 0.630474 | 0.579722 | 0.685370 |
| organsmnist_224 | 36 | 0.576703 | 0.526906 | 0.628866 |
| 全部 | 180 | 0.622065 | 0.415993 | 0.852967 |

Backbone 级均值如下：

| Backbone | Cells | 平均 Acc |
|---|---:|---:|
| ResNet | 45 | 0.615651 |
| ConvNeXt | 45 | 0.625863 |
| ViT-Tiny | 45 | 0.607281 |
| Swin-Tiny | 45 | 0.639463 |

新正式均值相对旧主表的 0.620988 提升至 0.622065，即提升约 0.11 个百分点。逐 cell 对比中，90 个提升、21 个持平、69 个下降；最大单 cell 降幅约 1.10 个百分点。因此应表述为总体性能未下降并略有提升，而不应声称每个 cell 都提升。

## 二、增量式模块间消融

| 设置 | Client-average cells | Mean Acc | Mean margin vs avg | Best/tied among four |
|---|---:|---:|---:|---:|
| avg | 60 | 0.220385 | 0.000000 | 0/60 |
| M1 only | 60 | 0.589122 | +0.368737 | 30/60 |
| LAMP-Merge | 60 | 0.622065 | +0.401680 | 45/60 |
| avg+M2 | 60 | 0.291759 | +0.071374 | 9/60 |

总体序列严格满足：

```text
avg 0.220385 < M1 only 0.589122 < LAMP-Merge 0.622065
```

M2 的数据集级变化如下：

| 数据集 | 不平衡强度 r | M1 only | LAMP-Merge | M2 增量 |
|---|---:|---:|---:|---:|
| bloodmnist_224 | 1.5587 | 0.818084 | 0.818084 | 0.000000 |
| chaoshengmnist_224 | 1.3874 | 0.461066 | 0.461066 | 0.000000 |
| dermamnist_224 | 4.6883 | 0.465780 | 0.623996 | +0.158216 |
| organcmnist_224 | 2.5315 | 0.625693 | 0.630474 | +0.004781 |
| organsmnist_224 | 2.7350 | 0.574988 | 0.576703 | +0.001715 |

统一阈值 $r>\tau=2.5$ 只在 Derma、Organ-C 和 Organ-S 上触发，三个数据集均取得正向数据集均值增量。Organ-C 与 Organ-S 的 ResNet 子组存在下降，因此论文应陈述“触发的三个数据集均值提升”，而不是“所有 backbone 均提升”。

`avg+M2` 的总体 Accuracy 仅为 0.291759，预测坍缩率达到 0.9551，说明 M2 不能脱离 M1 单独恢复全局诊断能力。

## 三、内部消融

| 设置 | Raw cells | Client-average mean Acc | Mean margin vs LAMP |
|---|---:|---:|---:|
| LAMP-Merge | 180 | 0.6221 | 0.0000 |
| M1 only | 180 | 0.5891 | -0.0329 |
| avg+M2 | 180 | 0.2918 | -0.3303 |
| Classifier-head aggregation | 180 | 0.2563 | -0.3658 |
| Global-feature mean | 180 | 0.2650 | -0.3570 |
| Support-only synthetic head | 180 | 0.1089 | -0.5132 |
| Shuffled-label prototype | 180 | 0.1957 | -0.4263 |
| Uniform client weight | 180 | 0.5833 | -0.0388 |
| Binary support only | 180 | 0.5518 | -0.0703 |
| Global client-size weight | 180 | 0.5939 | -0.0281 |
| No prevalence calibration | 180 | 0.5891 | -0.0329 |
| Uniform prevalence prior | 180 | 0.5891 | -0.0329 |
| Smoothed prevalence prior | 180 | 0.6220 | -0.0001 |

完整结果位于 `My_merge_ret/reports/lamp_merge_internal_ablation_full*.csv` 与 `lamp_merge_internal_ablation_full_summary.md`。

## 四、预测诊断

正式诊断表包含 5400 条 `status=OK` 记录：2340 条新 LAMP 记录、2160 条通用 merging baseline 记录和 900 条客户端记录。

| 设置 | Cases | BA | Macro-F1 | Collapse ratio | Effective classes | Pred-True TV |
|---|---:|---:|---:|---:|---:|---:|
| LAMP-Merge | 180 | 0.5422 | 0.5281 | 0.3063 | 7.2020 | 0.1242 |
| M1 only | 180 | 0.5737 | 0.5352 | 0.2307 | 8.0288 | 0.1492 |
| avg+M2 | 180 | 0.1255 | 0.0601 | 0.9551 | 1.1633 | 0.6938 |
| Global-feature mean | 180 | 0.1149 | 0.0447 | 1.0000 | 1.0000 | 0.7385 |
| Smoothed prevalence prior | 180 | 0.5423 | 0.5281 | 0.3060 | 7.2064 | 0.1243 |

M1 only 的类均衡指标更高；完整方法将 Pred-True TV 从 0.1492 降至 0.1242，并提高总体 Accuracy，符合“M1 恢复多类别判别、M2 进行有界患病率校准”的设计。正式 Accuracy 应以第三节消融评估为准；预测诊断中的退化构造可能因近似 logit 并列对 batch/GPU 数值顺序敏感。

## 五、Prototype Geometry 与 t-SNE

正式 LAMP-Merge 的全量原型几何均值为：

| Cases | Pairwise distance | Nearest-class distance | Consistency | Proto-client alignment | Evidence entropy |
|---:|---:|---:|---:|---:|---:|
| 180 | 0.1115 | 0.0355 | 0.8751 | 0.9419 | 0.2124 |

数值报告位于 `My_merge_ret/reports/lamp_merge_prototype_geometry*.csv`。5 个数据集 × 4 个 backbone 的 20 组联合 full-scope t-SNE 位于 `My_merge_ret/figures/lamp_merge_prototype_geometry/full_scope_tsne_comparison/`，每组同时提供 PNG 与 PDF。

## 六、证据文件

- 主表：`My_merge_ret/汇总表.md`
- 内部消融：`My_merge_ret/reports/lamp_merge_internal_ablation_full.csv`
- 预测诊断：`My_merge_ret/reports/prediction_diagnostics_full.csv`
- 预测诊断汇总：`My_merge_ret/reports/prediction_diagnostics_full/`
- 原型几何：`My_merge_ret/reports/lamp_merge_prototype_geometry_full.csv`
- 综合图：`My_merge_ret/figures/lamp_merge_internal_analysis/`
- 联合 t-SNE：`My_merge_ret/figures/lamp_merge_prototype_geometry/full_scope_tsne_comparison/`
