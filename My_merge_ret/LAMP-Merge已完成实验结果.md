# LAMP-Merge 已完成实验结果

本文档只汇总已经完成且通过一致性检查的 LAMP-Merge 实验结果。尚在运行或尚未重新验证的超参数全量扫描不纳入本文档；旧的 `my_merge` 候选池表、smoke/probe 表、public-test 表和旧图像已从正式结果目录移除，避免与 LAMP-Merge 发布版结果混淆。本文档的组织形式与 `LAMP-Merge模块内补充消融汇报.md` 保持一致：先说明 full-scope 覆盖范围和当前总体结论，再按实验项给出表格、证据文件和可写入论文的解释口径。

## 当前全量结果

当前已经完成的正式结果覆盖 `bloodmnist_224`、`dermamnist_224`、`organcmnist_224`、`organsmnist_224` 和 `chaoshengmnist_224` 五个医学图像数据集，覆盖 `resnet`、`convnext`、`vit_t` 和 `swin_tiny` 四种 backbone，覆盖 $K\in\{3,5,7\}$ 与 $\beta\in\{0,0.01,0.1\}$。因此，正式主表包含 180 个 raw cells；client-average 结果先在固定数据集、backbone 与 $K$ 的条件下对三个 $\beta$ 取均值，因此包含 60 个 client-average cells。

全量结果表明，在不删除任何基线的情况下，LAMP-Merge 在 57/60 个 client-average cells 上不低于最强非 LAMP-Merge 基线，在 154/180 个 raw cells 上不低于最强非 LAMP-Merge 基线。预测分布诊断进一步显示，LAMP-Merge 的提升并非来自多数类预测坍缩：在代表性 20 个诊断 case 中，LAMP-Merge 同时取得更高的 balanced accuracy 和 macro F1，并显著降低 collapse ratio 与 Pred-True TV。Prototype geometry 分析覆盖 180 个 raw cells，表明客户端上传的类别级参考原型在共享参考特征空间中具有稳定的类内一致性和类间分离性。

| 实验项 | 覆盖范围 | 当前状态 | 主要结论 |
|---|---:|---|---|
| 正式主表 | 180 raw cells / 60 client-average cells | 已完成，`main_table` 校验通过 | 211/240 全表不低于最强基线，57/60 client-average 不低于最强基线 |
| 预测坍缩诊断 | 20 diagnostic cases | 已完成 | LAMP-Merge 同时降低 collapse ratio 和 Pred-True TV，并提升 balanced accuracy 与 macro F1 |
| Prototype geometry | 180 raw cells | 已完成，`prototype_geometry` 校验通过 | 类别原型具有稳定的类内一致性、类间距离和客户端对齐性 |

## 一、正式主表

正式主表文件为 `My_merge_ret/汇总表.md`，对应的 LAMP-Merge 单方法快照为 `My_merge_ret/all_results_lamp_merge_client_local_20260708_193654.md`。当前表格中的 LAMP-Merge 行只来自 `outputs/lamp_merge_full_client_local_20260708_193654/resnet`、`outputs/lamp_merge_full_client_local_20260708_193654/convnext`、`outputs/lamp_merge_full_client_local_20260708_193654/vit_t` 和 `outputs/lamp_merge_full_client_local_20260708_193654/swin_tiny` 四个正式输出目录。

在不删除任何基线的情况下，LAMP-Merge 不低于最强非 LAMP-Merge 基线的统计如下。

| 统计范围 | 胜出或并列胜出 | 比例 |
|---|---:|---:|
| 全表 | 211/240 | 87.9% |
| Raw | 154/180 | 85.6% |
| Client Average | 57/60 | 95.0% |

Raw cell 上的 LAMP-Merge accuracy 统计如下。

| 数据集 | Cell 数 | 平均 Acc | 最小 Acc | 最大 Acc |
|---|---:|---:|---:|---:|
| bloodmnist_224 | 36 | 0.817410 | 0.761766 | 0.852675 |
| chaoshengmnist_224 | 36 | 0.457522 | 0.404313 | 0.489668 |
| dermamnist_224 | 36 | 0.628775 | 0.569576 | 0.674314 |
| organcmnist_224 | 36 | 0.627062 | 0.576801 | 0.687196 |
| organsmnist_224 | 36 | 0.574170 | 0.503115 | 0.628979 |
| 全部 | 180 | 0.620988 | 0.404313 | 0.852675 |

该结果由 `exp_analyze/validate_full_outputs.py --step main_table` 验证通过。该表用于论文主结果部分，说明 LAMP-Merge 在完整医学图像异步模型融合网格中具备稳定优势。

## 二、预测坍缩诊断

预测坍缩诊断固定 `clients=3`、`beta=0.01`、`seed=42`，覆盖 5 个医学数据集与 4 个 backbone，共 20 个诊断 case。该实验同时报告 Accuracy、balanced accuracy、macro F1、预测坍缩强度、有效预测类别数和预测分布总变差距离，用于检验 LAMP-Merge 的提升是否来自多数类坍缩。

总体均值如下。

| 方法 | Cases | Acc | Balanced Acc | Macro F1 | Collapse Ratio | Effective Classes | Pred-True TV |
|---|---:|---:|---:|---:|---:|---:|---:|
| client mean | 20 | 0.1804 | 0.1515 | 0.0638 | 0.8207 | 1.7370 | 0.7644 |
| client best | 20 | 0.3156 | 0.1810 | 0.1123 | 0.7619 | 2.0201 | 0.6168 |
| avg | 20 | 0.2304 | 0.1543 | 0.0751 | 0.8028 | 1.7666 | 0.7082 |
| best non-LAMP-Merge | 20 | 0.2585 | 0.1682 | 0.0886 | 0.7477 | 2.1360 | 0.6671 |
| LAMP-Merge | 20 | 0.5885 | 0.5747 | 0.5352 | 0.2294 | 8.0390 | 0.1510 |

按单 case 统计，LAMP-Merge 在 Accuracy 上达到 16/20 个最优或并列最优；在 balanced accuracy、macro F1 和 collapse ratio 上均为 20/20；在 Pred-True TV 上为 18/20。该结果表明，LAMP-Merge 并非通过放大多数类预测获得高 accuracy，而是在多数医学数据集上同时提高多类别召回、宏平均 F1，并显著降低单类预测坍缩。

| 证据类型 | 保留位置 |
|---|---|
| 诊断指标表 | `My_merge_ret/reports/prediction_diagnostics_4models_c3_b001/` |
| 预测分布图与总体比较图 | `My_merge_ret/figures/prediction_diagnostics_4models_c3_b001/` |

## 三、Prototype Geometry

Prototype geometry 分析覆盖正式 180 个 raw cells，比较正式原型、分类头聚合、全局特征均值、support-only synthetic head、打乱标签原型、客户端等权聚合和全局客户端规模加权等设置。该实验用于说明 LAMP-Merge 上传的类别级原型在共享参考特征空间中具有稳定的类内一致性和类间分离性。

正式 Full prototype 的数据集级统计如下。

| 数据集 | Cases | Pairwise Dist | Nearest-Class Dist | Consistency | Proto-Client Align | Evidence Entropy |
|---|---:|---:|---:|---:|---:|---:|
| bloodmnist_224 | 36 | 0.0705 | 0.0324 | 0.9409 | 0.9746 | 0.2406 |
| chaoshengmnist_224 | 36 | 0.0447 | 0.0227 | 0.8847 | 0.9487 | 0.2549 |
| dermamnist_224 | 36 | 0.0550 | 0.0298 | 0.8575 | 0.9393 | 0.2822 |
| organcmnist_224 | 36 | 0.1964 | 0.0567 | 0.8467 | 0.9325 | 0.2505 |
| organsmnist_224 | 36 | 0.1944 | 0.0387 | 0.8459 | 0.9280 | 0.2244 |

该结果由 `exp_analyze/validate_full_outputs.py --step prototype_geometry` 验证通过。该表用于论文机制分析部分，说明 LAMP-Merge 的类别原型并非任意统计量，而是在共享参考特征空间中保留了可聚合的诊断类别结构。

| 证据类型 | 保留位置 |
|---|---|
| 原型几何全量明细 | `My_merge_ret/reports/lamp_merge_prototype_geometry_full.csv` |
| 数据集级原型几何汇总 | `My_merge_ret/reports/lamp_merge_prototype_geometry_by_dataset.csv` |
| 原型几何文字汇总 | `My_merge_ret/reports/lamp_merge_prototype_geometry_summary.md` |
| 原型几何图与 t-SNE 图 | `My_merge_ret/figures/lamp_merge_prototype_geometry/` |

## 四、当前未纳入项

超参数全量扫描仍在独立进程中运行，当前结果不写入正式结论。模块内消融和超参数敏感性将在对应全量实验完成并通过验证后，按同一格式写入分析文档。后续新增表格应继续保持 full-scope 覆盖说明、主结论表、数据集级表、证据文件表和未纳入项说明这五类信息，避免与旧版 `my_merge` 中间结果混用。
