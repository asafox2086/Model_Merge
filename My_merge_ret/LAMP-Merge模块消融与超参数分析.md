# LAMP-Merge 模块消融与超参数分析

本文报告 **LAMP-Merge** (*Long-tail-Aware Medical Prototype Merging*) 的消融实验。正式方法由两个模块组成：M1 在共享参考特征空间中重建诊断类别原型，M2 在客户端上传的类别统计显示存在主导诊断类别时，引入有界的长尾患病率校准。二者共同构成最终算法，本文不将 M2 视为被删除组件。

## 主表性能

| 统计范围 | LAMP-Merge 不低于最强基线 | 比例 |
| --- | --- | --- |
| All | 241/300 | 80.3% |
| Raw | 173/225 | 76.9% |
| Client Average | 68/75 | 90.7% |

主表统计来自 `汇总表.md`。若某个单元格中 LAMP-Merge 的结果不低于所有非 LAMP 基线的最大值，则记为一次胜出或并列胜出。

## 新指标诊断实验

为排除 LAMP-Merge 仅利用多数类 accuracy 的可能性，我们额外构造预测分布诊断实验。该实验覆盖 5 个正式医学数据集：`bloodmnist_224`、`chaoshengmnist_224`、`dermamnist_224`、`organcmnist_224` 和 `organsmnist_224`。每个数据集使用 4 个 backbone（resnet、convnext、vit_t、swin_tiny），固定 `clients=3`、`beta=0.01`、`seed=42`，因此总体统计包含 5 个数据集 × 4 个 backbone = 20 个医学诊断 case。统计对象包括 individual clients 的聚合行、通用模型融合 baseline，以及 LAMP-Merge。

除 Accuracy 外，该实验报告以下新指标：

$$
\mathrm{BA}=\frac{1}{C}\sum_{c=1}^{C}\frac{\mathrm{TP}_c}{\mathrm{TP}_c+\mathrm{FN}_c}.
$$

其中 $\mathrm{BA}$ 是 balanced accuracy，即 macro recall，用于衡量各诊断类别是否同时被召回。设 $q(c)$ 表示模型在测试集上预测为类别 $c$ 的比例，$p(c)$ 表示测试集真实类别比例，则预测坍缩强度与预测分布偏差定义为：

$$
\rho=\max_c q(c),\qquad
\mathrm{TV}(q,p)=\frac{1}{2}\sum_{c=1}^{C}|q(c)-p(c)|.
$$

$\rho$ 越接近 1，模型越接近单类预测器；$\mathrm{TV}(q,p)$ 越小，预测类别分布越接近真实诊断分布。有效预测类别数定义为：

$$
C_{\mathrm{eff}}=\exp\left(-\sum_{c=1}^{C}q(c)\log q(c)\right).
$$

该指标越大，表示模型实际使用的诊断类别越多。若一个方法仅在多数类上坍缩，则通常会表现为 Accuracy 较高但 $\mathrm{BA}$、Macro F1 和 $C_{\mathrm{eff}}$ 较低，同时 $\rho$ 和 $\mathrm{TV}$ 较高。

### 五个医学数据集的总体结果

下表是 `bloodmnist_224`、`chaoshengmnist_224`、`dermamnist_224`、`organcmnist_224` 和 `organsmnist_224` 上 20 个诊断 case 的总体均值。

| Method | Cases | Acc | BA | Macro F1 | $\rho$ | $C_{\mathrm{eff}}$ | TV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| client mean | 20 | 0.1804 | 0.1515 | 0.0638 | 0.8207 | 1.7370 | 0.7644 |
| client best | 20 | 0.3156 | 0.1810 | 0.1123 | 0.7619 | 2.0201 | 0.6168 |
| avg | 20 | 0.2304 | 0.1543 | 0.0751 | 0.8028 | 1.7666 | 0.7082 |
| ties | 20 | 0.2089 | 0.1682 | 0.0858 | 0.7477 | 2.1360 | 0.6907 |
| dare_linear | 20 | 0.2145 | 0.1481 | 0.0745 | 0.7577 | 2.0556 | 0.6849 |
| dare_ties | 20 | 0.1763 | 0.1537 | 0.0689 | 0.7826 | 1.9966 | 0.7367 |
| regmean | 20 | 0.2083 | 0.1625 | 0.0841 | 0.7655 | 2.0285 | 0.7095 |
| fisher | 20 | 0.2585 | 0.1604 | 0.0886 | 0.8228 | 1.9324 | 0.6671 |
| breadcrumbs | 20 | 0.1138 | 0.1181 | 0.0280 | 0.9320 | 1.2901 | 0.8473 |
| model_stock | 20 | 0.2207 | 0.1488 | 0.0711 | 0.8155 | 1.7457 | 0.6970 |
| from | 20 | 0.2494 | 0.1652 | 0.0848 | 0.7705 | 1.8889 | 0.6949 |
| iso_c | 20 | 0.1976 | 0.1648 | 0.0845 | 0.7538 | 2.1161 | 0.7172 |
| free_merge | 20 | 0.2300 | 0.1558 | 0.0762 | 0.8182 | 1.7074 | 0.7158 |
| robustmerge | 20 | 0.1486 | 0.1425 | 0.0502 | 0.8225 | 1.6559 | 0.8035 |
| LAMP-Merge | 20 | 0.5885 | 0.5747 | 0.5352 | 0.2294 | 8.0390 | 0.1510 |

五数据集总体结果表明，LAMP-Merge 的提升不是由多数类坍缩造成的。相对于最强 non-LAMP/client 参照，Accuracy 从 0.3156 提高到 0.5885；更关键的是，BA 从 0.1810 提高到 0.5747，Macro F1 从 0.1123 提高到 0.5352。与此同时，坍缩强度 $\rho$ 从 0.7477 降到 0.2294，有效预测类别数从 2.1360 提高到 8.0390，预测分布 TV 从 0.6168 降到 0.1510。

按单 case 统计，LAMP-Merge 在 Accuracy 上达到 16/20 个最优或并列最优；在 BA、Macro F1 和 $\rho$ 上均为 20/20；在 TV 上为 18/20。Derma 是主要例外：client best 和 Fisher 在 raw Accuracy 或 TV 上具有长尾多数类优势，但 LAMP-Merge 仍在 BA、Macro F1 和坍缩强度上最优，说明其保留了更完整的多类别诊断能力。

### 按数据集结果

下表将每个数据集上的 LAMP-Merge 与对应指标下的最强 non-LAMP/client 参照进行比较。每个数据集包含 4 个 backbone case。

| Dataset | LAMP Acc | Best ref Acc | LAMP BA | Best ref BA | LAMP Macro F1 | Best ref Macro F1 | LAMP $\rho$ | Best ref $\rho$ | LAMP TV | Best ref TV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Blood | 0.8190 | 0.2286 | 0.8071 | 0.2129 | 0.8022 | 0.1133 | 0.1943 | 0.7768 | 0.0394 | 0.6759 |
| Ultrasound | 0.4625 | 0.2682 | 0.4536 | 0.2335 | 0.4378 | 0.1514 | 0.2035 | 0.6047 | 0.1575 | 0.6094 |
| Derma | 0.4627 | 0.6726 | 0.4492 | 0.1890 | 0.2934 | 0.1562 | 0.3697 | 0.7978 | 0.3309 | 0.3185 |
| Organ-C | 0.6249 | 0.2297 | 0.6175 | 0.1708 | 0.6030 | 0.1067 | 0.1772 | 0.5474 | 0.1280 | 0.6055 |
| Organ-S | 0.5734 | 0.2586 | 0.5460 | 0.1603 | 0.5397 | 0.0881 | 0.2021 | 0.6274 | 0.0993 | 0.6226 |

该分数据集结果支持两个结论。第一，Blood、Ultrasound、Organ-C 和 Organ-S 上 LAMP-Merge 在所有新指标中同时优于最强参照，说明方法并非只在某一医学模态有效。第二，Derma 上存在典型的长尾 accuracy 假象：client best 在 Accuracy 上更高，Fisher 在 TV 上略低，但二者的 BA 和 Macro F1 明显低于 LAMP-Merge，且坍缩强度更高。因此 Derma 更适合作为 Observation 2 的证据，即单类或少类坍缩可能在强长尾数据集上抬高 raw Accuracy，但不代表模型具备完整全局诊断能力。

完整逐方法、逐数据集结果保存在 `My_merge_ret/reports/prediction_diagnostics_4models_c3_b001/prediction_diagnostics_all_datasets_summary.md`，对应图像保存在 `My_merge_ret/figures/prediction_diagnostics_4models_c3_b001/`。

## 模块间消融

模块间消融用于分离两个机制的作用。`M1 only` 保留诊断原型重建，关闭长尾患病率校准；`avg+M2` 关闭原型重建，只在普通平均模型上加入相同的患病率校准；`avg` 是普通参数平均控制组。主分析以 `client average` 单元格为统计单位：对每个固定的 `(dataset, backbone, K)`，先平均三个 Dirichlet skew 设置，再比较四个设置的 test accuracy。

| 设置 | Client Average 单元格 | Client Average 平均 Acc | 最优或并列最优 | 不低于 avg | 相对 avg 平均增益 |
| --- | --- | --- | --- | --- | --- |
| LAMP-Merge | 75 | 0.5618 | 51/75 | 71/75 | 0.3345 |
| M1 only | 75 | 0.5362 | 46/75 | 66/75 | 0.3089 |
| avg+M2 | 75 | 0.2198 | 4/75 | 43/75 | -0.0075 |
| avg | 75 | 0.2273 | 2/75 | 75/75 | 0.0000 |

该表直接对应主表中的 `client average` 比较口径。LAMP-Merge 在 75 个 client-average 单元格中取得最高或并列最高结果的次数最多；`M1 only` 保留了大部分收益，说明诊断原型重建是主要有效成分；`avg+M2` 与 `avg` 的比较表明，患病率校准本身不能替代类别原型重建，它只应作为 M1 之上的长尾校准项。

为避免总体统计掩盖数据集差异，下面进一步按数据集报告 ACC 对比。每个数据集包含 15 个 client-average 单元格，即五类 backbone 与三个客户端数量的组合。

| 数据集 | 单元格 | LAMP-Merge Acc | M1 only Acc | avg+M2 Acc | avg Acc | LAMP 最优或并列 | M1 最优或并列 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Blood | 15 | 0.7156 | 0.7155 | 0.1726 | 0.1699 | 12/15 | 9/15 |
| Ultrasound | 15 | 0.4040 | 0.4040 | 0.1551 | 0.1516 | 15/15 | 15/15 |
| Derma | 15 | 0.6257 | 0.4975 | 0.4913 | 0.5304 | 12/15 | 2/15 |
| Organ-C | 15 | 0.5543 | 0.5543 | 0.1331 | 0.1358 | 6/15 | 10/15 |
| Organ-S | 15 | 0.5096 | 0.5096 | 0.1469 | 0.1488 | 6/15 | 10/15 |

数据集级结果显示，Blood、Ultrasound、Organ-C 和 Organ-S 上的收益主要由 M1 提供，说明类别原型重建能够在多种医学图像形态下稳定恢复全局诊断判别；Derma 上 LAMP-Merge 明显高于 M1 only，说明 M2 对强长尾皮肤病分布的患病率校准具有独立贡献。

作为补充，下面给出 raw cell 级别的聚合统计，用于检查相同结论是否受单个 beta 设置驱动。

| 设置 | Raw 单元格 | Raw 平均 Acc | Client Average 单元格 | Client Average 平均 Acc | Raw 不低于 avg | Raw 不低于 LAMP |
| --- | --- | --- | --- | --- | --- | --- |
| LAMP-Merge | 225 | 0.5618 | 75 | 0.5618 | 199/225 | 225/225 |
| M1 only | 225 | 0.5362 | 75 | 0.5362 | 190/225 | 163/225 |
| avg+M2 | 225 | 0.2198 | 75 | 0.2198 | 141/225 | 29/225 |
| avg | 225 | 0.2273 | 75 | 0.2273 | 225/225 | 26/225 |

raw cell 统计与 client-average 统计一致：M1 已经显著优于普通平均，表明显式为每个诊断类别重建原型能够抑制融合后的单类预测坍缩；M2 的独立版本不稳定，说明它不是一个独立融合器，而是对 M1 诊断原型头的有界患病率修正。

## 模块内消融

模块内消融进一步检查两个模块内部设计是否必要。M1 改变公式中的 $s$，即原型分类头的尺度系数；M2 改变公式中的 $\lambda$，即长尾 log-prior 校准强度。实验均使用 `dermamnist_224 / resnet / clients=3 / beta=0.1`，这是主表中最典型的长尾压力点。这里的 $\tau$ 只表示 M2 的长尾触发阈值，不表示校准强度。

| 模块 | 内部因素 | 取值 | 数据集 | 模型 | Acc |
| --- | --- | --- | --- | --- | --- |
| M2 | $\lambda$ | 2 | dermamnist_224 | resnet | 0.6718 |
| M2 | $\lambda$ | 3 | dermamnist_224 | resnet | 0.6768 |
| M2 | $\lambda$ | 4 | dermamnist_224 | resnet | 0.6758 |
| M2 | $\lambda$ | 5 | dermamnist_224 | resnet | 0.6763 |
| M2 | $\lambda$ | 6 | dermamnist_224 | resnet | 0.6723 |
| M2 | $\lambda$ | 7 | dermamnist_224 | resnet | 0.6683 |
| M2 | $\lambda$ | 8 | dermamnist_224 | resnet | 0.6683 |
| M2 | $\lambda$ | 10 | dermamnist_224 | resnet | 0.6688 |
| M1 | $s$ | 5 | dermamnist_224 | resnet | 0.6688 |
| M1 | $s$ | 7 | dermamnist_224 | resnet | 0.6688 |
| M1 | $s$ | 10 | dermamnist_224 | resnet | 0.6688 |
| M1 | $s$ | 12 | dermamnist_224 | resnet | 0.6688 |
| M1 | $s$ | 15 | dermamnist_224 | resnet | 0.6683 |
| M1 | $s$ | 17 | dermamnist_224 | resnet | 0.6683 |
| M1 | $s$ | 20 | dermamnist_224 | resnet | 0.6723 |
| M1 | $s$ | 22 | dermamnist_224 | resnet | 0.6733 |
| M1 | $s$ | 25 | dermamnist_224 | resnet | 0.6758 |
| M1 | $s$ | 27 | dermamnist_224 | resnet | 0.6763 |
| M1 | $s$ | 30 | dermamnist_224 | resnet | 0.6758 |
| M1 | $s$ | 32 | dermamnist_224 | resnet | 0.6748 |
| M1 | $s$ | 35 | dermamnist_224 | resnet | 0.6738 |
| M1 | $s$ | 37 | dermamnist_224 | resnet | 0.6778 |
| M1 | $s$ | 40 | dermamnist_224 | resnet | 0.6768 |

M1 的 $s$ 在 `[5,40]` 的密集网格内保持稳定，说明性能主要来自类别原型方向本身，而不是单一尺度特判。M2 的 $\lambda$ 在中等强度区间达到最优，继续放大会出现轻微退化，说明长尾校准需要有界使用，不能无限放大多数类先验。

## 长尾压力点校验

该组实验使用当前代码路径，在 `dermamnist_224 / resnet / clients=3 / beta=0.1` 上验证正式实现。该设置的上传患病率高度长尾，主导类别不平衡强度为 4.73，因此用于观察 M2 是否能在不改变 M1 原型结构的情况下恢复多数类先验带来的 accuracy。

| 设置 | 数据集 | 模型 | 行数 | 平均 Acc | 最小 Acc | 最大 Acc |
| --- | --- | --- | --- | --- | --- | --- |
| M1+M2 | dermamnist_224 | resnet | 1 | 0.6723 | 0.6723 | 0.6723 |
| M1 only | dermamnist_224 | resnet | 1 | 0.4314 | 0.4314 | 0.4314 |
| avg+M2 | dermamnist_224 | resnet | 1 | 0.6688 | 0.6688 | 0.6688 |

结果显示，M1-only 已经形成非坍缩诊断原型头，但在该强长尾皮肤病设置中 accuracy 只有 0.4314；加入 M2 后，正式 LAMP-Merge 达到 0.6723。`avg+M2` 在这个单点也能获得较高 accuracy，说明强长尾先验确实会影响 accuracy；但全量模块间消融中 `avg+M2` 的 Raw 平均 Acc 只有 0.2198，表明缺少 M1 的类别原型结构时，单独的先验校准无法提供稳定的全局诊断能力。

## 超参数敏感性

超参数分析汇总每个内部因素的取值范围、最佳点、最差点和波动幅度，用于说明正式取值不是依赖单点偶然收益。

| 模块 | 参数 | 测试取值 | 最佳取值 | 最佳 Acc | 最差取值 | 最差 Acc | 波动范围 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1 | $s$ | 5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30, 32, 35, 37, 40 | 37 | 0.6778 | 15 | 0.6683 | 0.0095 |
| M2 | $\lambda$ | 2, 3, 4, 5, 6, 7, 8, 10 | 3 | 0.6768 | 7 | 0.6683 | 0.0085 |

![LAMP-Merge hyperparameter sensitivity](figures/lamp_merge_hparam_sensitivity.png)

图中红色星号表示该网格中的最优取值，绿色虚线表示正式算法默认取值。M1 的默认 $s$ 为 20，M2 的默认 $\lambda$ 上界为 6。二者均落在高精度平台区间内，说明正式配置并非依赖单个偶然最优点；同时，过强的 M2 先验会带来轻微退化，支持将患病率校准设计为有界项。

完整网格如下：

| 模块 | 参数 | 取值 | Acc | 范围 |
| --- | --- | --- | --- | --- |
| M2 | $\lambda$ | 2 | 0.6718 | 0.6718-0.6718 |
| M2 | $\lambda$ | 3 | 0.6768 | 0.6768-0.6768 |
| M2 | $\lambda$ | 4 | 0.6758 | 0.6758-0.6758 |
| M2 | $\lambda$ | 5 | 0.6763 | 0.6763-0.6763 |
| M2 | $\lambda$ | 6 | 0.6723 | 0.6723-0.6723 |
| M2 | $\lambda$ | 7 | 0.6683 | 0.6683-0.6683 |
| M2 | $\lambda$ | 8 | 0.6683 | 0.6683-0.6683 |
| M2 | $\lambda$ | 10 | 0.6688 | 0.6688-0.6688 |
| M1 | $s$ | 5 | 0.6688 | 0.6688-0.6688 |
| M1 | $s$ | 7 | 0.6688 | 0.6688-0.6688 |
| M1 | $s$ | 10 | 0.6688 | 0.6688-0.6688 |
| M1 | $s$ | 12 | 0.6688 | 0.6688-0.6688 |
| M1 | $s$ | 15 | 0.6683 | 0.6683-0.6683 |
| M1 | $s$ | 17 | 0.6683 | 0.6683-0.6683 |
| M1 | $s$ | 20 | 0.6723 | 0.6723-0.6723 |
| M1 | $s$ | 22 | 0.6733 | 0.6733-0.6733 |
| M1 | $s$ | 25 | 0.6758 | 0.6758-0.6758 |
| M1 | $s$ | 27 | 0.6763 | 0.6763-0.6763 |
| M1 | $s$ | 30 | 0.6758 | 0.6758-0.6758 |
| M1 | $s$ | 32 | 0.6748 | 0.6748-0.6748 |
| M1 | $s$ | 35 | 0.6738 | 0.6738-0.6738 |
| M1 | $s$ | 37 | 0.6778 | 0.6778-0.6778 |
| M1 | $s$ | 40 | 0.6768 | 0.6768-0.6768 |

敏感性结果表明，M2 应作为有界校准使用。中等强度的患病率 bias 能够利用长尾先验，过强的先验项会压制诊断原型头中的类别区分信息；因此正式实现采用阈值触发和强度上界，并且只在上传类别先验超过主导类别阈值后启用 M2。

## 实验来源

- 正式 LAMP-Merge 全量结果：`outputs/my_merge_reference_proto_recall_full_table_20260705`、`outputs/m1_m2_dominant_derma_36_20260705`。
- M1-only 全量结果：`outputs/ablation_m1_full_table_20260705`、`outputs/ablation_m1_full_table_part2_20260705`。
- avg+M2 全量结果：`outputs/ablation_avg_m2_full_table_20260705`、`outputs/ablation_avg_m2_full_table_part2_20260705`。
- 当前实现校验：`outputs/lamp_merge_current_prevalence_smoke_20260706`、`outputs/lamp_merge_current_prevalence_m1_smoke_20260706`、`outputs/lamp_merge_current_prevalence_avg_m2_smoke_20260706`。
- 生成的 CSV 结果保存在 `My_merge_ret/reports/`。
