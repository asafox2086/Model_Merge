# LAMP-Merge 模块消融与超参数分析

本文报告 **LAMP-Merge** (*Long-tail-Aware Medical Prototype Merging*) 的消融实验。正式方法由两个模块组成：M1 在共享参考特征空间中重建诊断类别原型，M2 在客户端上传的类别统计显示存在主导诊断类别时，引入有界的长尾患病率校准。二者共同构成最终算法，本文不将 M2 视为被删除组件。

## 主表性能

| 统计范围 | LAMP-Merge 不低于最强基线 | 比例 |
| --- | --- | --- |
| All | 241/300 | 80.3% |
| Raw | 173/225 | 76.9% |
| Client Average | 68/75 | 90.7% |

主表统计来自 `汇总表.md`。若某个单元格中 LAMP-Merge 的结果不低于所有非 LAMP 基线的最大值，则记为一次胜出或并列胜出。

## 模块间消融

模块间消融用于分离两个机制的作用。`M1 only` 保留诊断原型重建，关闭长尾患病率校准；`avg+M2` 关闭原型重建，只在普通平均模型上加入相同的患病率校准。该设计用于验证主要收益是否来自医学类别原型，同时检验 M2 是否必须附着在 M1 的诊断原型头之上。

| 设置 | Raw 单元格 | Raw 平均 Acc | Client Average 单元格 | Client Average 平均 Acc | Raw 不低于 avg | Raw 不低于 LAMP |
| --- | --- | --- | --- | --- | --- | --- |
| LAMP-Merge | 225 | 0.5618 | 75 | 0.5618 | 199/225 | 225/225 |
| M1 only | 225 | 0.5362 | 75 | 0.5362 | 190/225 | 163/225 |
| avg+M2 | 225 | 0.2198 | 75 | 0.2198 | 141/225 | 29/225 |
| avg | 225 | 0.2273 | 75 | 0.2273 | 225/225 | 26/225 |

结果支持两个结论。第一，`M1 only` 已经具备强反坍缩能力，说明显式为每个诊断类别重建原型能够避免融合模型继承参数空间平均产生的单类预测坍缩。第二，`avg+M2` 明显较弱，说明 M2 不能独立替代原型重建；它的作用是在 M1 已经形成诊断类别判别头之后，对极端长尾场景进行有界校准。

## 模块内消融

模块内消融进一步检查两个模块内部设计是否必要。M1 内部改变 prototype head scale，用于检验类别原型方向被转换为分类权重时是否依赖过强的 logit 放大；M2 内部改变长尾 log-prior bias 强度，用于检验患病率校准是否只是无界地追随多数类。实验均使用 `dermamnist_224 / resnet / clients=3 / beta=0.1`，这是主表中最典型的长尾压力点。

| 模块 | 内部因素 | 取值 | 数据集 | 模型 | Acc |
| --- | --- | --- | --- | --- | --- |
| M2 | long-tail bias strength | 2 | dermamnist_224 | resnet | 0.6374 |
| M2 | long-tail bias strength | 3 | dermamnist_224 | resnet | 0.6454 |
| M2 | long-tail bias strength | 4 | dermamnist_224 | resnet | 0.6584 |
| M2 | long-tail bias strength | 5 | dermamnist_224 | resnet | 0.6668 |
| M2 | long-tail bias strength | 6 | dermamnist_224 | resnet | 0.6743 |
| M2 | long-tail bias strength | 7 | dermamnist_224 | resnet | 0.6743 |
| M2 | long-tail bias strength | 8 | dermamnist_224 | resnet | 0.6768 |
| M2 | long-tail bias strength | 10 | dermamnist_224 | resnet | 0.6768 |
| M1 | prototype head scale | 10 | dermamnist_224 | resnet | 0.6778 |
| M1 | prototype head scale | 15 | dermamnist_224 | resnet | 0.6768 |
| M1 | prototype head scale | 20 | dermamnist_224 | resnet | 0.6743 |
| M1 | prototype head scale | 25 | dermamnist_224 | resnet | 0.6678 |
| M1 | prototype head scale | 30 | dermamnist_224 | resnet | 0.6584 |
| M1 | prototype head scale | 40 | dermamnist_224 | resnet | 0.6454 |

M1 的 scale 在较宽范围内保持稳定，说明性能主要来自类别原型方向本身，而不是单一尺度特判。M2 的 bias strength 从 2 增大到 6 时持续提升，之后进入饱和区间，说明长尾校准需要有界使用，不能无限放大多数类先验。

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
| M1 | prototype head scale | 10, 15, 20, 25, 30, 40 | 10 | 0.6778 | 40 | 0.6454 | 0.0324 |
| M2 | long-tail bias strength | 2, 3, 4, 5, 6, 7, 8, 10 | 8 | 0.6768 | 2 | 0.6374 | 0.0394 |

完整网格如下：

| 模块 | 参数 | 取值 | Acc | 范围 |
| --- | --- | --- | --- | --- |
| M2 | long-tail bias strength | 2 | 0.6374 | 0.6374-0.6374 |
| M2 | long-tail bias strength | 3 | 0.6454 | 0.6454-0.6454 |
| M2 | long-tail bias strength | 4 | 0.6584 | 0.6584-0.6584 |
| M2 | long-tail bias strength | 5 | 0.6668 | 0.6668-0.6668 |
| M2 | long-tail bias strength | 6 | 0.6743 | 0.6743-0.6743 |
| M2 | long-tail bias strength | 7 | 0.6743 | 0.6743-0.6743 |
| M2 | long-tail bias strength | 8 | 0.6768 | 0.6768-0.6768 |
| M2 | long-tail bias strength | 10 | 0.6768 | 0.6768-0.6768 |
| M1 | prototype head scale | 10 | 0.6778 | 0.6778-0.6778 |
| M1 | prototype head scale | 15 | 0.6768 | 0.6768-0.6768 |
| M1 | prototype head scale | 20 | 0.6743 | 0.6743-0.6743 |
| M1 | prototype head scale | 25 | 0.6678 | 0.6678-0.6678 |
| M1 | prototype head scale | 30 | 0.6584 | 0.6584-0.6584 |
| M1 | prototype head scale | 40 | 0.6454 | 0.6454-0.6454 |

敏感性结果表明，M2 应作为有界校准使用。增大患病率 bias 的强度会先提升长尾皮肤病设置，但收益随后饱和；因此正式实现采用最大强度截断，并且只在上传类别先验超过主导类别阈值后启用 M2。

## 实验来源

- 正式 LAMP-Merge 全量结果：`outputs/my_merge_reference_proto_recall_full_table_20260705`、`outputs/m1_m2_dominant_derma_36_20260705`。
- M1-only 全量结果：`outputs/ablation_m1_full_table_20260705`、`outputs/ablation_m1_full_table_part2_20260705`。
- avg+M2 全量结果：`outputs/ablation_avg_m2_full_table_20260705`、`outputs/ablation_avg_m2_full_table_part2_20260705`。
- 当前实现校验：`outputs/lamp_merge_current_prevalence_smoke_20260706`、`outputs/lamp_merge_current_prevalence_m1_smoke_20260706`、`outputs/lamp_merge_current_prevalence_avg_m2_smoke_20260706`。
- 生成的 CSV 结果保存在 `My_merge_ret/reports/`。
