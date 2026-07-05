# MyMerge 模块消融记录

本记录用于固定当前消融结论。实验只使用客户端上传的聚合统计，不使用服务端私有原图、公开 probe 或候选池选择。

## 模块定义

**M1：诊断原型上传。** 客户端在共享 reference backbone 上，为每个诊断类别上传类别样本数和类别特征均值。

**M2：无坍缩原型头合成。** 服务端按类别聚合客户端原型，并用归一化原型构造 cosine prototype classifier。

**被删除组件：患病率 prior bias。** 服务端由客户端上传的类别样本数估计全局类别先验，并向 prototype classifier bias 加入 log-prior 校正。全量消融显示它减少最优格子数，因此不属于正式方法。

## 消融设置

代表性 smoke 使用 `resnet`、`beta=0.01`、5 个医学小图数据集和 3/5/7 个客户端，共 15 个格子。

比较四种设置：

| 设置 | 含义 |
| --- | --- |
| `avg` | 普通权重平均 |
| `my_merge` | M1 诊断原型上传 + M2 无坍缩原型头合成 |
| `avg+prior` | 普通权重平均后只加被删除的 prior bias |
| `+prior` | 正式 my_merge 后再加入被删除的 prior bias |

## 结果摘要

| 设置 | 15 格平均 Acc | 单格平均耗时 |
| --- | ---: | ---: |
| `avg` | 0.2912 | 11.69s |
| `my_merge` | 0.5744 | 11.81s |
| `avg+prior` | 0.2556 | 11.87s |
| `+prior` | 0.5956 | 11.75s |

`my_merge` 与 `avg` 的耗时基本一致，额外开销主要是一次分类头重建；端到端时间仍由测试集前向推理主导。

## 关键观察

正式两模块方法是主要收益来源。血液、超声、器官数据上，`my_merge` 已经显著超过 `avg`。例如 bloodmnist 从 0.19-0.36 提升到约 0.80，organcmnist 从 0.11-0.15 提升到约 0.61。

被删除的 prior bias 不应作为独立融合方法。`avg+prior` 基本不能修复已经坍缩的平均模型，甚至会在 dermamnist c5 上明显降低性能。

更重要的是，全量表显示 prior bias 的局部收益不稳定。虽然 dermamnist 个别格子会提升，但全表 Client Average 中，正式 `my_merge` 达到最优/并列最优 47/75，而 `+prior` 只有 43/75；Raw 中 `my_merge` 为 138/225，`+prior` 为 137/225。因此 prior bias 被判定为负优化并从正式方法删除。

## 结论

正式方法保留两个模块：

1. M1 负责把私有医学图像压缩为类别级诊断原型。
2. M2 负责把类别原型写成无 prior-bias 的 cosine prototype classifier。

被删除的 prior bias、`avg+prior` 和 `+prior` 都只作为消融行保留，用来说明：医学类别坍缩的核心修复来自诊断类别原型重建和原型头合成，而不是对平均模型做后验类别先验校正。
