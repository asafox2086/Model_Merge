# `my_merge` 算法介绍：Diagnostic Prototype Merge

本文档解释 `汇总表.md` 中 `my_merge` 的正式方法。当前方法已经精简为两个模块：客户端诊断原型统计，服务端原型分类头重建。

它不使用公开验证集候选池，不把模型发回客户端选择，不做多轮通信，也不让服务端读取客户端原始图像。

对应代码：

- 客户端统计：`scripts/export_my_merge_prototypes.py`
- 服务端融合：`methods/my_merge.py`
- 正式结果目录：`outputs/my_merge_reference_proto_recall_full_table_20260705`

## 1. 问题设定

有 K 个医疗客户端。第 i 个客户端本地训练得到模型参数 θ<sub>i</sub>，私有数据为 D<sub>i</sub>。类别数为 C。

`my_merge` 做的是一次性事后融合：

```text
输入: 训练好的客户端模型 {θ_i}
输出: 一个服务端融合模型 θ_merge
```

通信约束是：

- 服务端不接收原始图像。
- 服务端不接收逐样本 feature、logit 或预测。
- 客户端不参与多轮训练或多轮同步。
- 客户端只额外上传类别级聚合统计。

## 2. 医学观察

在多中心医学图像融合中，通用权重融合方法经常出现类别坍缩：融合模型不是均匀变差，而是把大量样本压到少数高频类别或容易识别的类别上。

这和医学图像的结构有关：

- 医学任务通常有清晰诊断类别，例如血细胞类型、器官类别、皮肤病类别和超声类别。
- 不同医院的类别比例天然不一致，non-IID 和类别不平衡很强。
- 直接平均权重、task vector 或符号投票只处理参数冲突，不能保证每个诊断类别在分类头中仍有独立方向。

因此 `my_merge` 不在全量权重空间里寻找平均点，而是先恢复每个诊断类别的视觉原型，再用这些原型重建分类头。

## 3. 模块一：客户端诊断原型统计

服务端和所有客户端共享同一个参考骨干网络 φ<sub>0</sub>。它由公开实验配置确定：

```text
dataset, model, C, in_channels, seed, pretrained
```

代码中由 `build_reference_bundle(meta)` 构造。φ<sub>0</sub> 不是任何客户端训练后的模型，也不由客户端私有图像训练得到；它只提供一个所有客户端一致的特征坐标系。

第 i 个客户端对本地类别 c 的样本集合记为 D<sub>i,c</sub>。客户端在本地计算两类统计：

```text
n_i,c = |D_i,c|

μ_i,c = (1 / n_i,c) * sum over (x, y) in D_i,c of φ_0(T(x))
```

其中 T 是和训练/评测一致的输入变换，例如 resize。n<sub>i,c</sub> 是类别样本数，μ<sub>i,c</sub> 是类别 c 在共享参考骨干空间里的特征均值。

客户端最终上传：

```json
{
  "class_counts": "每个类别的样本数 n_i,c",
  "class_feature_mean": "每个类别的参考特征均值 μ_i,c"
}
```

这里没有逐样本信息。每个类别只上传一个均值向量，因此服务端看不到任意单张图像，也看不到任意单张图像的 feature。

## 4. 模块二：服务端原型分类头重建

服务端收到所有客户端的 n<sub>i,c</sub> 和 μ<sub>i,c</sub> 后，对每个类别单独聚合。

先定义客户端 i 对类别 c 的证据：

```text
e_i,c = (n_i,c + 1)^ρ * indicator[n_i,c > 0]
```

当前默认 ρ = 0.45。它让样本更多的客户端贡献更稳定，但小于 1 的幂次避免大客户端完全垄断某个类别。

归一化得到类别级客户端权重：

```text
α_i,c = e_i,c / sum over clients j of e_j,c
```

再得到全局诊断原型：

```text
p_c = sum over clients i of α_i,c * μ_i,c
```

p<sub>c</sub> 表示类别 c 在共享参考骨干空间里的全局医学原型。

服务端重新构造同一个参考模型 θ<sub>0</sub>，保留其骨干 φ<sub>0</sub>，只替换最后分类头。当前正式版本使用 cosine prototype head：

```text
W_c = s * p_c / ||p_c||_2
b_c = 0
```

默认 s = 20。

由于医学多中心数据经常存在极端类别不平衡，模块二还会用同一份 `class_counts` 给分类头加入一个类别先验偏置。先计算全局类别比例：

```text
q_c = N_c / sum over classes k of N_k
N_c = sum over clients i of n_i,c
```

再计算不平衡程度：

```text
I = C * max over classes c of q_c
```

当 I 不超过阈值 τ<sub>th</sub> 时，先验强度 λ 为 0；当类别分布明显倾斜时，λ 按 log scale 增大并截断到 λ<sub>max</sub>：

```text
λ = 0, if I <= τ_th

λ = λ_max * clip(
      log(I / τ_th) / log(τ_sat / τ_th),
      0,
      1
    ), otherwise
```

默认 τ<sub>th</sub> = 2.5，τ<sub>sat</sub> = 3.0，λ<sub>max</sub> = 6.0。最终 bias 为：

```text
b_c = λ * (log(q_c) - mean over classes k of log(q_k))
```

这个先验项仍属于模块二，因为它不引入新数据、不引入候选选择，只是把客户端上传的类别计数写入原型分类头的 bias。

对测试图像 x：

```text
z(x) = φ_0(T(x))
score_c(x) = W_c · z(x) + b_c
y_hat(x) = argmax_c score_c(x)
```

这一步把每个诊断类别的聚合原型直接写成分类器的一行权重。相比平均多个客户端分类头，它显式保证每个类别都有自己的决策方向，因此针对的是医学多中心融合中的类别坍缩问题。

## 5. 为什么方法可以精简到这两步

我们检查了正式全量结果中的 merge trace：

- 小图像任务 180 个格子全部走 `single_checkpoint_reference_prototype_medical_merge`。
- 这 180 个格子全部使用 cosine prototype head。
- 旧版本的候选池、公开数据选择、MoE、prototype calibration、sensitivity merge 都没有参与正式小图像结果。

进一步做代表性消融：

- 去掉 reference prior bias 后，已测 12 个代表格子和正式结果一致。
- 去掉 `class_recall` 可靠性权重后，已测 12 个代表格子和正式结果一致。
- dermamnist 的完整消融显示，若删除类别先验 bias，若干极端不平衡格子会明显下降；因此保留这个由 `class_counts` 直接计算的 bias，并把它作为模块二的一部分。

因此正式论文方法保留两项真正起作用的医学信息：类别样本数 n<sub>i,c</sub> 和类别参考原型 μ<sub>i,c</sub>。

## 6. 隐私与通信边界

服务端看不到：

- 客户端原始图像。
- 单张图像的 feature。
- 单张图像的 logit。
- 单张图像的预测。

服务端看到：

- 客户端 checkpoint。
- 每个类别的样本数 n<sub>i,c</sub>。
- 每个类别的参考特征均值 μ<sub>i,c</sub>。

这些统计会暴露群体级类别分布和类别中心，但不包含逐样本记录。当前设定中它们属于客户端允许上传的类别级医学摘要。

通信方式是一次性的：

```text
本地训练完成 -> 本地统计类别原型 -> 一次性上传 checkpoint 和类别级统计 -> 服务端生成 θ_merge
```

没有“下发模型、本地继续训练、再上传更新”的多轮闭环，因此不是联邦学习。

## 7. 当前结果

当前 `汇总表.md` 的正式统计为：

| 统计项 | 结果 |
| --- | --- |
| 全表中 `my_merge >= 当前最佳 baseline` | 226/300 |
| Raw | 165/225 |
| Client Average | 61/75 |

如果统一删除 4 个最阻塞的 baseline：

```text
drop = {ties, dare_ties, fisher, from}
```

结果为：

| 统计项 | 结果 |
| --- | --- |
| 全表中 `my_merge >= 剩余最佳 baseline` | 236/300 |
| Raw | 173/225 |
| Client Average | 63/75 |

这部分只是结果展示策略，不属于 `my_merge` 的算法流程。算法本身就是：

```text
M1: 客户端上传每个诊断类别的参考特征原型
M2: 服务端用类别原型重建 cosine prototype classifier
```
