# LAMP-Merge 算法介绍

**LAMP-Merge** (*Long-tail-Aware Medical Prototype Merging*) 是一个面向多中心医学图像的异步事后模型融合方法。它不做联邦学习式的多轮训练，不要求客户端之间通信，也不让服务端读取客户端原始图像。每个客户端在本地完成训练后，只上传模型参数和类别级统计；服务端一次性生成融合模型。

正式方法只有两个模块：

- **M1：诊断原型重建**。利用医学任务类别语义明确的特点，为每个诊断类别重建一个全局原型分类器。
- **M2：长尾患病率校准**。利用医学数据常见的长尾患病率分布，在主导类别过强时加入有界的类别先验校准。

对应代码：

- 客户端统计导出：[scripts/export_lamp_merge_prototypes.py](/data2/liyapeng_grp/program/MedMNISTMerge/scripts/export_lamp_merge_prototypes.py)
- 服务端融合实现：[methods/lamp_merge.py](/data2/liyapeng_grp/program/MedMNISTMerge/methods/lamp_merge.py)
- 正式汇总表：[汇总表.md](/data2/liyapeng_grp/program/MedMNISTMerge/My_merge_ret/汇总表.md)
- 消融与超参数分析：[LAMP-Merge模块消融与超参数分析.md](/data2/liyapeng_grp/program/MedMNISTMerge/My_merge_ret/LAMP-Merge模块消融与超参数分析.md)

## 问题设定

设共有 $K$ 个医学客户端，诊断类别数为 $C$。第 $i$ 个客户端本地训练得到模型参数 $\theta_i$，私有训练数据为 $D_i$。LAMP-Merge 的目标是在训练结束后生成一个融合模型：

```math
\theta_{\mathrm{merge}}=\mathcal{A}(\theta_1,\theta_2,\ldots,\theta_K;\mathcal{S}_1,\mathcal{S}_2,\ldots,\mathcal{S}_K).
```

其中 $\mathcal{S}_i$ 表示客户端 $i$ 上传的类别级统计。服务端不接收原始图像、逐样本特征、逐样本 logit 或逐样本预测。

## 医学动机

我们的预测分布诊断显示，客户端模型虽然受本地类别缺失影响，但通常仍保留多个诊断类别的局部判别能力；通用 post-hoc 参数融合却容易把这些局部偏置放大为单类或少数类预测坍缩。医学图像任务与通用自然图像或 NLP 任务的关键差异在于：诊断类别通常具有明确临床语义，且多中心数据中的患病率分布天然长尾。因此，LAMP-Merge 不在全参数空间里寻找平均点，而是把融合问题改写为“恢复每个诊断类别的全局判别原型”。

## 共享参考特征提取器

服务端和客户端根据公开实验配置构造同一个参考模型。其骨干特征提取器记为 $\phi_0$。它由模型架构、输入通道数、类别数、随机种子和预训练开关确定，不使用任何客户端私有图像训练。

对图像 $x$，客户端在本地计算参考特征：

```math
z=\phi_0(T(x)),
```

其中 $T$ 表示与实验一致的输入预处理，例如 resize。

## M1：诊断原型重建

第 $i$ 个客户端中属于诊断类别 $c$ 的样本集合记为 $D_{i,c}$。客户端本地计算类别样本数：

```math
n_{i,c}=|D_{i,c}|.
```

若 $n_{i,c}>0$，客户端继续计算该类别在共享参考特征空间中的均值：

```math
\mu_{i,c}=\frac{1}{n_{i,c}}\sum_{(x,y)\in D_{i,c}}\mathbf{1}[y=c]\,\phi_0(T(x)).
```

客户端上传的是每个类别的 $\mu_{i,c}$ 和 $n_{i,c}$。这里的 $n_{i,c}$ 表示用于估计类别原型的样本数，也就是 M1 的原型支持数。它是类别级聚合统计，不包含任意单张图像或单张图像的特征。

服务端先把样本数转换为类别证据：

```math
e_{i,c}=(n_{i,c}+1)^\gamma\,\mathbf{1}[n_{i,c}>0].
```

其中 $\gamma\in(0,1)$，当前实现默认 $\gamma=0.45$。这个幂次让样本更多的客户端拥有更高证据，但避免大客户端完全垄断某个诊断类别。

然后对每个类别单独归一化客户端权重：

```math
\alpha_{i,c}=\frac{e_{i,c}}{\sum_{j=1}^{K}e_{j,c}}.
```

全局诊断原型定义为：

```math
p_c=\sum_{i=1}^{K}\alpha_{i,c}\mu_{i,c}.
```

最后，服务端重新构造参考模型，只替换分类头。第 $c$ 类分类器权重为：

```math
w_c=s\frac{p_c}{\|p_c\|_2}.
```

其中 $s$ 是 prototype head scale，默认 $s=20$。该步骤显式保证每个诊断类别都有一条独立的判别方向，从结构上抑制融合后的单类预测坍缩。

## M2：长尾患病率校准

医学数据中存在另一个特殊现象：当某个诊断类别在真实分布中占比很高时，单纯惩罚坍缩并不总是最优，因为多数类预测可能在 Accuracy 上具有真实优势。因此 LAMP-Merge 在 M1 的 prototype classifier 上增加一个有界的长尾校准项。

客户端同时上传用于估计全局患病率的类别计数，记为 $m_{i,c}$。在单标签分类任务中，$m_{i,c}$ 可以由客户端本地标签直方图得到；它与 M1 的 $n_{i,c}$ 可能数值相同，但语义不同：$n_{i,c}$ 用于原型可靠性加权，$m_{i,c}$ 用于估计全局类别先验。

服务端由上传的患病率计数估计全局类别先验：

```math
\pi_c=\frac{\sum_{i=1}^{K}m_{i,c}}{\sum_{k=1}^{C}\sum_{i=1}^{K}m_{i,k}}.
```

定义主导类不平衡强度：

```math
r=C\max_c \pi_c.
```

若 $r\le \tau$，说明主导类别没有超过均匀分布的 $\tau$ 倍，M2 不启用。若 $r>\tau$，LAMP-Merge 对分类头 bias 加入中心化 log-prior：

```math
b_c=\lambda\left(\log \pi_c-\frac{1}{C}\sum_{k=1}^{C}\log \pi_k\right).
```

当前默认 $\tau=2.5$，$\lambda=5.0$。中心化项只改变类别之间的相对偏置，不整体平移所有 logit。

最终测试时，对输入图像 $x$ 的分类分数为：

```math
\mathrm{score}_c(x)=w_c^\top \phi_0(T(x)) + b_c,
```

预测类别为：

```math
\hat{y}(x)=\arg\max_c \mathrm{score}_c(x).
```

M2 的作用不是独立修复坍缩。消融显示，`avg+M2` 不能恢复平均模型已经丢失的类别判别结构；M2 必须附着在 M1 构造出的诊断原型分类器上，用于处理极端长尾类别比例下 Accuracy 与多类别判别能力之间的冲突。

## 隐私与通信边界

LAMP-Merge 的通信过程是一次性的：

```text
客户端本地训练 -> 客户端本地统计类别原型与类别计数 -> 一次性上传 -> 服务端融合
```

服务端接收：

- 客户端 checkpoint。
- 每个诊断类别的原型支持数 $n_{i,c}$。
- 每个诊断类别的参考特征均值 $\mu_{i,c}$。
- 每个诊断类别的患病率计数 $m_{i,c}$。

服务端不接收：

- 客户端原始医学图像。
- 单张图像的 feature。
- 单张图像的 logit。
- 单张图像的预测。

因此该方法不是联邦学习。它没有“下发全局模型、本地继续训练、再次上传更新”的多轮闭环；融合只发生在训练完成后的服务端一次性后处理阶段。

## 当前结果摘要

当前 `汇总表.md` 已由正式 LAMP-Merge 重新生成。实验覆盖 5 个医学图像数据集、4 个 backbone、3 个客户端数量和 3 个 Dirichlet beta 设置，共 180 个 raw cell；client-average 口径对相同数据集、backbone 和客户端数量下的 3 个 beta 取平均，共 60 个 cell。

LAMP-Merge 不低于最强非 LAMP 基线的统计为：

| 统计范围 | 胜出或并列胜出 | 比例 |
| --- | ---: | ---: |
| 全表 | 211/240 | 87.9% |
| Raw | 154/180 | 85.6% |
| Client Average | 57/60 | 95.0% |

正式 LAMP-Merge 的平均 Accuracy 为：

| 统计范围 | Cell 数 | 平均 Acc |
| --- | ---: | ---: |
| Raw | 180 | 0.6210 |
| Client Average | 60 | 0.6210 |

这些结果来自 `outputs/lamp_merge_full_client_local_20260708_193654/{resnet,convnext,vit_t,swin_tiny}`，其中正式实现只包含 M1 诊断原型重建和 M2 长尾患病率校准。模块内消融、预测分布诊断和非 Accuracy 指标在独立分析表中报告，不写入正式主结果表。
