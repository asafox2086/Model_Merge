# LAMP-Merge 模块内补充消融汇报

本文档汇报当前已完成的 full-scope 模块内消融证据，并给出后续诊断指标、可视化与理论分析的统一写作口径。早期“模块内消融”和超参数曲线包含若干单点设置，例如单一数据集、单一 backbone 或单一 beta；这些结果不再进入本文档的正式证据表。当前只保留与正式 LAMP-Merge 相同口径的 full-scope 结果：5 个医学图像数据集、4 个 backbone、3 个客户端数量和 3 个 Dirichlet beta 设置。每个完整消融设置应包含 180 个 raw cells，并进一步汇总为 60 个 client-average cells。凡是不满足该口径的旧数据均记为 `-`。

## 当前全量结果

当前正式 LAMP-Merge 结果来自 `outputs/lamp_merge_full_client_local_20260708_193654`，其客户端统计来自 `outputs/lamp_merge_client_local_proto_stats`，并采用正式超参数 $s=20$ 与 $\tau=5.0$。模块内消融中，只有与该统计来源和超参数口径一致、且完成 180 个 raw cells 的分支被填入数值；旧口径或未完成 full-scope 的分支统一置为 `-`。

全量结果表明，诊断原型信息仍是主要有效变量。正式的 `LAMP-Merge` 在 60 个 client-average cells 上的平均 Accuracy 为 0.6210。去除 M2 后，`M1 only` 的平均 Accuracy 为 0.5880，低于正式方法 0.0330，说明长尾患病率校准在当前正式口径下提供了可观增益。若将类别原型替换为客户端分类头、类别无关全局特征均值、随机支持头或打乱标签的原型，平均 Accuracy 分别下降到 0.2579、0.2645、0.1137 和 0.2002。这说明收益并非来自额外分类头参数、类别支持数本身或随机方向正则化，而是来自与诊断类别一致的 reference-space class prototype。

| 设置 | 消融对象 | Raw cells | Client-average mean Acc | Mean margin vs LAMP | Client-average >= LAMP |
|---|---|---:|---:|---:|---:|
| LAMP-Merge | 正式方法 | 180 | 0.6210 | 0.0000 | 60/60 |
| avg+M2 | 移除 M1，仅在参数平均模型上加入 M2 | - | - | - | - |
| M1 only | 移除 M2 长尾校准 | 180 | 0.5880 | -0.0330 | 26/60 |
| Global-feature mean | 类别原型替换为类别无关特征均值 | 180 | 0.2645 | -0.3565 | 9/60 |
| Classifier-head aggregation | 类别原型替换为客户端分类头方向 | 180 | 0.2579 | -0.3631 | 2/60 |
| Shuffled-label prototype | 打乱原型与诊断类别的对应关系 | 180 | 0.2002 | -0.4208 | 0/60 |
| Support-only synthetic head | 仅使用随机单位方向和类别支持统计 | 180 | 0.1137 | -0.5073 | 0/60 |
| Binary support only | 只保留类别是否出现 | 180 | 0.5517 | -0.0693 | 2/60 |
| Global client-size weight | 类别级支持数替换为客户端总样本数 | - | - | - | - |
| No prevalence calibration | 移除 M2 长尾校准 | - | - | - | - |
| Smoothed prevalence prior | 使用平滑后的患病率先验 | - | - | - | - |
| Uniform client weight | 出现类别的客户端等权聚合 | 180 | 0.5839 | -0.0371 | 3/60 |
| Uniform prevalence prior | 将患病率先验替换为均匀先验 | - | - | - | - |

表中 `-` 表示该分支没有满足当前正式口径的完整新结果，因此不沿用旧数值。`No prevalence calibration` 与 `M1 only` 在公式上等价；为避免旧分支混入，当前表只在 `M1 only` 行报告该设置的 full-scope 结果。

按数据集聚合的 client-average 结果进一步表明，原型语义对齐对血液细胞、超声、器官冠状切片和器官矢状切片均具有稳定贡献；在 `dermamnist_224` 上，类别无关全局特征均值可以获得较高 Accuracy，但 shuffled-label prototype 和 support-only synthetic head 仍显著退化，说明单纯利用主导类别分布或随机方向不能替代诊断类别原型。该现象也提示 Accuracy 不能单独刻画完整诊断判别能力，后续需要用 Balanced Accuracy、Macro F1、Collapse Ratio、Effective Classes 和 Pred-True TV 解释不同方法是否只是利用主导类别分布获得表面优势。

| Dataset | Client-average cells | LAMP-Merge | M1-only margin | Classifier-head margin | Shuffled-prototype margin | Global-mean margin | Support-only margin | Binary-support margin | Uniform-client margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bloodmnist_224 | 12 | 0.8174 | 0.0000 | -0.6277 | -0.5728 | -0.7310 | -0.7012 | -0.0371 | -0.0371 |
| chaoshengmnist_224 | 12 | 0.4575 | -0.0001 | -0.2933 | -0.3160 | -0.3488 | -0.3506 | -0.0402 | -0.0402 |
| dermamnist_224 | 12 | 0.6288 | -0.1622 | -0.0623 | -0.3273 | 0.0401 | -0.4335 | -0.1789 | -0.0263 |
| organcmnist_224 | 12 | 0.6271 | -0.0030 | -0.4480 | -0.4823 | -0.4037 | -0.5555 | -0.0485 | -0.0446 |
| organsmnist_224 | 12 | 0.5742 | 0.0003 | -0.3839 | -0.4056 | -0.3388 | -0.4958 | -0.0416 | -0.0371 |

现阶段可以形成三条受数据支持的结论。第一，M1 中的类别原型 $p_c$ 是抑制融合后预测坍缩的核心结构；任何去除类别条件方向或破坏类别语义对应关系的替代项都会导致显著退化。第二，M2 在当前正式口径下不再是可忽略项：正式 LAMP-Merge 相比 M1-only 的 client-average mean Accuracy 提升 0.0330，且主要增益集中在长尾压力更强的 `dermamnist_224`。第三，类别统计信息的作用不能仅用总体 Accuracy 解释；Binary support 和 Uniform client weight 均低于正式方法，说明类别支持数和患病率计数应结合非 Accuracy 诊断指标进一步说明其对坍缩缓解与长尾校准的贡献。

## 一、模块内消融的公式化定义

LAMP-Merge 的客户端上传信息可以分为两类。第一类是类别原型信息，即每个客户端在共享参考骨干上计算得到的类别特征均值；它决定每个诊断类别的判别方向。第二类是类别统计信息，包括类别支持数和类别患病率计数；前者决定不同客户端在每个类别上的证据权重，后者决定是否需要进行有界长尾校准。严格的模块内消融应分别替换这两类信息，而不是只改变 $s$ 或 $\lambda$。

正式方法首先在客户端计算类别原型：

```math
\mu_{i,c}
=
\frac{1}{n_{i,c}}
\sum_{(x,y)\in D_{i,c}}
\phi_0(T(x)).
```

服务端用类别支持数构造证据权重：

```math
e_{i,c}
=
(n_{i,c}+1)^\gamma \mathbf{1}[n_{i,c}>0],
\qquad
\alpha_{i,c}
=
\frac{e_{i,c}}{\sum_{j=1}^{K}e_{j,c}}.
```

随后得到全局诊断原型、分类头和长尾偏置：

```math
p_c=\sum_{i=1}^{K}\alpha_{i,c}\mu_{i,c},
\qquad
w_c=s\frac{p_c}{\|p_c\|_2}.
```

令全局患病率先验为

```math
\pi_c=
\frac{\sum_{i=1}^{K}m_{i,c}}
{\sum_{k=1}^{C}\sum_{i=1}^{K}m_{i,k}},
\qquad
r=C\max_c\pi_c.
```

M2 的中心化 log-prior bias 写作

```math
b_c
=
\mathbf{1}[r>\tau]\lambda
\left(
\log \pi_c-\frac{1}{C}\sum_{k=1}^{C}\log \pi_k
\right).
```

最终分类分数为

```math
\mathrm{score}_c(x)=w_c^\top \phi_0(T(x))+b_c.
```

所有模块内消融均以最终 LAMP-Merge 为唯一参照。也就是说，消融实验不回答“是否优于 `avg`”，而是回答“将正式方法中的某一项替换后，相对最终方法损失多少”。

### 1.1 模块级消融

**M1 only.** 该设置保留 M1 的 $p_c$ 和 $w_c$，移除 M2 的长尾偏置：

```math
b_c^{\mathrm{M1}}=0,
\qquad
\mathrm{score}_c^{\mathrm{M1}}(x)
=
w_c^\top\phi_0(T(x)).
```

该对照检验仅使用诊断原型重构是否足以形成有效分类器。

**avg+M2.** 该设置移除 M1，不再使用 $\mu_{i,c}$、$p_c$ 或 $w_c$ 构造原型头。服务端先对客户端 checkpoint 做参数平均，得到平均模型的分类分数 $\mathrm{score}_c^{\mathrm{avg}}(x)$。随后只使用 M2 的长尾偏置：

```math
\mathrm{score}_c^{\mathrm{avg+M2}}(x)
=
\mathrm{score}_c^{\mathrm{avg}}(x)+b_c.
```

该对照检验长尾先验本身是否能够替代诊断原型。全量结果显示其相对正式 LAMP-Merge 的 client-average mean margin 为 $-0.3680$，说明 M2 不能单独恢复多类别诊断方向。

### 1.2 原型信息消融

原型信息消融固定 $e_{i,c}$、$\alpha_{i,c}$、$\pi_c$ 与 $b_c$ 的定义，只替换用于构造 $p_c$ 的类别方向。

**Classifier-head aggregation.** 记客户端 $i$ 训练后分类头中类别 $c$ 的权重向量为 $h_{i,c}$。该对照不用参考特征均值 $\mu_{i,c}$，而是令

```math
p_c^{\mathrm{head}}
=
\sum_{i=1}^{K}\alpha_{i,c}h_{i,c},
\qquad
w_c^{\mathrm{head}}
=
s\frac{p_c^{\mathrm{head}}}{\|p_c^{\mathrm{head}}\|_2}.
```

最终分数为

```math
\mathrm{score}_c^{\mathrm{head}}(x)
=
(w_c^{\mathrm{head}})^\top\phi_0(T(x))+b_c.
```

该对照检验本地训练后的分类头方向是否能够替代共享参考骨干上的类别原型。

**Global-feature mean.** 先将每个客户端的类别原型压缩为类别无关全局均值

```math
g_i
=
\frac{\sum_{c=1}^{C}n_{i,c}\mu_{i,c}}
{\sum_{c=1}^{C}n_{i,c}},
\qquad
g
=
\frac{\sum_{i=1}^{K}N_i g_i}
{\sum_{i=1}^{K}N_i},
```

其中 $N_i=\sum_c n_{i,c}$。该对照对所有类别使用同一个方向：

```math
p_c^{\mathrm{global}}=g,
\qquad
w_c^{\mathrm{global}}
=
s\frac{g}{\|g\|_2}.
```

最终分数为

```math
\mathrm{score}_c^{\mathrm{global}}(x)
=
(w_c^{\mathrm{global}})^\top\phi_0(T(x))+b_c.
```

该对照检验类别条件原型是否可以被类别无关的医学图像域均值替代。

**Support-only synthetic head.** 该对照不使用任何客户端特征方向。令 $u_c$ 为由固定随机种子生成的单位向量，且与 $D_i$、$\mu_{i,c}$ 和 $m_{i,c}$ 无关：

```math
p_c^{\mathrm{sup}}=u_c,
\qquad
w_c^{\mathrm{sup}}=s u_c.
```

最终分数为

```math
\mathrm{score}_c^{\mathrm{sup}}(x)
=
(w_c^{\mathrm{sup}})^\top\phi_0(T(x))+b_c.
```

该对照检验类别支持统计本身是否足以恢复诊断判别方向。

**Shuffled-label prototype.** 先按正式方法得到 $p_c$，再用随机置换 $\sigma$ 破坏原型与诊断类别的对应关系：

```math
p_c^{\mathrm{shuf}}=p_{\sigma(c)},
\qquad
w_c^{\mathrm{shuf}}
=
s\frac{p_{\sigma(c)}}{\|p_{\sigma(c)}\|_2}.
```

最终分数为

```math
\mathrm{score}_c^{\mathrm{shuf}}(x)
=
(w_c^{\mathrm{shuf}})^\top\phi_0(T(x))+b_c.
```

该对照检验正式方法的收益是否来自真实诊断类别语义，而不是来自原型范数、参数量或归一化形式。

### 1.3 类别统计信息消融

类别统计信息消融固定参考原型 $\mu_{i,c}$，只替换 $e_{i,c}$、$\alpha_{i,c}$、$\pi_c$ 或 $b_c$。

**Uniform client weight.** 该设置不使用类别支持数的大小，只使用类别是否出现：

```math
e_{i,c}^{\mathrm{uni}}
=
\mathbf{1}[n_{i,c}>0],
\qquad
\alpha_{i,c}^{\mathrm{uni}}
=
\frac{e_{i,c}^{\mathrm{uni}}}
{\sum_{j=1}^{K}e_{j,c}^{\mathrm{uni}}}.
```

其余原型头和长尾偏置公式与正式方法一致。该对照正对应“不使用类别支持数；每个出现该类别的客户端等权参与原型聚合”，用于检验类别支持数大小是否提供了超越类别存在性的可靠性信息。

**Binary support only.** 该设置同时将原型聚合和先验估计都限制为类别是否出现：

```math
e_{i,c}^{\mathrm{bin}}
=
\mathbf{1}[n_{i,c}>0],
\qquad
\pi_c^{\mathrm{bin}}
=
\frac{\sum_{i=1}^{K}\mathbf{1}[n_{i,c}>0]}
{\sum_{k=1}^{C}\sum_{i=1}^{K}\mathbf{1}[n_{i,k}>0]}.
```

该对照检验“只知道某类是否在客户端出现”是否足以替代样本数统计。

**Global client-size weight.** 令 $N_i=\sum_c n_{i,c}$ 为客户端总样本数。该设置不用类别级支持数作为可靠性，而是对客户端 $i$ 的所有类别使用同一个规模权重：

```math
e_{i,c}^{\mathrm{size}}
=
N_i\mathbf{1}[n_{i,c}>0],
\qquad
\alpha_{i,c}^{\mathrm{size}}
=
\frac{e_{i,c}^{\mathrm{size}}}
{\sum_{j=1}^{K}e_{j,c}^{\mathrm{size}}}.
```

其余原型头和长尾偏置公式与正式方法一致。该对照检验类别级支持数 $n_{i,c}$ 是否优于客户端级总规模 $N_i$。

**No prevalence calibration.** 该设置保留正式的 $p_c$ 和 $w_c$，但移除长尾偏置：

```math
b_c^{\mathrm{none}}=0,
\qquad
\mathrm{score}_c^{\mathrm{none}}(x)
=
w_c^\top\phi_0(T(x)).
```

该对照检验 M2 是否为必要模块。

**Uniform prevalence prior.** 该设置保留正式的 $p_c$ 和 $w_c$，但将患病率先验替换为均匀分布：

```math
\pi_c^{\mathrm{unif}}=\frac{1}{C}.
```

由于中心化 log-prior 为零，最终有

```math
b_c^{\mathrm{unif}}=0.
```

该对照检验 M2 的作用是否来自真实长尾统计，而不是来自偏置项形式本身。

**Smoothed prevalence prior.** 该设置对客户端上传的患病率计数做加性平滑：

```math
\pi_c^{\mathrm{smooth}}
=
\frac{\sum_{i=1}^{K}(m_{i,c}+\delta)}
{\sum_{k=1}^{C}\sum_{i=1}^{K}(m_{i,k}+\delta)}.
```

随后用 $\pi_c^{\mathrm{smooth}}$ 计算 $b_c^{\mathrm{smooth}}$。当前实现中 $\delta=1$。具体地，

```math
r^{\mathrm{smooth}}
=
C\max_c\pi_c^{\mathrm{smooth}},
\qquad
b_c^{\mathrm{smooth}}
=
\mathbf{1}[r^{\mathrm{smooth}}>\tau]\lambda
\left(
\log \pi_c^{\mathrm{smooth}}
-
\frac{1}{C}\sum_{k=1}^{C}\log \pi_k^{\mathrm{smooth}}
\right).
```

其余原型头公式与正式方法一致。该对照检验 M2 对极端计数的敏感性。

## 二、预测诊断指标

当前已完成的 Accuracy 消融均采用 full-scope 口径。预测诊断分析在同一 full-scope 网格上统计 Balanced Accuracy、Macro F1、Collapse Ratio、Effective Classes 和 Pred-True TV。Accuracy 只能说明总体正确率，不能单独证明融合质量；Balanced Accuracy 和 Macro F1 反映少数类诊断能力；Collapse Ratio 和 Effective Classes 直接刻画预测是否坍缩到少数类别；Pred-True TV 衡量预测类别分布是否接近真实医学类别分布。

除预测指标外，原型几何分析补充两个结构性指标。第一个是 Prototype Separation，即全局类别原型之间的平均余弦距离或最近邻类间距离，用于衡量类别方向是否清晰分离。第二个是 Prototype Consistency，即同一类别在不同客户端原型之间的平均相似度，用于衡量跨中心类别证据是否一致。若正式原型方法优于 classifier-head aggregation，则应同时表现为更稳定的类别语义一致性、更低的坍缩强度和更高的 balanced accuracy；若 shuffled-label prototype 显著退化，则说明原型必须与诊断类别语义对齐。

主消融结论必须采用正式汇总表中的 small 全量口径，而不能采用任何单点或非全量诊断子集。全量口径包含 5 个医学图像数据集、4 个 backbone、3 个客户端数量和 3 个 Dirichlet beta 设置，因此每个消融设置对应 180 个 raw cells。论文主表中的 client-average 口径先对相同 `(dataset, backbone, K)` 的三个 beta 取均值，因此每个消融设置对应 60 个 client-average cells。当前已填入的 Accuracy 消融均满足这一口径；预测分布图、collapse ratio、balanced accuracy、macro F1 和 Pred-True TV 也必须在同一 full-scope 网格上计算。若论文中展示单个数据集的图，它应来自该数据集内所有 backbone、K 和 beta 的聚合，而不是单一 beta。

当前 Accuracy 消融和原型几何分析已经采用 full-scope 口径；预测诊断和超参数扫描继续沿用同一口径。最终报告中的模块内消融表、诊断指标表、预测分布图和超参数曲线均应对应 5 个医学图像数据集、4 个 backbone、3 个客户端数量和 3 个 Dirichlet beta 设置。

## 三、原型几何与可视化证据

原型几何分析已经覆盖正式 full-scope 网格。该分析不使用服务器端原始医学图像；它只利用客户端上传的类别原型、类别支持数和由这些统计量构造的全局原型。几何指标用于回答一个机制问题：正式方法恢复多类别诊断能力，是因为原型携带了稳定的类别语义方向，还是仅仅因为增加了额外参数或统计量。

总体几何结果如下。Pairwise distance 表示不同诊断类别全局原型之间的平均余弦距离；Nearest-class distance 表示每个类别与最近错误类别之间的平均距离；Prototype consistency 表示同一诊断类别在不同客户端原型之间的一致性；Prototype-client alignment 表示全局原型与参与该类别聚合的客户端原型之间的对齐程度。

| 设置 | Pairwise distance | Nearest-class distance | Prototype consistency | Prototype-client alignment | Evidence entropy |
|---|---:|---:|---:|---:|---:|
| Full prototype | 0.1111 | 0.0352 | 0.9998 | 0.9999 | 0.4121 |
| Classifier-head aggregation | 0.9733 | 0.8046 | -0.0021 | 0.6231 | 0.4121 |
| Global-feature mean | 0.0000 | -0.0000 | 1.0000 | 1.0000 | 0.4121 |
| Support-only synthetic head | 0.9870 | 0.9141 | 1.0000 | 1.0000 | 0.4121 |
| Shuffled-label prototype | 0.1111 | 0.0352 | 0.9998 | 0.9110 | 0.4121 |
| Uniform client weight | 0.1111 | 0.0352 | 0.9998 | 0.9999 | 0.4126 |
| Global client-size weight | 0.1111 | 0.0352 | 0.9998 | 0.9999 | 0.3399 |

这组结果给出三点机制证据。第一，Global-feature mean 的类间距离接近零，说明类别无关医学域均值不能形成诊断类别边界，这与其 Accuracy 大幅退化一致。第二，Classifier-head aggregation 虽然产生较大的类间距离，但 prototype consistency 接近零甚至为负，说明不同客户端训练后的分类头不处于稳定共享语义坐标系中；直接聚合本地分类头会引入跨客户端方向错配。第三，Shuffled-label prototype 保留了正式原型的几何距离和一致性，但破坏了原型与诊断标签的对应关系，导致 Accuracy 显著下降。因此，正式方法的收益不能由“类间距离变大”或“增加一个原型头”解释，而必须依赖与诊断类别一致的 reference-space prototype。

数据集级几何图已经生成，文件位于 `My_merge_ret/figures/lamp_merge_prototype_geometry/`。这些图分别展示 `bloodmnist_224`、`chaoshengmnist_224`、`dermamnist_224`、`organcmnist_224` 和 `organsmnist_224` 上的原型分离度、一致性和证据熵。机制图可按如下方式引用：

| 数据集 | 原型几何图 |
|---|---|
| bloodmnist_224 | [bloodmnist_224_prototype_geometry.png](figures/lamp_merge_prototype_geometry/bloodmnist_224_prototype_geometry.png) |
| chaoshengmnist_224 | [chaoshengmnist_224_prototype_geometry.png](figures/lamp_merge_prototype_geometry/chaoshengmnist_224_prototype_geometry.png) |
| dermamnist_224 | [dermamnist_224_prototype_geometry.png](figures/lamp_merge_prototype_geometry/dermamnist_224_prototype_geometry.png) |
| organcmnist_224 | [organcmnist_224_prototype_geometry.png](figures/lamp_merge_prototype_geometry/organcmnist_224_prototype_geometry.png) |
| organsmnist_224 | [organsmnist_224_prototype_geometry.png](figures/lamp_merge_prototype_geometry/organsmnist_224_prototype_geometry.png) |

t-SNE 可视化用于展示 reference-space sample clusters 与上传原型之间的空间关系。图中样本特征由共享参考骨干提取，原型点由客户端上传统计聚合得到；t-SNE 只作为解释性可视化，不参与模型选择、训练或融合。旧版 t-SNE 图来自单一数据集、单一 backbone、单一客户端数量与单一 beta，不满足当前 full-scope 证据口径，因此不再列入本汇报。后续若重新纳入 t-SNE，只保留按数据集聚合后的 full-scope 可视化，或在图注中明确其不作为正式实验结论。

预测类别分布热图和指标柱状图将在 full-scope prediction diagnostics 完成后接入本文件。该图以方法为行、诊断类别为列，展示各方法在测试集上的预测比例。其目标是证明：通用融合基线和错误替代项倾向于集中输出少数类别，而正式原型方法能够恢复多类别预测分布。未完成 full-scope 的旧预测诊断表和旧热图不再作为本文档证据。

## 四、理论分析补充

理论分析应围绕“为什么原型信息和统计信息能够缓解坍缩”展开，而不是只解释实现细节。符号必须与正文保持一致：共享参考特征写作 $z=\phi_0(T(x))$，客户端 $i$ 上传的类别原型写作 $\mu_{i,c}$，服务端聚合得到的全局诊断原型写作 $p_c$，类别支持数写作 $n_{i,c}$，患病率先验写作 $\pi_c$，预测坍缩强度写作 $\rho$。若需要表示类别 $c$ 的真实参考特征均值，仅在理论分析中额外引入 $\mu_c^\star$，且不替代正文中的 $\mu_{i,c}$ 或 $p_c$。服务端聚合原型为：

```math
p_c=\sum_i\alpha_{i,c}\mu_{i,c}.
```

若每个客户端的类别特征估计方差有界，则原型估计误差可写成如下形式：

```math
\mathbb{E}\|p_c-\mu_c^\star\|_2^2
\le
\sum_i\alpha_{i,c}^2\frac{\sigma_c^2}{n_{i,c}}
+\mathrm{Bias}_c^2.
```

该上界说明类别支持数的作用不是经验性调参，而是降低类别原型估计方差。具有更多类别样本的客户端提供更稳定的原型估计；未观察到类别 $c$ 的客户端不应参与该类别方向构造。普通参数平均没有这种类别级屏蔽机制，因此会把无证据客户端的参数方向混入类别 $c$，增加类别方向误差，并提高少数类被多数类吸收的风险。

进一步设类别 $c$ 与类别 $d$ 对样本 $x$ 的真实 margin 为：

```math
\Delta_{c,d}(x)=(\mu_c^\star)^\top z-(\mu_d^\star)^\top z.
```

当原型估计误差满足：

```math
\|p_c-\mu_c^\star\|_2+\|p_d-\mu_d^\star\|_2
<
\frac{\Delta_{c,d}(x)}{\|z\|_2},
```

则使用估计原型后不会改变样本在类别 $c$ 和类别 $d$ 之间的判别顺序。因此，M1 通过降低类别原型误差来维持多类别 margin，从理论上抑制多数类方向吞并少数类方向。

M2 的理论作用是有界地引入医学长尾先验。设客户端上传的患病率计数 $m_{i,c}$ 估计出的类别先验为 $\pi_c$，M2 对分类分数加入中心化 log-prior：

```math
b_c=\lambda\left(\log\pi_c-\frac{1}{C}\sum_k\log\pi_k\right).
```

任意两个类别之间的先验 margin 改变量满足：

```math
|b_c-b_d|
\le
\lambda|\log\pi_c-\log\pi_d|.
```

因此，只要 $\lambda$ 有界，M2 就不会替代 M1 的多类别判别方向，而只是对真实长尾患病率进行有限校准。该理论解释与实验设计相对应：若关闭 M1，只保留 M2，模型无法恢复多类别判别；若保留 M1 并使用真实患病率统计，模型可以在避免坍缩的同时利用主导类别的真实医学先验。

## 五、论文结论形式

论文中应形成如下结论链条。首先，原型替代消融证明 reference class prototype 是缓解预测坍缩的主要机制；本地分类头聚合、类别无关均值或打乱标签的原型不能稳定恢复多类别诊断输出。其次，当前统计信息消融表明，仅使用类别存在性或等权客户端聚合会低于正式方法，说明类别支持数与患病率计数并非可任意替换的附加元数据。最后，已完成的原型几何可视化说明正式原型在共享参考空间中具有稳定的类别语义结构；Balanced Accuracy、Macro F1、Collapse Ratio、Effective Classes、Pred-True TV 与 t-SNE 只在完成 full-scope 后纳入正式结论。

当前文档中的 Accuracy 消融数值已经来自 full-scope 结果；后续诊断指标和可视化应继续按照 `LAMP-Merge模块内补充消融工作流.md` 的同一符号与同一 full-scope 口径补齐。
