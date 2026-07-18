# LAMP-Merge 模块内补充消融工作流

本文档定义 full-scope 模块内消融、可视化和理论分析流程。模块内消融的目标不是比较 LAMP-Merge 与通用基线，而是检验最终 LAMP-Merge 中每一类内部信息是否必要。因此，所有替代项均以最终 LAMP-Merge 为参照，并围绕两类上传信息展开：诊断原型信息与类别统计信息。

## 一、实验目标

补充实验需要证明三个命题。由于本实验是模块内消融，所有数值比较均以最终 `LAMP-Merge` 为参照；`avg`、`TIES`、`Fisher` 等通用方法只属于主实验基线，不作为判断某个内部变量是否有效的消融参照。

第一，诊断原型不是任意附加特征，而是抑制融合坍缩的核心信息。若去掉原型，只保留 checkpoint、分类头或类别无关特征统计，融合模型应更容易退化为少数类预测器，表现为 collapse ratio 上升、effective classes 下降、balanced accuracy 和 macro F1 下降。

第二，类别统计信息不是简单调参，而是医学长尾场景中必要的先验约束。若去掉类别支持数或患病率计数，模型要么无法可靠区分哪些客户端对某个类别具有证据，要么无法在强主导类别数据集上利用真实患病率，从而在 Derma 等长尾压力点上损失 accuracy 或出现分布偏移。

第三，原型信息与统计信息的作用机制应能够被可视化和理论解释。原型应表现为类间距离增大、类内聚合增强、预测分布更接近多类别诊断空间；统计信息应表现为客户端证据权重与类别覆盖相匹配，并在主导类别过强时产生有界而非坍缩式的 logit 偏置。

## 二、符号一致性规范

所有补充实验、图表说明和理论分析必须与正文 `lamp_merge_paper_sections.tex` 的符号保持一致。若需要新增理论量，必须显式说明它是补充分析符号，不能替换正文已有符号。后续写作中固定使用如下符号：

| 符号 | 正文含义 | 使用要求 |
|---|---|---|
| `K` | 客户端数量 | 不改写为 `N_c` 或其他客户端数量符号 |
| `C` | 诊断类别数量 | 不与 effective classes 混用 |
| `D_i` | 客户端 `i` 的本地数据集 | 保持为客户端数据集 |
| `D_{i,c}` | 客户端 `i` 中属于类别 `c` 的样本集合 | 用于定义类别支持 |
| `n_{i,c}` | M1 的类别支持数 | 只表示原型可靠性支持数 |
| `m_{i,c}` | M2 的类别患病率计数 | 只表示全局先验估计用计数 |
| `\phi_0` | 共享参考骨干 | 不写成 `\Theta_{\mathrm{base}}` 或其他模型参数符号 |
| `T(\cdot)` | 输入预处理 | 与 `\phi_0(T(x))` 一起使用 |
| `\mu_{i,c}` | 客户端 `i` 上传的类别 `c` 参考特征原型 | 不用于表示全局原型 |
| `e_{i,c}` | 类别证据权重 | 证据指数必须写作 `\gamma`，不能写作 `\rho` |
| `\alpha_{i,c}` | 类别级客户端可靠性权重 | 用于聚合客户端原型 |
| `p_c` | 服务端聚合得到的全局诊断原型 | 不写作 `hat_mu_c` |
| `w_c` | 原型分类头权重 | 与分类分数对应 |
| `s` | prototype head scale | 只表示 M1 尺度 |
| `\pi_c` | 由 `m_{i,c}` 估计的全局类别先验 | 不写作 `q_c` 或 `hat_pi_c` |
| `r` | 主导类不平衡强度 | 用于 M2 触发判断 |
| `tau` | M2 长尾触发阈值 | 不表示校准强度 |
| `lambda` | M2 有界长尾校准强度 | 不表示阈值 |
| `b_c` | M2 的中心化 log-prior bias | 只表示类别偏置 |
| `q(c)` | 测试集预测类别分布 | 只用于预测分布诊断，不表示类别先验 |
| `rho` | collapse ratio | 只表示预测坍缩强度，不能表示证据指数 |
| `C_eff` | effective predicted classes | 不与类别数 `C` 混用 |

其中类别支持数、证据权重和原型分类头分别定义为：

```math
n_{i,c}=|D_{i,c}|,
\qquad
e_{i,c}=(n_{i,c}+1)^\gamma\mathbf{1}[n_{i,c}>0],
\qquad
w_c=s\frac{p_c}{\|p_c\|_2}.
```

理论分析中若需要表示“真实参考特征均值”，统一使用补充符号 $\mu_c^\star$，并明确它只用于分析原型估计误差。全局融合原型仍必须写作 $p_c$。所有新增图表的 caption 和 markdown 表格也必须遵守该符号表。

## 三、模块内消融设计

### 3.1 原型信息消融

该组实验固定类别统计信息不变，只替换 M1 中的类别原型方向。所有设置使用相同的客户端支持数和患病率计数，以保证差异来自“类别方向如何构造”。

| 设置 | 说明 | 对应问题 |
|---|---|---|
| Full prototype | 正式方法。客户端上传每个类别在共享参考骨干中的特征均值，服务端按类别支持数加权重建全局原型 | 原型信息是否有效 |
| Classifier-head aggregation | 不使用特征原型，改为按类别支持数加权聚合各客户端 checkpoint 中的分类头权重 | 若只用本地分类头方向，是否足以避免坍缩 |
| Global-feature mean | 不使用类别原型，客户端只上传全体样本的参考特征均值，并将其作为所有类别共享的方向或共享校正项 | 类别条件信息是否必要 |
| Support-only synthetic head | 不上传特征方向，只根据类别支持数构造分类 bias 或单位随机方向控制组 | 仅有类别覆盖统计是否足以恢复判别方向 |
| Prototype shuffled-label control | 保持每个类别的原型数量和范数，但随机打乱类别原型与类别标签的对应关系 | 原型必须与诊断类别语义对齐，还是仅提供正则化 |

预期主要比较 `Full prototype`、`Classifier-head aggregation` 和 `Prototype shuffled-label control`。其中 classifier-head aggregation 是最接近原型的替代项，因为它同样提供每个类别的一条方向；若其明显弱于 Full prototype，可说明参考特征均值比本地训练后的分类头更稳定，原因是本地分类头已受到局部类别缺失影响，而参考原型直接来自本地真实类别样本。

### 3.2 统计信息消融

该组实验固定原型方向不变，只替换 M1/M2 中使用的类别统计信息。这里的统计信息包括两部分：原型支持数和患病率计数。支持数用于衡量某客户端对某类别原型估计的可靠性；患病率计数用于估计全局长尾先验。

| 设置 | 说明 | 对应问题 |
|---|---|---|
| Full statistics | 正式方法。M1 使用类别支持数加权原型，M2 使用类别患病率计数进行有界校准 | 统计信息是否有效 |
| Uniform client weight | 不使用类别支持数；每个出现该类别的客户端等权参与原型聚合 | 样本数可靠性是否必要 |
| Binary support only | 仅使用类别是否出现，不使用具体样本数 | 细粒度支持数是否优于类别存在性 |
| Global client-size weight | 使用客户端总样本数作为所有类别共享权重 | 类别级统计是否优于客户端级统计 |
| No prevalence calibration | 保留原型支持数，关闭 M2 患病率校准 | 患病率统计对长尾 accuracy 的作用 |
| Uniform prevalence prior | 保留 M2 形式，但将类别先验替换为均匀分布 | M2 的收益是否来自真实长尾统计 |
| Smoothed prevalence prior | 使用加性平滑后的患病率计数 | M2 是否依赖极端计数，平滑后是否稳定 |

预期主要比较 `Full statistics`、`Uniform client weight`、`Global client-size weight`、`No prevalence calibration` 和 `Uniform prevalence prior`。若 Full statistics 在 balanced accuracy、macro F1 和 collapse ratio 上优于 uniform/global 权重，说明支持数刻画了类别级可靠性；若在 Derma 等长尾压力点上 Full statistics 高于 no-prevalence 和 uniform-prior，说明 M2 使用的是医学长尾先验而不是任意 bias。

### 3.3 消融公式书写规范

每个消融项必须写明它替换了正式方法中的哪个量，以及替换后最终分类分数如何计算。正式方法固定为

```math
\mu_{i,c}
=
\frac{1}{n_{i,c}}
\sum_{(x,y)\in D_{i,c}}\phi_0(T(x)),
\qquad
e_{i,c}
=
(n_{i,c}+1)^\gamma\mathbf{1}[n_{i,c}>0],
```

```math
\alpha_{i,c}
=
\frac{e_{i,c}}{\sum_{j=1}^{K}e_{j,c}},
\qquad
p_c
=
\sum_{i=1}^{K}\alpha_{i,c}\mu_{i,c},
\qquad
w_c
=
s\frac{p_c}{\|p_c\|_2}.
```

患病率先验与最终分数固定为

```math
\pi_c
=
\frac{\sum_{i=1}^{K}m_{i,c}}
{\sum_{k=1}^{C}\sum_{i=1}^{K}m_{i,k}},
\qquad
b_c
=
\mathbf{1}[r>\tau]\lambda
\left(\log\pi_c-\frac{1}{C}\sum_{k=1}^{C}\log\pi_k\right),
```

```math
\mathrm{score}_c(x)=w_c^\top\phi_0(T(x))+b_c.
```

例如，`Uniform client weight` 不是一句“客户端等权”即可结束，而必须写成

```math
e_{i,c}^{\mathrm{uni}}
=
\mathbf{1}[n_{i,c}>0],
\qquad
\alpha_{i,c}^{\mathrm{uni}}
=
\frac{e_{i,c}^{\mathrm{uni}}}
{\sum_{j=1}^{K}e_{j,c}^{\mathrm{uni}}},
\qquad
p_c^{\mathrm{uni}}
=
\sum_{i=1}^{K}\alpha_{i,c}^{\mathrm{uni}}\mu_{i,c}.
```

然后继续写出

```math
w_c^{\mathrm{uni}}
=
s\frac{p_c^{\mathrm{uni}}}{\|p_c^{\mathrm{uni}}\|_2},
\qquad
\mathrm{score}_c^{\mathrm{uni}}(x)
=
(w_c^{\mathrm{uni}})^\top\phi_0(T(x))+b_c.
```

其他消融项也必须按同样格式写清楚，不能只写自然语言描述。若替换的是患病率先验，则必须写出替换后的 $\pi_c$、$b_c$ 和最终 $\mathrm{score}_c(x)$；若替换的是原型聚合权重，则必须写出替换后的 $\alpha_{i,c}$、$p_c$、$w_c$ 和最终 $\mathrm{score}_c(x)$。

## 四、评估指标

每个消融设置至少报告以下指标。

| 指标 | 作用 |
|---|---|
| Accuracy | 衡量总体分类性能，但需要与长尾指标联合解释 |
| Balanced Accuracy | 反映各诊断类别召回是否均衡，是检验非坍缩诊断能力的主要指标 |
| Macro F1 | 同时刻画各类别 precision 和 recall，避免多数类虚高 |
| Collapse Ratio | 预测最多类别的比例，直接度量单类或少数类坍缩 |
| Effective Classes | 预测分布熵对应的有效类别数，反映模型实际使用了多少诊断类别 |
| Pred-True TV | 预测分布与真实类别分布的总变差距离，衡量输出分布是否贴近医学数据分布 |
| Prototype Separation | 全局原型间平均余弦距离或最近邻类间距离，衡量类别方向是否分离 |
| Prototype Consistency | 同一类别跨客户端原型的平均相似度，衡量客户端类别证据是否一致 |

主消融结论、预测分布诊断、collapse ratio、balanced accuracy、macro F1、Pred-True TV、原型几何指标和超参数敏感性分析都必须使用正式汇总表中的 small 全量设置，而不能只使用单一 `beta=0.01` 或少量诊断 case。具体覆盖范围为 5 个医学图像数据集、4 个 backbone、3 个客户端数量和 3 个 Dirichlet beta 设置，即：

```text
datasets = bloodmnist_224, dermamnist_224, organcmnist_224, organsmnist_224, chaoshengmnist_224
backbones = resnet, convnext, vit_t, swin_tiny
K = 3, 5, 7
beta = 0, 0.01, 0.1
seed = 42
```

因此，每个消融设置和每个非客户端预测诊断方法都应产生 180 个 raw cells；在论文主表口径下，先对相同 `(dataset, backbone, K)` 的三个 beta 取均值，得到 60 个 client-average cells。所有“模块是否有效”的结论必须同时报告 raw cell 与 client-average cell，并以最终 LAMP-Merge 为参照报告 mean margin、胜出或持平 cell 数；不允许只报告某一个 beta，也不允许用是否超过 `avg` 来判断内部模块是否有效。`clients=3 / beta=0.01` 的 20 个 case 仅可作为 t-SNE 等机制可视化的代表子集，预测坍缩、balanced accuracy、macro F1 和 Pred-True TV 的正式结论必须来自 180 个 full-scope cases。为了分析 M2，还可以额外强调 `dermamnist_224 / resnet / clients=3 / beta=0.1 / seed=42` 作为长尾压力点，但它同样只用于机制解释。

后续执行中已将全量要求固化为独立脚本：

```text
scripts/run_lamp_merge_internal_ablation_full_parallel.sh
scripts/run_lamp_merge_prediction_diagnostics_full_parallel.sh
scripts/run_lamp_merge_hparam_full_parallel.sh
scripts/run_lamp_merge_full_evidence_queue.sh
scripts/summarize_lamp_merge_hparam_full.py
```

其中 `run_lamp_merge_full_evidence_queue.sh` 会等待模块内全量消融完成，再依次汇总全量消融、运行全量预测诊断、运行全量超参数敏感性分析。任何写入论文主文或消融表的结论都应来自这些 full-scope 输出；旧的单点超参数曲线只能标注为代表性现象，不作为主实验结论。

## 五、可视化产物

### 5.1 预测分布热图

每个数据集生成一张由全量 case 聚合得到的预测类别分布热图，行表示方法，列表示诊断类别。每个方法的分布先在每个 `(backbone,K,beta)` case 上计算，再在数据集内求平均。必须包含：

```text
avg
ties
fisher
LAMP-Merge
Full prototype
Classifier-head aggregation
Prototype shuffled-label control
Uniform client weight
No prevalence calibration
Uniform prevalence prior
```

该图用于说明 Full prototype 能把预测分布从少数类坍缩恢复到多类别诊断空间；统计信息替代项若失败，应表现为预测分布重新集中或偏离真实类别先验。若为了版面只展示若干代表图，caption 必须说明对应图来自全量数据集级聚合，而不是单一 beta。

### 5.2 指标柱状图

每个数据集生成 Accuracy、Balanced Accuracy、Macro F1、Collapse Ratio、Effective Classes 和 Pred-True TV 的柱状图。柱状图应分为两组：原型信息消融和统计信息消融。所有柱状图数值均来自全量聚合；论文主文可展示总体均值与两个代表数据集，附录展示全部数据集。

### 5.3 原型几何可视化

对每个方法计算全局类别方向矩阵，报告类间余弦相似度热图和最近邻类间距离。Full prototype 应表现为更清晰的类别分离；classifier-head aggregation 若受到本地训练偏置污染，其类间方向可能更相似或更不稳定。

### 5.4 t-SNE 降维可视化

t-SNE 使用共享参考骨干的测试集特征与全局类别原型共同降维。每张图包含测试样本点和类别原型点，样本点按真实类别着色，原型点使用更大的 marker。必须输出以下图：

```text
Full prototype
Classifier-head aggregation
Prototype shuffled-label control
Uniform client weight
No prevalence calibration
```

可视化解释口径为：若原型有效，则全局类别原型应落在对应类别测试样本簇附近；若使用本地分类头或打乱原型标签，原型点会偏离真实类别簇，导致分类边界与医学类别结构不一致。

## 六、理论分析框架

理论部分以预测坍缩风险为目标，给出类别原型与类别统计信息为何能够缓解坍缩的上界式解释。

设 $z=\phi_0(T(x))$ 为共享参考骨干特征。正文中客户端 $i$ 上传的类别原型为 $\mu_{i,c}$，服务端聚合得到的全局诊断原型为 $p_c$。为了分析估计误差，额外引入真实参考特征均值 $\mu_c^\star$。该符号只用于理论分析，不替代正文中的 $\mu_{i,c}$ 或 $p_c$。服务端聚合原型为

```math
p_c=\sum_i \alpha_{i,c}\mu_{i,c}.
```

在每个客户端类别样本独立且特征二阶矩有界的条件下，可得到原型估计误差的形式：

```math
\mathbb{E}\|p_c-\mu_c^\star\|_2^2
\le
\sum_i \alpha_{i,c}^2\frac{\sigma_c^2}{n_{i,c}}+\mathrm{Bias}_c^2.
```

该式说明类别支持数加权并非任意设计。若 $n_{i,c}$ 较大，则该客户端原型估计方差较小，应赋予更高权重；若某客户端未观察到类别 $c$，即 $n_{i,c}=0$，则其不应对该类方向产生贡献。相比之下，普通参数平均没有这种按类别屏蔽机制，会把无证据客户端的分类头或参数方向混入类别 $c$，从而增加该类方向误差。

进一步设类别 $c$ 与类别 $d$ 的真实 margin 为

```math
\Delta_{c,d}(x)
=
(\mu_c^\star)^\top z-(\mu_d^\star)^\top z.
```

当原型估计误差满足

```math
\|p_c-\mu_c^\star\|_2+\|p_d-\mu_d^\star\|_2
<
\frac{\Delta_{c,d}(x)}{\|z\|_2},
```

则替换为估计原型后不会改变样本 $x$ 在类别 $c$ 与 $d$ 之间的判别顺序。因此，降低每个类别的原型估计误差可以直接降低跨类误判和多数类吸收少数类的风险。

对于 M2，正文中上传患病率计数 $m_{i,c}$ 得到的类别先验为 $\pi_c$，最终分类分数为

```math
\mathrm{score}_c(x)=w_c^\top z+\lambda\left(\log \pi_c-\frac{1}{C}\sum_k\log \pi_k\right).
```

若 $\lambda$ 有界，则先验项对任意两个类别的 margin 改变量满足

```math
|b_c-b_d|
\le
\lambda\left|\log \pi_c-\log \pi_d\right|.
```

该式说明 M2 只能对 M1 已经形成的类别方向进行有限校准，而不能无限制地把所有样本推向多数类。理论叙述应强调：M1 提供多类别判别方向，M2 只在这些方向之上加入有界长尾先验；因此 LAMP-Merge 能同时避免普通融合的类别坍缩，并保留医学长尾分布中真实存在的多数类统计信息。

## 七、执行顺序

1. 实现原型信息替代项：classifier-head aggregation、global-feature mean、support-only synthetic head、shuffled-label prototype。
2. 实现统计信息替代项：uniform client weight、binary support only、global client-size weight、no prevalence calibration、uniform prior、smoothed prior。
3. 对所有替代项运行 small 全量设置，输出 180 个 raw cells 和 60 个 client-average cells。
4. 使用同一套结果生成总体均值、按数据集均值、raw cell 统计、client-average cell 统计；所有模块内消融表同时报告相对最终 LAMP-Merge 的 mean margin 和胜出或持平 cell 数。
5. 在同一 full-scope 网格上重新计算 Accuracy、Balanced Accuracy、Macro F1、Collapse Ratio、Effective Classes、Pred-True TV 和预测分布热图。
6. 在同一 full-scope 网格上扫描 $s\in[5,40]$ 与 $\lambda$ 的候选值，并用 60 个 client-average cells 报告敏感性。
7. 使用全量聚合结果生成预测分布热图、指标柱状图和原型相似度热图；若展示 t-SNE，caption 中必须说明其为机制图，不替代全量数值结论。
8. 汇总数值表：总体均值、按数据集均值、raw cell 统计、client-average cell 统计；明确区分“全量数值证据”和“机制可视化”。
9. 按“符号一致性规范”逐项检查 CSV 字段、图表 caption、markdown 表格和理论公式，确保与正文符号一致。
10. 将理论解释写入论文分析文档，使用“原型估计误差上界”和“有界先验 margin 改变量”支撑方法合理性。

## 八、最终交付文件

建议最终产物如下：

```text
My_merge_ret/LAMP-Merge模块内补充消融汇报.md
My_merge_ret/reports/lamp_merge_internal_ablation.csv
My_merge_ret/reports/lamp_merge_internal_ablation_summary.md
My_merge_ret/figures/lamp_merge_internal_ablation/
My_merge_ret/figures/lamp_merge_internal_ablation/tsne/
```

该组结果完成后，应把原有 `LAMP-Merge模块消融与超参数分析.md` 中的“模块内消融”替换为本实验，而将原来的 $s$ 与 $\lambda$ 扫描归入“超参数敏感性”。
