# LAMP-Merge 模块内补充消融汇报

本文档汇报当前模块内消融需要补充的全量证据，并给出后续实验、可视化与理论分析的执行口径。现有版本已经证明了 `LAMP-Merge` 相对于通用融合基线的整体优势，也完成了 `M1 only`、`avg+M2`、`avg` 等模块间消融；但早期“模块内消融”和超参数曲线包含若干单点设置，例如单一数据集、单一 backbone 或单一 beta。此类结果只能作为机制观察，不能作为论文主结论。后续所有正式消融、诊断指标和超参数敏感性分析均采用 full-scope 网格：5 个医学图像数据集、4 个 backbone、3 个客户端数量和 3 个 Dirichlet beta 设置。每个方法或消融项对应 180 个 raw cells，并进一步汇总为 60 个 client-average cells。

## 当前阶段性结果

当前已经完成的模块内消融覆盖 `bloodmnist_224`、`dermamnist_224`、`organcmnist_224`、`organsmnist_224` 和 `chaoshengmnist_224` 五个医学图像数据集，覆盖 `resnet`、`convnext`、`vit_t` 和 `swin_tiny` 四种 backbone，覆盖 \(K\in\{3,5,7\}\) 与 \(\beta\in\{0,0.01,0.1\}\)。因此，一个完整消融设置包含 180 个 raw cells；client-average 结果先在固定数据集、backbone 与 \(K\) 的条件下对三个 \(\beta\) 取均值，因此包含 60 个 client-average cells。当前 `avg+M2` 仍处于运行状态，仅有部分 cells 完成，因此不纳入以下正式结论。

阶段性结果表明，诊断原型信息是当前方法的主要有效变量。正式的 `LAMP-Merge` 在 60 个 client-average cells 中有 53 个不低于 `avg`，平均 Accuracy 为 0.5884；若将类别原型替换为类别无关全局特征均值、支持数构造头或打乱标签的原型，平均 Accuracy 分别下降到 0.0833、0.0860 和 0.1440，且没有任何 client-average cell 能达到正式方法。若使用客户端本地分类头方向替代共享参考骨干上的类别原型，平均 Accuracy 为 0.2378，仅有 7/60 个 client-average cells 不低于正式方法。这说明收益并非来自额外分类头参数、类别支持数本身或随机方向正则化，而是来自与诊断类别一致的 reference class prototype。

| 设置 | 消融对象 | Raw cells | Client-average mean Acc | Client-average >= avg | Client-average >= LAMP |
|---|---:|---:|---:|---:|---:|
| LAMP-Merge | 正式方法 | 180 | 0.5884 | 53/60 | 60/60 |
| M1 only | 移除 M2 长尾校准 | 180 | 0.5884 | 53/60 | 60/60 |
| Global-feature mean | 类别原型替换为类别无关特征均值 | 180 | 0.0833 | 7/60 | 0/60 |
| Classifier-head aggregation | 类别原型替换为客户端分类头方向 | 180 | 0.2378 | 37/60 | 7/60 |
| Shuffled-label prototype | 打乱原型与诊断类别的对应关系 | 180 | 0.1440 | 24/60 | 0/60 |
| Support-only synthetic head | 仅使用类别支持统计构造分类头 | 180 | 0.0860 | 6/60 | 0/60 |
| Binary support only | \(n_{i,c}\) 替换为类别是否出现 | 180 | 0.5884 | 53/60 | 47/60 |
| Global client-size weight | 类别级 \(n_{i,c}\) 替换为客户端总样本数 | 180 | 0.5885 | 53/60 | 43/60 |
| No prevalence calibration | 移除 \(\pi_c\) 对应的 M2 校准 | 180 | 0.5884 | 53/60 | 60/60 |
| Smoothed prevalence prior | 使用平滑后的 \(\pi_c\) | 180 | 0.5884 | 53/60 | 60/60 |
| Uniform client weight | 出现类别的客户端等权聚合 | 180 | 0.5884 | 53/60 | 47/60 |
| Uniform prevalence prior | 将 \(\pi_c\) 替换为均匀先验 | 180 | 0.5884 | 53/60 | 60/60 |

按数据集聚合的 client-average 结果进一步表明，`LAMP-Merge` 对血液细胞、超声、器官冠状切片和器官矢状切片均显著高于 `avg`；`dermamnist_224` 是当前 full-scope 网格中的主要长尾压力点，`avg` 的平均 Accuracy 为 0.5042，而正式方法为 0.4619。该现象与本文的第二个经验观测一致：当某个诊断类别在测试分布中占据主导比例时，坍缩式预测可能在 Accuracy 上获得表面优势。因此，后续需要用 Balanced Accuracy、Macro F1、Collapse Ratio、Effective Classes 和 Pred-True TV 进一步解释 Accuracy 与完整诊断判别能力之间的差异。

| Dataset | Client-average cells | avg | LAMP-Merge | Classifier-head aggregation | Shuffled-label prototype |
|---|---:|---:|---:|---:|---:|
| bloodmnist_224 | 12 | 0.1714 | 0.8188 | 0.1849 | 0.2450 |
| chaoshengmnist_224 | 12 | 0.1564 | 0.4618 | 0.1598 | 0.1442 |
| dermamnist_224 | 12 | 0.5042 | 0.4619 | 0.5167 | 0.0976 |
| organcmnist_224 | 12 | 0.1286 | 0.6249 | 0.1674 | 0.1116 |
| organsmnist_224 | 12 | 0.1414 | 0.5744 | 0.1600 | 0.1217 |

现阶段可以形成两条受数据支持的结论。第一，M1 中的类别原型 \(p_c\) 是抑制融合后预测坍缩的必要结构；任何去除类别条件方向或破坏类别语义对应关系的替代项都会导致显著退化。第二，当前 full-scope 网格中，类别支持统计的不同实现形式与正式方法数值接近，说明主要判别收益来自 \(p_c\) 所提供的医学类别方向；统计信息的作用需要结合长尾压力点和非 Accuracy 指标进一步分析，而不能仅用总体 Accuracy 解释。

## 一、需要补充的模块内消融

LAMP-Merge 的客户端上传信息可以分为两类。第一类是类别原型信息，即每个客户端在共享参考骨干上计算得到的类别特征均值；它决定每个诊断类别的判别方向。第二类是类别统计信息，包括类别支持数和类别患病率计数；前者决定不同客户端在每个类别上的证据权重，后者决定是否需要进行有界长尾校准。严格的模块内消融应分别替换这两类信息，而不是只改变 \(s\) 或 \(\lambda\)。

原型信息消融应至少包含三个关键对照。第一，正式的 reference prototype，即客户端上传类别特征均值，服务器按支持数加权得到全局类别原型。第二，classifier-head aggregation，即不用参考特征原型，而是按类别支持数直接聚合客户端分类头中的对应类别权重；该对照检验“只使用本地训练后的类别头方向”是否足以替代原型。第三，shuffled-label prototype，即保留原型的范数和数量，但打乱原型与诊断类别标签的对应关系；该对照检验原型收益是否来自真实医学类别语义，而不是来自额外参数或归一化形式。进一步可以加入 global-feature mean 和 support-only synthetic head，用于证明类别条件特征方向不可由类别无关统计或支持数本身替代。

统计信息消融应区分支持数和患病率两种统计。对于支持数，正式方法使用 \(n_{i,c}\) 构造类别级客户端权重；对照项包括 uniform client weight、binary support only 和 global client-size weight。它们分别检验“客户端等权”“只知道类别是否出现”和“只知道客户端总规模”是否足以替代类别支持数。对于患病率，正式方法使用客户端上传的类别计数估计全局长尾先验；对照项包括 no prevalence calibration、uniform prevalence prior 和 smoothed prevalence prior。它们分别检验 M2 的收益是否确实来自真实医学长尾统计，而不是来自任意 bias 项。

## 二、需要补充的数值指标

每个替代项都必须报告 Accuracy、Balanced Accuracy、Macro F1、Collapse Ratio、Effective Classes 和 Pred-True TV。Accuracy 只能说明总体正确率，不能单独证明融合质量；Balanced Accuracy 和 Macro F1 反映少数类诊断能力；Collapse Ratio 和 Effective Classes 直接刻画预测是否坍缩到少数类别；Pred-True TV 衡量预测类别分布是否接近真实医学类别分布。

除预测指标外，还应补充两个原型几何指标。第一个是 Prototype Separation，即全局类别原型之间的平均余弦距离或最近邻类间距离，用于衡量类别方向是否清晰分离。第二个是 Prototype Consistency，即同一类别在不同客户端原型之间的平均相似度，用于衡量跨中心类别证据是否一致。若正式原型方法优于 classifier-head aggregation，则应表现为更高的类别分离度、更低的坍缩强度和更高的 balanced accuracy；若 shuffled-label prototype 显著退化，则说明原型必须与诊断类别语义对齐。

主消融结论必须采用正式汇总表中的 small 全量口径，而不能只采用 `clients=3 / beta=0.01` 的代表性诊断子集。全量口径包含 5 个医学图像数据集、4 个 backbone、3 个客户端数量和 3 个 Dirichlet beta 设置，因此每个消融设置对应 180 个 raw cells。论文主表中的 client-average 口径先对相同 `(dataset, backbone, K)` 的三个 beta 取均值，因此每个消融设置对应 60 个 client-average cells。后续报告必须同时给出 raw cell 和 client-average cell 的统计结果。预测分布图、collapse ratio、balanced accuracy、macro F1 和 Pred-True TV 也必须在同一 full-scope 网格上计算；若论文中展示单个数据集的图，它应来自该数据集内所有 backbone、K 和 beta 的聚合，而不是单一 beta。

当前 full-scope 任务已固化为独立队列。`scripts/run_lamp_merge_full_evidence_queue.sh` 会等待模块内全量消融完成，随后运行 `scripts/run_lamp_merge_prediction_diagnostics_full_parallel.sh` 与 `scripts/run_lamp_merge_hparam_full_parallel.sh`。因此，最终报告中的模块内消融表、诊断指标表、预测分布图和超参数曲线将来自同一 full-scope 实验口径。

## 三、需要补充的可视化

第一类可视化是预测类别分布热图。该图以方法为行、诊断类别为列，展示各方法在测试集上的预测比例。它应同时包含通用融合基线、正式 LAMP-Merge、原型替代项和统计替代项。该图的目标是证明：普通融合基线和错误替代项倾向于集中输出少数类别，而正式原型方法能够恢复多类别预测分布。

第二类可视化是指标柱状图。每个数据集至少绘制 Balanced Accuracy、Macro F1、Collapse Ratio 和 Effective Classes。柱状图应分成“原型信息消融”和“统计信息消融”两组，避免把不同问题混在同一张图里。主文可展示总体均值和两个代表数据集；附录展示全部数据集。

第三类可视化是原型几何图。应绘制全局类别原型之间的余弦相似度矩阵，或者展示每个类别与最近错误类别的距离。若原型信息起作用，正式方法应形成更清晰的类间结构；若使用本地分类头聚合或打乱标签，类间相似度应更混乱，且更容易对应预测坍缩。

第四类可视化是 t-SNE 降维图。t-SNE 应使用共享参考骨干提取的测试样本特征，并把全局类别原型作为额外点一起投影到二维空间。样本点按真实类别着色，原型点使用更大的 marker。若诊断原型有效，则每个原型点应靠近其对应类别的样本簇；classifier-head aggregation 或 shuffled prototype 若失败，应表现为类别方向偏离样本簇或与错误类别混合。建议至少在 Blood、Derma、Organ-C 上绘制 t-SNE，因为它们分别代表血细胞形态、强长尾皮肤病和器官切片结构。

## 四、理论分析补充

理论分析应围绕“为什么原型信息和统计信息能够缓解坍缩”展开，而不是只解释实现细节。符号必须与正文保持一致：共享参考特征写作 \(z=\phi_0(T(x))\)，客户端 \(i\) 上传的类别原型写作 \(\mu_{i,c}\)，服务端聚合得到的全局诊断原型写作 \(p_c\)，类别支持数写作 \(n_{i,c}\)，患病率先验写作 \(\pi_c\)，预测坍缩强度写作 \(\rho\)。若需要表示类别 \(c\) 的真实参考特征均值，仅在理论分析中额外引入 \(\mu_c^\star\)，且不替代正文中的 \(\mu_{i,c}\) 或 \(p_c\)。服务端聚合原型为：

$$
p_c=\sum_i\alpha_{i,c}\mu_{i,c}.
$$

若每个客户端的类别特征估计方差有界，则原型估计误差可写成如下形式：

$$
\mathbb{E}\|p_c-\mu_c^\star\|_2^2
\le
\sum_i\alpha_{i,c}^2\frac{\sigma_c^2}{n_{i,c}}
+\mathrm{Bias}_c^2.
$$

该上界说明类别支持数的作用不是经验性调参，而是降低类别原型估计方差。具有更多类别样本的客户端提供更稳定的原型估计；未观察到类别 \(c\) 的客户端不应参与该类别方向构造。普通参数平均没有这种类别级屏蔽机制，因此会把无证据客户端的参数方向混入类别 \(c\)，增加类别方向误差，并提高少数类被多数类吸收的风险。

进一步设类别 \(c\) 与类别 \(d\) 对样本 \(x\) 的真实 margin 为：

$$
\Delta_{c,d}(x)=(\mu_c^\star)^\top z-(\mu_d^\star)^\top z.
$$

当原型估计误差满足：

$$
\|p_c-\mu_c^\star\|_2+\|p_d-\mu_d^\star\|_2
<
\frac{\Delta_{c,d}(x)}{\|z\|_2},
$$

则使用估计原型后不会改变样本在类别 \(c\) 和类别 \(d\) 之间的判别顺序。因此，M1 通过降低类别原型误差来维持多类别 margin，从理论上抑制多数类方向吞并少数类方向。

M2 的理论作用是有界地引入医学长尾先验。设客户端上传的患病率计数 \(m_{i,c}\) 估计出的类别先验为 \(\pi_c\)，M2 对分类分数加入中心化 log-prior：

$$
b_c=\lambda\left(\log\pi_c-\frac{1}{C}\sum_k\log\pi_k\right).
$$

任意两个类别之间的先验 margin 改变量满足：

$$
|b_c-b_d|
\le
\lambda|\log\pi_c-\log\pi_d|.
$$

因此，只要 \(\lambda\) 有界，M2 就不会替代 M1 的多类别判别方向，而只是对真实长尾患病率进行有限校准。该理论解释与实验设计相对应：若关闭 M1，只保留 M2，模型无法恢复多类别判别；若保留 M1 并使用真实患病率统计，模型可以在避免坍缩的同时利用主导类别的真实医学先验。

## 五、预期写入论文的结论形式

补充实验完成后，论文中应形成如下结论链条。首先，原型替代消融证明 reference class prototype 是缓解预测坍缩的主要机制；本地分类头聚合、类别无关均值或打乱标签的原型不能稳定恢复多类别诊断输出。其次，统计信息消融证明类别支持数和患病率计数分别承担不同作用：支持数决定每个客户端对每个诊断类别的可靠性，患病率计数在强长尾医学数据中提供有界先验校准。最后，t-SNE 和原型几何可视化应直观显示，正式原型位于对应类别样本簇附近，并形成更清晰的类间结构；这解释了为什么 LAMP-Merge 能在 balanced accuracy、macro F1、collapse ratio 和 effective classes 上同时优于通用融合基线。

当前文档是补充实验设计与论文汇报框架，尚不声称这些全量消融已经完成。后续需要按照 `LAMP-Merge模块内补充消融工作流.md` 实现替代项、在 small 全量口径下运行实验、生成图像，并用真实数值替换本文档中的待验证结论。
