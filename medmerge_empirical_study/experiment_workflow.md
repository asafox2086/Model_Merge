# 医学客户端模型融合经验研究工作流

## 1. 研究目标

本研究不再围绕自定义融合方法展开，而是使用已经训练好的医学客户端模型和已有融合基线，系统评估一个问题：

> 在医学客户端模型融合中，仅依赖模型权重或公开同类型数据是否足以完成可靠融合？

核心假设是：

> 现有后训练模型融合方法在医学异质客户端上并不稳定；可靠融合往往需要与客户端源域分布匹配的图像统计。相同医学任务类型的公开数据集，即使类别相近，也不能稳定替代原始源域验证图像。

本文的贡献应定位为 empirical study，而不是新算法论文。

## 2. 实验资产

使用现有仓库中的训练好客户端和基线方法，不重新训练客户端。

### 2.1 客户端模型

从已有 `model_hub` 读取客户端 checkpoint。

覆盖维度：

- 数据集：BloodMNIST、DermaMNIST、OrganAMNIST、OrganCMNIST、OrganSMNIST、PathMNIST、ChaoshengMNIST
- 客户端数量：3、5、7
- 异质性设置：Dirichlet beta = 0、0.01、0.1
- backbone：ResNet、ConvNeXt、ViT、Swin 等已有模型
- VLM：已有 CLIP 设置，如果结果稳定可放入补充实验

### 2.2 融合方法

分为两类：

**权重/几何型方法**

这些方法主要依赖 checkpoint、reference model 或 task vector，不直接依赖源域图像统计。

- Avg
- TIES
- DARE-Linear
- DARE-TIES
- Breadcrumbs
- Model Stock
- FROM
- ISO-C
- FreeMerge
- RobustMerge

**图像统计型方法**

这些方法需要使用验证图像、无标签图像或 activation statistics。

- Fisher
- RegMean
- AdaMerging
- 融合后 BN recalibration

## 3. 总体实验流程

整体分为三组主实验。

1. 大规模基线测试：证明已有融合方法在医学客户端上不稳定。
2. 客户端能力分析：证明问题不是客户端本身太弱，而是权重融合破坏或无法恢复知识。
3. 源域图像统计替代实验：证明同类型公开医学数据不能稳定替代原始源域验证图像。

## 4. 实验一：已有融合方法的大规模基线测试

### 4.1 目的

证明现有后训练模型融合方法在医学异质客户端场景下缺乏稳定性。

### 4.2 设置

对每个实验 setting：

```text
dataset x backbone x num_clients x beta x seed
```

运行所有已有融合方法，并在固定测试集上评估。

### 4.3 指标

不能只看 accuracy。需要记录：

- Accuracy
- Balanced accuracy
- Macro F1
- Per-class recall
- Test loss / NLL
- Prediction entropy
- 预测类别分布
- Collapse rate
- Average rank
- Win count
- Delta vs Avg

### 4.4 Collapse rate 定义

建议至少使用两个 collapse 指标：

```text
majority_prediction_ratio = max_k count(predicted_class = k) / total_samples
```

如果 `majority_prediction_ratio` 很高，说明模型几乎只预测一个类别。

另一个指标：

```text
effective_predicted_classes = exp(entropy(predicted_class_distribution))
```

如果有效预测类别数远小于真实类别数，说明模型发生预测塌缩。

### 4.5 期望结果

期望观察到：

- 没有一个融合方法在所有数据集和模型结构上稳定领先。
- 某些方法在局部 setting 有效，但跨数据集、backbone、client 数后不稳定。
- 普通 accuracy 会掩盖类别塌缩，尤其在类别不均衡数据集上。
- Balanced accuracy、Macro F1、per-class recall 会暴露塌缩问题。
- 客户端数量越多、数据异质性越强，融合结果越不稳定。

### 4.6 该实验要支撑的结论

> 现有权重空间融合方法直接用于医学客户端模型时，并不可靠。医学模型融合不能只用 accuracy 或单一 setting 判断有效性。

## 5. 实验二：客户端能力与融合损失分析

### 5.1 目的

证明融合失败不是因为所有客户端都训练得很差，而是因为权重融合过程无法可靠保留或恢复客户端中的有效知识。

### 5.2 对照对象

对每个 setting，额外评估：

- 每个 single client
- Best single client
- Mean single client
- Prediction ensemble
- Oracle per-sample client

### 5.3 Prediction ensemble

不合并权重，只平均各客户端预测：

```text
prob_ensemble(x) = mean_i softmax(f_i(x))
```

或：

```text
logit_ensemble(x) = mean_i f_i(x)
```

这个 baseline 用来判断客户端预测层面是否互补。

### 5.4 Oracle per-sample client

不可部署，只作为上界：

```text
oracle(x) = 选择在该样本上预测正确或 loss 最低的客户端
```

如果 oracle 明显高于融合模型，说明客户端池里存在可利用信息。

### 5.5 关键比较

对每个 setting 计算：

```text
merged_model - best_single_client
merged_model - prediction_ensemble
merged_model - oracle_client
```

### 5.6 期望结果

期望观察到：

- 很多 setting 中，融合模型低于 best single client。
- Prediction ensemble 通常比权重融合更稳。
- Oracle per-sample client 明显高于普通融合模型。
- 说明客户端池里有可用知识，但权重空间融合不能稳定恢复。

### 5.7 该实验要支撑的结论

> 融合失败不是单纯来自本地训练失败，而是来自后训练参数融合在医学异质场景下的不稳定性。

## 6. 实验三：同类型公开数据能否替代源域验证集

### 6.1 目的

这是论文最关键的实验。

目标不是证明错误数据集会导致差结果，而是证明：

> 即使公开数据集属于相同医学任务类型，只要不是同源分布，也不能稳定替代原始源域验证图像。

因此本实验不把“错误数据集”作为主对照。

### 6.2 参与方法

只选择需要图像统计或校准图像的方法：

- Fisher
- RegMean
- AdaMerging
- BN recalibration after merge

### 6.3 校准数据来源

固定 client checkpoint、merge 方法、测试集，只替换统计来源。

主要比较：

| 校准数据来源 | 作用 |
|---|---|
| 源域验证集 | 正样本，代表原始源域图像统计 |
| 同类型公开医学数据 | 关键对照，模拟无法访问原始图像、只能使用公开同类数据 |
| 不使用图像统计 | 下界 baseline |

可选增强对照：

| 校准数据来源 | 作用 |
|---|---|
| 公开数据按源域类别比例重采样 | 排除类别比例差异的解释 |
| 公开数据预处理完全对齐 | 排除 resize、归一化、通道数等工程差异 |
| 公开数据数量增加 | 检验更多 non-source public data 是否能弥补分布差异 |

### 6.4 不建议作为主实验的对照

不建议把错误医学数据集作为主结果，例如：

```text
用 Blood 图像校准 Derma 模型
```

这类结果太容易被认为是 trivial negative control。它最多放 appendix，不能作为证明原图必要性的主证据。

### 6.5 公平性控制

为了让结果有说服力，需要控制：

- 校准样本数量一致。
- 类别映射一致，不能映射的公开类别不要硬塞。
- 类别比例尽量一致，至少报告 public-original 和 public-reweighted 两个版本。
- 图像预处理一致，包括 resize、crop、normalization、灰度/RGB 处理。
- 测试集固定不变。
- checkpoint 固定不变。
- 方法超参数固定不变。

### 6.6 期望结果

最理想结果：

```text
源域验证集校准 > 同类型公开医学数据校准 ≈ 不使用图像统计
```

较温和但仍可接受的结果：

```text
源域验证集校准 > 同类型公开医学数据校准 > 不使用图像统计
```

关键不是公开数据完全没用，而是：

```text
公开同类数据不能稳定达到源域验证集的效果。
```

### 6.7 该实验要支撑的结论

> 医学模型融合需要的是源域分布匹配的图像统计，而不是任意同类型公开医学图像。公开数据集即使任务类型相同，也无法稳定替代原始源域图像。

## 7. BN Recalibration 子实验

### 7.1 目的

证明融合后的模型有时不是权重完全无效，而是缺少正确的源域统计状态。

### 7.2 流程

1. 用已有方法完成权重融合。
2. 冻结所有可学习参数。
3. 将模型设置为 train 模式，仅更新 BatchNorm running mean / variance。
4. 分别使用不同校准数据：
   - 源域验证集
   - 同类型公开医学数据
   - 不校准
5. 在同一测试集上评估。

### 7.3 期望结果

期望观察到：

- 源域验证图像可以显著恢复部分 CNN 融合模型性能。
- 同类型公开数据效果弱于源域验证集。
- 不校准时更容易出现类别塌缩或不稳定预测。

### 7.4 该实验要支撑的结论

> 融合后的医学模型不仅需要合并权重，还需要恢复与源域匹配的模型状态统计。

### 7.5 公开验证集选择客户端

#### 目的

检验一个非常强但简单的替代策略：

> 不做模型融合，而是用同类型公开验证集给每个已训练客户端打分，直接采用 public validation 上表现最好的客户端。

#### 流程

对每个 `dataset / backbone / num_clients / beta / seed`：

1. 固定所有已训练客户端 checkpoint。
2. 在 `PublicMedLabeled_data` 的 validation split 上评估每个客户端。
3. 选择 public validation accuracy 最高的客户端。
4. 在源域 test set 上评估该客户端。
5. 与以下选择规则对比：
   - `source_val_accuracy_best`：用源域 validation set 选客户端。
   - `source_val_balanced_accuracy_best`：用源域 balanced accuracy 选客户端。
   - `oracle_test_accuracy_best`：用源域 test set 选客户端，只作为不可用上界。

#### 关键指标

- `source-test accuracy / balanced accuracy / macro F1`：最终落到源域测试集的效果。
- `selection regret`：

```text
regret = source-test acc(source-val selected client)
       - source-test acc(public-val selected client)
```

- `Spearman(public-val score, source-test score)`：public 排名是否能代表源域测试排名。

#### 期望结果

如果 public validation 可以替代源域图像，那么 public-selected client 应接近 source-val selected client，且 public score 与 source-test score 排名相关性较高。

如果 public-selected client 明显差于 source-val selected client，则说明：

> 即使公开数据类别覆盖完整、输入格式一致，它仍不能稳定承担源域模型选择的角色。

### 7.6 公开数据集作为 Test Set 的汇总表2

#### 目的

检验一个更直接的问题：

> 如果训练好的客户端和融合过程仍然来自源域，但最终测试集换成同类型公开医学数据，模型融合方法的 accuracy 和排名是否会大幅改变？

这不是用错误数据集做测试，而是把同类型公开医学数据当作外部 test set，观察模型是否只在源域 test 上表现稳定。

#### 流程

对 `bloodmnist_224 / dermamnist_224 + resnet + 3 clients + beta in {0, 0.01, 0.1} + seed 42`：

1. 使用 `Med_data` 作为融合/统计数据来源。
2. 使用 `PublicMedFingerprint_data/test` 作为公开测试集。
3. 运行 12 个已有融合基线：
   - Avg, TIES, DARE-Linear, DARE-TIES
   - RegMean, Fisher
   - Breadcrumbs, Model Stock, FROM, ISO-C, FreeMerge, RobustMerge
4. 记录 accuracy、balanced accuracy、macro F1、预测类别塌缩指标。
5. 与源域 formal accuracy 表中同 setting 的结果比较：
   - `source mean acc` vs `public mean acc`
   - `public - source`
   - 每个 setting 的 top method 是否切换
   - rank Spearman

注意：公开 test 图像不应参与融合统计。若为了检查数据依赖方法额外跑 `PublicMedFingerprint_data` 作为统计来源的版本，必须标注为辅助检查，不作为主结论。

#### 期望结果

如果医学模型融合只学到了源域局部解，换成同类型公开 test 后应观察到：

- 整体 accuracy 明显下降。
- 最优方法从源域 test 到 public test 大量切换。
- rank Spearman 较低，说明方法排名不稳定。
- balanced accuracy / macro F1 暴露出 accuracy 被类别分布影响的问题。

该实验支撑的结论是：

> 医学模型融合的有效性高度依赖源域分布；即使公开数据类别覆盖相同，作为外部 test set 时也会暴露出融合方法的鲁棒性不足。

## 8. 主表和图设计

### 8.1 主表一：整体 benchmark

每个方法报告：

- Mean accuracy
- Mean balanced accuracy
- Mean macro F1
- Average rank
- Win count
- Collapse rate
- Low-performance rate

重点不要只按 accuracy 排名。

### 8.2 主图一：方法稳定性

x 轴：

```text
dataset / backbone / beta / num_clients
```

y 轴：

```text
balanced accuracy 或 average rank
```

展示没有方法稳定领先。

### 8.3 主图二：客户端能力差距

画以下 gap：

```text
merged - best_single_client
merged - prediction_ensemble
merged - oracle_client
```

如果大多数 gap 为负，说明权重融合损失明显。

### 8.4 主图三：公开数据替代实验

对 Fisher、RegMean、AdaMerging、BN recalibration 画柱状图：

```text
source validation
same-type public data
same-type public data reweighted
no image statistics
```

y 轴使用 balanced accuracy、macro F1、collapse rate。

### 8.5 主图四：类别塌缩诊断

展示某些 accuracy 看似不低的 setting 中：

- 预测类别直方图
- per-class recall
- confusion matrix

用来说明 accuracy 在不均衡医学数据上会误导。

### 8.6 机制补充图：医学坍缩的特异点

在发现 AVG 或其他融合方法出现单类预测后，额外记录以下机制指标：

- 每层 feature effective rank：`bn1 / layer1 / layer2 / layer3 / layer4 / classifier_input`
- dominant class pairwise margin：

```text
logit_dominant - logit_competitor
```

- margin 的 `mean / std / p05 / min / pct_positive`
- logit 分解：

```text
logit_d - logit_c = (w_d - w_c)^T h + (b_d - b_c)
```

- classifier、BN running stats、layer4 的 client-to-avg parameter dispersion
- client task-vector cosine：

```text
delta_i = client_i - reference/init
cosine(delta_i, delta_j)
```

- feature/head 互换：

```text
client_i feature + client_j head
avg feature + client_j head
avg feature + avg head
```

- frozen AVG feature 的 linear probe：

```text
冻结 avg feature extractor，只重新训练 linear head
```

当前 DermaMNIST 与 CIFAR-10 3-client 对照的初步结论是：

> 医学坍缩的直接机制不是 classifier bias，而是融合后最终特征在 dominant-vs-competitor 方向上的投影低尾部全部为正。自然图像也有 dominant-class drift，但 margin 仍有负尾部，所以不会完全单类坍缩。

后续 E13 诊断进一步修正：

> 医学 AVG 不是把 feature 完全毁掉；linear probe 能恢复 Derma BAcc `0.142857 -> 0.573081`。更准确的失败点是 feature extractor、BN running statistics、classifier head 的配套关系被 AVG 打乱。Derma 的 `layer4` task-vector cosine 只有 `0.030976`，而 CIFAR 是 `0.960982`，说明医学高层更新方向几乎没有共同方向。

该机制图不替代主 benchmark，而是解释为什么 accuracy 和 pred_counts 会变成单类预测。

### 8.7 严格控制变量：医学图像 vs 自然图像

为回应“唯一的区别只能是医学图像和自然图像”的控制变量要求，新增一个严格对照实验。

#### 设置

医学端固定使用：

```text
model_hub/small/dermamnist_224/resnet/clients_3/beta_0/seed_42
```

自然图像端构造 `cifar7_derma_strict`，强制匹配 DermaMNIST 的：

- 7 个类别。
- train/val/test 每类样本数。
- 3 个客户端。
- 每个客户端拥有的类别集合。
- 每个客户端内每个类别的样本数。
- ResNet、224 输入、pretrained、50 epochs、batch size 64、lr 0.001、weight decay 0.0001。
- 权重 AVG 融合。

具体分布：

| split/client | counts |
| --- | --- |
| train classes 0-6 | `[228, 359, 769, 80, 779, 4693, 99]` |
| val classes 0-6 | `[33, 52, 110, 12, 111, 671, 14]` |
| test classes 0-6 | `[66, 103, 220, 23, 223, 1341, 29]` |
| client 0 | `3:80, 4:779, 0:228` |
| client 1 | `2:769, 1:359` |
| client 2 | `6:99, 5:4693` |

技术限制：CIFAR class 5 因唯一图片数量不足，使用了有放回采样；论文中要披露。

#### 结果

| domain | accuracy | balanced accuracy | collapse ratio | effective predicted classes | pred counts |
| --- | ---: | ---: | ---: | ---: | --- |
| DermaMNIST | 0.668828 | 0.142857 | 1.000000 | 1 | `[0, 0, 0, 0, 0, 2005, 0]` |
| CIFAR strict | 0.699252 | 0.191548 | 0.906733 | 2 | `[0, 0, 0, 0, 187, 1818, 0]` |

严格控制后的解释要写成：

```text
同样的 partial-label + long-tail 分布会让自然图像也出现强 dominant-class drift；
但医学图像在相同条件下坍缩更彻底，直接退化为单类预测。
```

不要再写成：

```text
自然图像不会坍缩，医学图像才会坍缩。
```

#### 机制

Derma AVG 的 class 5 对所有竞争类的 margin 在所有测试样本上都为正：

| compare | p05 delta | min delta | pct positive |
| --- | ---: | ---: | ---: |
| 5-0 | 0.635670 | 0.359340 | 1.000000 |
| 5-1 | 0.411551 | 0.123354 | 1.000000 |
| 5-2 | 0.268922 | 0.133353 | 1.000000 |
| 5-3 | 0.381957 | 0.080625 | 1.000000 |
| 5-4 | 0.463657 | 0.071431 | 1.000000 |
| 5-6 | 0.688538 | 0.283340 | 1.000000 |

CIFAR strict 的关键竞争类 `5-4` 仍有负尾部：

| compare | p05 delta | min delta | pct positive |
| --- | ---: | ---: | ---: |
| 5-4 | -0.061839 | -0.344914 | 0.906733 |

因此 CIFAR strict 仍能把 187 个样本判成 class 4，而 Derma 全部判成 class 5。

高层 task-vector cosine：

| group | Derma | CIFAR strict |
| --- | ---: | ---: |
| BN running stats | 0.607558 | 0.932364 |
| layer3 | 0.030653 | 0.958356 |
| layer4 | 0.030976 | 0.960622 |

这支持以下机制表述：

```text
非IID和样本不均衡提供坍缩压力；
医学客户端的高层更新方向更不兼容，AVG 后 feature extractor、BN running statistics 和 classifier head 的配套关系被打乱；
最终 AVG head 把剩余特征证据全部读成 class 5。
```

linear probe 结果要同时报告，避免过度声称“医学特征被完全破坏”：

| domain | original AVG BAcc | frozen AVG feature + linear head BAcc |
| --- | ---: | ---: |
| Derma | 0.142857 | 0.551806 |
| CIFAR strict | 0.191548 | 0.774773 |

详细报告：

```text
medmerge_empirical_study/results/experiment14_strict_domain_control/strict_domain_control_report_cn.md
```

## 9. 论文叙事结构

### 9.1 引言

医学模型融合常被视为 checkpoint-level 或 weight-space 问题，但医学客户端之间存在强烈分布差异，包括设备、采集协议、病人群体、染色方式、组织区域和类别比例差异。

因此，问题不只是如何合并权重，而是合并后的模型是否拥有正确的源域统计。

### 9.2 主要发现

建议总结为四点：

1. 现有后训练融合方法在医学客户端上不稳定，没有方法跨 setting 稳定领先。
2. 很多高 accuracy 结果其实来自多数类预测或类别塌缩。
3. 客户端池中仍有有用知识，但权重融合无法稳定恢复。
4. 同类型公开医学数据不能稳定替代源域验证图像，说明可靠融合依赖源域分布匹配的图像统计。

### 9.3 结论表述

推荐使用稳健表述：

> Reliable medical model merging requires source-distribution-matched image statistics. Same-task public medical datasets are often insufficient substitutes for the original source-domain validation images.

中文对应：

> 可靠医学模型融合需要与源域分布匹配的图像统计。相同任务类型的公开医学数据，通常不能充分替代原始源域验证图像。

避免过强表述：

```text
所有医学模型融合都必须使用原始训练图像。
```

更合理的表述是：

```text
原始图像或等价的源域统计对于可靠融合非常关键。
```

## 10. 最小可执行版本

如果时间有限，优先完成以下实验：

1. 从正式结果中补充 balanced accuracy、macro F1、collapse rate。
2. 评估 single client、best single client、prediction ensemble。
3. 对 Fisher、RegMean、BN recalibration 做三种校准来源比较：
   - 源域验证集
   - 同类型公开医学数据
   - 不使用图像统计

这三个实验已经足够支撑论文主线。

## 11. 最终期望结论

最终论文应得到如下结论：

> 医学客户端模型融合的关键瓶颈不是缺少更复杂的权重合并技巧，而是现有后训练融合方法在缺少源域图像统计时，无法可靠处理医学客户端之间的分布冲突。即使使用相同类型的公开医学数据，也不能稳定替代原始源域验证图像。
