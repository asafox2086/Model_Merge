# 医学与自然图像融合坍缩的对齐诊断结果

本实验用四个诊断检查一个观点：

> 医学客户端之间学到的高层判别规则更不一致，AVG 把这些规则硬平均后，分类头读不懂融合后的特征，最后被 dominant class 接管；自然图像客户端虽然也是 non-IID，但高层更新更有共同方向，所以融合后还保留部分共同视觉结构。

实验对象：

```text
Medical: model_hub/small/dermamnist_224/resnet/clients_3/beta_0/seed_42
Natural: medmerge_empirical_study/results/experiment12_natural_image_3client_imbalance_probe/natural_probe_hub/small/cifar10_32/resnet/clients_3/partial_label/seed_42
```

结果目录：

```text
medmerge_empirical_study/results/experiment13_alignment_diagnostics
```

汇总图：

```text
medmerge_empirical_study/results/experiment13_alignment_diagnostics/figures/alignment_diagnostics_summary.png
```

## 结论先写清楚

这组实验把之前的说法修正成更准确的版本：

> 医学 AVG 不是把图像特征完全毁掉。融合后的特征里仍然有类别信息，因为重新训练一个 linear head 后 balanced accuracy 可以从 `0.142857` 恢复到 `0.573081`。真正的问题是：医学客户端的高层参数更新方向几乎不一致，AVG 得到的 `feature extractor + BN stats + classifier head` 不再配套；原来的 `W_avg` 不能正确读取 `h_avg`，于是把所有样本稳定读成 class 5。

自然图像也被 AVG 弄坏了，但它的高层更新方向更一致，尤其 `layer4` 和 BN running statistics，所以平均后还有共同结构。结果是偏向 class 8，但没有完全单类坍缩。

一句话：

> 医学的高层判别规则变得彼此不一致了；高层规则本来是用来把图像变成可分类证据的；由于 AVG 把这些不一致规则硬平均，融合后的分类头读错了仍然存在的图像证据，导致所有图都被读成 class 5。

## 1. CKA：没有证明“医学 activation 更不对齐”

CKA 用同一批 test images 比较不同 client 在同一层的 activation 相似度。

| domain | bn1 | layer1 | layer2 | layer3 | layer4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Derma medical | 0.994642 | 0.842579 | 0.800724 | 0.624328 | 0.316591 |
| CIFAR natural | 0.992404 | 0.837278 | 0.233668 | 0.364615 | 0.239942 |

这个结果不能支持“医学 client activation 更不对齐”。相反，Derma 的 layer2/layer3/layer4 CKA 还高于 CIFAR。

因此不能把论文故事写成：

```text
医学 activation CKA 更低，所以医学坍缩。
```

更合理的解释是：activation 相似度不是决定坍缩的充分指标。真正拉开差异的是参数更新方向、BN 统计一致性、以及 AVG head 是否能读懂 AVG feature。

## 2. Task Vector Cosine：强烈支持“医学高层更新方向不一致”

task vector 定义为：

```text
delta_i = client_i 参数 - reference/init 参数
```

然后计算不同 client 的 `delta_i` cosine similarity。cosine 越高，说明不同 client 从共同初始模型出发，朝更相似的方向更新；cosine 接近 0，说明更新方向几乎正交。

| group | Derma medical cosine | CIFAR natural cosine |
| --- | ---: | ---: |
| classifier | 0.107055 | 0.219918 |
| BN running stats | 0.607558 | 0.985049 |
| layer3 | 0.030653 | 0.599215 |
| layer4 | 0.030976 | 0.960982 |

这是最支持“医学特异点”的表。

解释：

- `layer3/layer4` 本来负责形成高层语义/类别判别特征。
- CIFAR 的 `layer4 cosine = 0.960982`，说明不同自然图像 client 虽然类别不同、样本数不均衡，但高层卷积参数仍朝非常相似的方向更新。
- Derma 的 `layer4 cosine = 0.030976`，几乎是正交，说明不同医学 client 的高层判别规则没有共同方向。
- BN running stats 本来用于保存特征分布统计。CIFAR 是 `0.985049`，Derma 是 `0.607558`，说明医学 client 的中间特征统计更不一致。

所以这里可以这样说：

> 医学的高层参数更新方向变得几乎互不相干了。高层参数本来用来提取类别证据；由于每个医学 client 只在自己的局部病灶类别上训练，它们把高层参数推向不同方向。AVG 把这些方向硬平均后，得到的高层表示和分类头不再配套。

## 3. Head-Feature 互换：医学 AVG feature 被本地 head 读成单类

模型可以拆成：

```text
x -> feature h -> head W -> logits
```

这个实验把 feature source 和 head source 交叉组合。

关键看 `avg feature + different heads`：

| domain | feature | head | collapse | effective classes | top class | pred counts |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Derma | avg | client_0 | 0.943641 | 3 | 4 | [6, 0, 0, 107, 1892, 0, 0] |
| Derma | avg | client_1 | 0.993516 | 2 | 2 | [0, 13, 1992, 0, 0, 0, 0] |
| Derma | avg | client_2 | 1.000000 | 1 | 5 | [0, 0, 0, 0, 0, 2005, 0] |
| Derma | avg | avg | 1.000000 | 1 | 5 | [0, 0, 0, 0, 0, 2005, 0] |
| CIFAR | avg | client_0 | 0.438600 | 3 | 1 | [2385, 4386, 3229, 0, 0, 0, 0, 0, 0, 0] |
| CIFAR | avg | client_1 | 0.485500 | 3 | 3 | [0, 0, 0, 4855, 3039, 2106, 0, 0, 0, 0] |
| CIFAR | avg | client_2 | 0.998800 | 2 | 8 | [0, 0, 0, 0, 0, 0, 0, 0, 9988, 12] |
| CIFAR | avg | avg | 0.781000 | 8 | 8 | [254, 421, 329, 593, 210, 235, 0, 0, 7810, 148] |

解释：

- Derma 的 `avg feature` 被任何 head 读取时，都几乎变成该 head 的局部默认类。特别是 `avg head` 直接全 class 5。
- CIFAR 的 `avg feature` 被非 dominant 的 client_0/client_1 head 读取时，还能保留各自 3 个局部类别，不是单类。
- CIFAR dominant head `client_2` 本来就因为 `8=4000, 9=80` 极度不均衡，所以它读成 class 8 不奇怪。

所以更准确的结论是：

> 医学 AVG feature 不是没有任何信息，而是对原有本地 head 来说更像一个被推到局部默认类附近的表示；AVG head 进一步把它读成 class 5。

## 4. Seen-Class 子表：只看每个 head 自己见过的类别

为了避免“client 没见过其他类”的干扰，又只在每个 head 自己见过的类别上评价。

| domain | head | own feature local BAcc | avg feature local BAcc | avg feature collapse | avg feature key recalls |
| --- | --- | ---: | ---: | ---: | --- |
| Derma | client_0 [3,4,0] | 0.371565 | 0.408332 | 0.939103 | class 4 recall 0.964, class 0 recall 0 |
| Derma | client_1 [2,1] | 0.560437 | 0.533981 | 0.978328 | class 2 recall 1.000, class 1 recall 0.068 |
| Derma | client_2 [6,5] | 0.735028 | 0.500000 | 1.000000 | class 5 recall 1.000, class 6 recall 0 |
| CIFAR | client_0 [0,1,2] | 0.767333 | 0.639667 | 0.424667 | 0/1/2 recalls all nonzero |
| CIFAR | client_1 [3,4,5] | 0.591000 | 0.481333 | 0.523000 | 3/4/5 recalls all nonzero |
| CIFAR | client_2 [8,9] | 0.577500 | 0.499000 | 0.999000 | class 9 recall 0 |

这张表说明：

- Derma 的 avg feature 到 dominant client_2 head 后，`class 6` 完全消失，只剩 class 5。
- Derma 的 client_1 head 到 avg feature 后，也几乎只剩 class 2。
- CIFAR 的非 dominant client_0/client_1 head 读 avg feature 时，仍能区分自己的多个局部类别。

这支持“自然图像平均后还保留局部可读结构，医学更容易退化到局部默认类”。

## 5. Linear Probe：医学 feature 没彻底坏，是原 AVG head 读错了

这个实验冻结 AVG feature extractor，只重新训练一个线性分类头。

| domain | model | BAcc | Macro F1 | collapse | effective classes | pred counts |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Derma | AVG original head | 0.142857 | 0.114508 | 1.000000 | 1 | [0, 0, 0, 0, 0, 2005, 0] |
| Derma | AVG feature + retrained linear head | 0.573081 | 0.560969 | 0.667830 | 7 | [52, 142, 207, 35, 205, 1339, 25] |
| CIFAR | AVG original head | 0.110500 | 0.070777 | 0.781000 | 8 | [254, 421, 329, 593, 210, 235, 0, 0, 7810, 148] |
| CIFAR | AVG feature + retrained linear head | 0.431400 | 0.427458 | 0.120200 | 10 | [1080, 1029, 856, 748, 1001, 1139, 1202, 1005, 934, 1006] |

这是很重要的修正：

> 如果 h_avg 真的完全废掉，重新训练 linear head 也救不回来。但 Derma 的 BAcc 从 `0.142857` 到 `0.573081`，说明 h_avg 里仍有可用类别信息。

因此不能说：

```text
AVG 把医学 feature 完全破坏了。
```

应该说：

```text
AVG 破坏了 feature extractor、BN 统计和 classifier head 的配套关系。特征里还有信息，但原来的 averaged head 读不出来，并且稳定读成 class 5。
```

## 最终可写进论文的版本

建议这样表述：

> 在医学 partial-label non-IID 场景中，不同客户端的高层参数更新方向几乎正交，BN running statistics 也更不一致。简单权重平均把这些局部判别规则硬合在一起后，得到的 AVG head 不能正确解释 AVG feature。虽然 AVG feature 中仍保留可被重新线性读出的类别信息，但原始 averaged head 会把所有样本映射到 dominant class 一侧，形成单类坍缩。自然图像客户端的高层更新方向和 BN 统计明显更一致，因此平均后仍保留部分共享视觉结构，表现为 dominant-class drift 而不是完全坍缩。

按用户要求的因果句式：

> 医学模型的高层更新方向变得彼此不一致了；这些高层参数本来是用来提取类别判别特征的；由于 AVG 把不一致的局部规则硬平均，融合后的特征和分类头不再配套；分类头本来是用来读取图像类别证据的，但现在它把剩余的图像信息稳定读成 class 5，导致所有图像都预测为 class 5。

## 需要注意的边界

这组实验没有证明“医学 activation CKA 一定比自然图像更低”。相反，本 case 里 Derma 的 high-layer CKA 高于 CIFAR。

所以论文里不要把 CKA 当主证据。主证据应该是：

1. task-vector cosine：医学 layer3/layer4 更新方向接近 0，自然图像很高。
2. BN running stats：医学一致性明显低于自然图像。
3. head-feature swap：医学 avg feature 被各个 head 读成局部默认类。
4. linear probe：AVG feature 仍可恢复，说明失败点主要是 averaged head/readout 与 feature 不配套。
