# 医学图像融合坍缩的特异点分析

本轮目标是回答一个更具体的问题：

> 为什么同样是 partial-label non-IID 加样本不均衡，医学图像 AVG 融合会单类坍缩，而自然图像 CIFAR-10 主要表现为 dominant-class drift，却没有完全单类坍缩？

结论先写在前面：

> 当前 DermaMNIST case 的医学特异点不在输入图像阶段，也不是 classifier bias 直接把模型推到 class 5；关键发生在融合后最终特征到 classifier 的投影上。医学 AVG 的 class 5 对所有竞争类的 logit margin 在每一个 test 样本上都为正，说明图像相关变化已经不足以翻过 class 5 的全局优势。自然图像也被 class 8 拉偏，但仍有负尾部 margin，所以一部分样本还能被其他类赢回去。

## 实验设置

医学 case：

```text
model_hub/small/dermamnist_224/resnet/clients_3/beta_0/seed_42
```

自然图像对照：

```text
medmerge_empirical_study/results/experiment12_natural_image_3client_imbalance_probe/natural_probe_hub/small/cifar10_32/resnet/clients_3/partial_label/seed_42
```

CIFAR-10 3-client 设置用于控制 client 数和 AVG 权重：

```text
client_0: classes 0,1,2, each 300 samples
client_1: classes 3,4,5, each 300 samples
client_2: classes 8,9, class 8 = 4000, class 9 = 80
```

机制诊断输出目录：

```text
medmerge_empirical_study/results/experiment12_natural_image_3client_imbalance_probe/domain_comparison
```

关键图：

```text
medmerge_empirical_study/results/experiment12_natural_image_3client_imbalance_probe/domain_comparison/figures/medical_vs_natural_mechanism_rank_margin.png
```

## 1. 最终输出：医学是单类坍缩，自然图像是强漂移

| domain | avg accuracy | balanced accuracy | collapse ratio | effective pred classes | top class | pred counts |
| --- | --- | --- | --- | --- | --- | --- |
| DermaMNIST medical | 0.668828 | 0.142857 | 1.000000 | 1 | 5 | [0, 0, 0, 0, 0, 2005, 0] |
| CIFAR-10 natural 3-client | 0.110500 | 0.110500 | 0.781000 | 8 | 8 | [254, 421, 329, 593, 210, 235, 0, 0, 7810, 148] |

这说明自然图像在 3-client、dominant client 权重同为 1/3 时也会被大类 client 拉偏，但它仍然不是医学那种单类分类器。

## 2. 差异从 feature extractor 中后段开始扩大

| domain | bn1 rank | layer1 rank | layer2 rank | layer3 rank | layer4 rank |
| --- | --- | --- | --- | --- | --- |
| DermaMNIST medical | 2.344272 | 6.550472 | 11.029681 | 21.386566 | 31.610453 |
| CIFAR-10 natural 3-client | 2.446746 | 21.313923 | 35.495064 | 109.482483 | 237.550751 |

早期 `bn1` 的 effective rank 很接近，说明问题不是一开始就完全不同。差异从 `layer1` 开始拉开，到 `layer4/classifier_input` 时自然图像保留了更高维的样本级变化，而医学 AVG 的最终特征维度有效秩只有 31.61。

这不是说 effective rank 单独导致坍缩，而是说明医学 AVG 的图像相关变化被压到更窄的表示空间里。后面只要 classifier 方向上出现一个全局 dominant offset，就更容易覆盖所有样本。

## 3. 决定性证据：dominant class margin 的低尾部

| domain | compare | mean delta | std delta | p05 delta | min delta | pct positive |
| --- | --- | --- | --- | --- | --- | --- |
| DermaMNIST | 5-2 | 0.570603 | 0.203555 | 0.268922 | 0.133353 | 1.000000 |
| DermaMNIST | 5-4 | 0.727216 | 0.171161 | 0.463657 | 0.071431 | 1.000000 |
| DermaMNIST | 5-6 | 1.869700 | 0.723917 | 0.688538 | 0.283340 | 1.000000 |
| CIFAR-10 | 8-1 | 0.915266 | 0.711141 | -0.049698 | -1.550948 | 0.937600 |
| CIFAR-10 | 8-3 | 0.780737 | 0.696047 | -0.119339 | -3.122090 | 0.917300 |
| CIFAR-10 | 8-9 | 1.535806 | 1.024280 | 0.169226 | -1.132782 | 0.974100 |

这个表是目前最关键的证据。

Derma 的 class 5 对近邻竞争类 2、4、6 的 margin，不只是均值为正，而是 `min_delta` 也全部大于 0。因此每张 test 图像都会被 class 5 赢掉。

CIFAR 的 class 8 也有明显均值优势，但 8-1、8-3、8-9 都存在负 margin 样本，说明图像内容仍然可以让其他类在部分样本上翻盘。

## 4. Logit 分解：不是 classifier bias，而是 feature projection

对最终 logit 做分解：

```text
logit_d - logit_c = (w_d - w_c)^T h + (b_d - b_c)
```

| domain | compare | projection mean | projection p05 | projection min | bias delta | total p05 | total min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DermaMNIST | 5-2 | 0.566755 | 0.265074 | 0.129505 | 0.003848 | 0.268922 | 0.133353 |
| DermaMNIST | 5-4 | 0.761412 | 0.497853 | 0.105627 | -0.034195 | 0.463657 | 0.071431 |
| DermaMNIST | 5-6 | 1.867584 | 0.686422 | 0.281224 | 0.002116 | 0.688538 | 0.283340 |
| CIFAR-10 | 8-1 | 0.979989 | 0.015025 | -1.486224 | -0.064724 | -0.049698 | -1.550948 |
| CIFAR-10 | 8-3 | 0.826021 | -0.074054 | -3.076805 | -0.045285 | -0.119339 | -3.122090 |
| CIFAR-10 | 8-9 | 1.531149 | 0.164569 | -1.137439 | 0.004657 | 0.169226 | -1.132782 |

医学的 `bias_delta` 很小，甚至 5-4 是负的，所以不能解释 class 5 为什么全赢。真正的差异是 projection 项：

- 医学：`(w_5 - w_c)^T h` 的 5% 分位和最小值都为正。
- 自然图像：`(w_8 - w_c)^T h` 的低尾部会低于 0，尤其 8-1、8-3、8-9。

因此 class 5 不是靠 bias 硬推出来的，而是 AVG 后的最终特征 `h` 被映射到 class 5 相对所有竞争类都占优的半空间。

## 5. 参数层面：医学的 head 和 BN 统计更不一致，但参数本身不是完整解释

| domain | group | weighted relative L2 to avg |
| --- | --- | --- |
| DermaMNIST | classifier | 0.787614 |
| CIFAR-10 3-client | classifier | 0.477090 |
| DermaMNIST | BN running stats | 0.207082 |
| CIFAR-10 3-client | BN running stats | 0.129962 |
| DermaMNIST | layer4 | 0.064024 |
| CIFAR-10 3-client | layer4 | 0.149379 |

医学的 classifier 和 BN running statistics 在 client 之间更不一致，这与医学坍缩一致。但 layer4 的参数分散度反而是 CIFAR 更高，所以不能简单说“参数差异越大越坍缩”。

更准确的说法是：

> 参数冲突是否危险，要看它传播到最终 feature projection 后，是否让 dominant class margin 对所有样本都为正。

## 6. 当前可写进论文的机制叙事

现在能支撑的故事是：

1. Partial-label non-IID 会让每个 client 的 head 学到局部类别空间。
2. 样本不均衡会让某个 client 的 dominant class 形成强 margin。
3. 自然图像融合后仍保留较丰富的样本级特征变化，所以 dominant class 只是漂移优势，不能覆盖所有样本。
4. 医学图像融合后最终特征有效秩较低，且 head/BN 统计冲突更强，导致样本级图像证据不足以翻过 dominant class 的投影优势。
5. 因此医学坍缩的特异点是：融合后的最终特征在 dominant-vs-competitor 方向上没有负尾部，所有样本都落在 dominant class 一侧。

一句话版本：

> 医学图像不是因为 non-IID 本身坍缩，而是因为不均衡 client 诱导出的 dominant-class 投影优势，在融合后的低变化医学特征空间里覆盖了全部样本；自然图像也有这个优势，但仍保留足够的图像相关变化让部分样本跨过决策边界。

## 7. 还需要继续验证的点

当前最强证据来自 DermaMNIST `resnet / clients_3 / beta_0 / seed_42` 和 CIFAR-10 控制组。为了把“医学特异点”写得更稳，需要继续做：

- 在 BloodMNIST、OrganMNIST、ChaoshengMNIST 上复算同样的 margin-tail 指标。
- 对 Fisher、RegMean、AdaMerging 的 merged model 也计算同样的 pairwise margin decomposition。
- 对 BN recalibration 前后比较 effective rank 和 dominant margin tail，验证源域图像统计是否能恢复负尾部。
- 对 balanced Derma clients 计算同样指标，确认类别均衡后 class 5 margin 的 `min_delta/p05_delta` 不再全为正。
