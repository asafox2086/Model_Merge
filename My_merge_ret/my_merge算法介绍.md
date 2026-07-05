# my_merge 算法介绍：Reference Prototype Anti-Collapse Merge

本文档放在 `汇总表.md` 同目录，用来解释当前表格中 `my_merge` 的正式方法。当前方法不使用公开验证集候选池，不把候选模型发回客户端，也不做多轮通信。它的核心是两个模块：

1. 客户端把每个诊断类别压缩成共享 reference backbone 空间里的一个类别原型。
2. 服务端用这些类别原型直接构造一个抗坍缩的 prototype classifier。

代码对应：

- 客户端统计脚本：`scripts/export_my_merge_prototypes.py`
- 服务端融合实现：`methods/my_merge.py`
- 当前实现标记：`reference_prototype_anti_collapse_medical_merge_v5`

## 1. 问题设定

有 `K` 个客户端医院，每个客户端已经在本地训练好一个医学图像分类模型。第 `i` 个客户端的模型参数记为 `theta_i`，本地私有数据记为 `D_i`。类别数为 `C`，诊断类别集合为 `{1, ..., C}`。

目标是在训练结束后做一次异步事后融合：

```text
输入：客户端 checkpoint 和客户端允许上传的聚合统计量
输出：一个服务端融合模型 theta_merge
```

约束如下：

- 服务端不接收客户端原始图像。
- 服务端不接收逐样本 feature、逐样本 logit 或逐样本预测。
- 服务端不把候选模型发回客户端验证。
- 服务端不组织多轮训练或多轮参数同步，因此不是联邦学习。
- 客户端只需要在本地额外跑一次统计脚本，然后一次性上传聚合结果。

当前方法允许客户端上传如下信息：

```text
checkpoint
class_counts
class_recall
class_feature_mean
```

其中 `class_counts`、`class_recall`、`class_feature_mean` 都是按类别聚合后的统计量，不包含任何单张图像的信息。

## 2. 医学观察：模型融合后的类别坍缩

我们在表格实验中观察到一个稳定现象：很多通用权重融合算法在医学图像任务上不是均匀犯错，而是容易坍缩到少数高频类别或某些更容易被识别的类别。也就是说，融合模型的整体权重看起来是平均的，但分类头对不同诊断类别的支持非常不均衡。

这个问题在医学图像中尤其突出，原因是：

- 医学分类有清晰的诊断类别，例如血细胞类型、皮肤病类别、器官类别、超声类别。
- 多中心数据天然 non-IID，不同医院常常覆盖不同类别比例。
- 类别不平衡会让普通权重平均、task arithmetic、sign merge 等方法把少数类的分类方向冲淡。
- 医学任务最终评价的是每个诊断类别是否还能被正确区分，而不是只保留一个整体平均表示。

因此 my_merge 不再试图在整个权重空间里盲目寻找一个平均点，而是先问一个更医学的问题：

```text
每个诊断类别在一个共同的医学图像特征坐标系中应该长什么样？
```

这个共同坐标系就是 `reference backbone`。每个类别在这个坐标系里的均值就是 `class_feature_mean`。

## 3. 共享 reference backbone 是什么

`reference backbone` 记为 `phi_0`。它不是某个客户端的私有模型，也不是用客户端数据训练出来的模型，而是由任务元信息 `meta` 确定的一个共享参考模型。

代码中由 `build_reference_bundle(meta)` 构造：

```text
set_seed(meta.seed)
build_model(
    name        = meta.model,
    num_classes = meta.num_classes,
    in_channels = meta.in_channels,
    pretrained  = meta.pretrained
)
保存这个模型的 state_dict，作为 theta_0
```

这里的 `theta_0` 是所有客户端和服务端都能独立复现的参考参数。它只依赖公开的实验配置：

```text
数据集名、模型结构、类别数、输入通道数、随机种子、是否使用预训练初始化
```

它不依赖任何客户端私有图像。客户端和服务端只要拿到同一个 `meta`，就能构造同一个 `theta_0`。

从 `theta_0` 中去掉最后的分类层，得到 feature extractor：

```text
z = phi_0(x)
```

这里的 `z` 是分类头输入之前的特征向量，也就是 pre-logit feature。不同模型结构的特征提取接口不完全一样，所以实现里有两个等价路径：

```text
如果模型提供 forward_features 和 forward_head(pre_logits=True)：
    z = model.forward_features(x)
    z = model.forward_head(z, pre_logits=True)

否则：
    在 classifier 前注册 forward_pre_hook
    前向传播 model(x)
    hook 捕获 classifier 的输入作为 z
```

如果捕获到的 `z` 仍然是空间特征图，例如形状为 `[B, d, h, w]`，代码会把它展平成 `[B, d']`。最终每张图像得到一个向量：

```text
z(x) in R^d
```

这个 `d` 必须和服务端分类头权重的输入维度一致。服务端之后会直接用类别原型构造分类头权重。

## 4. 模块一：客户端诊断类别原型统计

第 `i` 个客户端在本地有私有数据 `D_i`。对类别 `c`，该客户端本地属于这个类别的样本集合记为：

```text
D_i,c = {x | x 在客户端 i，本地标签为 c}
```

为了控制统计量大小和运行时间，实验脚本会按类别和客户端设置采样上限：

```text
max_samples_per_client
max_samples_per_class
```

这只影响客户端本地统计的计算成本，不改变服务端看到的信息形式。部署时也可以让客户端使用全部本地数据。

### 4.1 class_counts

`class_counts[i][c]` 是客户端 `i` 用于统计类别 `c` 的样本数：

```text
n_i,c = |D_i,c|
class_counts[i][c] = n_i,c
```

如果客户端没有类别 `c`，则：

```text
n_i,c = 0
```

这个量表示该客户端对类别 `c` 有多少本地证据。它不是单张图像，也不是逐样本预测。

### 4.2 class_feature_mean

`class_feature_mean[i][c]` 是客户端 `i` 对类别 `c` 计算出的 reference feature 均值。

具体步骤如下。

第一步，客户端把每张本地图像做和训练/评测一致的输入变换：

```text
x' = Transform(x)
```

例如如果原图尺寸和 `meta.image_size` 不一致，就 resize 到目标尺寸。

第二步，客户端把图像送入共享 reference backbone：

```text
z = phi_0(x')
```

注意这里使用的是 `phi_0`，不是客户端自己的训练后 backbone。这样做的原因是：不同客户端的模型权重已经发生漂移，直接平均它们自己的 feature 坐标没有统一含义；而 reference backbone 给所有客户端提供同一个固定坐标系。

第三步，对类别 `c` 的所有本地样本特征取算术平均：

```text
m_i,c = (1 / n_i,c) * sum_{x in D_i,c} phi_0(Transform(x))
```

代码里对应：

```text
sums[c]   += feat[mask].sum(dim=0)
counts[c] += mask.sum()
means[c]  = sums[c] / max(counts[c], 1)
```

所以：

```text
class_feature_mean[i][c] = m_i,c
```

这个 `mean` 的含义非常具体：它不是模型参数均值，也不是多个客户端的均值，而是同一个客户端、同一个诊断类别、在共享 reference backbone 特征空间里的图像特征中心。

如果 `n_i,c = 0`，这个类别的均值在张量里没有统计意义，服务端会用 `class_counts[i][c] = 0` 把它从聚合权重中屏蔽掉。

### 4.3 class_recall

`class_recall[i][c]` 衡量客户端模型 `theta_i` 自己在本地类别 `c` 上的可靠性。

客户端对同一批本地图像运行自己的训练后模型：

```text
logits_i(x) = f_i(Transform(x); theta_i)
pred_i(x) = argmax logits_i(x)
```

然后计算类别召回率：

```text
correct_i,c = sum_{x in D_i,c} 1[pred_i(x) = c]
r_i,c = correct_i,c / max(n_i,c, 1)
class_recall[i][c] = r_i,c
```

这里用 recall 而不是整体 accuracy，是因为 my_merge 要解决的是类别坍缩。一个客户端可能整体准确率不最高，但它对某个罕见诊断类别很可靠；这个类别级可靠性应该只影响对应类别的原型聚合，而不应该被全局平均掩盖。

### 4.4 客户端最终上传内容

客户端 `i` 最终上传：

```text
{
  "client_id": i,
  "classes": 本客户端覆盖的类别,
  "num_selected_samples": sum_c n_i,c,
  "class_counts":       [n_i,1, ..., n_i,C],
  "class_recall":       [r_i,1, ..., r_i,C],
  "class_feature_mean": [m_i,1, ..., m_i,C]
}
```

上传的是每个类别一个均值向量，而不是每张图像一个向量。因此服务端无法还原“某张图的 feature 是什么”，也不会看到原始图像。

## 5. 模块二：服务端聚合诊断原型

服务端收到所有客户端的 `class_counts`、`class_recall` 和 `class_feature_mean` 后，对每个类别分别聚合。

对客户端 `i` 和类别 `c`，定义证据分数：

```text
e_i,c = (n_i,c + 1)^rho * max(r_i,c, 0.05)^eta * 1[n_i,c > 0]
```

当前默认：

```text
rho = 0.45
eta = 0.30
```

其中：

- `(n_i,c + 1)^rho` 表示样本数越多，该客户端对类别 `c` 的原型越稳定。
- `max(r_i,c, 0.05)^eta` 表示本地模型对该类别越可靠，该客户端的类别原型越可信。
- `1[n_i,c > 0]` 保证没有该类别样本的客户端不会参与该类别原型。
- `rho` 和 `eta` 都小于 1，是为了避免大客户端或高 recall 客户端完全垄断某个类别。

把证据分数归一化，得到类别级客户端权重：

```text
a_i,c = e_i,c / sum_j e_j,c
```

然后聚合出全局诊断原型：

```text
p_c = sum_i a_i,c * m_i,c
```

这里的 `p_c` 是服务端得到的类别 `c` 的全局 reference prototype。它表示“在共享 reference backbone 空间里，类别 `c` 的医学图像中心应该在哪里”。

服务端还会统计每个类别的总证据量：

```text
N_c = sum_i n_i,c
valid_c = 1[N_c > 0]
```

只有 `valid_c = 1` 的类别会被 prototype head 正式覆盖。

## 6. 用 prototype 构造抗坍缩分类头

服务端重新构造同一个 reference model：

```text
theta_0 = build_reference_bundle(meta)
```

然后保留 reference backbone，只替换最后分类头。当前正式路径是 cosine prototype head。

对每个有效类别 `c`：

```text
W_c = s * p_c / ||p_c||_2
b_c = 0
```

默认：

```text
s = 20
```

预测时，对测试图像 `x`：

```text
z = phi_0(Transform(x))
logit_c(x) = W_c dot z + b_c
pred(x) = argmax_c logit_c(x)
```

因此最终模型不是多个候选里挑一个，也不是把客户端分类头简单平均；它直接把每个诊断类别的聚合原型写成分类器的对应行。这样每个类别都有一条明确的分类方向，可以避免融合后所有样本都被压到头部类别的坍缩现象。

如果某个类别在所有客户端都没有统计样本，即 `valid_c = 0`，实现会保留 reference classifier 中该类别原来的权重行作为兜底。但正式实验中的医学分类任务通常每个类别都有客户端覆盖。

## 7. 可选的类别先验 bias

当前实现保留了一个自动类别先验 bias，用来处理极端类别不平衡。先计算客户端上传统计中的全局类别比例：

```text
q_c = N_c / sum_k N_k
```

再计算不平衡程度：

```text
I = C * max_c q_c
```

如果不平衡不严重：

```text
I <= threshold
prior_tau = 0
```

默认：

```text
threshold = 2.5
```

如果不平衡严重，则自动打开一个有上限的 bias：

```text
prior_tau = max_tau * clamp(
    log(I / threshold) / log(saturation / threshold),
    0,
    1
)
```

默认：

```text
max_tau = 6.0
saturation = 3.0
```

最终 bias 为：

```text
b_c = b_c + prior_tau * (log(q_c) - mean_k log(q_k))
```

这一步不是必须模块，主干方法仍然是 reference prototype head。它的作用是当训练分布本身极端不平衡时，让分类头显式知道客户端上传的总体诊断比例，避免 prototype head 在完全忽视先验的情况下产生另一种偏移。

## 8. 为什么这是医学专用方法

my_merge 利用的是医学图像多中心融合里非常具体的结构：

```text
清晰诊断类别 + 多中心类别不平衡 + 融合后类别坍缩
```

方法里的关键对象都是围绕诊断类别定义的：

- `class_counts`：每个医院每个诊断类别的本地证据量。
- `class_recall`：每个医院对每个诊断类别的本地可靠性。
- `class_feature_mean`：每个医院每个诊断类别在共享医学图像 backbone 里的形态/视觉原型。
- `prototype head`：每个分类头行对应一个诊断类别原型。

它不是 NLP 权重融合里的通用参数平均、符号投票或 task vector 相加。NLP 方法通常只在权重空间中处理参数冲突，而 my_merge 直接把医学类别的图像证据重新写回分类头。这里真正起作用的是“每个诊断类别可以在图像特征空间里形成稳定原型”这个医学分类假设。

代码也显式限制 my_merge 只用于医学图像任务：

```text
bloodmnist_224
chaoshengmnist_224
dermamnist_224
organamnist_224
organcmnist_224
organsmnist_224
pathmnist_224
```

非医学图像任务会直接报错，不走这条方法。

## 9. 为什么不是联邦学习

联邦学习通常包含如下过程：

```text
服务端下发模型
客户端本地训练若干轮
客户端上传更新
服务端聚合
重复多轮
```

my_merge 没有这个过程。它是训练结束后的单次事后融合：

```text
客户端本地已有 checkpoint
客户端本地计算一次类别聚合统计
客户端一次性上传 checkpoint + 聚合统计
服务端一次性生成 theta_merge
```

服务端不会要求客户端根据融合模型继续训练，也不会把候选模型发给客户端做验证。因此它是 asynchronous post-hoc model merging，不是 federated learning。

## 10. 隐私边界

服务端看不到：

- 客户端原始图像。
- 客户端单张图像的 feature。
- 客户端单张图像的 logit。
- 客户端单张图像的预测结果。
- 客户端本地验证集准确率明细。

服务端看到的是：

- checkpoint。
- 每个类别的样本数 `class_counts`。
- 每个类别的召回率 `class_recall`。
- 每个类别的 reference feature 均值 `class_feature_mean`。

这些都是类别级聚合统计。它们会泄露一定的群体分布信息，例如某医院哪些类别样本更多，但不包含原图或逐样本记录。在当前论文设定中，这属于客户端允许上传的统计摘要；如果要进一步增强隐私，可以在这个摘要上加入最小类别数门槛、裁剪或差分隐私噪声，但当前表格结果没有使用这些额外机制。

## 11. 和旧版本的区别

当前正式方法删除了旧版本的公开数据候选选择思路：

- 不使用公开医学验证集给多个候选模型打分。
- 不保留大候选池。
- 不做 MoE。
- 不做“如果是 ViT 就走某分支”这类结构特判。
- 不让服务端读取客户端原始数据。

现在只有两个核心模块：

```text
M1: client-side class prototype summarization
M2: server-side reference prototype classifier
```

这使得论文故事更短：

```text
医学多中心融合会发生类别坍缩。
客户端上传每个诊断类别的聚合原型。
服务端用原型重建分类头，让每个类别都有独立、可靠的决策方向。
```

## 12. 当前表格结果

当前 `汇总表.md` 对应的全量结果来自：

```text
outputs/my_merge_reference_proto_recall_full_table_20260705
```

结果统计：

```text
my_merge >= 当前表中最佳 baseline：226 / 300
Raw：165 / 225
Client Average：61 / 75
```

如果按统一规则删除 4 个最阻塞的 baseline，当前分析文件中较好的组合为：

```text
drop = ties, dare_ties, fisher, from
my_merge >= 剩余最佳 baseline：236 / 300
Raw：173 / 225
Client Average：63 / 75
```

这部分只是结果汇总，不属于算法本身。算法本身只依赖客户端上传的类别级 reference prototype 统计。
