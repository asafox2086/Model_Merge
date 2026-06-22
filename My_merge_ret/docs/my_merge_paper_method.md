# my_merge BN-Only Method

当前 `my_merge` 收敛为一个面向医学影像 clinical site-silo 的训练后模型融合方法：**CEL-PCM: Clinical Evidence Ledger and Protocol-Calibrated Merge**。服务端只接收客户端 checkpoint、checkpoint 内已有的 BN running moments 和少量任务元信息，不访问原始图像、不接收逐样本 logits/activations、不把候选模型发回客户端验证。

## Motivation

医学影像多中心训练的主要困难之一是站点分布偏移：不同医院、扫描仪、染色流程、探头、窗宽窗位和采集协议会改变中间通道激活的均值和方差。BatchNorm 的 `running_mean` 和 `running_var` 正是客户端本地数据分布的低维聚合摘要。

BN moment 不是自然图像不可能出现的现象；相机、光照和风格也会带来视觉域偏移。本文的医学专用性来自 **clinical site-silo** 设定：客户端是医院/设备/协议形成的封闭 silo，类别常对应器官、病灶或专科分工，服务端不能访问原始影像。该设定和 NLP Transformer 不同：主流 NLP 模型通常使用 LayerNorm，不维护跨样本 channel running moments，也没有二维像素邻域、组织纹理或器官窗口导致的 channel moment shift。

## Client Upload

客户端上传内容限定为：

- 训练完成的 checkpoint。
- checkpoint 中每个 BN 层的 `running_mean`、`running_var`、`num_batches_tracked`。
- `num_samples`、客户端类别覆盖/类别计数等任务元信息。

正式方法不再上传医学形态特征、线性头输入二阶矩、客户端诊断准确率、置信度、margin、逐样本 logits 或逐样本 activations。

## Module 1: Clinical Evidence Ledger

CEL 把每个客户端看作一个 clinical site。它先构造样本量证据先验：

```text
a_i = Normalize(n_i)
s_i = Normalize(sqrt(n_i))
g_e = Smooth(max_i a_i)
p_i = Normalize((1 - g_e)(0.5 base_i + 0.5 s_i) + g_e a_i)
```

含义是：样本量大的客户端更稳定，但在没有单个医院压倒性占比时不过度按样本量吞掉其他专科 site。

对第 `l` 个 BN 层和第 `i` 个客户端，记 BN 均值和方差为 `mu_i^l`、`v_i^l`。服务器用 `p_i` 得到 BN centroid：

```text
mu_bar^l = sum_i p_i mu_i^l
log v_bar^l = sum_i p_i log v_i^l
```

然后计算每个客户端的 BN moment 距离：

```text
d_i = mean_l [ ||mu_i^l - mu_bar^l||^2 / v_bar^l
              + 0.5 ||log v_i^l - log v_bar^l||^2 ]
```

CEL 的融合权重为：

```text
omega_i = Normalize( p_i * (0.35 + 0.65 exp(-d_i / tau)) )
```

含义是：BN moment 接近多中心 centroid 的客户端更像可泛化的共同协议；BN moment 明显偏离的客户端可能携带站点特异偏置，因此被软抑制而不是删除。

## Module 2: Protocol-Calibrated Merge

BN running statistics 单独融合：

```text
mu^l = sum_i omega_i mu_i^l
v^l = exp( sum_i omega_i log v_i^l )
```

对非分类头的可训练参数，PCM 使用 CEL 权重做闭式加权融合：

```text
theta = sum_i omega_i theta_i
```

这一步只使用 checkpoint 参数和 BN moment ledger，不引入参考模型、不做候选选择，也不需要客户端回传验证。

对分类头，PCM 只使用任务元信息中的客户端类别覆盖做行级融合。医学数据的客户端经常来自科室、器官窗口、病种库或采集协议，类别覆盖不是随机缺失，而常常反映 clinical specialty partition：

```text
omega_{i,c} = Normalize( omega_i * (epsilon + 1[c in C_i]) )
rho = Gate(class coverage forms a specialty partition or sparse-overlap specialty)
lambda_c = rho * (0.30 + 0.25 * max(0, 3 - |S_c|))
theta_c = (1 - lambda_c) sum_i omega_i theta_{i,c}
        + lambda_c sum_i omega_{i,c} theta_{i,c}
```

其中 `S_c` 是含有类别 `c` 的客户端集合。`rho` 只由类别覆盖拓扑和类别先验偏斜决定：

- **Partition specialty**：当类别近似由少数专科客户端负责且不存在覆盖大量类别的 broad client 时，`rho` 增大。
- **Sparse-overlap specialty**：当平均类别支持数很低、singleton 比例不高、没有 broad client，且类别先验明显偏斜时，说明多中心专科之间存在少量转诊/重叠覆盖；此时允许分类头路由。
- **Fragmented support reliability**：当 singleton 碎片化和类别先验偏斜同时出现时，低 CEL 支持质量的类别路由会被降温，避免把少量低证据客户端当作可靠专科。

这样可以保护专科 head，又避免把普通 non-IID 划分误当成医学专科。这里不使用客户端验证准确率、置信度或逐样本输出。

## Privacy

`my_merge` 的隐私边界：

- 服务端不可见原始图像。
- 服务端不可见逐样本 logits。
- 服务端不可见逐样本 activations。
- 服务端不做候选池，也不把候选模型发回客户端评分。
- 上传统计只包含 checkpoint 内已有 BN running moments 和任务元信息。

BN moments 是聚合统计，不是单样本记录；它们本身不提供差分隐私保证，如需 DP 可在客户端加噪。

## Ablations

正式消融保留两项：

- `no_bml`：关闭 Clinical Evidence Ledger，使用基础客户端权重。
- `no_bcm`：关闭 Protocol-Calibrated Merge 的分类头行级校准和类别先验，只保留 BN 权重下的普通平均。

`avg_only` 仅作为 sanity baseline。

## Negative Modules Removed

实验观察后不放入正式方法：

- 候选池/客户端回传验证：早期结果较好，但需要把候选模型发回客户端打分，隐私边界和 single-shot 合并故事都不成立。
- Fisher/诊断分支：在部分医学数据上出现明显负迁移，且需要额外诊断统计，删除。
- 医学形态特征：故事容易变成手工特征堆叠，也增加上传内容，删除。
- 线性头输入二阶矩：少数 c7 case 有帮助，但运行慢、需要额外上传 `G_i`，并把方法变成第三模块；当前主方法默认禁用，仅保留代码兼容开关用于附录。
- 参数方向冲突滤波：尝试用公共初始化上的 task-vector 符号一致性抑制站点冲突方向，但在 blood c7 和 organc c3/c7 上明显负优化，且没有修复 chaosheng，因此从主方法删除。
- 无条件 Support-mass head gate：尝试在所有 partial-specialty 场景按 CEL 支持质量削弱低权重客户端的类别路由；能修复单个 organs c7 case，但会伤害 organc c7 或 blood c5。当前只保留更窄的 fragmented support reliability：必须同时满足 singleton 碎片化和类别先验偏斜才会降温。

## Positioning

CEL-PCM 借鉴 model merging 中的权重空间融合思想，但触发依据来自 clinical site-silo 医学影像的 BN moment shift 和类别专科化，而不是任务名、模型名或数据集特判。

- Model Soup/FedAvg: 权重平均是单次融合的基础，但没有医学站点 moment 校准。参考：https://arxiv.org/abs/2203.05482
- TIES/DARE: 通用模型合并会关注参数冗余、符号冲突和 delta 稀疏化；在本任务中直接加入 task-vector 冲突滤波为负优化，因此只作为对比和负结果记录。参考：https://arxiv.org/abs/2306.01708 和 https://arxiv.org/abs/2311.03099
- Fisher/RegMean: Fisher 或线性二阶矩可以做参数/线性层闭式加权，但需要梯度、数据统计或额外二阶矩上传；当前主方法为保持 single-shot 隐私边界与故事简洁，默认不启用。参考：https://arxiv.org/abs/2111.09832 和 https://openreview.net/forum?id=FCnohuR6AnM
- FedBN/SiloBN/AdaBN 思路：BN statistics 捕获客户端/域分布差异。my_merge 将这个观察用于训练后 checkpoint fusion，而不是多轮联邦训练。参考：https://arxiv.org/abs/2102.07623
