# my_merge 方法设计

## 一句话定义

`my_merge` 是一个面向多中心医学图像任务的训练后 checkpoint 融合方法：各医院/客户端模型先独立训练完成，之后只读取这些 checkpoint 和合法验证集信息，生成一个融合模型。

它不是联邦学习，也不依赖多轮通信。

## 解决的医学问题

真实医疗多中心数据存在几个典型问题：

- 各中心采集设备、扫描协议、操作者习惯不同，导致图像分布不一致。
- 医学图像诊断依赖边界、形态、纹理、局部对比、病灶区域等像素空间证据。
- 类别往往不均衡，少数病种或困难样本对临床更关键。
- 各中心训练好的模型可能各自擅长不同图像风格或不同类别。

普通平均融合会把所有 checkpoint 等权合并，容易抹掉这些差异。`my_merge` 的目标是在不重新训练所有模型、不做联邦通信的前提下，从验证集估计“哪个模型在当前医学证据上更可靠”，再做更保守的 checkpoint 融合。

## 为什么不是联邦学习

联邦学习通常要求：

- 各客户端参与同一训练协议。
- 多轮同步或半同步通信。
- 服务器持续聚合每轮更新。
- 客户端训练流程、通信轮次和聚合规则被统一控制。

当前任务不是这个场景。这里的前提是：模型已经在不同中心训练完了，之后再拿 checkpoint 做融合。医院之间可以不同步，训练时也不需要彼此通信。因此方法叙事应写成“训练后模型融合 / checkpoint merging”，不能写成联邦学习套壳。

## 为什么不能直接迁移到 NLP

`my_merge` 的医学专属性来自图像空间和成像物理，而不是一般的任务调参。

M1 使用或围绕这些医学图像证据：

- 局部对比度。
- 纹理异质性。
- 边界和形态。
- 前景/病灶相关区域。
- 类别稀缺和困难样本。
- 跨中心成像差异。

这些概念依赖像素网格、空间邻域、成像噪声和病灶形态。NLP 文本没有超声散斑、影像边界、ROI 形态、声影伪影、局部灰度纹理这些对象。因此该方法不能原样迁移到 NLP。若迁移，需要重新定义文本领域的证据函数，那已经不是当前医学图像方法本身。

## 当前两模块设计

### M1：医学诊断客户端信息估计

输入：

- 各客户端 checkpoint。
- 任务 metadata。
- 合法的 `val` split。

输出：

- `overall_weights`：整体客户端权重。
- `morphology_weights`：形态/纹理相关权重。
- `class_weights`：按类别的客户端权重。
- 诊断信息：`label_coverage_ratio`、`evidence_support_gate`、`client_specialization_ratio` 等。

核心原则：

- 只看 `val`，不看 `test`。
- 不按数据集名称硬编码参数。
- 权重不能被证据噪声过度放大。

当前 M1 证据包括前景面积、边界强度、局部对比度、纹理异质性、形状紧致性、诊断显著性和证据可靠性。2026-06-03 曾尝试加入 `speckle_noise`、`acoustic_shadow`、`hyperechoic_response`、`ultrasound_profile`，但 smoke 证明该 profile 会在 `dermamnist_224` 误触发、在 `chaoshengmnist_224` 不触发，造成明显负优化，因此已从代码删除。

### M2：验证集驱动的保守 checkpoint 融合

输入：

- 平均融合 checkpoint。
- M1 权重。
- 可选候选 checkpoint。
- 合法的 `val` split。

当前候选方向：

- `avg`：普通平均融合。
- `medical_weighted_fusion`：使用 M1 估计出的医学图像客户端可靠性做轻量加权融合；实现上以 reference checkpoint 为原点写成 delta 融合。
- `sign_consistent_delta`：符号一致的 delta 融合，delta 合并权重使用 M1 的 `overall_weights` 和 `morphology_weights` 形成的医学共识权重，而不是无脑基础权重。
- `avg_sign_blend_0p25`：平均融合与 sign delta 的小比例混合，同样继承医学共识权重。该候选只保留在非超声通用分支；超声 45 格里该类 blend 候选 0 次被选中，已从超声分支删除。
- `ultrasound_subset_top{k}_overall`：仅在 `chaoshengmnist_224 + my_merge_ultrasound_specialist` 下启用。候选只保留按 M1 `overall_weights` 排序得到的 top-k 客户端子集，`k=2..min(4, n-1)`。旧版枚举的 `drop_client*`、`morph/consensus top-k` 已删除，避免候选池变成无语义编号搜索。
- `top2_soup:*`：当验证集上前两名候选分数接近时，对两个候选做 0.5/0.5 soup，并重新在同一 val 批次上评估；只有 soup 的 `selection_score` 不低于当前第一名时才接管。

选择规则：

- 每个候选先按最终输出路径做同样的 BN recalibration。
- 在 `val` 上计算候选表现。
- 用验证指标排序候选；如果前两名分数差不超过默认 `0.05`，构造 top-2 soup，再用同一验证指标做保守接收。
- 最终只在 `test` 上做一次评估。

实现备注：

- 对归一化权重而言，`reference + sum_i w_i * (client_i - reference)` 与直接 `sum_i w_i * client_i` 在数学上等价。因此“delta 写法”本身不是万能改进；真正有行为差异的改动是：M1 医学共识权重进入 `sign_consistent_delta`，以及 M2 从单一 winner 转为接近候选的 guarded top-2 soup。
- delta helper 必须避免原地修改 `reference_state`；2026-06-03 的 smoke 已经证明 reference 污染会让 `medical_weighted_fusion` 变成负优化。
- 超声 profile 改造已经按 smoke 结果删除。后续不能只凭图像统计 profile 给某个数据集开特殊路径；必须先证明医学域证据能正确识别目标模态，并且不能在其他医学数据集上误触发。
- 可控超声单独处理通过 `my_merge_ultrasound_specialist` 开关启用，默认关闭。该分支只对 `chaoshengmnist_224` 生效，不改数据读取和 test 使用方式；主要做全量 val 统计、M1 权重平滑、M2 保守候选扩展、默认禁用超声 top-2 soup，以及降低 selection score 中通用医学加权准确率的占比。
- 曾尝试在超声 M1 证据前加入 `my_merge_ultrasound_denoise_evidence`：只处理 M1 形态证据灰度图，不改模型 forward、候选验证和 test 图像。45 格消融相对当前 sparse-only 基线为 `5/29/11`、mean delta `-0.000419`，因此已删除代码和 CLI，不属于当前方法。
- 曾尝试用局部梯度方向一致性做超声 M1 噪声感知 reliability，不改超参数、不改图像输入。45 格消融相对当前 sparse-only 基线为 `9/20/16`、mean delta `-0.004213`，且 `resnet` 明显受损，因此已删除代码，不属于当前方法。
- 曾尝试超声 robust validation selection：把 val 按 even/odd 拆分并用较差子集 score 排序。45 格消融相对 head repair 为 `4/39/2`、mean delta `-0.000060`，属于收益不足且轻微负优化，已删除代码和 CLI。
- 当前保留超声 head prior repair：在 M2 候选评估中，对有 classifier bias 的候选用 val 标签先验 `pi` 和候选平均预测先验 `p_hat` 做 `bias += log(pi) - log(p_hat)`，再作为独立候选进入同一 validation selection。45 格消融相对当前 sparse-only 基线 `27/18/0`、mean delta `+0.036658`，相对 formal best `37/0/8`、mean delta `+0.055405`，因此保留。
- 曾做 `avg + head prior repair` 受控消融，用来验证超声收益是否只来自分类头先验修正。45 格相对当前 full headrepair sparse 为 `0/17/28`、mean delta `-0.042568`，平均 accuracy `0.234062` 低于 full 的 `0.276630`；该实验说明当前 M1/M2 候选池在超声上仍有实际选择价值。该 ablation 开关已删除，只保留实验记录。
- 当前保留超声 client-subset soup：依据 Model Soups 的验证集候选选择思想，但不启用不受控 top-2 权重 soup，而是新增可解释的客户端 top-k 子集平均候选。旧版 45 格全量相对 full headrepair sparse 为 `20/25/0`、mean delta `+0.017390`，相对 formal best 为 `39/0/6`、mean delta `+0.072796`。随后按候选选择频次裁剪，删除所有 0 次被选中的 blend 候选和无语义 `drop_client*` 枚举；裁剪后 ResNet 9 格 smoke 仍为 `6/3/0`、mean delta `+0.038834`，候选池均值从 `31.33` 降到 `12.67`。
- 超声子项只保留 `my_merge_ultrasound_sparse_sign` 可控开关：它新增稀疏 sign-delta 候选，不替换原候选；45 格消融显示 sparse-only 相对当前超声基线 `3/42/0`、mean delta `+0.002915`，因此在超声分支下默认开启。旧版 `ultrasound_avg_sparse_sign_blend_0p10_0p2` 在后续 full subset 候选池中 0 次被选中，已删除。`my_merge_ultrasound_early_detox` 曾尝试让 `medical_weighted_fusion` 的 early 层回到 base prior，但单独消融相对当前超声基线 `7/29/9`、mean delta `-0.000599`，已从代码和 CLI 删除，只在进度文档中保留失败记录。
- gate 默认从更保守的高阈值降低为 `DEFAULT_RELIABILITY_THRESHOLD=0.10`、`DEFAULT_SEPARATION_THRESHOLD=0.05`，让医学证据在弱但稳定时也能影响 M1 权重。

当前应避免的旧设计：

- anchor 强行单模型接管。
- 稀疏残差注入。
- 复杂层路由。
- prototype head。
- specialist candidate。
- 根据某个数据集结果追加特判。

## 算法流程

```text
训练完成的客户端 checkpoint
        |
        v
加载任务 metadata 和 val loader
        |
        v
M1: 估计医学图像客户端信息
        |
        v
构造候选融合 checkpoint
        |
        v
M2: 只在 val 上选择候选
        |
        v
输出最终 merged checkpoint
        |
        v
只在 test 上报告最终指标
```

## 实验原则

- 超声是第一优先级。
- 超声不提升时，不跑全量，先改代码或删除负优化。
- 超声提升后，再检查其他医学图像数据集是否大面积下降。
- full、-M1、-M2、avg_only 必须一起看。
- 结果表和图必须脚本生成，不能手动填。

## 当前风险

2026-05-31 晚间已经修复两个关键问题：

- 超声任务上 sign/blend 候选被过窄 gate 误杀。
- 候选评分没有经过与最终输出一致的 BN recalibration，导致 `val` 选择和最终 checkpoint 表现不一致。

2026-06-02 新一轮改动后，`full/no_client_information/no_fusion_selection/avg_only` 的 2026-06-01 全量结果已作为旧版本基线。下一步需要先跑小规模 smoke，确认医学共识 sign-delta 和 top-2 soup 是否改善超声与已知坏例，再决定是否重跑全量。
