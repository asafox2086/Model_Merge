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
- 类别覆盖和预测置信边界。
- 跨中心成像差异。

这些概念依赖像素网格、空间邻域、成像噪声和病灶形态。NLP 文本没有超声散斑、影像边界、ROI 形态、声影伪影、局部灰度纹理这些对象。因此该方法不能原样迁移到 NLP。若迁移，需要重新定义文本领域的证据函数，那已经不是当前医学图像方法本身。

## 当前三模块设计

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

当前 M1 证据精简为前景面积、边界强度、局部对比度、纹理异质性、诊断显著性和证据可靠性。`shape_compactness`、class-rarity、hard/focal 样本权重和 domain-focus 已删除：没有最新全量子项消融支撑，且与显著性、margin 和类别覆盖信息重复。2026-06-03 曾尝试加入 `speckle_noise`、`acoustic_shadow`、`hyperechoic_response`、`ultrasound_profile`，但 smoke 证明该 profile 会在 `dermamnist_224` 误触发、在 `chaoshengmnist_224` 不触发，造成明显负优化，因此已从代码删除。

### M2：医学证据 checkpoint 融合

输入：

- 平均融合 checkpoint。
- M1 权重。
- reference checkpoint。

输出基础候选：

- `medical_weighted_fusion`：使用 M1 估计出的医学图像客户端可靠性做轻量加权融合；实现上以 reference checkpoint 为原点写成 delta 融合。
- `sign_consistent_delta`：符号一致的 delta 融合，delta 合并权重使用 M1 的 `overall_weights` 和 `morphology_weights` 形成的医学共识权重，而不是无脑基础权重。

M2 不再包含普通 `avg`、`avg_sign_blend_0p25`、权重空间 top-2 soup、anchor、prototype、specialist 或稀疏残差。`avg_only` 只作为显式对照。

### M3：自适应候选生成与验证选择

输入：

- M2 基础候选。
- M1 权重。
- 合法的 `val` split。

当前 M3 默认开启，消融标签为 `no_adaptive_candidates` 或 `no_m3`。

M3 额外生成三类候选：

- `adaptive_subset_top{k}_overall`：按 M1 `overall_weights` 排序，只保留 top2/top3 客户端做子集平均。它用医学证据选择客户端，不按数据集名称特判。
- `adaptive_sign_sparse_0p2`：在符号一致 delta 融合前只保留幅度最大的 20% delta，用于抑制客户端私有噪声和方向冲突。
- `head_prior_repair:*`：对每个候选的分类头 bias 做类别先验校正，使用 `bias += log(pi_val) - log(p_pred)`，只修正类别偏置，不改主体特征层。

M3 选择规则：

- 每个候选先按最终输出路径做同样的 BN recalibration。
- 在 `val` 上计算候选表现。
- 用验证指标排序候选，直接选择分数最高的医学候选。
- 最终只在 `test` 上做一次评估。

实现备注：

- 对归一化权重而言，`reference + sum_i w_i * (client_i - reference)` 与直接 `sum_i w_i * client_i` 在数学上等价。因此“delta 写法”本身不是万能改进；真正有行为差异的改动是：M1 医学共识权重进入 `sign_consistent_delta`，以及 M2 在医学加权、sign delta、top-k 子集和 head repair 候选之间做验证选择。
- delta helper 必须避免原地修改 `reference_state`；2026-06-03 的 smoke 已经证明 reference 污染会让 `medical_weighted_fusion` 变成负优化。
- 统一 adaptive candidate 不判断具体数据集。早期超声专用开关和 `ultrasound_*` 候选已从当前代码路径删除，相关失败/收益记录只保留在进度文档中。
- M3 默认开启，因为用户认为现有结果可接受；默认全量消融现在包含 `no_adaptive_candidates` 作为 `-M3`。
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
M2: 构造基础医学融合候选
        |
        v
M3: 增加自适应候选并在 val 上选择
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
- full、-M1、-M2、-M3、avg_only 必须一起看。
- 结果表和图必须脚本生成，不能手动填。

## 当前风险

2026-05-31 晚间已经修复两个关键问题：

- 超声任务上 sign/blend 候选被过窄 gate 误杀。
- 候选评分没有经过与最终输出一致的 BN recalibration，导致 `val` 选择和最终 checkpoint 表现不一致。

2026-06-02 新一轮改动后，`full/no_client_information/no_fusion_selection/avg_only` 的 2026-06-01 全量结果已作为旧版本基线。下一步需要先跑小规模 smoke，确认医学共识 sign-delta 和 top-2 soup 是否改善超声与已知坏例，再决定是否重跑全量。
