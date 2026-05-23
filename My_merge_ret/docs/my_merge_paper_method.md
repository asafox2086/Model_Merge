# my_merge Two-Module Method

当前 `my_merge` 已整理为两个模块，不再把旧的医学先验提取作为独立 M1。

## Module 1: Diagnostic Evidence Client Information Estimation

M1 从验证集图像中估计每个客户端携带的诊断信息。医学专用性来自医学影像中普遍存在的证据，而不是针对某个数据集写分支：

- 前景结构比例
- 边界强度
- 局部对比度
- 纹理异质性
- 形状紧致度
- 诊断显著性
- 类别稀有度
- 难例正确率和预测 margin

这些量依赖医学图像的空间结构、病灶/组织边界、局部纹理和类别不均衡，因此是医学影像专用的；它不直接迁移到 NLP，因为 NLP token 序列没有同样的二维前景、边界、局部纹理和形态显著性。

M1 输出三类权重：

- `overall_weights`：客户端整体诊断可靠性。
- `morphology_weights`：客户端在高医学证据样本上的可靠性。
- `class_weights`：每个类别对应的客户端专长权重。

## Module 2: Medical Evidence Guided Fusion and Selection

M2 使用 M1 的诊断信息做参数融合和候选选择：

- 层级融合：早期层偏向形态证据强的客户端，后期层更多考虑整体诊断可靠性。
- 分类头融合：按类别使用 `class_weights`，保留类别专长。
- 稀疏残差注入：保留最有诊断贡献的客户端差异。
- 候选池选择：在 `avg`、医学证据融合、形态锚点、专科客户端、reference delta、prototype head、consensus 等候选中选择验证分数最高者。

候选选择分数同时考虑普通准确率、balanced accuracy、医学证据加权准确率和难例准确率。

## Default Ablations

默认全量实验只保留两模块消融：

- `full`：完整两模块方法。
- `no_client_information`：关闭 M1，回退到基础客户端权重和中性医学证据。
- `no_fusion_selection`：关闭 M2，回退到平均融合。
- `avg_only`：同时关闭 M1 和 M2。

旧的 `no_medical_prior`、`no_domain_preprocess`、`no_modality_features`、`no_medical_preprocess` 仍保留兼容入口，但都映射到关闭整个 M1，不再代表独立第三模块。
