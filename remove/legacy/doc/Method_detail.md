# Method Detail

本文档记录仓库中各融合方法的来源、官方核心流程，以及本仓库当前如何对齐。

约定：

- `core aligned`：核心算法骨架与论文/官方实现一致。
- `adapted`：保留了论文或官方实现的核心思想，但为了适配本仓库的 full-parameter、多 client、single-dataset 或 small/VLM 统一接口，做了工程化扩展或裁剪。
- 当前正式对比脚本默认跑 12 个方法：`avg`、`ties`、`dare_linear`、`dare_ties`、`regmean`、`fisher`、`breadcrumbs`、`model_stock`、`from`、`iso_c`、`free_merge`、`robustmerge`。
- 仓库里还额外保留了 `adamerging` 和 `iso_cts`，因此下面一并说明。

## avg

- 论文链接：无单一官方论文。这里指标准的加权模型平均 baseline。
- 官方实现要点：
  - 对每个浮点参数做加权平均。
  - 不依赖 `base model`、`task vector`、`sign election` 或统计量。
- 本仓库如何对齐：
  - 直接对每个浮点参数做加权平均。
  - 非浮点 buffer 保留第一个 checkpoint 的值，避免无意义平均。
- 当前状态：`core aligned`
- 适配说明：
  - 非浮点参数不平均，这是工程性选择，不影响 baseline 主体语义。

## ties

- 论文链接：https://arxiv.org/abs/2306.01708
- 官方实现链接：https://github.com/prateeky2806/ties-merging
- 官方实现要点：
  - 用 `base model` 构造每个 client 的 `task vector`。
  - 对 task vector 做 magnitude trimming。
  - 对每个参数位置做 sign election。
  - 只保留与 elected sign 一致的更新，再做 disjoint merge。
- 本仓库如何对齐：
  - 使用 reference model 构造 task vector。
  - 做全局向量级的 magnitude trimming。
  - sign election 直接取每个位置所有 task vector 求和后的符号；求和为 0 时保持为 0，不再引入全局多数符号回填。
  - disjoint merge 仍按权重归一后执行。
- 当前状态：`core aligned`
- 适配说明：
  - 统一接入本仓库的 `merge.py` dispatcher 和 `model_hub` 输入格式。

## dare_linear

- 论文链接：https://arxiv.org/abs/2311.03099
- 官方实现要点：
  - 先构造 task vector。
  - 对 task vector 做随机稀疏保留。
  - 用 `1 / preserve_density` 做重标定，保证期望不偏。
  - 对剪枝后的 task vector 做线性加权合并。
- 本仓库如何对齐：
  - 与论文核心流程一致。
  - 随机掩码、密度超参和重标定都保留。
- 当前状态：`core aligned`
- 适配说明：
  - 只是统一到了本仓库的 full checkpoint merge API，没有改变方法主体。

## dare_ties

- 论文链接：https://arxiv.org/abs/2311.03099
- 官方实现要点：
  - 先做 DARE 随机稀疏化。
  - 再做 TIES 风格的 trimming、sign election 和 disjoint merge。
- 本仓库如何对齐：
  - 先随机 prune，再做 magnitude trim、sign election 和 disjoint merge。
  - 现在 sign tie 位置保持 0，不做全局多数符号补零。
- 当前状态：`core aligned`
- 适配说明：
  - 与 `ties` 共享公共 task-vector 工具函数。

## fisher

- 论文链接：https://arxiv.org/abs/2111.09832
- 官方实现链接：https://github.com/mmatena/model_merging
- 官方实现要点：
  - 先为每个模型收集 Fisher diagonal。
  - 对每个参数位置做 Fisher-weighted average。
  - 常见实现会加数值稳定项。
- 本仓库如何对齐：
  - 按 checkpoint 分别计算 Fisher diagonal。
  - 对每个参数执行 Fisher 对角加权平均。
  - 提供 `eps` 与最小 Fisher 权重稳定项。
  - 保留了一个可选的 Fisher L2 归一化开关，用于平衡不同 client 的 Fisher 尺度。
- 当前状态：`core aligned`
- 适配说明：
  - 额外的 Fisher 归一化是工程选项，不改变主公式。

## regmean

- 论文链接：https://arxiv.org/abs/2212.09849
- 官方实现链接：未锁定单一官方 repo，本仓库按论文闭式解对齐。
- 官方实现要点：
  - 为线性层收集输入协方差。
  - 合并时求解 `(\sum C_i)^{-1} (\sum C_i W_i)` 形式的闭式表达。
  - 主要针对二维线性层权重。
- 本仓库如何对齐：
  - 仅对二维 `.weight` 参数执行 RegMean 闭式解。
  - 协方差来自校验集统计。
  - 通过伪逆和 `eps` 处理数值稳定性。
- 当前状态：`core aligned`
- 适配说明：
  - 只处理二维线性层，这是 RegMean 在 vision/CLIP 场景下最稳妥的落地方式。
  - 额外支持 `reduce_non_diagonal_ratio`，用于缩减非对角项影响。

## breadcrumbs

- 论文链接：https://arxiv.org/abs/2412.06754
- 官方实现链接：未在本次整理中锁定单一官方 repo，按论文核心“middle magnitude breadcrumb”思想对齐。
- 官方实现要点：
  - 基于 `base model` 构造 task vector。
  - 过滤掉最小和最大的更新，只保留中等幅度的“breadcrumbs”。
  - 再对过滤后的 task vector 做合并。
- 本仓库如何对齐：
  - 现在在全局 task-vector 空间上处理，而不是按 tensor 最后一维局部筛选。
  - 对每个 client 的 task vector 先 remove 再 keep，保留中间幅度带。
  - 过滤后的 task vector 按 merge 权重加权平均后再加回 base。
- 当前状态：`adapted`
- 适配说明：
  - 当前实现对齐的是论文核心筛选思路，但仍通过本仓库统一的 full-parameter task-vector API 落地。
  - 之前版本存在“按最后一维筛选、直接累加不归一”的偏差，已修正。

## model_stock

- 论文链接：https://arxiv.org/abs/2403.19522
- 官方实现链接：未在本次整理中锁定单一官方 repo，本仓库按论文几何缩放核心做多 client 扩展。
- 官方实现要点：
  - 以 `base + ratio * delta` 的几何形式做修正。
  - `ratio` 由更新方向的夹角或余弦关系决定。
  - 原始分析更接近少量模型场景。
- 本仓库如何对齐：
  - 保留 `ratio(cosine)` 的几何缩放核心。
  - 对两个模型时直接使用两者余弦。
  - 对多模型时使用“各 client delta 与加权 mean delta 的余弦”的加权平均，作为多 client 扩展。
  - `mean_delta` 现在按 merge weights 计算。
- 当前状态：`adapted`
- 适配说明：
  - 多 client 扩展是本仓库为了支持 `clients=3/5/7` 必须做的工程化适配，不是原始少量模型场景的逐字复刻。

## adamerging

- 论文链接：https://arxiv.org/abs/2310.02575
- 官方实现链接：未在本次整理中锁定单一官方 repo，本仓库按论文“通过无标签数据学习合并系数”对齐。
- 官方实现要点：
  - 把每个 task model 的 delta 作为可学习组合项。
  - 在无标签数据上最小化熵或相关无监督目标。
  - 学习每个任务分支的系数后再形成 merged model。
- 本仓库如何对齐：
  - 用 `base + \sum lambda_i delta_i` 的形式构造 merged params。
  - 在 `stats_split` 上通过熵最小化学习 `lambda_i`。
  - 保留 `adamerging_epochs / lr / prior / max_batches` 等核心控制项。
- 当前状态：`adapted`
- 适配说明：
  - 当前仅支持 `small`，不支持 `vlm`。
  - 当前按单个数据集运行，不是多任务多数据集训练版。

## from

- 论文链接：https://arxiv.org/abs/2410.18959
- 官方实现链接：本次整理未锁定单一官方 repo，本仓库按论文中的 closed-form scaling 思路对齐。
- 官方实现要点：
  - 根据每个任务更新的强度或几何量，计算闭式融合系数。
  - 再用这些系数对各 task vector 做加权合并。
- 本仓库如何对齐：
  - 以 task-vector 的全局范数幂 `||tau_i||^k` 作为基础缩放项。
  - 再乘以 merge weights，形成最终系数并重新归一。
  - 用这些系数对 delta 做线性组合后加回 base。
- 当前状态：`adapted`
- 适配说明：
  - 当前实现明确是 `task_vector_scaling` 版本，不再声称包含 FFT。
  - 因未锁定官方 repo，文档中按论文核心思想表述，而不声称逐仓库复刻。

## iso_c

- 论文链接：https://arxiv.org/abs/2502.04959
- 官方实现链接：未在本次整理中锁定单一官方 repo，本仓库按 ISO-C 核心流程对齐。
- 官方实现要点：
  - 在公共子空间假设下，对合并更新做各向同性约束。
  - 对矩阵更新做 SVD，并对奇异值进行统一化处理。
- 本仓库如何对齐：
  - 对每个参数先形成加权平均 delta。
  - 仅对二维参数执行 SVD 和奇异值拉平。
  - 其余参数直接走加权平均 delta。
- 当前状态：`adapted`
- 适配说明：
  - 当前只在二维参数上做 ISO 处理。
  - `text_projection` 被排除，以避免 CLIP 特定投影层带来不稳定数值问题。

## iso_cts

- 论文链接：https://arxiv.org/abs/2502.04959
- 官方实现链接：未在本次整理中锁定单一官方 repo，本仓库按 ISO-CTS 核心流程对齐。
- 官方实现要点：
  - 将更新分解为 common subspace 和 task-specific subspace。
  - 再对公共与任务特定部分进行重构与等方差处理。
- 本仓库如何对齐：
  - 对二维参数先形成加权组合更新。
  - 在公共空间和 task-specific 空间之间做切分、重构、再正交化。
  - 最终对奇异值拉平后重构 merged delta。
- 当前状态：`adapted`
- 适配说明：
  - 当前只在二维参数上启用 ISO-CTS 主体流程。
  - 该方法保留在仓库中，但不是正式 12 方法对比的一部分。

## free_merge

- 论文链接：https://arxiv.org/abs/2507.01839
- 官方实现链接：未在本次整理中锁定单一官方 repo，本仓库按 FREE-Merging 的频域核心做简化对齐。
- 官方实现要点：
  - 使用频域视角分离共享和任务特定更新。
  - 完整方法通常包含更明确的 shared-backbone / expert / router 设计。
- 本仓库如何对齐：
  - 对选中的二维参数 delta 做 FFT 频域滤波。
  - 先 `fftshift`，再按频带掩码保留低频与中频部分。
  - 过滤后的 delta 按 merge weights 加权求和，再加回 base。
- 当前状态：`adapted`
- 适配说明：
  - 当前版本明确是 `without_router` 的简化实现。
  - 它保留了频域过滤这个核心，但没有复刻完整的 expert/router 结构。

## robustmerge

- 论文链接：https://arxiv.org/abs/2506.19879
- 官方实现链接：未在本次整理中锁定单一官方 repo，本仓库按 RobustMerge 的稳健聚合核心做 full-parameter 扩展。
- 官方实现要点：
  - 对更新先做幅值裁剪或掩码。
  - 再根据更新强度或注意力样指标做重加权。
  - 重点是稳健地抑制异常方向。
- 本仓库如何对齐：
  - 对二维参数先堆叠所有 delta。
  - 做幅值裁剪与 mask。
  - 基于裁剪前后强度比构造 `scale`。
  - 再结合 `fuse_weight` 与 merge weights 做加权融合。
- 当前状态：`adapted`
- 适配说明：
  - 当前实现是 full-parameter matrix 版本，不是参数高效微调场景下的原始 LoRA 形态。
  - 非二维参数仍退化为普通加权平均。

## 小结

- 明确 `core aligned` 的方法：
  - `avg`
  - `ties`
  - `dare_linear`
  - `dare_ties`
  - `fisher`
  - `regmean`
- 明确 `adapted` 的方法：
  - `breadcrumbs`
  - `model_stock`
  - `adamerging`
  - `from`
  - `iso_c`
  - `iso_cts`
  - `free_merge`
  - `robustmerge`

这些 `adapted` 方法并不是“随意实现”，而是保留论文或官方核心后，为了适配当前仓库的 full-parameter、多 client、single-dataset、small/VLM 统一接口而做的工程化版本。后续如果要进一步追求与某个外部官方 repo 的逐细节一致，需要针对对应 repo 再做专项对照。
