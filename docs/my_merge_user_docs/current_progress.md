# 当前进度

更新时间：2026-05-31

## 当前状态总览

`my_merge` 仍处于“方法方向已明确，但实现还需要修稳”的阶段。2026-05-31 晚上已经完成一轮针对超声退化的定位和修复，不建议立刻跑全量，但可以进入更系统的 smoke。

已经完成的事情：

- 明确 `my_merge` 是训练后 checkpoint 模型融合，不是联邦学习。
- 明确只能用 `val` 做统计、候选选择和权重估计，`test` 只做最终评估。
- 确认 `my_merge`、`fisher`、`regmean` 的统计数据读取都走 `utils.runtime.build_runtime`，默认 `stats_split=val`；`avg` merge 阶段不读数据，所有方法最终 eval 都走 test。
- 识别出原先 M2 的负优化来源：权重二次 softmax、anchor、稀疏残差、复杂层路由等容易把噪声放大。
- 主路径已经转向更保守的两模块设计。
- 修复了超声 convnext 部分格子候选池只剩 `avg` 的问题。
- 把 M1 诊断权重融合恢复为 M2 的一个候选 `medical_weighted_fusion`，由 `val` 选择，而不是无条件替代 avg。
- 修复候选选择和最终输出不一致的问题：候选现在先按最终路径做 BN recalibration，再在 `val` batch 上评分；选中的 state 不再重复 BN recalibration。
- 写好了交接指南和当前方法设计文档。
- 代码语法检查通过：`python3 -m py_compile methods/my_merge.py merge.py scripts/run_all_avg_eval.py`。

## 2026-05-31 晚间修复记录

### 读数据公平性

已确认当前 `my_merge` 的 M1 统计和 M2 候选选择只使用 `stats_split=val`。这条路径与 `fisher/regmean` 的统计入口一致，都是通过 `utils.runtime.build_runtime` 构建模型、loader 和 forward 函数。`avg` 在 merge 阶段不需要读数据，但 eval 路径相同，都是 `test`。

因此本轮问题不是数据泄漏或读取标准不同导致的，而是候选构造和候选评分逻辑导致的。

### 问题 1：sign/blend 候选被过窄 gate 误杀

坏例子：

- `outputs/codex_current_probe_20260531_190252`
- `chaoshengmnist_224 / convnext / clients=3 / beta=0`
- `avg acc=0.105121`
- 当时 `my_merge acc=0.105121`，`candidate_pool=['avg']`

原因是 `_use_sign_consistent_delta` 要求 client 数量和类别数接近，还要求接近 disjoint partition。这个条件对医学图像任务过窄，导致需要 sign/blend 候选的超声格子没有候选。

当前修复：

- `_use_sign_consistent_delta` 改为小型医学图像任务统一允许构造 sign/blend 候选。
- 仍然只在 `MEDICAL_IMAGE_DATASETS` 内生效，不迁移到 NLP。

### 问题 2：只放开 sign/blend 仍无法修复所有超声格子

在 `chaoshengmnist_224 / convnext / clients=7 / beta=0.01` 上，诊断评估显示：

- `avg test acc=0.105121`
- `sign_consistent_delta test acc=0.105121`
- `avg_sign_blend_0p25 test acc=0.105121`

这说明该格子不是选择规则误选，而是候选集合本身缺少有效候选。历史结果中同格 `0.161725` 来自旧的 M1 诊断权重融合路径。

当前修复：

- 恢复轻量版 `_medical_weighted_state_merge`。
- 只把它作为候选 `medical_weighted_fusion` 放进已有 M2 candidate pool。
- 不恢复 anchor、specialist、prototype、稀疏残差等旧模块。
- 最终仍由 `val` 评分决定是否选择该候选。

### 问题 3：候选评分必须和最终 BN 路径一致

恢复 `medical_weighted_fusion` 后，`organsmnist_224 / resnet / clients=3 / beta=0` 一度从约 `0.50` 掉到 `0.360145`。原因不是 weighted candidate 必然差，而是候选打分在未做 BN recalibration 的 state 上执行，最终输出却会再做 BN recalibration。

当前修复：

- 每个候选先按最终路径做 BN recalibration。
- 再用相同 `val` batch 计算 `candidate_metrics`。
- 选中的候选已经是 prepared state，最终阶段不重复 recalibration。

修复后该格子恢复到：

- `outputs/codex_bn_aligned_probe_20260531_2030`
- `organsmnist_224 / resnet / clients=3 / beta=0`
- selected=`avg`
- test acc=`0.507534`

## 已有结果

### 2026-05-31 晚间最终代码 smoke

运行环境限制：

- 当前机器 PyTorch 检测到 `cuda False 0`，本轮 probe 都是在 CPU 上跑的。
- smoke 只代表局部格子，不是全量结论。
- 所有候选选择仍只用 `val`，表里的 acc 是最终 `test`。

超声关键格子：

来源：

- `outputs/codex_bn_aligned_ultrasound_20260531_2040`

| dataset | model | clients | beta | selected | test acc | 备注 |
|---|---|---:|---:|---|---:|---|
| chaoshengmnist_224 | convnext | 3 | 0.0 | avg_sign_blend_0p25 | 0.173405 | 从坏例 `0.105121` 恢复 |
| chaoshengmnist_224 | convnext | 5 | 0.01 | medical_weighted_fusion | 0.173405 | 从坏例 `0.105121` 恢复 |
| chaoshengmnist_224 | convnext | 7 | 0.01 | medical_weighted_fusion | 0.161725 | avg/sign/blend 都是 `0.105121`，需要 weighted candidate |

非超声代表格子：

来源：

- `outputs/codex_bn_aligned_cross_dataset_20260531_2055`

| dataset | model | clients | beta | selected | test acc | 备注 |
|---|---|---:|---:|---|---:|---|
| bloodmnist_224 | resnet | 3 | 0.0 | avg_sign_blend_0p25 | 0.552470 | 高于旧 smoke `0.4680` |
| dermamnist_224 | resnet | 3 | 0.0 | avg_sign_blend_0p25 | 0.673815 | 略低于旧 smoke `0.6793`，仍高于文档中 fisher `0.6718` |
| organcmnist_224 | resnet | 3 | 0.0 | medical_weighted_fusion | 0.487463 | 接近旧 smoke `0.4877`，高于文档中 robustmerge `0.3824` |
| organsmnist_224 | resnet | 3 | 0.0 | avg | 0.507534 | 修复 BN 候选评分后从 `0.360145` 恢复，高于旧 smoke `0.4945` |

当前判断：

- 超声退化的核心原因已经定位并修复：候选池不能只剩 avg，且 `medical_weighted_fusion` 对部分超声格子是必要候选。
- 其他医学图像代表格子没有出现大面积回退。
- 下一步应跑更完整的 ultrasound convnext 小网格，再扩展到其他模型和数据集。

### 小规模 smoke

参考文件：

- `reference_reports/selection_fix_small_smoke_summary.md`

范围：

- `small / resnet / clients=3 / beta=0 / seed=42`

结论：

- 5 个 checked dataset 均高于同格已有 best formal result。
- `chaoshengmnist_224` 上 full 高于 `-M1`、`-M2` 和 `avg_only`。
- 这说明当前方向在小样本 smoke 上是有希望的。

重要限制：

- 这不是全量结果。
- 不能拿它当最终论文结果。

### 超声 convnext probe

表现较好的一轮：

- `outputs/my_merge_probe_validated_select_20260531_150416`

其中 `chaoshengmnist_224 / convnext` 多个格子达到或接近 `0.173405`。

暴露问题的一轮：

- `outputs/my_merge_chaosheng_convnext_signfix2_20260531_150557`

其中部分格子退化到 `0.105121`。2026-05-31 晚间已经定位为候选 gate 过窄和候选池缺少 `medical_weighted_fusion` 两个问题。当前代码已修复这些问题，但还需要更完整小网格验证。

## 当前代码里的关键风险

### 1. 候选池已修复，但需要扩展验证

当前 M2 候选池包括：

- `avg`
- `medical_weighted_fusion`
- `sign_consistent_delta`
- `avg_sign_blend_0p25`

候选会先按最终路径做 BN recalibration，再用 `val` batch 评分。已验证的关键超声格子不再退化到 `0.105121`，但还没有跑完整 convnext 小网格。

下一步应优先检查每个 `meta.json` 中的：

- `selected_candidate`
- `candidate_pool`
- `candidate_metrics`
- `overall_weights`
- `label_coverage_ratio`
- `fallback_reason`

### 2. 主方法仍然偏长

`methods/my_merge.py` 里还保留了一些历史函数。虽然默认主路径已经转向保守候选选择，但文件还没有真正压缩到“正常方法”的长度。

后续应删除不再使用的旧模块，把统计和可视化放在脚本里。

### 3. 不能再加无效功能

后续优化原则不是叠模块，而是：

- 找负优化。
- 删负优化。
- 小跑验证。
- 如果结果差，回到代码逻辑继续修。

## 推荐下一步

### 第一步：跑完整超声 convnext 小网格

先跑：

- dataset：`chaoshengmnist_224`
- model：`convnext`
- clients/beta：`(3,0)`, `(3,0.01)`, `(3,0.1)`, `(5,0)`, `(5,0.01)`, `(5,0.1)`, `(7,0)`, `(7,0.01)`, `(7,0.1)`
- 单 GPU
- `num_workers=0`
- `stats_split=val`
- `my_merge_stats_max_batches=16`
- `my_merge_bn_batches=4`

目标：

- 不再出现明显退化格子。
- full 不低于 avg 太多，并尽量超过已有方法。
- M1、M2 消融有可见差异。

### 第二步：普通数据集 smoke

超声稳定后，再检查：

- `bloodmnist_224`
- `dermamnist_224`
- `organcmnist_224`
- `organsmnist_224`

目标：

- 不能大面积下降。
- 少数格子不超过已有方法可以接受。

### 第三步：全量与汇总

只有当前两步都过关后，再跑全量。

全量完成后：

- 自动更新 `summary_table.md` 的来源表。
- 图放进独立图目录。
- 权重分析用熵、集中度、比值图，不要堆不可读大表。

## 不要做的事

- 不要直接跑全量。
- 不要偷看 test 选参数。
- 不要按数据集名称写特判。
- 不要把方法重新写成联邦学习。
- 不要继续堆新模块掩盖负优化。
- 不要开高并发让服务器过载。
