# my_merge 交接文档

生成时间：2026-05-31

## 0. 2026-05-31 晚间更新

本轮工作的目标是修复 `chaoshengmnist_224` 超声数据集上 `my_merge` 退化到 avg 的问题，同时保证其他医学图像数据集不出现大面积回退。

### 0.1 已确认的数据使用方式

`my_merge` 的统计、候选选择和权重估计走 `utils.runtime.build_runtime`，默认 `stats_split=val`。`fisher/regmean` 的统计也走同一个 runtime 入口。`avg` 在 merge 阶段不读数据，但所有方法最终 eval 都走 `test`。

因此当前结果差异不是由于 `my_merge` 偷看 test 或数据读取标准不同造成的。后续不要用 test 做选择、调参或特殊回退。

### 0.2 关键问题与修复

问题 1：`_use_sign_consistent_delta` 的 gate 过窄。

- 坏例：`outputs/codex_current_probe_20260531_190252`
- `chaoshengmnist_224 / convnext / clients=3 / beta=0`
- `my_merge` 与 `avg` 都是 `0.105121`
- `candidate_pool=['avg']`

修复：

- 小型医学图像任务统一允许构造 `sign_consistent_delta` 和 `avg_sign_blend_0p25`。
- 仍然限制在医学图像数据集集合中，不扩展到 NLP。

问题 2：只靠 sign/blend 不足以修复所有超声格子。

- `chaoshengmnist_224 / convnext / clients=7 / beta=0.01`
- 诊断候选 test probe 显示 `avg`、`sign_consistent_delta`、`avg_sign_blend_0p25` 都是 `0.105121`
- 历史可用结果 `0.161725` 来自 M1 诊断权重融合路径

修复：

- 恢复轻量版 `medical_weighted_fusion`，但只作为 M2 候选池中的一个候选。
- 不恢复旧的 anchor、specialist、prototype、稀疏残差等复杂模块。
- 最终仍由 `val` 候选评分选择，避免把 weighted fusion 写成无条件主路径。

问题 3：候选评分和最终输出路径不一致。

- 恢复 weighted candidate 后，`organsmnist_224 / resnet / clients=3 / beta=0` 一度掉到 `0.360145`
- 原因是候选评分用未 BN recalibration 的 state，但最终输出会 BN recalibration

修复：

- 每个候选先按最终路径做 BN recalibration。
- 再用同一批 `val` batch 评分。
- 选中的 state 已经是 prepared state，最终阶段不重复 recalibration。

### 0.3 当前代码 smoke 结果

超声关键格子，来源 `outputs/codex_bn_aligned_ultrasound_20260531_2040`：

| dataset | model | clients | beta | selected | test acc |
|---|---|---:|---:|---|---:|
| chaoshengmnist_224 | convnext | 3 | 0.0 | avg_sign_blend_0p25 | 0.173405 |
| chaoshengmnist_224 | convnext | 5 | 0.01 | medical_weighted_fusion | 0.173405 |
| chaoshengmnist_224 | convnext | 7 | 0.01 | medical_weighted_fusion | 0.161725 |

非超声代表格子，来源 `outputs/codex_bn_aligned_cross_dataset_20260531_2055`：

| dataset | model | clients | beta | selected | test acc |
|---|---|---:|---:|---|---:|
| bloodmnist_224 | resnet | 3 | 0.0 | avg_sign_blend_0p25 | 0.552470 |
| dermamnist_224 | resnet | 3 | 0.0 | avg_sign_blend_0p25 | 0.673815 |
| organcmnist_224 | resnet | 3 | 0.0 | medical_weighted_fusion | 0.487463 |
| organsmnist_224 | resnet | 3 | 0.0 | avg | 0.507534 |

### 0.4 接手后优先事项

下一步不要直接全量。建议顺序：

1. 跑更完整的 `chaoshengmnist_224 / convnext` 小网格：`(3,0)`, `(3,0.01)`, `(3,0.1)`, `(5,0)`, `(5,0.01)`, `(5,0.1)`, `(7,0)`, `(7,0.01)`, `(7,0.1)`。
2. 如果超声稳定，再跑其他数据集同等 smoke。
3. 如果发现个别格子回退，优先看 `candidate_pool`、`candidate_metrics`、`selected_candidate` 和 BN 后 val 分数，不要先加新模块。
4. 当前候选 BN 对齐会增加 merge 时间；如果全量太慢，可以考虑引入单独的候选 BN batch 参数，但不要改变 val/test 边界。

## 1. 用户要求复述

这次工作的核心不是继续堆功能，而是把 `my_merge` 做成一个可信、简洁、可复现的医学图像模型融合方法。用户明确要求如下。

### 方法定位

- `my_merge` 必须是模型融合方法，不是联邦学习。
- 正确流程是：各客户端/各中心模型已经训练完成，之后读取训练好的 checkpoint 做 post-hoc merge。
- 论文叙事需要强调：真实医疗多中心场景中，医院数据和训练过程通常不同步，联邦学习的多轮同步通信、统一训练协议和持续协同假设很难成立；本方法解决的是“训练后 checkpoint 融合”问题。
- 不要把“异步模拟”硬塞进代码。模型融合天然允许各模型先独立训练，再在训练结束后合并。

### 数据使用合法性

- 必须和 `avg`、`fisher`、`regmean` 等正式方法一样“看数据”。
- 统计、选择、权重估计只能看 `val`，最终指标只能在 `test` 上评估。
- 不能用 test 做候选选择、参数选择或特殊回退。
- 不能因为看了某个数据集结果而写硬编码特判；数据分析只能用于理解通用特征。

### 模块设计

- 不要一直加功能。发现负优化模块后，应优先删除或关闭，而不是再叠一个新模块补救。
- `my_merge` 应该像正常融合方法一样短而清楚；统计、可视化、诊断分析应放在额外脚本里保留，不要塞进主方法。
- 当前主方法应保留两个有意义的模块：
  - M1：从验证集估计医学诊断相关的客户端信息/权重。
  - M2：基于验证集选择或执行保守 checkpoint 融合。
- M1、M2 都应该在消融中体现作用；如果某模块无作用或负优化，应修改或删除。

### 医学专属性

- 方法要有医学图像专属性，但不能变成某个数据集的特判。
- 医学专属性应来自影像域共性：形态、边界、纹理、局部对比、类别稀缺、困难样本、跨中心成像差异等。
- 不能迁移到 NLP 的论据要讲清楚：这些证据特征依赖像素空间、成像物理和病灶形态；文本任务没有对应的局部边界、声学噪声、ROI 形态和成像伪影。

### 实验优先级

- 第一优先级是优化超声数据集。
- 如果超声没有优化，没有必要跑其他数据集，应撤回或继续改代码。
- 超声提升后，再检查普通医学图像数据集不能明显下降。
- 可以接受一两个格子没有超过现有方法，但不能大面积退化。
- 全量运行前必须先小规模 smoke，确认方向正确。

### 服务器与运行约束

- 可以用 GPU。
- 不要让服务器过载。
- 跑法应保守：单 GPU、少进程、`num_workers=0`，先跑小规模，再扩展。
- 以前全量能跑通，说明不要通过盲目加并发或自定义数据读取把机器压爆。

### 结果整理

- 汇总表和图必须自动生成，不能手动填表。
- `My_merge_ret/汇总表.md` 放在 `My_merge_ret` 最外层，便于查看。
- 图统一放在文件夹里，不要把大量图片散落在最外层。
- 本体和消融都要输出模型权重/候选信息，用于深入分析。
- 权重分析不要堆大表；需要用更可读的比值、集中度、熵或图展示“方法是否真的区分模型”。

### 其他已提出要求

- 复现其他方法只需要证明能成功复现，不需要把所有基线重跑一遍；`result` 里的已有模型结果视为确定基线。
- 曾要求制作 PPT 和演讲稿，逻辑必须简洁：发现什么医学问题，设计什么方法，流程图和算法核心是什么，为什么不是联邦学习，为什么不能用于 NLP。
- `my_merge_extra` 曾被要求做成像 `avg` 一样可直接调用的完整方法，但当前用户后来明确：`extra` 先不管，第一任务是把 `my_merge` 做好。

## 2. 当前代码状态

主要文件：

- `methods/my_merge.py`
- `merge.py`
- `scripts/run_all_avg_eval.py`
- `My_merge_ret/汇总表.md`
- 诊断/可视化脚本在 `scripts/` 和 `My_merge_ret/reports/` 下。

当前 `methods/my_merge.py` 已经从“复杂医学融合大杂烩”收敛为更保守的两模块方向：

### M1：Diagnostic Evidence Client Information Estimation

入口大致在 `_module1_diagnostic_client_information_estimation`。

作用：

- 使用正式 runtime 读取验证集数据。
- 提取医学图像证据特征、类别稀缺、困难样本和预测 margin。
- 计算 `overall_weights`、`morphology_weights`、`class_weights`。
- 记录 `label_coverage_ratio`、`evidence_support_gate`、`client_specialization_ratio` 等诊断信息。

重要改动：

- `DEFAULT_STATS_MAX_BATCHES` 改为 `16`。
- 原因是发现 `chaoshengmnist_224` 的 `val` 前缀类别覆盖不足：
  - 1 batch 基本只有 0 类；
  - 4 batch 仍覆盖不全；
  - 16 batch 才能覆盖更多类别。
- 这不是偷看 test，而是为了避免在合法 `val` 统计时被类别顺序误导。

### M2：Validated Conservative Checkpoint Fusion

入口大致在 `_validated_standard_checkpoint_merge` 和 `_module2_medical_evidence_guided_fusion_and_selection`。

当前思路：

- 先构造 `avg` checkpoint。
- 在满足条件时构造 `sign_consistent_delta` 和 `avg_sign_blend_0p25`。
- 只在验证集 batch 上评估候选，按 `selection_score = 0.85 * val_acc + 0.15 * val_medical_acc - 0.001 * val_loss` 选候选。
- 之后可做 BN recalibration。
- 不再使用原先容易过拟合/负优化的层路由、锚点强制、稀疏残差、prototype head、specialist candidate 等复杂机制。

注意：

- 当前代码还保留了一些历史函数，例如旧的 layer/head routing 相关函数，但默认主路径已转向保守候选选择。
- 这些历史函数之后应继续清理，避免主方法看起来过长。

## 3. 已发现的问题

### 3.1 原 M2 存在负优化

原先的 `_stable_evidence_weights` 会对 M1 权重做二次 softmax。实际观察中，中等差异的权重可能被放大成极端权重，例如从约 `0.43` 放大到约 `0.92`，导致单客户端接管融合。

这在超声上尤其危险：超声存在强散斑噪声、低对比、跨设备增益差异。如果 M1 证据本身不稳定，M2 再强行放大，就会把噪声风格当成诊断优势。

### 3.2 超声验证集前缀有类别覆盖问题

小 batch smoke 曾经被 `chaoshengmnist_224` 的 `val` 排列误导。只取前几个 batch 时类别不全，导致权重/候选选择失真。

因此当前把 M1 统计 batch 默认提高到 16。这个改动仍然只看 `val`，符合合法数据使用方式。

### 3.3 过多模块会制造过拟合风险

之前堆了候选池、anchor、稀疏残差、prototype、specialist 等模块。用户明确指出这会导致：

- 方法不像正常融合方法；
- 代码过长；
- 容易对单个数据集形成捷径；
- 消融中可能出现“去掉模块反而更好”。

当前方向是删除负优化功能，而不是继续加功能。

## 4. 已跑结果与可信状态

### 4.1 比较可信的小规模 smoke

文件：

- `My_merge_ret/reports/selection_fix_small_smoke_summary.md`
- `My_merge_ret/reports/selection_fix_small_smoke_summary.csv`

范围：

- `small / resnet / clients=3 / beta=0 / seed=42`

结论：

- 5 个 checked dataset 均高于同格已有 best formal result。
- `chaoshengmnist_224` 上：
  - full：`0.3890`
  - no_client_information：`0.3720`
  - no_fusion_selection：`0.2552`
  - avg_only：`0.2552`
- 该 smoke 中 M1、M2 都有可见作用。

注意：

- 这是 smoke，不是全量。

### 4.2 超声 convnext probe 中较有价值的一轮

目录：

- `outputs/my_merge_probe_validated_select_20260531_150416`

结果摘录：

| dataset | model | clients | beta | acc |
|---|---|---:|---:|---:|
| chaoshengmnist_224 | convnext | 3 | 0.0 | 0.173405 |
| chaoshengmnist_224 | convnext | 5 | 0.0 | 0.173405 |
| chaoshengmnist_224 | convnext | 5 | 0.01 | 0.172507 |
| chaoshengmnist_224 | convnext | 5 | 0.1 | 0.173405 |
| chaoshengmnist_224 | convnext | 7 | 0.0 | 0.173405 |
| chaoshengmnist_224 | convnext | 7 | 0.1 | 0.161725 |

这轮相对稳定，说明“保守候选 + val 选择”的方向有价值。

### 4.3 2026-05-31 晚间已修复的不稳定点

目录：

- `outputs/my_merge_chaosheng_convnext_signfix2_20260531_150557`

观察：

- 有些格子退化到 `0.105121`，例如：
  - clients=3, beta=0.0
  - clients=5, beta=0.01
  - clients=7, beta=0.01
- 这轮现在作为历史坏例保留。2026-05-31 晚间已经定位为两类问题：
  - `_use_sign_consistent_delta` gate 过窄，导致候选池只剩 `avg`。
  - `clients=7,beta=0.01` 需要 M1 诊断权重融合候选，仅靠 sign/blend 不够。

下一位接手时，应优先对比：

- `outputs/my_merge_probe_validated_select_20260531_150416`
- `outputs/my_merge_chaosheng_convnext_signfix2_20260531_150557`
- `outputs/codex_bn_aligned_ultrasound_20260531_2040`

重点检查每个 `meta.json` 里的：

- `selected_candidate`
- `candidate_pool`
- `candidate_metrics`
- `overall_weights`
- `label_coverage_ratio`
- `fallback_reason` 是否存在

## 5. 当前运行状态

2026-05-31 晚间最终检查时没有发现后台 Python 实验进程在跑。

如需接着跑，先再次检查进程，避免重复任务：

```bash
ps -ww -C python -C python3 -o pid,ppid,stat,etime,pcpu,pmem,args
```

代码语法检查已通过：

```bash
python3 -m py_compile methods/my_merge.py merge.py scripts/run_all_avg_eval.py
```

## 6. 下一步建议

### 第一步：跑完整超声小 probe

先不要跑全量。

建议先跑：

- `chaoshengmnist_224`
- `convnext`
- clients/beta 关键格子：`(3,0)`, `(3,0.01)`, `(3,0.1)`, `(5,0)`, `(5,0.01)`, `(5,0.1)`, `(7,0)`, `(7,0.01)`, `(7,0.1)`
- `num_workers=0`
- 单 GPU
- `stats_split=val`
- `my_merge_stats_max_batches=16`
- `my_merge_bn_batches=4`

目标：

- 不再出现 `0.105121` 这种明显退化格子。
- full 至少不低于 avg/已有方法太多，并尽量超过。
- 消融中 M1、M2 要有可见贡献。

### 第二步：再跑普通数据集 smoke

超声稳定后，再跑普通医学图像数据集小规模检查：

- `bloodmnist_224`
- `dermamnist_224`
- `organcmnist_224`
- `organsmnist_224`

目标：

- 不能大面积下降。
- 少数格子没超过可以接受，但不能为了超声把其他数据集改坏。

### 第三步：清理主方法长度

用户明确要求 `my_merge` 像正常方法，不要近 2000 行。

建议：

- 主方法只保留融合必需逻辑。
- 权重统计、图像源分析、可视化、候选诊断移到 `scripts/`。
- 删除不再走默认路径的历史复杂函数。
- 保留 `my_merge_extra` 但当前先不作为重点。

### 第五步：全量与汇总

只有当超声和普通数据集 smoke 都过关后，再跑全量。

全量完成后：

- 用脚本更新 `My_merge_ret/汇总表.md`。
- 图放在 `My_merge_ret/figures/` 或其子目录。
- 权重分析放在 `My_merge_ret/reports/` 和对应图目录中。
- 不手动填表。

## 7. 交接提醒

- 不要随便 `git checkout` 或 `git reset`，当前仓库有大量用户/历史改动。
- 不要删除已有结果目录，除非用户明确要求。
- 不要先跑大规模任务。先修逻辑，再小跑。
- 当前方向的关键不是“再发明一个复杂医学网络”，而是让 checkpoint merge 本身合理、简洁、可验证。
- 用户最在意的是可信：合法看数据、删除负优化、不特判、不把联邦学习写成模型融合。
