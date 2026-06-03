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

## 2026-06-01 全量运行记录

用户已经要求直接看全量数据。基于 2026-05-31 晚间修复后的 smoke 结果，本轮进入全量实验，但需要记录一个环境限制：

- `nvidia-smi` 当前不能连接 NVIDIA driver。
- `.gpuenv` 内 PyTorch 检测为 `cuda=False, device_count=0`。
- 因此本轮先按 CPU 启动全量；如果 GPU 恢复，可以用相同输出目录和 `--resume` 续跑。

本轮全量优先跑 `my_merge` 自身及消融，不重新全量复现所有 baseline。对比 baseline 仍使用仓库已有 `result/all_results.md`，这样可以更快看到当前方法代码在所有任务上的表现。

计划配置：

- 数据集：`bloodmnist_224 dermamnist_224 organcmnist_224 organsmnist_224 chaoshengmnist_224`
- small 模型：`resnet convnext vit_t swin_tiny`
- VLM：`openai/clip-vit-base-patch32`
- 消融：`full no_client_information no_fusion_selection avg_only`
- 统计 split：`val`
- `my_merge_stats_max_batches=16`
- `my_merge_bn_batches=4`
- `my_merge_eval_max_batches=1` 目前代码中未实际使用，保留脚本参数记录。
- `num_workers=0`
- `merge_weight_mode=equal`

本轮不再继续加新模块。若全量发现坏格子，优先检查候选选择、BN 对齐、数据读取和负优化模块，再决定删改。

启动记录：

- 启动时间：2026-06-01 09:26:56 CST
- 主输出：`outputs/codex_my_merge_full_cpu_20260601_0924/my_merge_ablation_grid`
- 总日志：`logs/codex_my_merge_full_cpu_20260601_0924`
- `RUN_REPRO=false`，本轮不重复跑 baseline 复现。
- `PUBLISH_RESULTS=false`，先不覆盖 `My_merge_ret/汇总表.md`；等结果检查后再决定是否发布。

CPU 目录状态：

- 由于沙箱内 PyTorch 看不到 GPU，CPU 每个 resnet 格子耗时约 6 到 12 分钟。
- 该 CPU 运行已停止；停止前完成 `full/small_resnet` 的 17 个格子，可作为参考，但不作为正式全量目录。
- 正式全量改用沙箱外 GPU 运行，避免 CPU 目录半截结果混入最终汇总。

GPU 正式全量计划：

- 运行目录：`outputs/codex_my_merge_full_gpu_20260601_1400`
- 日志目录：`logs/codex_my_merge_full_gpu_20260601_1400`
- 启动时间：2026-06-01 13:59:45 CST
- 设备：2 x NVIDIA GeForce GTX 1080 Ti
- PyTorch CUDA 状态：`cuda=True, device_count=2`
- 并行：`GPU_IDS="0 1"`, `MAX_PARALLEL_JOBS=2`

阶段性进度：

- 2026-06-01 14:08 CST：`full` 已完成 13 个 small 格子，其中 resnet 8 个、convnext 5 个。
- GPU 运行耗时明显低于 CPU：resnet 单格约 58 到 65 秒，convnext 单格约 90 到 100 秒。
- 早期 `bloodmnist_224/convnext/c3_b0` 的 `0.1947` 与 baseline `avg=0.1947` 对齐，不是新异常。
- 2026-06-01 14:33 CST：`full/resnet` 45 个格子完成，GPU0 已切到 `vit_t`；`full/convnext` 完成 24/45。
- 关键复现点：`chaoshengmnist_224/convnext/c7_b0.01=0.161725`，不再退回坏例 `0.105121`。
- 2026-06-01 15:00 CST：`full` small 完成 117/180；resnet 45/45，convnext 39/45，vit_t 33/45。
- 2026-06-01 15:12 CST：`full/convnext` 和 `full/vit_t` 均完成；GPU1 开始 `swin_tiny`，GPU0 开始 VLM `clip-vit-base-patch32`。
- 2026-06-01 15:18 CST：`full` 已完成 141/225；`swin_tiny` 3/45，VLM 3/45。计数方式为 `resnet45 + convnext45 + vit_t45 + swin_tiny + VLM`。
- 2026-06-01 15:34 CST：`full` 已完成 165/225；`swin_tiny` 15/45，VLM 15/45。
- 2026-06-01 15:59 CST：`full` 已完成 200/225；`swin_tiny` 33/45，VLM 32/45。日志仍在正常推进，没有出现失败记录。
- 2026-06-01 16:07 CST：`full` 已完成 211/225；`swin_tiny` 38/45，VLM 38/45。两个任务都已进入 `organsmnist_224` 后段。
- 2026-06-01 16:19 CST：`full` 225/225 完成，脚本自动进入 `no_client_information`。下一步先保留 `full` 完整表用于和 baseline 对齐，再继续等待消融结果。
- `full` 快照汇总：`reports/ablation_summary_snapshot_after_full.md`。相对 `result/all_results.md` 中每格最佳原方法，整体 `mean_delta=-0.0079`，W/T/L=`54/56/115`，说明当前方法还没有达到“整体基本超过最优 baseline”的目标。
- 分模型初判：`resnet` 平均 `+0.0780`，`convnext` 约 `-0.0016`；`vit_t=-0.0338`、`swin_tiny=-0.0432`、VLM `clip-vit-base-patch32=-0.0389`。后续优化应优先查架构敏感的候选选择和校准，而不是继续堆新模块。
- 当前最大坏例包括 `vit_t/dermamnist_224/c3_b0.1`、`swin_tiny/dermamnist_224/c5_b0.1`、VLM `bloodmnist_224/c5_b0.01`、以及 `organsmnist_224` 的若干低值。先等 `no_client_information/no_fusion_selection/avg_only` 消融确认哪一块在拖后腿。
- 2026-06-01 16:25 CST：`no_client_information` 已开始；`resnet` 7/45，`convnext` 4/45。该组用于判断 M1 client 信息加权是否造成负优化。
- 2026-06-01 16:27 CST：抽查坏例 `merge_result.json`。`vit_t/dermamnist_224/c3_b0.1` 选择 `sign_consistent_delta`，候选 val 分最高也只有约 `0.082`；`swin_tiny/dermamnist_224/c5_b0.1` 四个候选 val 几乎并列约 `0.1035`；VLM `bloodmnist_224/c5_b0.01` 选择 `medical_weighted_fusion`，val 从 `avg=0.0625` 提到 `0.1426`，但 test 仍落后旧最佳。结论：这些坏例不像单纯权重微调，可能是候选池缺少能复现旧最佳偏置/退化解的医学合理版本，等消融后再决定是否收紧或删候选。
- 2026-06-01 16:32 CST：`no_client_information` 已完成 35/225。关键超声 `convnext/chaoshengmnist_224/c3_b0.01=0.173405`，与 `full` 一致，说明该超声修复不是只靠 M1 client 信息权重。
- 2026-06-01 16:43 CST：`no_client_information/resnet` 45/45 完成。与 `full/resnet` 对齐后平均 `delta=-0.0168`，45 格中 14 格变化；M1 client 信息对 resnet 总体有正贡献，不能简单删除。
- 2026-06-01 18:31 CST：半小时巡检。`no_client_information` 225/225 完成；`no_fusion_selection` 已开始并完成 58/225。中间汇总 `reports/ablation_summary_snapshot_1831.md` 显示 `no_client_information` 平均 `delta_vs_full=-0.0393`，W/T/L=`35/39/151`，因此 M1 client 信息总体应保留。`no_fusion_selection` 当前局部 `delta_vs_full=-0.1503`，先等全量后确认，但初步说明 M2 选择机制也不是无效模块。
- 2026-06-01 19:04 CST：按用户要求只做运行巡检，不继续分析或改代码。`no_fusion_selection` 126/225；`full` 和 `no_client_information` 均已完成，最终汇总尚未生成。
- 2026-06-01 19:35 CST：`no_fusion_selection` 183/225；当前后台仍在跑 `swin_tiny` 和 VLM，最终汇总尚未生成。
- 2026-06-01 20:06 CST：`no_fusion_selection` 219/225；只剩 VLM 6 个格子，最终汇总尚未生成。
- 2026-06-01 22:27 CST：按用户指示，`avg_only` 不再实际跑 GPU，直接复用 `result/all_results.md` 中 `avg` 行生成 `avg_only` 结果 CSV。四组结果均为 225/225：`full`、`no_client_information`、`no_fusion_selection`、`avg_only`。最终汇总已生成：`reports/ablation_summary.md` 和 `reports/all_results_ablation_combined.md`。后台 `run_all_avg_eval.py` 已无进程。
- 不要偷看 test 选参数。
- 不要按数据集名称写特判。
- 不要把方法重新写成联邦学习。
- 不要继续堆新模块掩盖负优化。
- 不要开高并发让服务器过载。

## 2026-06-02 M1/M2 改造记录

用户提出的新方向包括：M1 从绝对空间层级加权转向 delta 空间权重，M2 从单候选硬筛选转向 top-2 soup，并把 M1 医学权重注入 sign-delta。

本轮代码改动：

- `methods/my_merge.py` 增加 `DEFAULT_RELIABILITY_THRESHOLD=0.10`、`DEFAULT_SEPARATION_THRESHOLD=0.05`，降低 M1 权重 gate。
- `medical_weighted_fusion` 的实现改成 `_medical_delta_weighted_state_merge`，以 reference checkpoint 为原点融合 delta。
- `sign_consistent_delta` 不再使用 `base_weights`，改用 `_medical_consensus_weights(overall_weights, morph_weights, base_weights)`。
- M2 候选排序后，如果前两名 `selection_score` 差距不超过 `0.05`，生成并选择 `top2_soup:<候选1>+<候选2>`。
- 实现版本标记更新为 `medical_evidence_two_module_posthoc_merge_v8_delta_soup`。

判断记录：

- 单纯把归一化加权平均写成 `reference + sum_i w_i * delta_i` 在数学上等价于直接加权平均，不应把它当成必然提升点。
- 本轮真正可能改变结果的点是：M1 医学共识权重进入 sign-delta，以及 M2 由单一 winner 改为接近候选的 top-2 soup。

验证记录：

- 2026-06-02 00:48 CST：`py_compile` 通过：`methods/my_merge.py merge.py scripts/run_all_avg_eval.py`。

## 2026-06-03 smoke 结果与修复记录

改版后第一次 GPU smoke 因用户中断只完成了 `convnext` 的 12/18 个格子，`vit_t` 和 `swin_tiny` 未跑完。有效格子相对 2026-06-01 旧版 full：10 个持平、1 个小幅提升、1 个严重下降。

严重下降格子：

- `dermamnist_224 / convnext / clients=3 / beta=0.01`：旧版 test acc `0.668828`，新版 smoke test acc `0.109726`。
- 旧版该格子选择 `medical_weighted_fusion`，其 val acc 为 `0.687500`。
- 新版该格子选择 `top2_soup:avg+avg_sign_blend_0p25`，且新版 `medical_weighted_fusion` 的 val acc 掉到 `0.103516`。

定位结论：

- `_weighted_delta_for_key` 使用 `reference_value.detach().float().zero_()` 会在 float32 reference tensor 上原地清零，污染 `reference_state`。这会破坏后续候选，尤其是 sign-delta，也会让 delta 版本的 `medical_weighted_fusion` 行为不可信。
- top-2 soup 原逻辑在前两名接近时无条件接管，即使 soup 自己的 `selection_score` 低于第一名，也会被选中。这不符合保守验证模块的目标。

本步修复：

- `_weighted_delta_for_key` 改为 `torch.zeros_like(reference_float)` 新建累加张量，不再修改 reference。
- top-2 soup 仍会被构造和评估，但只有 `soup.selection_score >= top.selection_score + my_merge_soup_min_score_delta` 时才选中；默认 `my_merge_soup_min_score_delta=0.0`。
- 实现版本标记更新为 `medical_evidence_two_module_posthoc_merge_v8_delta_soup_guarded`。

修复后 smoke：

- 路径：`outputs/codex_my_merge_v8_guarded_smoke_20260603_0025/full_convnext_critical`。
- 范围：`dermamnist_224`、`chaoshengmnist_224`，`convnext`，共 18 个格子。
- 状态：18/18 OK。
- 相对 2026-06-01 旧版 full：`wins/ties/losses = 1/17/0`，平均 delta `+0.000050`。
- 刚才的坏例 `dermamnist_224 / convnext / clients=3 / beta=0.01` 已从坏版 `0.109726` 回到 `0.668828`，与旧版 full 持平。
- 超声 `chaoshengmnist_224 / convnext` 基本无变化，只有 `clients=5 / beta=0.01` 从旧版 `0.172507` 到 `0.173405`，其余持平。因此当前修复解决了负优化 bug，但还没有证明能改善超声主问题。
