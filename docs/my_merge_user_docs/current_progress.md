# 当前进度

更新时间：2026-06-05

## 2026-06-05 精简与全量前状态

本轮目标是把 `my_merge` 收成清晰的科研代码，并启动新的全量实验。当前已经完成：

- `methods/my_merge.py` 从上一轮约 1500 行压到 780 行，主路径只保留 M1/M2/M3 三个模块。
- M1 保留医学图像证据和客户端诊断权重估计；删除对结果没有实际作用的覆盖率混合字段。
- M2 保留两个基础融合候选：医学权重增量融合 `medical_weighted_fusion` 和符号一致增量融合 `sign_consistent_delta`。
- M3 默认开启，只做自适应候选生成和验证集选择；正式消融为 `no_adaptive_candidates`。
- 删除了 M3 子候选的隐藏命令行开关，避免通过超参数绕过论文里的模块定义。
- `scripts/generate_ablation_combined_results_table.py` 只生成总表，不再把超声单独分析、smoke、权重分析写到 `汇总表.md`。
- `My_merge_ret/汇总表.md` 已按旧全量结果重新生成一次，目前只保留 `Ablation Rows`、`Small`、`VLM` 总表结构。

当前设计判断：

- 不再给 `chaoshengmnist_224` 单独分支，所有医学数据集走同一套 M1/M2/M3。
- `avg_only` 只作为 `-M1,-M2,-M3` 的消融基线，正式全量不再运行；总表生成时直接复用 baseline `avg` 行。
- 后续只看总表判断是否负优化；额外诊断表不再发布到主汇总表。

全量运行：

- 已启动：`codex_my_merge_compact_full_20260605`
- 输出目录：`outputs/codex_my_merge_compact_full_20260605/my_merge_ablation_grid`
- 发布总表：`My_merge_ret/汇总表.md`
- 运行消融：`full no_client_information no_fusion_selection no_adaptive_candidates`
- 不运行：`avg_only`，由总表生成脚本复用 baseline `avg`。
- 初始检查：`full/small/resnet` 和 `full/small/convnext` 已启动并占用 GPU，输出目录已产生首个 `eval.json`。
- 完成状态：2026-06-06 已补齐到 `900/900` 个 eval，并重新生成 `My_merge_ret/汇总表.md`。
- 全量结论：`full` 平均准确率 `0.3191`，相对已有最佳原始方法均值差 `-0.0012`，W/T/L=`85/38/102`。
- 消融结论：去掉 M2 平均下降 `0.0964`，去掉 M3 平均下降 `0.0457`，去掉 M1 平均下降 `0.0095`。总体上 M2、M3 有用，M1 较弱。
- 超声结论：`chaoshengmnist` 仍是主要问题，`full` 只有 `1/0/44`，且 `-M3` 比 full 高 `0.0551`、`-M2` 比 full 高 `0.0381`，说明当前 M2/M3 在超声上仍有负优化。

## 2026-06-06 精简回归定位

用户指出精简前超声结果更好。回看旧代码和新结果后确认一个明确回归：

- 精简前 `_validated_standard_checkpoint_merge` 会把 `avg` 放入候选池，验证集可以在 `avg`、医学加权融合、sign delta、子集候选之间选择。
- 精简后 `_build_m2_m3_candidates` 只放入 M2/M3 候选，遗漏了 `avg`，导致只要 M2/M3 打开，超声就没有退回平均模型的机会。
- 这解释了为什么 `chaoshengmnist` 上 M2/M3 看起来负优化：验证选择被迫在一组可能已经坏掉的候选里选最不坏的，而不能选 `avg`。

已修复：

- 在候选池中恢复 `avg`。
- `module2_candidate_pool` 仍只记录 M2 候选，不把 `avg` 当成 M2 模块。
- 语法检查通过：`py_compile methods/my_merge.py scripts/run_all_avg_eval.py scripts/generate_ablation_combined_results_table.py`。
- 后续超声全量显示 full 仍为 `0.1170`，原因是 M3 的 `head_prior_repair` 在一个很小的 val batch 上把 `val_acc` 修到 `1.0`，但 test 崩掉。
- 已删除 `head_prior_repair`。单格 smoke `chaoshengmnist_224 / resnet / c3_b0` 从 `0.1087` 恢复到 `0.3046`。

已启动超声全量验证：

- run tag：`codex_my_merge_chaosheng_no_head_repair_full_20260606`
- 数据集：`chaoshengmnist_224`
- 范围：small 的 4 个 backbone + VLM CLIP，4 个消融，共 `180` 个 eval。
- 输出目录：`outputs/codex_my_merge_chaosheng_no_head_repair_full_20260606/my_merge_ablation_grid`
- VLM smoke 已确认 `avg` 回到候选池，且至少一个格子重新选回 `avg`。

结果显示这还没有真正恢复：

- `full` 超声 45 格均值为 `0.1742`，仍低于 2026-06-01 旧全量的 `0.2253`。
- 关键原因不是单纯候选池，而是本轮全量脚本默认用了 `my_merge_stats_max_batches=1`、`my_merge_bn_batches=1`。同一个 `chaoshengmnist_224/resnet/c3_b0` 格子里，候选验证集 `val_acc` 全部为 `0`，选择退化成按 loss 排名，M1 权重也退化成均匀。
- 2026-06-01 旧全量配置是 `my_merge_stats_max_batches=16`、`my_merge_bn_batches=4`，该口径下 M1 权重非均匀，候选验证有效。

已进一步修复：

- `scripts/run_validated_my_merge_full.sh` 和 `scripts/run_my_merge_ablation_split_grid.sh` 默认恢复为 `stats=16`、`bn=4`。
- 删除 M3 中的 `adaptive_subset_top2/top3` 和 `adaptive_sign_sparse_0p2` 搜索候选。
- 恢复旧版有效的保守候选 `avg_sign_blend_0p25`，作为精简后的唯一 M3 候选：在 `avg` 和 `sign_consistent_delta` 之间做 25% 小步混合。
- `head_prior_repair`、subset、sparse 相关代码均已删除，`methods/my_merge.py` 当前为 727 行。
- 语法检查通过：`py_compile methods/my_merge.py scripts/run_all_avg_eval.py`。

修复后 smoke：

- `chaoshengmnist_224 / resnet / 9格 / stats=16 / bn=4`
- 输出目录：`outputs/codex_my_merge_blend_stats16_chaosheng_resnet9_20260606`
- 新均值 `0.3883`，高于 2026-06-01 旧全量同 9 格均值 `0.3726`，远高于错误口径 `0.2687`。
- 候选池恢复为 `avg`、`medical_weighted_fusion`、`sign_consistent_delta`、`avg_sign_blend_0p25`。

已启动新的超声全量：

- run tag：`codex_my_merge_chaosheng_blend_stats16_full_20260606`
- 数据集：`chaoshengmnist_224`
- 范围：small 的 4 个 backbone + VLM CLIP，4 个消融，共 `180` 个 eval。
- 口径：`my_merge_stats_max_batches=16`、`my_merge_bn_batches=4`、`stats_split=val`。
- 输出目录：`outputs/codex_my_merge_chaosheng_blend_stats16_full_20260606/my_merge_ablation_grid`

全量结果：

- 已完成 `180/180`，日志未发现 `fail (`。
- `full` 均值 `0.2317`，相对已有最佳原始方法均值 `+0.0105`，W/T/L=`18/8/19`。
- 2026-06-01 旧版超声 `full` 均值为 `0.2253`，本轮高 `+0.0064`。
- 错误口径 no-head-repair 版超声 `full` 均值为 `0.1742`，本轮高 `+0.0575`。

按模型拆分：

| model | new full | old 2026-06-01 | bad stats1 | new-old |
|---|---:|---:|---:|---:|
| convnext | 0.1708 | 0.1707 | 0.1251 | +0.0001 |
| resnet | 0.3883 | 0.3726 | 0.2687 | +0.0158 |
| swin_tiny | 0.2359 | 0.2251 | 0.1444 | +0.0108 |
| vit_t | 0.2008 | 0.1952 | 0.1827 | +0.0056 |
| CLIP ViT-B/32 | 0.1628 | 0.1631 | 0.1503 | -0.0003 |

消融结果：

| ablation | mean_acc | delta_vs_full | W/T/L |
|---|---:|---:|---:|
| full | 0.2317 | 0.0000 | 18/8/19 |
| no_adaptive_candidates (-M3) | 0.2302 | -0.0015 | 16/8/21 |
| no_client_information (-M1) | 0.2019 | -0.0298 | 7/9/29 |
| no_fusion_selection (-M2) | 0.1551 | -0.0766 | 1/5/39 |

候选选择：

- `full`：`medical_weighted_fusion` 19 次、`avg` 11 次、`avg_sign_blend_0p25` 8 次、`sign_consistent_delta` 7 次。
- `-M3`：`medical_weighted_fusion` 23 次、`avg` 13 次、`sign_consistent_delta` 9 次。
- M3 的贡献很小，但不再负优化；M1、M2 已重新表现为明确正贡献。

生成文件：

- 消融汇总：`outputs/codex_my_merge_chaosheng_blend_stats16_full_20260606/my_merge_ablation_grid/reports/ablation_summary.md`
- 本轮 combined 总表：`outputs/codex_my_merge_chaosheng_blend_stats16_full_20260606/my_merge_ablation_grid/reports/all_results_ablation_combined.md`
- 主汇总表已更新：`My_merge_ret/汇总表.md` 保留 `outputs/codex_my_merge_compact_full_20260605` 的非超声结果，并用本轮 `codex_my_merge_chaosheng_blend_stats16_full_20260606` 覆盖 `chaoshengmnist_224` 的 `my_merge` 四个消融行。

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

## 2026-06-03 超声 profile 改造尝试与回删记录

用户提出 M1/M2 在超声上可能把 speckle 噪声当作诊断证据，导致 M1 gate 和 M2 delta/soup 放大噪声。代码核对后确认这些风险在当前实现中存在：

- M1 `_generic_medical_features` 使用 Sobel、局部对比和局部方差，缺少 speckle-aware 去噪，也没有声影/强回声证据。
- M1 focal 权重固定为 `(1 - margin)^1.35`，超声低 margin 时容易无差别抬高 hard/focal 权重。
- M2 sign-delta 只用全局 `preserve_density=0.5`，没有区分浅层纹理层和深层语义层。
- M2 selection score 固定 `0.85 * val_acc + 0.15 * val_medical_acc - 0.001 * val_loss`，top-2 soup 只做分数守门，未考虑两个 checkpoint 的权重距离。

尝试过的 v9 代码改动：

- M1 特征从 7 维扩展到 11 维，新增 `speckle_noise`、`acoustic_shadow`、`hyperechoic_response`、`ultrasound_profile`。
- 超声 profile 不是用 `dataset == chaoshengmnist_224` 触发，而是由图像边缘残差和 Lee-like despeckle 后的 speckle 估计得到；profile 高时降低 Sobel/局部方差的主导性，增加声束方向衰减和强回声响应。
- M1 scoring 根据 `ultrasound_profile` 自动降低 evidence floor、降低 focal 幂次和 focal 占比、提高 diagnostic weight softmax temperature，并加重未见类别惩罚。
- M2 selection score 根据 `ultrasound_profile` 把 `val_medical_acc` 权重从 `0.15` 逐步提高到最高 `0.40`。
- M2 在 `ultrasound_profile >= 0.35` 时加入 `sign_ultrasound_sparse` 和 `sign_ultrasound_balanced` 两个层级稀疏候选：浅层更激进裁剪，深层保留更高密度。
- Guarded Soup 在 profile 高时要求至少 `+0.003` 的 selection score 增益，并限制 top-2 候选的相对权重距离不超过 `0.08`。

验证结果：

- 路径：`outputs/codex_my_merge_v9_ultrasound_smoke_20260603_1200/full_convnext_critical`。
- 范围：`dermamnist_224`、`chaoshengmnist_224`，`convnext`，共 18 个格子。
- 状态：18/18 OK，但相对 v8 guarded smoke 明显负优化。
- 代表坏例：`dermamnist_224 / convnext / clients=3 / beta=0.01` 从 v8 的 `0.668828` 掉到 `0.109726`。
- 相对 v8 guarded：`wins/ties/losses = 0/1/17`，平均 delta `-0.031061`。

失败原因：

- 图像统计 profile 触发方向错误：`chaoshengmnist_224` 的 `ultrasound_profile` 只有约 `0.022`，没有触发超声分支；`dermamnist_224` 反而达到约 `0.519`，错误触发超声分支。
- 这说明用当前的边缘残差和 Lee-like despeckle 统计去识别“超声物理特性”不可靠，会把皮肤镜纹理误当作 speckle。
- 层级稀疏和更严 soup guard 的想法本身仍可作为后续方向，但必须先有可靠的医学域证据；不能建立在误触发 profile 上。

处理结论：

- v9 的 `speckle_noise`、`acoustic_shadow`、`hyperechoic_response`、`ultrasound_profile`、`sign_ultrasound_sparse`、`sign_ultrasound_balanced` 已从代码删除。
- 当前代码回到 v8 guarded：保留 delta reference 修复、M1 医学共识权重注入 sign-delta、guarded top-2 soup。
- 后续若继续做超声专用优化，优先用可验证的 ultrasound-only 数据增强/验证协议或明确的成像元信息，不再用这版图像统计 profile 作为 gate。

## 2026-06-03 可控超声单独处理开关

用户允许对 `chaoshengmnist_224` 单独处理，但要求可控、可关闭，并验证是否有效。

本步设计：

- 新增显式开关 `my_merge_ultrasound_specialist`，默认关闭。
- 批量脚本新增 `--my-merge-ultrasound-specialist` / `--no-my-merge-ultrasound-specialist`。
- 开关只有在 `meta["dataset"] == "chaoshengmnist_224"` 时生效；其他医学数据集即使传入开关也不走超声分支。
- 不改数据读取、不改 split、不看 test 做选择，仍然只用 `val` 选择候选，`test` 只做最终评估。
- 超声开关打开且未显式传 `my_merge_stats_max_batches` 时，M1/M2 统计默认使用全量 `val`，避免只看前 16 个 batch 时类别覆盖不足；仍可通过命令行覆盖。
- 不再引入 v9 的图像统计 `ultrasound_profile`，避免把 `dermamnist_224` 误判为超声。

本步代码策略：

- M1：超声开关打开时，对 hard/focal 权重和 M1 权重 softmax 做轻度平滑，减少 speckle/低 margin 样本无差别放大。
- M2：增加保守候选 `ultrasound_avg_weighted_blend_0p5`、`ultrasound_avg_weighted_blend_0p75`、`ultrasound_avg_sign_blend_0p10`，让 val 可以在 avg、医学加权、轻量 sign 注入之间选择。
- M2：超声开关打开时默认禁用 top-2 soup，避免小验证集近分数候选直接平均造成测试回退。
- M2 selection score：超声开关打开时默认把 `val_medical_acc` 权重从 `0.15` 降到 `0.08`，更偏向普通 val acc，减少通用医学证据在超声噪声上的误导。

待验证：

- 先跑 `chaoshengmnist_224 / convnext` smoke，对比 v8 guarded。
- 如果没有明显负优化，再跑 `chaoshengmnist_224` 45 格 full。

convnext smoke 结果：

- 路径：`outputs/codex_my_merge_ultrasound_specialist_convnext_20260603`。
- 范围：`chaoshengmnist_224 / convnext`，9 个格子。
- 状态：9/9 OK。
- 相对 v8 guarded convnext smoke：`wins/ties/losses = 0/9/0`，平均 delta `0.0`。
- 相对 `result/all_results.md` formal best：`wins/ties/losses = 7/0/2`，平均 delta `-0.00129`。
- 具体结果：`c3_b0=0.173405`、`c3_b0.01=0.173405`、`c3_b0.1=0.173405`、`c5_b0=0.173405`、`c5_b0.01=0.173405`、`c5_b0.1=0.173405`、`c7_b0=0.173405`、`c7_b0.01=0.161725`、`c7_b0.1=0.161725`。
- 结论：开关生效但 convnext 不改善；未见 v9 那种负优化。继续跑 `chaoshengmnist_224` 45 格，检查其他架构是否受益。

45 格 formal 对齐验证：

- 路径：`outputs/codex_my_merge_ultrasound_specialist_chaosheng45_20260603`。
- 说明：本次命令实际按 manifest 跑了 81 格，其中 45 格与 `result/all_results.md` formal 表对齐；主结论只统计这 45 格。
- 相对 v8 guarded 当前 full：`wins/ties/losses = 21/17/7`，平均 delta `+0.011720`。
- 相对 `result/all_results.md` formal best：`wins/ties/losses = 29/0/16`，平均 delta `+0.015833`。
- 旧 v8 guarded 在同一 45 格上相对 formal best 是 `20/0/25`，平均 delta `+0.004112`；因此超声开关把 formal best 对比从 20 胜提升到 29 胜，平均优势增加约 `+0.01172`。

按模型分解：

| model | mean specialist | mean v8 | mean formal best | vs v8 W/T/L | vs best W/T/L | delta vs best |
|---|---:|---:|---:|---:|---:|---:|
| resnet | 0.414196 | 0.372567 | 0.307178 | 7/0/2 | 9/0/0 | +0.107018 |
| convnext | 0.170810 | 0.170710 | 0.172100 | 1/8/0 | 7/0/2 | -0.001290 |
| vit_t | 0.200559 | 0.195168 | 0.229711 | 5/0/4 | 4/0/5 | -0.029152 |
| swin_tiny | 0.237796 | 0.225117 | 0.217744 | 7/2/0 | 7/0/2 | +0.020051 |
| clip-vit-base-patch32 | 0.161925 | 0.163123 | 0.179389 | 1/7/1 | 2/0/7 | -0.017464 |

候选选择分布：

- `medical_weighted_fusion`：11/45。
- `avg`：9/45。
- `sign_consistent_delta`：8/45。
- `ultrasound_avg_weighted_blend_0p75`：6/45。
- `ultrasound_avg_weighted_blend_0p5`：6/45。
- `avg_sign_blend_0p25`：5/45。

补充：

- 额外 manifest 模型也跑完：`densenet` mean `0.453729`、`efficientnet` mean `0.454228`、`mobilenet` mean `0.360387`、`resnet34` mean `0.426375`。这些不在 `result/all_results.md` formal 表中，暂不纳入论文主对比。
- 结论：这个可控超声分支有价值，应该保留开关；它改善了 formal 对齐 45 格的总体结果，但 `vit_t` 和 `clip-vit` 仍低于现有 best，需要后续专门分析。

## 2026-06-03 超声分支 M2 受控增强预注册

用户提出三条后续建议：early layer detox、hard sparsification、提高 selection medical weight。为避免“面向 test 结果编程”，本轮先固定实验协议再跑数据。

预注册规则：

- 数据读取、`val/test` split、最终 test 评估路径不变。
- 只在 `my_merge_ultrasound_specialist=True` 且 `dataset == chaoshengmnist_224` 时启用新增子项。
- 不按单个 test 格子写特判，不根据某个 beta/client 结果改规则。
- 主指标只看与 `result/all_results.md` 对齐的 45 格：相对当前超声分支基线、相对 v8 guarded、相对 formal best 的 W/T/L 和 mean delta。
- 保留标准：45 格 mean delta 相对当前超声分支不能下降；不能出现某个 formal 模型族明显崩塌；新增候选必须在 val 选择中有实际被选中或至少不伤害总体。
- 若新增项负优化，保留实验记录；无实际价值的代码路径直接删除，避免后续误用。

待试改动：

- Early Layer Detox：超声分支下 `medical_weighted_fusion` 的 early 层使用 `base_weights`，不再混入 `morph_weights`。
- Sparse Sign Candidate：新增超声稀疏 sign-delta 候选，默认 `preserve_density=0.20`，只作为候选进入 val 选择，不替换现有 sign。
- Selection Medical Weight：暂不直接固定为 `0.40/0.50`。先跑结构改动；如需要，再用已有参数 `--my-merge-ultrasound-selection-medical-weight` 做显式小网格。

受控消融结果：

| variant | path | vs current ultrasound baseline | vs v8 guarded | vs formal best | decision |
|---|---|---:|---:|---:|---|
| current ultrasound baseline | `outputs/codex_my_merge_ultrasound_specialist_chaosheng45_20260603` | 0/45/0, +0.000000 | 21/17/7, +0.011720 | 29/0/16, +0.015833 | baseline |
| early detox only | `outputs/codex_my_merge_ultrasound_early_detox_only_chaosheng45_20260603` | 7/29/9, -0.000599 | 20/17/8, +0.011121 | 27/0/18, +0.015234 | removed |
| early detox + sparse sign | `outputs/codex_my_merge_ultrasound_detox_sparse_chaosheng45_20260603` | 10/26/9, +0.002316 | 21/17/7, +0.014036 | 27/0/18, +0.018149 | not default because early detox hurts some cells |
| sparse sign only | `outputs/codex_my_merge_ultrasound_sparse_only_chaosheng45_20260603` | 3/42/0, +0.002915 | 22/17/6, +0.014635 | 29/0/16, +0.018748 | default on |

事实结论：

- 建议一 Early Layer Detox 的直觉有道理，但 45 格消融不支持保留；它单独相对当前超声分支 mean delta 为 `-0.000599`，且 W/T/L 为 `7/29/9`。按“无用模块直接删除”的原则，已移除代码和 CLI 开关，只保留本实验记录。
- 建议二 Sparse Sign Candidate 有事实支持；sparse-only 相对当前超声分支 `3/42/0`，没有 loss cell，mean delta `+0.002915`。
- Sparse candidate 被实际选中：`ultrasound_avg_sparse_sign_blend_0p10_0p2` 3/45，`ultrasound_sign_sparse_0p2` 2/45，不是无效候选。
- 当前默认策略改为：超声分支下保留 `my_merge_ultrasound_sparse_sign=True`；不再提供 early detox 开关。
- 建议三 Selection Medical Weight 暂未固定调高；因为当前 sparse-only 已经带来无负优化提升，且通用 morphology 权重在超声上仍可能被 speckle 误导。若继续试，只做显式参数扫描，不把 `0.40/0.50` 写死。

## 2026-06-03 超声 M1 证据降噪预注册

用户追问“能不能先处理掉噪声，然后再扔进 M1 学习依据”。当前判断：

- 这个方向有价值，但必须限定在 M1 证据提取阶段，不能改变模型训练/验证/test 图像输入，否则会破坏与 `avg` 等方法的数据口径一致性。
- 不能把它写成 test 特判；只在 `my_merge_ultrasound_specialist=True` 且 `dataset == chaoshengmnist_224` 时启用，并提供显式开关。
- 不引入复杂新模块。先做轻量、确定性的 log-domain edge-preserving smoothing：先把超声强度做 log 压缩，使乘性 speckle 更接近加性扰动；再用 5x5 Gaussian 平滑，并用 Sobel 边缘作为保护门控，边缘处少平滑、同质区域多平滑。
- M1 仍输出相同 7 维特征，不改后续融合接口。降噪只改变这些特征的估计来源；候选模型的 val/test forward 仍使用原图。
- 若后续 45 格消融显示负优化，删除代码而不是保留无效模块。

实现状态：

- 曾短暂实现 `my_merge_ultrasound_denoise_evidence`，只在超声 specialist 分支下生效：超声分支先生成 denoised gray，再提取同样 7 维 M1 证据。
- 真实 45 格输出：`outputs/codex_my_merge_ultrasound_denoise_sparse_chaosheng45_20260603`。
- 相对当前 sparse-only 基线：`5/29/11`，mean delta `-0.000419`。分模型看，`resnet` 为 `1/3/5, -0.001398`，`swin_tiny` 为 `1/6/2, -0.000799`，`vit_t` 为 `3/3/3, +0.000200`，VLM 为 `0/8/1, -0.000100`，`convnext` 不变。
- 相对 formal best：`30/0/15`，mean delta `+0.018328`，低于 sparse-only 的 `+0.018748`。
- 结论：这版轻量 log-domain edge-preserving smoothing 不能作为有效改进；按“无用模块直接删除”的规则，已删除代码、CLI 和 trace，只保留本失败记录。

## 2026-06-03 超声 M1 噪声感知可靠性预注册

用户指出“既然问题是噪声，为什么降噪失败”。新的判断：

- 像素平滑会同时削弱 speckle 和真实高频诊断结构，失败不代表噪声假设错，而是“先降噪再提证据”太粗。
- 本轮不再调 temperature、floor、focal power 等超参数。
- 只加入一个结构性判据：局部梯度方向一致性。真实解剖边界通常在局部窗口内方向更一致；随机 speckle 的梯度方向更杂乱。
- M1 仍在原图上提 `edge/local_contrast/texture`，不替换输入图；只用方向一致性对 evidence reliability 和 diagnostic salience 打折。
- 若 45 格相对当前 sparse-only 基线不提升，删除代码，只保留失败记录。

实验结果：

- 真实 45 格输出：`outputs/codex_my_merge_ultrasound_noiseaware_sparse_chaosheng45_20260603`。
- 相对当前 sparse-only 基线：`9/20/16`，mean delta `-0.004213`。
- 相对未加 sparse 的超声 specialist 基线：`11/19/15`，mean delta `-0.001298`。
- 相对 formal best：`25/0/20`，mean delta `+0.014535`，明显低于 sparse-only 的 `+0.018748`。
- 分模型相对 sparse-only：`convnext 0/9/0, +0.000000`；`clip-vit 2/7/0, +0.002496`；`resnet 1/1/7, -0.018668`；`swin_tiny 2/2/5, -0.004293`；`vit_t 4/1/4, -0.000599`。
- 结论：方向一致性把 evidence reliability 压得过低，M1 更频繁选错 `medical_weighted_fusion` 相关候选，尤其伤害 `resnet`。该分支已删除，只保留本失败记录。

## 2026-06-03 超声 Head Prior Repair 预注册

动机：

- 当前超声结果里不少格子反复落在 `0.173405` 附近，像类别先验/分类头塌缩，而不是 backbone 彻底不可用。
- 本轮不调任何 temperature、floor、density、focal 等超参数。

实现规则：

- 只在超声 specialist 分支下新增 M2 候选，不改变基础候选。
- 对每个已准备好的候选，在 val 上计算平均预测概率 `p_hat`，用 val 标签分布 `pi` 做标准 prior correction：对 classifier bias 加 `log(pi) - log(p_hat)`。
- 只处理 state_dict 里形状为 `[num_classes]` 且名称属于 `fc/classifier/head/proj` 的 bias；没有 classifier bias 的候选自动跳过。
- 所有修正候选仍通过同一个 val selection 选择，最终 test 只评估被选候选。
- 若 45 格相对当前 sparse-only 基线不提升，删除代码，只保留失败记录。

实验结果：

- 真实 45 格输出：`outputs/codex_my_merge_ultrasound_headrepair_sparse_chaosheng45_20260603`。
- 相对当前 sparse-only 基线：`27/18/0`，mean delta `+0.036658`，无 loss cell。
- 相对未加 sparse 的超声 specialist 基线：`27/18/0`，mean delta `+0.039573`。
- 相对 formal best：`37/0/8`，mean delta `+0.055405`。
- 分模型相对 sparse-only：`convnext 2/7/0, +0.002596`；`clip-vit 0/9/0, +0.000000`；`resnet 9/0/0, +0.049216`；`swin_tiny 7/2/0, +0.035240`；`vit_t 9/0/0, +0.096236`。
- 候选选择：head repair 候选被选中 `31/45`，其中 `ultrasound_head_prior_repair:medical_weighted_fusion` 13 次、`:sign_consistent_delta` 6 次、`:ultrasound_sign_sparse_0p2` 4 次、`:avg` 4 次。
- 结论：该分支明确正向，保留。它解释了之前大量 `0.173405` 类别塌缩现象；超声问题至少有一部分来自分类头/类别先验，而不是 M1 图像噪声。

## 2026-06-03 超声 Robust Validation Selection 预注册

动机：

- head repair 强正向后，剩余风险主要是 M2 在 val 上选到不稳定候选。
- 本轮不使用图像扰动，不调 selection 权重，不新增阈值。

实现规则：

- 在同一次 candidate val forward 里，除了全量 val 指标，同时按样本全局序号拆成 even/odd 两个确定性子集。
- 每个子集单独计算 `selection_score`。
- 超声分支候选排序不再只看全量 `selection_score`，而是先看 `min(even_score, odd_score)`，再用全量 `selection_score`、`val_acc`、`-val_loss` 做后续排序。
- 若 45 格相对 head-repair 基线不提升，删除该排序规则，只保留 head repair。

## 2026-06-04 Ultrasound Robust Selection Summary

- candidate: `outputs/codex_my_merge_ultrasound_headrepair_robustselect_chaosheng45_20260603`。
- vs head_repair: n=45 W/T/L=4/39/2, mean_delta=-0.000060。
- vs formal best: n=45 W/T/L=37/0/8, mean_delta=0.055345。
- 汇总表：`outputs/codex_my_merge_ultrasound_headrepair_robustselect_chaosheng45_20260603/reports/robustselect_summary.md`，明细：`outputs/codex_my_merge_ultrasound_headrepair_robustselect_chaosheng45_20260603/reports/robustselect_vs_headrepair.csv`。
- 结论：相对 head repair 是轻微负优化，且收益不足以抵消复杂度；已删除 robust selection 排序逻辑和 CLI，只保留 head prior repair 与 sparse sign。

## 2026-06-04 Ultrasound Avg HeadRepair Only Summary

- candidate: `outputs/codex_my_merge_ultrasound_avg_headrepair_chaosheng45_20260604`。
- vs full_headrepair_sparse: n=45 W/T/L=0/17/28, mean_delta=-0.042568。
- vs sparse_only: n=45 W/T/L=11/14/20, mean_delta=-0.005910。
- vs formal best: n=45 W/T/L=27/0/18, mean_delta=0.012838。
- 已写入总汇总表：`My_merge_ret/汇总表.md`。对比报告：`outputs/codex_my_merge_ultrasound_avg_headrepair_chaosheng45_20260604/reports/avg_headrepair_vs_full_headrepair.md`，明细：`outputs/codex_my_merge_ultrasound_avg_headrepair_chaosheng45_20260604/reports/avg_headrepair_vs_full_headrepair.csv`。
- 平均 accuracy 为 `0.234062`，低于 full headrepair sparse 的 `0.276630`。
- 结论：`avg + head prior repair` 不能替代当前 full candidate bank；M1/M2 候选在超声上仍提供有效选择空间。该受控 ablation 只用于验证假设，负优化后已从代码中删除，不进入最终方法。

## 2026-06-04 近年模型融合论文调研后的下一步候选

调研依据：

- Model Soups (ICML 2022)：验证集上表现好的 checkpoint 做贪婪/均匀 soup，通常比只选单模型更稳。
- Fisher Merging (NeurIPS 2022)：用参数后验/Fisher 重要性做加权平均，适合解释“哪些参数不能随便平均”。
- Git Re-Basin (2022)、ZipIt (2023)、FedMA (ICLR 2020)：核心问题是神经元/通道排列对齐；这和超声 CNN/ConvNeXt 塌缩高度相关。
- TIES-Merging (NeurIPS 2023)、DARE (2023)、DELLA (2024)、Breadcrumbs (ECCV 2024)：通过稀疏 delta、符号一致或随机 drop/rescale 降低任务向量干扰。
- RegMean (ICLR 2023)、AdaMerging (ICLR 2024)：用激活统计或无监督目标学习 layer-wise/task-wise 融合系数。

结合当前事实后的优先级：

1. Greedy Client-Subset Soup + Head Repair。对每个设置枚举或贪婪选择客户端子集，先做子集 avg，再做 head prior repair，用同一 val selection 选最终候选。理由：当前 `avg+headrepair` 低于 full，但 full 里大量收益来自“避开坏候选/坏客户端”的选择空间；子集 soup 是最小、最可控的扩展。
2. Activation/Channel Alignment for CNN。只对 `resnet/convnext` 先试，用超声 val feature map 做通道相关性匹配，再平均对应层；目标是解决 permutation/channel mismatch，而不是再调医学特征。风险是架构适配成本高，先做单模型族 smoke。
3. Head Weight Repair。现有 head prior repair 只修 bias；下一步可冻结 backbone，用 val 特征对 classifier weight+bias 做闭式 ridge/prototype 修复。它是医学图像分类头校准，不是 NLP 通用融合；但会更依赖 val labels，必须严格对照 avg 同等处理。
4. Low-Dim AdaMerging。只学习少量 layer group/client 系数，不学习全参数；目标函数用 val loss 或 entropy，带强正则和早停。该方向比调 temperature 更合理，但有小验证集过拟合风险。
5. DELLA/DARE Random Sparse Delta Bank。把当前固定 `sparse_sign_density=0.20` 扩展为少量随机 magnitude-drop 候选，由 val 选择。它成本低，但更像工程增强，优先级低于子集 soup 和通道对齐。

明确不优先：

- 继续在 M1 上做超声 speckle 降噪或边缘特征堆叠。已有 denoise/noise-aware 45 格负优化。
- 直接把 top-2 soup 默认打开。超声上权重空间差异大时 soup 可能破坏特征。
- 大规模搜索超参数。当前收益瓶颈更像错位/坏客户端/头塌缩，不是阈值没调好。

## 2026-06-04 超声 Client-Subset Soup 预实验

动机：

- 近年 Model Soups / greedy soup 的核心启发是：不要只做单一平均，应该让验证集在多个可解释 checkpoint soup 中选择。
- 当前 `avg+headrepair` 负优化，而 full headrepair 的收益来自候选选择空间，说明超声上存在坏客户端或坏方向；排除部分客户端可能比继续调 M1 特征更直接。
- 本轮不调超参数，不看 test 单格写规则。候选只来自客户端子集平均：drop-one 子集，以及按 M1 `overall/morph/consensus` 取 top-k 的子集。所有候选继续走同一个 val selection 和 head prior repair。

ResNet 9 格 smoke：

- 输出：`outputs/codex_my_merge_ultrasound_subset_soup_resnet9_20260604`。
- 相对当前 full headrepair sparse：`6/3/0`，mean delta `+0.038834`，无负格。
- 相对 formal best：`9/0/0`，mean delta `+0.202156`。
- 新增候选实际被选中：`ultrasound_head_prior_repair:ultrasound_subset_top2_overall` 5 次，`ultrasound_subset_top2_overall` 1 次。
- 结论：该方向有事实信号，进入 45 格全量验证。若全量负优化或只在 resnet 上成立，后续需要收窄到 top2 overall 或直接删除。

## 2026-06-04 Ultrasound Subset Soup Summary

- candidate: `outputs/codex_my_merge_ultrasound_subset_soup_chaosheng45_20260604`。
- vs full_headrepair_sparse: n=45 W/T/L=20/25/0, mean_delta=0.017390。
- vs formal best: n=45 W/T/L=39/0/6, mean_delta=0.072796。
- 汇总表：`outputs/codex_my_merge_ultrasound_subset_soup_chaosheng45_20260604/reports/subset_soup_vs_full_headrepair.md`，明细：`outputs/codex_my_merge_ultrasound_subset_soup_chaosheng45_20260604/reports/subset_soup_vs_full_headrepair.csv`。

## 2026-06-04 论文调研后的实现收敛

调研记录：

- Model Soups, ICML 2022, https://arxiv.org/abs/2203.05482 。可借鉴点是：验证集上表现好的 checkpoint 不必只选一个，可以做 soup；但超声 top-2 权重 soup 已被设置为默认禁用，因为权重空间差异大时容易破坏特征。当前采用的是更保守的 client-subset soup，让验证集在多个子集平均 checkpoint 中选择。
- TIES-Merging, NeurIPS 2023, https://arxiv.org/abs/2306.01708 ；DARE, 2023, https://arxiv.org/abs/2311.03099 ；DELLA, 2024, https://arxiv.org/abs/2406.11617 。共同启发是减少 task delta 干扰，保留高置信方向。当前保留的超声 sparse sign candidate 属于这一类，45 格相对旧超声分支无负格提升。
- Git Re-Basin, 2022, https://arxiv.org/abs/2209.04836 ；ZipIt, 2023, https://arxiv.org/abs/2305.03053 。核心启发是通道/神经元错位会让普通权重平均失败。当前未贸然加入通道匹配模块，因为跨 ResNet/ConvNeXt/ViT/CLIP 适配成本高，且需要单独 smoke 证明。
- RegMean, ICLR 2023, https://openreview.net/forum?id=KelmQq0t3u ；AdaMerging, ICLR 2024, https://arxiv.org/abs/2310.02575 。启发是用激活统计或验证目标估计融合系数。当前只把这个思想落到低风险的验证集候选选择上，没有做大规模可学习系数搜索，避免小验证集过拟合。

基于数据的取舍：

- 全量 subset soup 的收益来自 `top2_overall`、少量 `top3_overall` 和 drop-one 子集。被选候选统计显示 `ultrasound_head_prior_repair:ultrasound_subset_top2_overall` 11 次，`ultrasound_subset_top2_overall` 3 次，`top3_overall` 相关 3 次，drop-one 相关 12 次。
- `morph/consensus` top-k subset 候选没有作为最终候选出现。按“无用模块删除”的原则，代码已裁剪为只保留 drop-one 和 M1 overall top-k。
- 超声 specialist 下 `my_merge_ultrasound_subset_soup` 现在默认开启；可以通过 `--no-my-merge-ultrasound-subset-soup` 关闭，便于继续做受控消融。

裁剪后 smoke：

- 输出：`outputs/codex_my_merge_ultrasound_subset_pruned_resnet9_20260604`。
- 相对 full headrepair sparse：`6/3/0`，mean delta `+0.038834`。
- 相对 formal best：`9/0/0`，mean delta `+0.202156`。
- 候选选择与 full subset 的 ResNet 9 格一致，说明删掉 morph/consensus top-k 没有损失这部分收益。

## 2026-06-04 超声候选池二次裁剪

用户指出候选池太大、太杂，不像一个好的方法。按最近 45 格 full subset soup 的 `selected_candidate` 做统计：

- 0 次被选中但大量进入候选池：`avg_sign_blend_0p25`、`ultrasound_avg_sign_blend_0p10`、`ultrasound_avg_sparse_sign_blend_0p10_0p2`、裸 `medical_weighted_fusion`、裸 `sign_consistent_delta`、`ultrasound_avg_weighted_blend_0p5/0p75`。
- 真正带来正 delta 的主要是 `ultrasound_subset_top2_overall` 及其 head-repair 版本；少量收益来自 `top3_overall` 和旧版 `drop_client*`。
- `drop_client*` 的编号没有医学含义，而且很多只是和 `top-k overall` 等价但因为生成顺序被命名成 drop-one。因此不再保留编号枚举。

代码裁剪：

- 超声候选池只保留：`avg`、`medical_weighted_fusion`、`sign_consistent_delta`、`ultrasound_sign_sparse_0p2`、`ultrasound_subset_top{k}_overall`。
- 删除超声分支的所有 blend 候选：`ultrasound_avg_weighted_blend_*`、`avg_sign_blend_0p25`、`ultrasound_avg_sign_blend_0p10`、`ultrasound_avg_sparse_sign_blend_*`。
- 删除 `ultrasound_subset_drop_client*` 枚举；只保留有语义的 M1 overall top-k 子集。
- Head prior repair 仍作为统一校准步骤作用在剩余候选上。

验证：

- 输出：`outputs/codex_my_merge_ultrasound_candidate_prune_resnet9_20260604`。
- 相对 full headrepair sparse：`6/3/0`，mean delta `+0.038834`。
- 相对 formal best：`9/0/0`，mean delta `+0.202156`。
- ResNet 9 格 accuracy 与裁剪前一致。
- 候选池规模从裁剪前 ResNet smoke 的均值 `31.33` 降到 `12.67`，候选池结构明显更干净。

## 2026-06-04 去掉 Chaosheng 数据集特判的测试

动机：

- 用户指出当前设计把 `chaoshengmnist_224` 和其他数据集割裂，容易被质疑是数据集特判。
- 本轮先做两个对照：完全不开超声 specialist 的通用版，以及不看数据集名的统一 adaptive candidate 版。

完全无特殊待遇：

- 命令不传 `--my-merge-ultrasound-specialist`，也不传 adaptive 开关。
- 输出：`outputs/codex_my_merge_no_special_chaosheng_resnet9_20260604`。
- 候选池均值：`4.44`，主要只在 `medical_weighted_fusion` 和 `sign_consistent_delta` 中选择。
- 相对精简 specialist：`0/0/9`，mean delta `-0.088350`。
- 相对 formal best：`9/0/0`，mean delta `+0.113807`。
- 结论：完全去掉额外候选后，超声 ResNet 9 格明显下降，说明提升不是普通 M1/M2 自然产生的。

统一 adaptive candidate 版：

- 新增开关：`--my-merge-adaptive-candidates`。
- 该开关只判断是否是医学图像任务，不判断 `dataset == chaoshengmnist_224`。
- 输出：`outputs/codex_my_merge_adaptive_candidates_chaosheng_resnet9_20260604`。
- 候选命名变为 `adaptive_subset_top{k}_overall`、`adaptive_sign_sparse_0p2`、`head_prior_repair:*`，不再带 `ultrasound_` 前缀。
- 候选池均值：`12.67`，与精简 specialist 一致。
- 相对完全无特殊待遇：`9/0/0`，mean delta `+0.087052`。
- 相对精简 specialist：`1/6/2`，mean delta `-0.001298`，只有 c3 的两个格子轻微下降，c5/c7 核心收益完全保住。
- 相对 formal best：`9/0/0`，mean delta `+0.200859`。
- 结论：可以把“超声特判”改写成统一医学 adaptive candidate 机制。至少 ResNet 9 格上，`chaoshengmnist_224` 没有数据集名特殊待遇时仍基本保留收益。

## 2026-06-04 非超声 adaptive candidates 负优化检查

目的：

- 检查 `--my-merge-adaptive-candidates` 放到非超声医学数据集后是否造成负优化。
- 本轮先跑 `resnet`，数据集为 `bloodmnist_224`、`dermamnist_224`、`organcmnist_224`、`organsmnist_224`，clients `{3,5,7}`，beta `{0,0.01,0.1}`，共 36 格。

输出：

- 默认 my_merge：`outputs/codex_my_merge_adaptive_check_default_resnet36_20260604`。
- adaptive candidates：`outputs/codex_my_merge_adaptive_check_adaptive_resnet36_20260604`。

总体结果：

- W/T/L：`33/1/2`。
- 默认 mean acc：`0.493506`。
- adaptive mean acc：`0.575111`。
- mean delta：`+0.081606`。
- 候选池均值：`12.67`，候选类型仍是 `avg`、`medical_weighted_fusion`、`sign_consistent_delta`、`adaptive_sign_sparse_0p2`、`adaptive_subset_top{k}_overall` 及其 `head_prior_repair:*`。

分数据集：

| dataset | W/T/L | mean delta |
| --- | ---: | ---: |
| `bloodmnist_224` | 9/0/0 | +0.148625 |
| `dermamnist_224` | 9/0/0 | +0.019839 |
| `organcmnist_224` | 9/0/0 | +0.089433 |
| `organsmnist_224` | 6/1/2 | +0.068527 |

负优化格：

| dataset | model | clients | beta | default | adaptive | delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `organsmnist_224` | `resnet` | 3 | 0.01 | 0.447264 | 0.413277 | -0.033987 |
| `organsmnist_224` | `resnet` | 7 | 0.0 | 0.327291 | 0.314376 | -0.012915 |

结论：

- 在非超声 ResNet 36 格上，adaptive candidates 总体显著正向，不是大面积负优化。
- 但它不是无损模块：`organsmnist_224` 有 2 个负格。因此不能贸然默认开启；更稳妥的论文/代码设计是继续保留显式开关，或增加“repair/subset 接受门控”，只有 adaptive 候选在 val 上超过基础候选一定 margin 时才接管。

## 2026-06-04 汇总表覆盖旧 my_merge 全量消融数据

用户要求：

- 更新 `My_merge_ret/汇总表.md`，覆盖旧的 `my_merge` 及其消融数据。

数据来源确认：

- 采用最新完整全量消融源：`outputs/codex_my_merge_full_gpu_20260601_1400/my_merge_ablation_grid`。
- 该源包含 4 个 ablation：`full`、`no_client_information`、`no_fusion_selection`、`avg_only`。
- 每个 ablation 覆盖 5 个模型表，每个模型 45 格，共 `225` 条 eval 结果；4 个 ablation 共 `900` 条 eval 结果。
- 另一个目录 `outputs/codex_my_merge_full_cpu_20260601_0924/my_merge_ablation_grid` 虽然配置写了全量，但实际只完成了 `full/small_resnet`，因此没有作为覆盖源。

覆盖方式：

- 保留 `My_merge_ret/汇总表.md` 顶部的 2026-06-04 Chaosheng 控制实验区块和 smoke 记录。
- 从 `## Ablation Rows` 开始，用 GPU 全量目录生成的 `reports/all_results_ablation_combined.md` 整段替换旧表。
- 同步更新：
  - `My_merge_ret/reports/all_results_ablation_combined.md`
  - `My_merge_ret/reports/ablation_summary.md`
  - `My_merge_ret/reports/validated_run_config.txt`

校验：

- `My_merge_ret/汇总表.md` 从 `## Ablation Rows` 开始与源文件 `outputs/codex_my_merge_full_gpu_20260601_1400/my_merge_ablation_grid/reports/all_results_ablation_combined.md` 完全一致。
- `My_merge_ret/reports/all_results_ablation_combined.md` 与源文件二进制一致。
- 解析表结构正常：外层 `汇总表.md` 共 15 张表，其中顶部 Chaosheng 控制区 5 张，标准全量区 10 张。
- 一个明确变化样例：`Small/resnet/bloodmnist_224/c3_b0` 的 `my_merge full` 从旧表 `0.4607` 覆盖为最新 `0.5525`。

最新全量消融摘要：

| ablation | rows | mean_acc | delta_vs_full | delta_vs_best_original | W/T/L |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full` | 225 | 0.3124 | 0.0000 | -0.0079 | 54/56/115 |
| `no_client_information` | 225 | 0.2731 | -0.0393 | -0.0472 | 35/39/151 |
| `no_fusion_selection` | 225 | 0.2227 | -0.0897 | -0.0976 | 4/33/188 |
| `avg_only` | 225 | 0.2273 | -0.0851 | -0.0930 | 0/35/190 |

解释：

- 下方标准全量表现在已经是 6/1 GPU 全量消融数据，不再沿用 5/23 的旧 `my_merge` 表。
- 顶部 Chaosheng 6/4 控制实验区块是额外的超声专项对照，不用 6/1 全量覆盖；它用于记录后续超声候选池和 adaptive candidate 的受控实验。

## 2026-06-04 正常候选池移除 avg 保底

用户反馈：

- 当前是科研方法，不需要把工程鲁棒性作为主要目标。
- 正常 `my_merge` 候选池里继续保留 `avg` 会削弱方法叙事，也会让 M2 变成“保底搜索”，不够干净。

代码修改：

- 在 `_validated_standard_checkpoint_merge` 中删除正常候选池初始化时的 `avg` 候选。
- 同时删除默认非 extra 模式里的 `avg_sign_blend_0p25`，避免继续保留显式 avg 混合候选。
- 以前 `len(candidates) == 1` 会直接返回 `avg`；现在改为只有 `len(candidates) == 0` 才返回 `avg`。因此只剩一个医学候选时，会继续走候选评估并选择这个医学候选。

保留项：

- `avg_only` 消融仍然直接返回 `avg`，因为它是明确对照。
- 非医学图像任务或框架信息缺失时仍然 fallback 到 `avg`，这不参与医学方法主实验。
- `no_fusion_selection` 这类关闭 M2 的消融仍然会退回平均，用于对比。

验证：

- `./.gpuenv/bin/python -m py_compile methods/my_merge.py` 通过。
- 静态检查确认正常候选池里已经没有 `candidates["avg"]` 和 `avg_sign_blend_0p25`。

## 2026-06-04 M1/M2 算法精简

用户反馈：

- 当前代码超过一千多行，候选池和 M1 细项过多，不像一个清晰科研方法。
- 需要根据已有结果删除用处不大的 M1 参数和低采用率 M2 方法。

已有数据依据：

- 最新 6/1 全量 full 结果中，M2 最终选择统计为：
  - `medical_weighted_fusion`: 117/225。
  - `sign_consistent_delta`: 29/225。
  - `avg`: 47/225，已从正常候选池删除。
  - `avg_sign_blend_0p25`: 32/225，属于 avg 混合保底，已删除。
- adaptive/超声 smoke 中，最终被选中的主要是 `head_prior_repair:*`、`adaptive_subset_top2_overall`、`adaptive_sign_sparse_0p2`、`medical_weighted_fusion`、`sign_consistent_delta`。
- `adaptive_subset_top4_overall` 只在非超声 ResNet 36 格中出现 1 次，Chaosheng ResNet 9 格和精简超声 smoke 中没有成为主要来源；因此 top-k 子集只保留 `top2/top3`。
- `top2_soup:*` 在当前 6/1 全量、adaptive ResNet 36 格和 Chaosheng ResNet 9 格里没有作为最终候选出现；超声方向上权重空间 top-2 soup 还容易引入不稳定，因此删除。
- 最新全量只有 `full/no_client_information/no_fusion_selection/avg_only`，没有 `no_focal/no_rarity/no_domain_focus` 的子项消融。因此 M1 细项按“下游是否实际独立使用、是否与保留指标重复、是否增加复杂度”裁剪。

M1 删除项：

- 删除 `shape_compactness`：需要一整段空间协方差/特征值计算，但只间接进入 reliability/salience，没有独立消融证据。
- 删除 class-rarity 权重：类别层已经保留 `seen_classes` 覆盖信息和 per-class accuracy/margin，rarity 是重复调制。
- 删除 hard/focal 样本分支：`hard_acc` 和 `focal_acc` 与 `morph_acc + margin_score` 高度重叠，且没有最新子项消融支撑。
- 删除 domain-focus 二次权重：它本质仍是 boundary/contrast/texture 的再加权，和 `diagnostic_salience` 重复。

M1 保留项：

- 图像证据：前景面积、边界强度、局部对比度、纹理异质性、诊断显著性、证据可靠性。
- 客户端评分：普通准确率、医学加权准确率、预测 margin、类别覆盖与 per-class 表现。
- 输出权重：`overall_weights`、`morphology_weights`、`class_weights`。

M2 删除项：

- 删除普通 `avg` 候选和 `avg_sign_blend_0p25`。
- 删除权重空间 `top2_soup:*` 及其 `_blend_state_dicts` helper。
- 删除 `adaptive/ultrasound_subset_top4_overall`，只保留采用依据更明确的 `top2/top3`。

M2 保留项：

- `medical_weighted_fusion`：全量中采用最多。
- `sign_consistent_delta`：ResNet 上采用较多，并且承载 M1 医学共识权重。
- `adaptive_subset_top{k}_overall` / `ultrasound_subset_top{k}_overall`：只保留 `k=2,3`，是超声和 adaptive smoke 的主要收益来源。
- `adaptive_sign_sparse_0p2` / `ultrasound_sign_sparse_0p2`：少量但有效，尤其配合 head prior repair。
- `head_prior_repair:*`：adaptive/超声 smoke 中频繁作为最终候选。

代码结果：

- `methods/my_merge.py` 从 1755 行降到 1587 行，净删约 168 行。
- `./.gpuenv/bin/python -m py_compile methods/my_merge.py` 通过。
- 设计文档 `docs/my_merge_user_docs/method_design.md` 已同步为精简后的 M1/M2 叙事。

最小 smoke：

- 命令范围：`small / bloodmnist_224 / resnet / limit=1`，即 `clients=3, beta=0, seed=42`。
- 输出：`outputs/codex_my_merge_simplified_smoke_20260605`。
- test acc：`0.413329`，test loss：`1.555681`。
- 正常候选池：`medical_weighted_fusion`、`sign_consistent_delta`。
- 最终选择：`medical_weighted_fusion`。
- implementation：`medical_evidence_two_module_posthoc_merge_v9_delta_validated`。

补充同配置单格验证：

- 为了和 6/1 全量更可比，另跑 `bloodmnist_224/resnet/c3_b0`，设置 `my_merge_stats_max_batches=16`、`my_merge_bn_batches=4`。
- 输出：`outputs/codex_my_merge_simplified_smoke_blood_fullcfg_20260605`。
- 精简后 test acc：`0.386437`，最终选择 `medical_weighted_fusion`，候选池为 `medical_weighted_fusion`、`sign_consistent_delta`。
- 6/1 旧全量同格 test acc：`0.552470`，最终选择 `avg_sign_blend_0p25`，候选池为 `avg`、`medical_weighted_fusion`、`sign_consistent_delta`、`avg_sign_blend_0p25`。
- 旧版同格 `medical_weighted_fusion` 的 val acc 是 `0.378906`，精简后同候选 val acc 是 `0.386098`，权重也接近：旧 overall `[0.4757, 0.3346, 0.1897]`，新 overall `[0.4862, 0.3392, 0.1746]`。
- 结论：这个格子的下降主要来自按用户要求删除 `avg/avg_sign_blend` 保底候选，而不是 M1 精简导致医学权重大幅跑偏。

另一个旧版本来就选择 `medical_weighted_fusion` 的格子：

- `organcmnist_224/resnet/c3_b0` 快速 smoke 输出：`outputs/codex_my_merge_simplified_smoke_organc_20260605`。
- 精简后 test acc：`0.566821`，旧 6/1 全量同格 test acc：`0.487342`。
- 该格说明保留的医学加权主路径仍能正常工作；是否全量正向需要后续跑完整 grid 判断。

## 2026-06-05 三模块重构：M3 自适应默认开启

用户判断：

- 当前 adaptive 结果可以接受。
- 代码需要改成更简洁的科研代码，不再保留从未触发的异常兜底。
- adaptive 不应作为超声特判，而应作为第三个模块 M3，并加入正式消融。

代码改动：

- `DEFAULT_ADAPTIVE_CANDIDATES` 从 `False` 改为 `True`，full 默认开启 M3。
- 新增 `MODULE3_COMPONENTS = {"adaptive_candidates"}`。
- 新增消融标签：`no_adaptive_candidates`、`no_adaptive_candidate_generation`、`no_m3`，都表示关闭 M3。
- 默认全量 ablation 从 `full no_client_information no_fusion_selection avg_only` 改为 `full no_client_information no_fusion_selection no_adaptive_candidates avg_only`。
- 删除 `merge_my_merge` 内部的 `try/except` 平均 fallback；非医学任务或缺少 metadata/checkpoints/config 现在直接报错。
- 从 `my_merge` 主逻辑删除 `ultrasound_specialist`、`ultrasound_sparse_sign`、`ultrasound_subset_soup`、`ultrasound_selection_medical_weight` 等数据集特判路径。
- `scripts/run_all_avg_eval.py` 删除旧的 `--my-merge-ultrasound-*` 参数入口。

结构整理：

- M1：`_module1_diagnostic_client_information_estimation`，只负责医学图像证据和客户端信息估计。
- M2：`_module2_base_medical_candidates`，只生成两个基础医学候选：`medical_weighted_fusion` 和 `sign_consistent_delta`。
- M3：`_module3_add_adaptive_candidates` 和 `_module3_validate_and_select`，默认加入 `adaptive_subset_top2/top3_overall`、`adaptive_sign_sparse_0p2`、`head_prior_repair:*`，再统一做 BN recalibration 和 val selection。

保留原则：

- `avg` 仍只用于 `avg_only` 和 `no_fusion_selection` 这类显式对照，不进入正常 full 候选池。
- M3 不看数据集名称；`chaoshengmnist_224` 不再有单独代码路径。
- M3 的消融 `no_adaptive_candidates` 是正式 `-M3`，用于衡量 adaptive 候选增量。

验证：

- `./.gpuenv/bin/python -m py_compile methods/my_merge.py scripts/run_all_avg_eval.py scripts/generate_ablation_combined_results_table.py scripts/update_summary_with_chaosheng_raw.py scripts/monitor_my_merge_progress.py scripts/plot_client_weight_dashboard.py scripts/plot_client_weight_ratios.py` 通过。
- 静态检查确认 `methods/my_merge.py` 已无 `ultrasound_*` 主逻辑、无 `try/except` 方法级异常兜底。

继续精简：

- 删除旧 `hard_case_accuracy`、`focal_hard_case_accuracy`、`diagnostic_information_vector` 和 `fusion_confidence` 输出字段。
- 同步更新 `scripts/generate_my_merge_diagnostics.py`，诊断表只保留 `acc_A`、`medical_acc_M`、`margin_Q`。
- 删除旧 component alias：`clip_denorm`、`vlm_denorm`、`layerwise_merge`、`classwise_head`、`medical_prior` 等，仅保留 M1/M2/M3 相关别名。
- `methods/my_merge.py` 从本轮开始时 1587 行降到 1545 行；本轮 diff 统计为 454 insertions / 568 deletions，净删 114 行。

smoke 验证：

- full 默认 M3：`outputs/codex_my_merge_m3_default_smoke_20260605`。
  - 单格：`bloodmnist_224/resnet/c3_b0`。
  - test acc：`0.625840`。
  - implementation：`medical_evidence_three_module_posthoc_merge_v10_adaptive`。
  - M2 基础候选：`medical_weighted_fusion`、`sign_consistent_delta`。
  - M3 后完整候选池：上述两个基础候选 + `adaptive_subset_top2_overall` + `adaptive_sign_sparse_0p2` + 四个 `head_prior_repair:*`。
  - 最终选择：`head_prior_repair:adaptive_subset_top2_overall`。
- `-M3`：`outputs/codex_my_merge_m3_off_smoke_20260605`。
  - test acc：`0.413329`。
  - 候选池只剩 `medical_weighted_fusion`、`sign_consistent_delta`。
  - 最终选择：`medical_weighted_fusion`。
- `-M2`：`outputs/codex_my_merge_m2_off_smoke_20260605`。
  - test acc：`0.301680`。
  - disabled components 包含 `medical_weighted_fusion`、`sign_consistent_delta`、`validated_selection`、`bn_recalibration`、`adaptive_candidates`。
  - 候选池为空，最终选择 `avg`，消融逻辑正确。
- `-M1`：`outputs/codex_my_merge_m1_off_smoke_20260605`。
  - test acc：`0.625840`。
  - disabled components 为 `diagnostic_client_information`、`diagnostic_evidence`、`image_space`。
  - M3 仍默认开启，候选池与 full 相同，最终选择 `head_prior_repair:adaptive_subset_top2_overall`。

## 2026-06-05 my_merge 紧凑重写

用户反馈：

- 1545 行仍然太长；这个方法的核心不应需要一千五百行。

处理：

- 直接重写 `methods/my_merge.py`，不再小修小补。
- 保留 M1/M2/M3、必要消融、必要诊断输出和 small/VLM 两类医学图像入口。
- 删除旧配置框架式写法、旧兼容字段、旧超声分支、旧异常 fallback 和冗长包装函数。
- 文件行数从 1545 行降到 813 行。

紧凑版结构：

- M1：`_module1`，收集 val 批次、提取 6 维医学图像证据、估计 `overall_weights`、`morphology_weights`、`class_weights`。
- M2：`_build_m2_m3_candidates` 的前半部分，生成 `medical_weighted_fusion` 和 `sign_consistent_delta`。
- M3：同一函数后半部分加入 `adaptive_subset_top2/top3_overall` 和 `adaptive_sign_sparse_0p2`；`_select_candidate` 负责 BN 校准、`head_prior_repair:*` 和 val selection。

验证：

- `./.gpuenv/bin/python -m py_compile methods/my_merge.py scripts/run_all_avg_eval.py scripts/generate_my_merge_diagnostics.py scripts/generate_ablation_combined_results_table.py` 通过。
- full smoke：`outputs/codex_my_merge_compact_full_smoke_20260605`，acc `0.625840`，候选池 8 个，最终 `head_prior_repair:adaptive_subset_top2_overall`。
- `-M3` smoke：`outputs/codex_my_merge_compact_m3_off_smoke_20260605`，acc `0.416837`，候选池只剩 `medical_weighted_fusion` 和 `sign_consistent_delta`，最终 `medical_weighted_fusion`。
- `-M2` smoke：`outputs/codex_my_merge_compact_m2_off_smoke_20260605`，acc `0.301680`，候选池为空，最终 `avg`。
- `-M1` smoke：`outputs/codex_my_merge_compact_m1_off_smoke_20260605`，acc `0.625840`，M1 disabled 后仍能正常用 base weights 跑 M2/M3。
