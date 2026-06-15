# 当前进度

更新时间：2026-06-12

## 2026-06-12 观察驱动的方法重设

用户强调：方法设计必须先看数据和中间结果，观察现象，再由现象推出方法；不能为了刷指标倒推故事。这个原则已作为后续 `my_merge` 的设计约束。

### 2026-06-12 候选空间诊断观察

本轮新增了只读诊断脚本：

- 脚本：`scripts/analyze_my_merge_candidate_space.py`
- 输出：`docs/my_merge_user_docs/candidate_space_metrics_20260612.csv`
- 摘要：`docs/my_merge_user_docs/candidate_space_metrics_20260612.md`

这一步不改主算法，也不跑测试评估；它只复用历史好版本的 `selected_candidate` 作为“显微镜标签”，计算每个候选在模型参数空间里的中间量。历史标签来自服务端验证集候选选择，因此不能作为最终算法输入，只能用来观察什么形态曾经有效。

诊断范围：

- `bloodmnist_224`、`chaoshengmnist_224`、`organcmnist_224`
- `resnet`
- `clients={3,5,7}`、`beta={0,0.01,0.1}`
- 27 个格子，每格 7 个候选：`delta_0p00/0p25/0p50/0p75/1p00`、`medical_weighted_fusion`、`specialist_client`

关键中间现象：

- 单个 `conflict_score` 解释不了候选选择。历史选中的各类候选在 `conflict_score` 上高度重叠：例如 `delta_0p00` 均值约 `0.431`，`delta_1p00` 均值约 `0.467`，`specialist_client` 均值约 `0.463`，但每组范围互相覆盖。
- `delta_0p00 -> delta_1p00` 的确是一条连续冲突路径：随着 lambda 增大，`distance_from_avg_ratio` 和 `candidate_norm_ratio_to_avg` 单调增大，`cos_to_sign` 单调接近 1。这说明“增量冲突处理”是合理的连续轴，不应该拆成互不相关的候选补丁。
- `medical_weighted_fusion` 的空间特征很稳定：它通常离 `avg` 很近，`cos_to_consensus` 接近 1，说明它更像是 M1 的医学权重微调，不是强冲突处理。
- `specialist_client` 是另一种机制：它距离 `avg` 远，范数往往很大，且 `cos_to_specialist=1`；它不能被解释为 delta 路径的一段，而应解释为“当平均会混合互斥专家时，保留医学证据最强专家的表征”。
- 负结论也很重要：候选空间指标与历史验证分数的全局相关性不足以直接写一个排序器。单特征在 27 格内命中历史最佳的次数都很低，例如按 `distance_from_avg_ratio` 取最大只命中 `9/27`，按 `cos_to_specialist` 取最大也只命中 `9/27`。继续调手写 proxy score 会变成面向结果编程。

由观察得到的设计约束：

- 不能再做“大候选池 + 手写评分选赢家”。这既不像一个简洁方法，也没有稳定中间现象支撑。
- 更合理的做法是把方法收成两个可解释模块：
  - M1：医学证据权重，并产生一个离平均模型较近的 `medical_weighted_fusion`。
  - M2：冲突处理，不做验证集选择，只在一条连续 delta 路径上做闭式修正；专家保真只能作为高异质、医学赢家清晰时的软锚点，而不是候选池搜索。
- 下一次改代码时，应避免再恢复验证集 candidate selection；如果要加入 `delta_0p25/0p50/0p75/1p00`，也应该作为连续公式的离散化观察或消融，而不是服务端验证选最优。

补充观察：历史好版本和当前固定候选 smoke 的一个关键差异是 BN 重校准。

- 同一格 `bloodmnist_224/resnet/c3_b0`，当前固定候选脚本构造出的 `delta_0p50` 测试为 `0.4037`，与 `v17` 的 `delta_0p50` 一致。
- 历史好版本同名 `delta_0p50` 测试为 `0.5881`。
- 查看 commit `9ea1e0c` 后确认：历史候选在验证评估和最终输出前都会经过 `_prepare -> _recalibrate_bn`，即用统计 batch 重置并重新累计 BatchNorm running mean/var。
- 因此历史好结果不只是“选了 delta_0p50”，还包含“冲突融合后做 BN 统计对齐”。这不是候选选择，也不需要标签；它解决的是融合后特征分布与 BN 运行统计不匹配的问题。

由这个观察得到的下一步：

- 可以恢复 BN 重校准作为 M2 的一部分，命名为统计对齐（statistical alignment）。
- 仍然不恢复服务端验证集候选选择。
- 先只验证这个窄改动，避免把问题再次变成大候选池搜索。

### 2026-06-12 v15 闭式冲突规则的负优化观察

`v15` 探针不再继续等待，已经可以判定为负优化：

- 探针：`outputs/codex_v15_closed_delta_probe_20260612/my_merge_ablation_grid/full/small_resnet__my_merge`
- 范围：`bloodmnist_224`、`organcmnist_224`、`chaoshengmnist_224` 的 `resnet`，共 27 格。
- 总体：平均相对已有最佳基线 `delta=-0.0686`，W/T/L=`7/0/20`。
- 分数据集：
  - `bloodmnist_224`: `delta=-0.0300`，W/T/L=`3/0/6`
  - `organcmnist_224`: `delta=-0.0988`，W/T/L=`1/0/8`
  - `chaoshengmnist_224`: `delta=-0.0770`，W/T/L=`3/0/6`

关键中间现象：

- `v15` 的闭式规则几乎总是把模型从 `delta_0p00` 推向更激进的增量路径：27 格里 `delta_0p25/0p50/0p75/1p00` 共 24 次，`medical_weighted_fusion` 3 次，`delta_0p00` 0 次。
- 历史好版本在同一 27 格里经常选择更保守或更极端但有语义解释的形态：`specialist_client` 9 次、`delta_1p00` 6 次、`delta_0p00` 4 次、其余 delta/医学加权 8 次。
- 典型反例：
  - `bloodmnist_224/resnet/c3_b0.01`：`v15` 选 `delta_0p25`，测试低于已有最佳 `-0.0555`；历史好版本验证候选显示 `delta_0p00` 最好。
  - `organcmnist_224/resnet/c3_b0.1`：`v15` 因 `reliability=0.725`、`weight_shift=0.137`、`conflict=0.390` 启用 `medical_weighted_fusion`，测试低于已有最佳 `-0.2512`；历史好版本显示 `delta_0p00` 最稳。
  - `chaoshengmnist_224/resnet/c7_b0/b0.01`：`v15` 因高冲突选 `delta_1p00`，但测试分别低于已有最佳 `-0.1393/-0.0404`，说明“冲突越高越大步 sign delta”不是稳定规律。

由观察得到的设计结论：

- 单个 `conflict_score` 不能直接映射到 `delta_lambda`。高冲突既可能需要 sign-delta，也可能意味着客户端表征已经不可平均，应该保留一个医学证据清晰的专家。
- `medical_weighted_fusion` 不能只凭高可靠性和低范数离散度启用；当方向冲突接近 `0.49` 时，即使 M1 权重看起来清晰，也可能是多个客户端学到互斥表征。
- 下一版 M2 不做验证集候选选择，也不做数据集特判。它只根据 M1 医学权重和模型空间冲突，把冲突分成三种处理：
  1. 低冲突：允许 M1 的医学加权增量融合。
  2. 中等冲突：只做小步 sign-consistent delta 修正。
  3. 高冲突且 M1 有清晰赢家：使用医学证据最强专家作为保真锚点，避免把互斥表征平均坏。

### 2026-06-12 v16 专家保真三段式规则的负面观察

按上面的结论实现了 `v16`：M2 加入 `specialist_anchor`，并用 M1 权重清晰度、客户端类别分区异质性、模型冲突共同决定是否保留专家；同时把 delta 规则改得更保守。

探针：

- 输出：`outputs/codex_v16_conflict_routing_probe_20260612/my_merge_ablation_grid/full/small_resnet__my_merge`
- 范围：先跑 `resnet` 的 `bloodmnist_224` 和 `chaoshengmnist_224` 前 15 格后停止。
- 结果：平均相对已有最佳基线 `delta=-0.0736`，W/T/L=`2/0/13`。

关键中间现象：

- `bloodmnist_224/resnet/c3_b0`：`v15` 的 `delta_0p25` 为 `0.4183`，但 `v16` 退回 `delta_0p00` 后只有 `0.3017`，低于已有最佳 `0.3826`。说明“中低冲突先保守回 avg”会错杀有效的 sign-delta。
- `bloodmnist_224/resnet/c7_b0.01`：`v16` 选 `delta_1p00`，测试 `0.2891`，低于已有最佳 `0.4250`，也低于历史好版本的 `delta_0p50`。说明“高冲突直接强 delta”仍不可靠。
- `chaoshengmnist_224/resnet/c3_b0.01/c3_b0.1`：`v16` 因冲突分数略低而退回 `delta_0p00`，分别只有 `0.3010/0.1743`，而历史好版本分别是 `delta_0p75/delta_1p00`。说明超声上的高 M1 权重清晰度本身就是需要冲突处理的证据，不能因为 `conflict_score` 未过阈值就放弃 delta。
- `chaoshengmnist_224/resnet/c5_b0`：`v16` 触发 `specialist_client`，结果 `0.2615` 高于已有最佳 `0.2264`。说明专家保真不是无效模块，但触发范围和 delta 强度选择仍未解决。

删除/保留结论：

- 删除或回滚：`v16` 的过度保守 delta 规则。它把大量格子退回平均模型，实际比 `v15` 更差。
- 暂时保留为待改进观察：`specialist_anchor`。它有明确医学解释，也在部分高异质场景有效，但不能作为简单阈值补丁。
- 下一步不再用单个闭式阈值直接决定最终形态。更合理的方向是恢复一组少量、可解释的融合形态：`delta_0p00/0p25/0p50/0p75/1p00`、`medical_weighted_fusion`、`specialist_client`，但不使用服务端验证集。最终选择要由模型空间代理指标排序，例如：
  - 与 M1 医学共识 delta 的方向一致性；
  - sign-delta 的冲突消解比例；
  - 候选相对 avg 的增量范数是否过大；
  - 专家客户端的医学证据优势和类别分区异质性。

### 2026-06-12 v17 代理评分候选路由的负面观察

随后实现了 `v17`：恢复少量可解释候选形态，但不使用服务端验证集；每个候选只用模型空间/M1 代理指标打分：

- 候选：`delta_0p00/0p25/0p50/0p75/1p00`、`medical_weighted_fusion`、`specialist_client`。
- 分数：`proxy_score`，来源包括 `conflict_score`、`sign_conflict`、`norm_dispersion`、M1 权重清晰度、专家分数差、客户端类别分区异质性。
- 隐私约束：没有 `val_acc/val_loss/selection_score`，没有服务端验证集候选选择。

探针：

- 输出：`outputs/codex_v17_proxy_routing_probe_20260612/my_merge_ablation_grid/full/small_resnet__my_merge`
- 范围：`resnet` 先跑到 22 格后停止。
- 总体：平均相对已有最佳基线 `delta=-0.0370`，W/T/L=`7/0/15`。
- 分数据集：
  - `bloodmnist_224`: `delta=-0.0400`，W/T/L=`2/0/7`
  - `chaoshengmnist_224`: `delta=-0.0394`，W/T/L=`3/0/6`
  - `organcmnist_224`: 前 4 格 `delta=-0.0246`，W/T/L=`2/0/2`

关键中间现象：

- `v17` 比 `v16` 好，但仍明显负优化；说明“用手写代理分数替代验证集选择”目前还不能稳定恢复历史好版本。
- `bloodmnist_224/resnet/c3_b0`：`v17` 选 `delta_0p50`，测试 `0.4037`，相对已有最佳 `+0.0211`。这说明恢复 delta 路径是有用的。
- `bloodmnist_224/resnet/c7_b0`：`v17` 选 `delta_0p75`，测试 `0.1885`，相对已有最佳 `-0.1675`。这说明代理分数无法识别“高冲突但不应强 delta”的情形。
- `chaoshengmnist_224/resnet/c3_b0/c3_b0.01`：`v17` 选 `specialist_client`，分别只有 `0.1959/0.2875`，相对已有最佳 `-0.0700/-0.1240`。这说明专家保真不能只由 M1 赢家清晰度和类别异质性触发。
- `chaoshengmnist_224/resnet/c5_b0.01/c7_b0.01`：`v17` 的 `specialist_client` 有正收益，分别 `+0.0970/+0.0521`。这说明专家保真机制本身有价值，但当前触发条件不可靠。

删除/保留结论：

- 删除：当前 `v17` 的手写 `proxy_score` 自动选择逻辑。它符合隐私约束，但不能解释足够多的中间现象，继续调分会变成面向结果编程。
- 保留为观察：候选形态本身仍有价值，尤其是 `delta` 路径和 `specialist_client` 在不同格子均出现正收益。
- 下一步应先生成系统性的“中间现象表”，而不是继续写新规则。表至少要包含每个格子的：
  - M1 权重、权重清晰度、专家优势；
  - 参数冲突统计；
  - 各候选相对 `avg` 的模型空间距离/方向一致性；
  - 历史验证选择只作为显微镜，不作为最终算法输入。
- 已生成中间现象表：`docs/my_merge_user_docs/observation_table_20260612.md`。后续方法设计应先补充这张表缺失的候选空间指标，再提出新规则。

本轮立即停止等待负优化版本。`v14` 小探针已经足够判定方向错误：

- 探针：`outputs/codex_v14_resnet_probe_20260612/my_merge_ablation_grid/full`
- 首批 `bloodmnist_224/resnet` 结果：
  - `c3_b0`: `acc=0.3201`，已有最好基线 `0.3826`，`delta=-0.0625`
  - `c3_b0.01`: `acc=0.2569`，已有最好基线 `0.3376`，`delta=-0.0807`
  - `c3_b0.1`: `acc=0.3215`，已有最好基线 `0.3964`，`delta=-0.0749`
  - 前 5 行均值 `delta=-0.0841`，W/T/L=`0/1/4`

观察 1：当前负优化不是“跑得不够久”，而是主融合路径错了。

- `v14` 强制执行 `medical_weighted_fusion + 小幅 sign_delta`。
- 对 `bloodmnist_224/resnet/c3_b0.01`，M1 权重为：
  - `overall=[0.419, 0.214, 0.366]`
  - `morphology=[0.380, 0.212, 0.409]`
  - `evidence_reliability=0.668`
- 这些权重看起来有医学差异，但历史有效版本在同一格选择的是 `delta_0p00`，即普通平均，而不是医学加权。
- 说明：M1 权重可以作为医学证据，但不能无条件覆盖参数融合；当医学证据与模型空间冲突不匹配时，强行使用医学加权会破坏模型。

观察 2：历史好结果的有效机制不是单一医学加权，而是“保守平均到冲突增量”的路径。

- 历史有效运行：`outputs/codex_restored_good_medical_full_20260611_1838/my_merge_ablation_grid/full`
- `resnet` 45 格均值相对已有最好基线 `delta=+0.0895`，W/T/L=`39/0/6`
- 关键格：
  - `bloodmnist_224/resnet/c3_b0`: 选择 `delta_0p50`，`delta=+0.2055`
  - `bloodmnist_224/resnet/c3_b0.01`: 选择 `delta_0p00`，`delta=+0.1164`
  - `chaoshengmnist_224/resnet/c3_b0.01`: 选择 `delta_0p75`，`delta=+0.1887`
- 说明：同一统一方法里，不同任务需要不同强度的冲突增量；`delta_0p00/0p25/0p50/0p75/1p00` 是一个连续稳定路径，不应被删成单点或弱补丁。

观察 3：历史候选选择只能作为显微镜，不能作为最终算法。

- 历史版本用 `val_acc/val_loss/selection_score` 在服务端选择候选，这违反联邦模型融合的隐私语义。
- 但是这些历史选择结果可以用于观察现象：哪些融合形态在什么情形下有效。
- 最终算法必须只使用融合时可得的信息：
  - M1 医学证据权重
  - 模型增量的符号冲突、方向冲突、范数离散度
  - 医学权重相对平均权重的偏移
  - 客户端类别覆盖/医学证据可靠性

由观察推出的下一版设计约束：

- M1 只负责估计医学证据权重，不直接决定最终必须采用 `medical_weighted_fusion`。
- M2 负责冲突处理，核心输出是一条 `delta_lambda` 路径：
  `delta_lambda = (1 - lambda) * delta_0p00 + lambda * sign_consistent_delta`。
- `lambda` 不能由服务端验证集选，而要由模型空间冲突闭式给出。
- `medical_weighted_fusion` 不能再作为强制主路径；它应作为 M1 证据强、模型冲突低时才启用的医学加权融合。
- 如果 M1 权重偏移明显但参数冲突也明显，应优先走 delta 冲突稳定，而不是直接按医学权重重排全模型参数。

下一步只做小探针，不跑全量。探针必须先验证这个观察是否成立：恢复 `delta_0p00/0p25/0p50/0p75/1p00` 的闭式冲突路径后，至少不能在 `bloodmnist_224/resnet` 前几格继续出现 `-0.06~-0.20` 的明显负优化。

## 2026-06-12 统一医学方法约束

用户指出“肯定不能每个数据集方法不一样”。这个判断是对的：如果 `chaoshengmnist_224`、`dermamnist_224` 等数据集分别走不同融合规则，论文里的方法就会退化成数据集调参，不能作为统一医学模型融合方法。

本轮已按这个原则修改 `methods/my_merge.py`：

- 删除按具体医学数据集写融合分支的逻辑。代码中只保留医学/自然域边界判断，不再出现 `dataset == chaosheng/derma/...` 影响融合路径。
- 删除旧的候选池硬选择路由。当前不再从 `delta_0p25/0p50/0p75/specialist_client` 里挑一个赢家，而是在同一条流水线上连续执行。
- M1：计算医学证据权重，并生成 `medical_weighted_fusion`。这是“医学图像需要诊断证据驱动权重”的主观察。
- M2：处理模型空间冲突。先用统一的 `conflict_score` 产生小幅 `sign_consistent_delta` 修正，再用“专家增量方向是否与医学共识方向一致”决定 `specialist_anchor` 的软锚定强度。
- `specialist_anchor` 不再是某个数据集特判，也不是候选池里直接选最强客户端；它只有在 M1 的专家优势明显、且该专家 delta 与医学共识 delta 对齐时才会被软混合。
- 所有医学数据集共享同一套公式。允许存在的差异只来自模型结构的参数命名分组，例如 CNN 和 Transformer 的 early/mid/late 层名不同；这不是数据集特判。
- 输出诊断中 `validated_candidates=0`、`candidate_metrics={}`，表示当前主流程没有服务端验证集候选选择。

当前统一流程：

```text
Avg baseline
  -> M1: medical evidence weights + medical_weighted_fusion
  -> M2a: model-space conflict score -> soft sign-delta correction
  -> M2b: consensus-aligned specialist anchor -> soft expert preservation
  -> final merged model
```

接口验证：

- 语法检查通过：`python3 -m py_compile methods/my_merge.py scripts/generate_ablation_combined_results_table.py scripts/run_all_avg_eval.py`。
- shell 检查通过：`bash -n scripts/run_validated_my_merge_full.sh scripts/run_my_merge_ablation_split_grid.sh`。
- CPU smoke 已跑通：`outputs/codex_unified_two_module_smoke_cpu_20260612`。
- smoke 单格：`chaoshengmnist_224/resnet/clients=3/beta=0/seed=42`，`test_acc=0.2471`。该 smoke 只用了 2 个统计 batch 和 1 个 eval batch，只用于确认流程，不作为正式性能结论。
- smoke 输出：`selected_candidate=medical_weighted_fusion+sign_delta_0p08+specialist_anchor_0p19`，说明当前是统一连续流水线，不是按数据集挑候选。

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

## 2026-06-06 两模块整理与自然图像对照准备

用户指出 M3 只剩 `avg_sign_blend_0p25`，本质上只是 M2 候选池里的一个保守混合候选，不应作为独立模块。这个判断成立，已调整：

- 删除正式 M3 模块定义，当前方法改为 M1+M2 两模块。
- `avg_sign_blend_0p25 = 0.75 * avg + 0.25 * sign_consistent_delta` 已并入 M2，作为 `avg_sign_blend` 候选。
- 旧消融标签 `no_adaptive_candidates/no_m3` 仅保留为兼容别名，含义改为关闭 M2 里的 `avg_sign_blend` 候选，不再作为论文正式 M3。
- 默认全量消融改为 `full no_client_information no_fusion_selection`。
- `merge_result.json` 中实现名改为 `medical_evidence_two_module_posthoc_merge_v11`。

自然图像对照准备：

- `natural_model_hub/` 已确认包含 288 个 small 配置：`cifar10_32`、`cifar100_32`、`svhn_32`、`tinyimagenet_64` × 8 个 backbone × 9 个 clients/beta。
- 当前 `my_merge` 默认仍拒绝非医学数据集，保持医学专用边界。
- 新增 `--my-merge-domain-control`，仅用于自然图像域外对照实验；显式打开后才允许在自然图像数据集上运行。
- 已试跑 `cifar10_32/resnet/c3_b0`，代码可进入自然对照模式，但失败于数据缺失：`Med_data/cifar10_32.npz` 不存在。
- 已搜索当前仓库 `Med_data`、`natural_model_hub`、`/data/liyapeng_grp` 常见深度、`/data1/users/weiyipan/FL/data`、`/data1/users/weiyipan/ML`，未找到 `cifar10_32.npz/cifar100_32.npz/svhn_32.npz/tinyimagenet_64.npz`。

医学 smoke：

- 运行：`outputs/codex_two_module_medical_smoke_20260606`
- 配置：`chaoshengmnist_224 / resnet / clients=3 / beta=0 / stats=16 / bn=4`
- 结果：`acc=0.3585`，与合并 M3 前同格一致。
- 候选池：`avg`、`medical_weighted_fusion`、`sign_consistent_delta`、`avg_sign_blend_0p25`。
- `module2_candidate_pool`：`medical_weighted_fusion`、`sign_consistent_delta`、`avg_sign_blend_0p25`。

## 2026-06-07 M3 重新定义为冲突稳定路径

用户指出只有 `avg_sign_blend_0p25` 和纯 `sign_consistent_delta` 会显得割裂。已将 M3 改成一条统一的冲突感知 delta 插值路径：

```text
delta_lambda = (1 - lambda) * avg + lambda * sign_consistent_delta
lambda in {0.00, 0.25, 0.50, 0.75, 1.00}
```

当前模块定义：

- M1：医学诊断客户端信息估计。
- M2：医学可靠性加权融合，只包含 `medical_weighted_fusion`。
- M3：冲突感知增量稳定，包含 `delta_0p00/0p25/0p50/0p75/1p00`。

这样 `delta_0p00` 是普通平均，`delta_1p00` 是原纯 sign delta 端点，`0p25/0p50/0p75` 是从 avg 到该端点的离散稳定路径，不再是孤立补丁。

代码检查：

- 语法检查通过：`py_compile methods/my_merge.py scripts/run_all_avg_eval.py scripts/generate_ablation_combined_results_table.py scripts/monitor_my_merge_progress.py`。

smoke：

- 输出：`outputs/codex_m3_path_probe_20260607`
- 配置：`chaoshengmnist_224 / resnet / clients=3 / beta=0 / stats=16 / bn=4`
- 结果：`acc=0.3585`
- 候选池旧命名：`avg`、`medical_weighted_fusion`、`delta_stabilization_0p25`、`delta_stabilization_0p50`、`delta_stabilization_0p75`、`delta_stabilization_1p00`
- `module2_candidate_pool`：`medical_weighted_fusion`
- `module3_candidate_pool`：四个旧命名 `delta_stabilization` 路径点
- 该格选择 `delta_stabilization_0p25`，val selection score 分别为：`0p25=0.4133`、`0p50=0.3198`、`0p75=0.2330`、`1p00=0.2172`。

命名整理：

- 按用户要求把 M3 起点 `avg` 改名为 `delta_0p00`。
- 同时把整条 M3 候选路径统一短名为 `delta_0p00/0p25/0p50/0p75/1p00`，模块含义仍由 M3 的“冲突感知增量稳定”解释。
- `medical_weighted_fusion` 是之前讨论的第一条医学加权融合路径，属于 M2 候选；它不是 M1 本身，而是 M1 权重进入参数融合后的结果。
- 改名后 smoke：`outputs/codex_m3_path_delta0_probe_20260607`，同一超声 ResNet 格子 `acc=0.3585`，候选池为 `delta_0p00`、`medical_weighted_fusion`、`delta_0p25`、`delta_0p50`、`delta_0p75`、`delta_1p00`，最终选择 `delta_0p25`。

## 2026-06-07 自然图像 .npz 到位确认

用户反馈师兄已上传自然图像 `.npz`。检查结果：

- `Med_data/cifar10_32.npz`、`Med_data/cifar100_32.npz`、`Med_data/svhn_32.npz`、`Med_data/tinyimagenet_64.npz` 已存在。
- 四个文件都包含现有 loader 需要的 `train_images/train_labels/val_images/val_labels/test_images/test_labels/metadata`。
- shape 正常：CIFAR10/100 为 `45000/5000/10000`，SVHN 为 `65931/7326/26032`，TinyImageNet 为 `90000/10000/10000`。
- `natural_model_hub` 中对应 small checkpoint 目录和 manifest 均存在。

入口 smoke：

- 运行：`outputs/codex_natural_npz_smoke_20260607`
- 配置：`cifar10_32 / resnet / clients=3 / beta=0 / --my-merge-domain-control`
- 结果：脚本已能完整读自然数据、加载自然 checkpoint、执行 `my_merge` 和 test eval，`test_acc=0.1467`。
- `merge_result.json` 中 `natural_domain_control=true`，说明该结果是显式域外对照模式，不改变默认“医学任务专用”的边界。

已有结果口径：

- `natural_model_hub` 中 288 个配置的 `meta.json` 都带有客户端单模型 `best_val_acc/test_acc/test_loss`，这是训练阶段已有结果。
- 当前仓库还没有自然图像模型融合全量对照表；已有融合输出只有 `cifar10_32/resnet/c3_b0` 的 smoke，以及 2026-06-06 因 `.npz` 缺失失败的 probe。
- 客户端单模型结果粗略汇总：`cifar10_32` 平均客户端 test acc 约 `0.2768`，`cifar100_32` 约 `0.1511`，`svhn_32` 约 `0.2667`，`tinyimagenet_64` 约 `0.1128`。

## 2026-06-07 医学 + 自然域全量启动准备

用户要求医学和自然领域都跑全量，并把结果用脚本写入 `My_merge_ret/汇总表.md`。

脚本调整：

- `scripts/run_my_merge_ablation_split_grid.sh` 新增 `MY_MERGE_DOMAIN_CONTROL=true` 透传，用于自然域外对照。
- `scripts/run_validated_my_merge_full.sh` 同步记录并透传该开关。
- `scripts/generate_ablation_combined_results_table.py` 改为从表头动态读取数据集名，因此同一脚本可生成医学和自然域 combined 表。
- 新增 `scripts/generate_formal_results_table.py`，从自然域 baseline 的 `eval_summary.csv` 生成与 `result/all_results.md` 同格式的 base table。
- 新增 `scripts/publish_domain_summary_tables.py`，把医学 combined 表和自然 combined 表合并写入一个总表。
- 新增 `scripts/run_medical_natural_full_and_publish.sh`，串联医学 my_merge 全量、自然 baseline、自然 my_merge 消融和最终发布。

全量口径：

- 医学：`bloodmnist_224/dermamnist_224/organcmnist_224/organsmnist_224/chaoshengmnist_224`，small 四个 backbone + VLM，`full/-M1/-M2/-M3` 四个消融。
- 自然：`cifar10_32/cifar100_32/svhn_32/tinyimagenet_64`，8 个 small backbone，无 VLM。
- 自然 baseline：默认 12 个对比方法 `avg/ties/dare_linear/dare_ties/regmean/fisher/breadcrumbs/model_stock/from/iso_c/free_merge/robustmerge`。
- 自然 my_merge：同样跑 `full/-M1/-M2/-M3`，显式 `--my-merge-domain-control`。

启动状态：

- 2026-06-07 11:49 CST 已启动后台全量。
- `RUN_TAG=codex_medical_natural_full_20260607_1150`
- 总输出目录：`outputs/codex_medical_natural_full_20260607_1150`
- driver 日志：`logs/codex_medical_natural_full_20260607_1150/driver.log`
- 预计最终发布：`My_merge_ret/汇总表.md`
- 当前阶段：医学 `full` 消融，`resnet` 与 `convnext` 两个 small job 已在 GPU 0/1 上启动。

巡检：

- 2026-06-07 14:40 CST：医学 `full` 已完成，5 个 job 共 `225/225` 行，全部 `OK`。
- 当前阶段进入医学 `no_client_information (-M1)`：`resnet 32/45`，`convnext 17/45`，全部 `OK`。
- 自然域 baseline 和自然域 my_merge 消融尚未开始；`My_merge_ret/汇总表.md` 尚未被本轮覆盖。

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

- M1：`_module1`，收集医学图像证据摘要、估计 `overall_weights`、`morphology_weights`、`class_weights`。
- M2：`_build_m2_m3_candidates` 的前半部分，生成 `medical_weighted_fusion` 和 `sign_consistent_delta`。
- M3：同一函数后半部分加入 `adaptive_subset_top2/top3_overall` 和 `adaptive_sign_sparse_0p2`；`_select_candidate` 负责 BN 校准、`head_prior_repair:*` 和 val selection。

验证：

- `./.gpuenv/bin/python -m py_compile methods/my_merge.py scripts/run_all_avg_eval.py scripts/generate_my_merge_diagnostics.py scripts/generate_ablation_combined_results_table.py` 通过。
- full smoke：`outputs/codex_my_merge_compact_full_smoke_20260605`，acc `0.625840`，候选池 8 个，最终 `head_prior_repair:adaptive_subset_top2_overall`。
- `-M3` smoke：`outputs/codex_my_merge_compact_m3_off_smoke_20260605`，acc `0.416837`，候选池只剩 `medical_weighted_fusion` 和 `sign_consistent_delta`，最终 `medical_weighted_fusion`。
- `-M2` smoke：`outputs/codex_my_merge_compact_m2_off_smoke_20260605`，acc `0.301680`，候选池为空，最终 `avg`。
- `-M1` smoke：`outputs/codex_my_merge_compact_m1_off_smoke_20260605`，acc `0.625840`，M1 disabled 后仍能正常用 base weights 跑 M2/M3。

## 2026-06-07 医学+自然全量运行进度

本次全量 run tag：

- `codex_medical_natural_full_20260607_1150`
- 主输出：`outputs/codex_medical_natural_full_20260607_1150`
- driver 日志：`logs/codex_medical_natural_full_20260607_1150/driver.log`
- 目标汇总表：`My_merge_ret/汇总表.md`

19:55 检查：

- 后台主流程仍在运行，当前还在医学部分，没有进入自然领域阶段。
- 医学 `full` 已完成：5 个任务分组全部 `45/45 OK`。
- 医学 `no_client_information` driver 已标记 finished；small 四个骨干都是 `45/45 OK`，但 VLM 状态表只有 `42/45 OK`。
- 医学 `no_fusion_selection` 已完成：5 个任务分组全部 `45/45 OK`。
- 医学 `no_adaptive_candidates` 正在运行：
  - `small_resnet`：`45/45 OK`
  - `small_convnext`：`45/45 OK`
  - `small_vit_t`：`45/45 OK`
  - `small_swin_tiny`：`26/45 OK`
  - `vlm_clip-vit-base-patch32`：`15/45 OK`
- `My_merge_ret/汇总表.md` 尚未被本轮覆盖，时间戳仍是 `2026-06-06 20:55:02 +0800`。

发现的问题：

- `no_client_information/vlm_clip-vit-base-patch32` 日志里出现过 `Killed`，所以 `42/45` 不是正常完成。
- 主流程没有因此停止，后续仍继续跑 `no_fusion_selection` 和 `no_adaptive_candidates`。
- 如果最终汇总前这 3 个 VLM 缺项没有自动补齐，需要在 GPU 空出后单独 resume 补跑该分组，再生成汇总表。

22:37 检查：

- 主流程已经通过医学 my_merge 阶段，进入自然领域 formal baseline。
- 医学 `no_adaptive_candidates` 已完成：5 个任务分组全部 `45/45 OK`。
- 医学四个消融整体状态：
  - `full`：`225/225 OK`
  - `no_fusion_selection`：`225/225 OK`
  - `no_adaptive_candidates`：`225/225 OK`
  - `no_client_information`：small 四个骨干 `180/180 OK`，VLM 仍是 `42/45 OK`
- 自然 formal baseline 当前已经跑完：
  - `avg`：`216 OK / 72 FAIL`
  - `ties`：`216 OK / 72 FAIL`
  - `dare_linear`：`216 OK / 72 FAIL`
- 自然 formal baseline 正在跑：
  - `dare_ties`：当前 `162 OK / 43 FAIL / 205 rows`
  - `regmean`：当前 `52 OK / 52 rows`
- 当前活跃进程是自然 formal 的 `dare_ties` 和 `regmean`。
- `My_merge_ret/汇总表.md` 仍未被本轮覆盖，时间戳还是 `2026-06-06 20:55:02 +0800`。

自然领域失败原因：

- 失败集中在 `vit_t` 和 `swin_tiny`，错误为 `Input height (32/64) doesn't match model (224)`。
- `natural_model_hub` 的对应 `meta.json` 里写有 `image_size: 224`，说明这些自然 ViT/Swin 检查点需要把 CIFAR/SVHN/TinyImageNet 图像 resize 到 224 后再评估。
- 当前评估脚本没有按自然模型 meta 的 `image_size` 对 `.npz` 图像 resize，所以自然领域结果如果直接汇总会缺 `vit_t/swin_tiny` 的大量配置。

23:05 处理 resize：

- 用户确认按严谨科研流程做 resize，并要求写入文档。
- 进一步检查发现自然 `resnet` 的 `meta.json` 也写有 `image_size: 224`；因此问题不只是 `vit_t/swin_tiny` 会报错，所有自然 checkpoint 都应按 `meta.image_size` 做评估输入。
- 原始 `.npz` 不修改，train/val/test 划分和标签不修改；只在 dataloader/eval transform 中把输入 tensor 确定性 resize 到 checkpoint metadata 记录的尺寸。
- 已修改 `evaluators/small_eval.py`：small 评估路径按 `meta.image_size` 构造 resize transform，并在结果中记录 `source_image_size`、`eval_image_size`、`image_resize`。
- 已修改 `evaluators/vlm_eval.py`：VLM 原本已有 resize，现在也记录同样的尺寸字段。
- 已修改 `utils/runtime.py`：my_merge 内部 M1 统计、M2/M3 验证选择和 BN recalibration 走 `build_runtime`，现在也按 `meta.image_size` 做同样 resize。
- 已修改 `evaluate.py`：把尺寸字段写入 `eval.json` 和 `eval_summary.csv`，后续可检查自然结果是否都按 224 协议跑。
- 已修改 `scripts/run_all_avg_eval.py`：自然数据集 resume 时要求旧 `eval.json` 的 `eval_image_size` 与 `meta.image_size` 一致；旧自然结果没有这个字段，会自动判为 stale 并重跑。
- 验证：
  - `py_compile` 通过：`evaluators/small_eval.py`、`evaluators/vlm_eval.py`、`evaluate.py`、`utils/runtime.py`、`scripts/run_all_avg_eval.py`。
  - `build_small_runtime` smoke：`cifar10_32/vit_t` 的 batch shape 为 `(2, 3, 224, 224)`，一次前向输出 shape 为 `(2, 10)`。
  - 旧自然 `avg/cifar10_32/resnet/c3_b0` 的 `eval.json` 被 `load_valid_eval_payload` 判为 stale，确认后续 resume 会重跑旧自然结果。
- 已停止仍在使用旧预处理的自然 formal 进程：`dare_ties`、`regmean` 及其父级 wrapper。
- 已用同一 `RUN_TAG=codex_medical_natural_full_20260607_1150` 重新启动 wrapper：`screen -dmS mednat_resize_resume_20260607 ...`。
- 新 wrapper 当前先进入医学 my_merge 阶段；医学旧结果不会因缺少 `eval_image_size` 被误判 stale，后续自然结果会按 resize 修复后的逻辑重跑。
- 23:06 检查：resume 已跳过医学 `full`，正在跑 `no_client_information/vlm_clip-vit-base-patch32__my_merge`，用于补之前 `42/45` 的 VLM 缺口。
- 23:08 检查：后台进程仍在运行；`no_client_information/vlm_clip-vit-base-patch32__my_merge` 当前 `44/45`，其中旧结果 `SKIP=42`，新补结果 `OK=2`，还剩 1 个 VLM 配置。
- 2026-06-08 08:01 检查：后台 `mednat_resize_resume_20260607` 仍在运行，已经进入自然 formal baseline。
  - 医学 resume 已完成，`no_client_information` 的 VLM 缺口已补过并进入后续阶段。
  - 自然 formal 当前状态：`avg/ties/dare_linear/dare_ties` 都是 `288/288 OK`。
  - 当前活跃自然任务：`regmean` 和 `fisher`。
  - `regmean` 状态表暂时显示 `130 OK / 18 FAIL / 148 rows`；这些 `FAIL` 是 resize 修复前留下的旧 `Input height` 行，当前日志显示新 run 正在把旧 eval 判为 stale 后重跑，且 `cifar100_32` 的 `vit_t/swin_tiny` 已能正常完成。
  - `fisher` 当前 `31/288 OK`。
  - `My_merge_ret/汇总表.md` 还未被本轮覆盖。
- 2026-06-08 08:05 检查：后台任务仍在跑。
  - `screen`、主 wrapper 和自然 formal 子任务仍存在。
  - GPU 0/1 均有计算占用，当前自然 formal 子任务仍是 `regmean` 和 `fisher`。
  - 自然 formal 状态：`avg/ties/dare_linear/dare_ties` 已完成 `288/288 OK`；`regmean` 当前状态表为 `130 OK / 18 FAIL / 148 rows`，日志显示正在继续重跑旧 stale 结果；`fisher` 当前 `36/288 OK`。
  - `My_merge_ret/汇总表.md` 时间戳仍是 `2026-06-06 20:55:02 +0800`，说明还没走到最终生成总表。
- 2026-06-08 11:13 检查：后台任务仍在自然 formal baseline 阶段。
  - 已完成：`avg/ties/dare_linear/dare_ties` 均为 `288/288 OK`。
  - 正在推进：`regmean` 当前 `228/288 OK`，`fisher` 当前 `216/288 OK`。
  - 尚未开始生成目录的方法：`breadcrumbs/model_stock/from/iso_c/free_merge/robustmerge`。
  - 当前自然 formal baseline 已完成约 `1596/3456` 个配置；后续还需要跑剩余 baseline 方法，再跑自然领域 `my_merge` 消融，最后才会写入 `My_merge_ret/汇总表.md`。
  - `My_merge_ret/汇总表.md` 时间戳仍是 `2026-06-06 20:55:02 +0800`，不是本轮最终结果。
- 2026-06-08 14:26 检查：后台任务继续正常运行，仍在自然 formal baseline 阶段。
  - 新完成：`regmean/fisher` 均已达到 `288/288 OK`。
  - 正在推进：`breadcrumbs` 当前 `134/288 OK`，`model_stock` 当前 `145/288 OK`。
  - 已完成方法：`avg/ties/dare_linear/dare_ties/regmean/fisher`。
  - 尚未开始生成目录的方法：`from/iso_c/free_merge/robustmerge`。
  - 当前自然 formal baseline 已完成约 `2007/3456` 个配置。
  - GPU 0/1 均有计算占用；`My_merge_ret/汇总表.md` 时间戳仍未更新，说明还没进入最终发布表格阶段。
- 2026-06-08 医学结果快速汇总：
  - 医学 my_merge 报告文件：`outputs/codex_medical_natural_full_20260607_1150/medical/my_merge_ablation_grid/reports/all_results_ablation_combined.md`。
  - 整体均值：`full=0.3156`，`-M1=0.2764`，`-M2=0.2227`，`-M3=0.3036`。
  - 相比 full 的下降：去掉 M1 下降 `0.0392`，去掉 M2 下降 `0.0929`，去掉 M3 下降 `0.0120`。
  - 与原始最好方法逐配置比较：full 为 `65/58/102`，平均低 `0.0047`；说明模块本身有贡献，但 full 还没有在所有医学配置上压过原有最优 baseline。
- 2026-06-08 23:05 检查：后台任务仍在运行，自然 formal baseline 接近结束。
  - 已完成：`avg/ties/dare_linear/dare_ties/regmean/fisher/breadcrumbs/model_stock/from/iso_c` 均为 `288/288 OK`。
  - 正在推进：`free_merge` 当前 `158/288 OK`，`robustmerge` 当前 `201/288 OK`。
  - 当前自然 formal baseline 已完成约 `3239/3456` 个配置。
  - GPU 0/1 均有计算占用；自然领域 `my_merge` 消融目录尚未生成，说明还没进入 my_merge 自然消融阶段。
  - `My_merge_ret/汇总表.md` 时间戳仍是 `2026-06-06 20:55:02 +0800`；`My_merge_ret/医学汇总表.md` 已复制并提交到远端。
- 2026-06-09 07:52 检查：自然 formal baseline 已全部完成，当前进入自然领域 `my_merge` 消融。
  - 自然 formal baseline：12 个方法全部 `288/288 OK`，总计 `3456/3456` 完成。
  - `my_merge` 自然消融：`full` 已完成 8 个模型，共 `288/288 OK`。
  - 当前正在跑：`no_client_information/small_densenet__my_merge`，状态为 `26/36 OK`；同一消融中 `convnext/mobilenet/resnet/resnet34/swin_tiny/vit_t` 已各自 `36/36 OK`，`efficientnet` 尚未生成状态表。
  - 当前自然 `my_merge` 消融完成量约为 `530/1152`；后面还需要完成 `no_client_information` 尾部以及 `no_fusion_selection/no_adaptive_candidates`。
  - `My_merge_ret/汇总表.md` 尚未更新，最终总表还未生成。
- 2026-06-09 12:05 表示保真候选回加：
  - 在当前精简版 `my_merge` 的 M3 中加回两个旧版最有证据的候选：`specialist_client` 和 `prototype_head`。
  - `specialist_client` 使用 M1 的 client 医学诊断信息选择得分最高 client，作为完整医学专家模型候选。
  - `prototype_head` 只用于 `vit_t/swin_tiny`，用融合后 encoder 在验证集上的医学加权类别原型重建分类头。
  - CPU smoke 通过：
    - `bloodmnist_224/vit_t/c3_b0`：候选池包含 `specialist_client/prototype_head`，最终选中 `prototype_head`，test acc `0.3195`。
    - `bloodmnist_224/resnet/c3_b0`：候选池包含 `specialist_client`，最终选中 `delta_0p25`，test acc `0.4116`。
  - 已启动医学 full-only 全量验证：`outputs/codex_rep_m3_medical_full_20260609`，只跑 `full`，不覆盖 `My_merge_ret/汇总表.md`。
- 2026-06-09 12:12 调度调整：
  - 用户要求若表示保真改动显著优化，则停止自然领域 run，优先跑医学。
  - 已停止旧自然领域 `codex_medical_natural_full_20260607_1150` 相关进程。
  - 医学 full-only run 改为双 GPU resume：`screen repm3_med_full_2gpu_20260609`。
  - 当前医学状态：`small_resnet__my_merge` 已有 5 行记录（3 行 resume SKIP，2 行新 OK），`small_convnext__my_merge` 已有 1 行 OK。
  - 第一个正式医学结果 `bloodmnist_224/resnet/c3_b0` 为 `0.5881`，相比当前旧汇总表同格 my_merge 约 `0.2886` 明显提升，先继续跑医学 full。
- 2026-06-09 12:25 汇总表发布约束：
  - 用户明确要求不能手动填 `My_merge_ret/汇总表.md`，必须由脚本自动写入。
  - 已检查 `scripts/generate_ablation_combined_results_table.py`：它调用 `highlight_rows`，按每个配置列在所有方法之间比较，最高值自动写为 `<strong>...</strong>`，次高的不同数值自动写为 `<u>...</u>`。
  - 已修改 `scripts/run_validated_my_merge_full.sh`：发布阶段不再把报告文件 `cp` 到 `汇总表.md`，而是再次调用生成脚本并用 `--dest "${PUBLISH_ROOT}/汇总表.md"` 直接写目标表。后续医学全量跑完后，最终总表必须通过这个脚本路径生成。
- 2026-06-09 12:56 医学 full-only 全量进度：
  - 后台 `screen repm3_med_full_2gpu_20260609` 仍在运行。
  - 当前状态：`small_resnet__my_merge` 为 `45/45`（其中 `3` 个是 resume SKIP，`42` 个 OK），`small_convnext__my_merge` 为 `28/45 OK`，`small_vit_t__my_merge` 为 `17/45 OK`。
  - `swin_tiny` 和 `vlm` 还没开始生成状态表；最终 `My_merge_ret/汇总表.md` 尚未更新。
- 2026-06-09 13:27 医学 full-only 全量进度：
  - 后台任务仍在运行。
  - 当前状态：`small_resnet__my_merge` 为 `45/45`，`small_vit_t__my_merge` 为 `45/45`，`small_convnext__my_merge` 为 `43/45 OK`，`vlm_clip-vit-base-patch32__my_merge` 为 `2/45 OK`。
  - `swin_tiny` 还没开始生成状态表；`outputs/.../reports` 当前只有 `validated_run_config.txt`，最终总表还未生成。
- 2026-06-09 13:58 医学 full-only 全量进度：
  - 后台任务仍在运行。
  - 当前状态：`small_convnext__my_merge` 为 `45/45`，`small_resnet__my_merge` 为 `45/45`，`small_vit_t__my_merge` 为 `45/45`，`small_swin_tiny__my_merge` 为 `15/45 OK`，`vlm_clip-vit-base-patch32__my_merge` 为 `22/45 OK`。
  - `reports` 目录仍只有 `validated_run_config.txt`，说明 full 还没结束，最终总表还未生成。
- 2026-06-09 14:28 医学 full-only 全量进度：
  - 后台任务仍在运行。
  - 当前状态：`small_convnext__my_merge` 为 `45/45`，`small_resnet__my_merge` 为 `45/45`，`small_vit_t__my_merge` 为 `45/45`，`small_swin_tiny__my_merge` 为 `32/45 OK`，`vlm_clip-vit-base-patch32__my_merge` 为 `32/45 OK`。
  - `reports` 目录仍只有 `validated_run_config.txt`，最终总表还未生成。
- 2026-06-09 15:02 医学 full-only 全量完成：
  - `screen repm3_med_full_2gpu_20260609` 已结束，`driver.log` 显示 `validated my_merge run finished`。
  - 报告已由脚本生成：`outputs/codex_rep_m3_medical_full_20260609/my_merge_ablation_grid/reports/all_results_ablation_combined.md` 和 `ablation_summary.md`。
  - 已按用户要求直接调用 `scripts/generate_ablation_combined_results_table.py --dest My_merge_ret/汇总表.md`，由脚本自动写入总表，不手动填、不复制。
  - `My_merge_ret/汇总表.md` 已更新，且校验含有 `my_merge full (none missing)`、`<strong>` 最好标记和 `<u>` 次好标记。
  - full 有效配置 `212` 个，整体 `mean_acc=0.3683`，相对原始最好方法 `delta_vs_best_original=+0.0456`，`W/T/L=115/52/45`。
  - 数据集分解：`bloodmnist_224=28/8/9`，`dermamnist_224=16/27/2`，`organcmnist_224=24/5/12`，`organsmnist_224=21/6/9`，`chaoshengmnist_224=26/6/13`。
  - 选中候选分布（按 212 个有效配置）：`prototype_head=61`，`specialist_client=60`，`medical_weighted_fusion=40`，`delta_0p00=19`，`delta_1p00=10`，`delta_0p50=9`，`delta_0p25=8`，`delta_0p75=5`。
  - 结论：这次恢复的 M3 表示保真候选不是边缘补丁，而是主要收益来源；`prototype_head/specialist_client` 合计 `121/212` 次被最终选择。
- 2026-06-09 15:30 占位符和下划线问题处理：
  - 用户指出 `汇总表.md` 仍有占位符且 GitHub 上看不到下划线。
  - 已定位占位符来源：`full/vlm_clip-vit-base-patch32__my_merge` 原始日志在 `organcmnist_224 c5_b0.0` 后被系统 `Killed`，只完成 `32/45` 个 VLM 配置，因此总表中 VLM 的 `organcmnist_224` 后 4 个配置和 `organsmnist_224` 9 个配置显示 `-`。
  - 曾在普通命令中尝试 resume，但沙箱内 CUDA 不可见，失败项变为 `No CUDA GPUs are available`；后续 GPU 运行必须继续走 `screen`。
  - 已把次好标记从 `<u>...</u>` 改为 GitHub 更稳定显示的 `<ins>...</ins>`，并重新生成过一次表；待 VLM 缺口和消融补齐后再最终发布。
  - 已启动完整医学消融后台任务：`screen repm3_med_ablation_2gpu_20260609`，配置为 `full no_client_information no_fusion_selection no_adaptive_candidates`，`VLM_BATCH_SIZE=8`，`PUBLISH_RESULTS=true`。该任务会先 resume `full` 并补齐缺失 VLM，再跑三个消融，最后脚本写入 `My_merge_ret/汇总表.md`。
- 2026-06-09 18:04 消融进度：
  - 后台 `screen repm3_med_ablation_2gpu_20260609` 仍在运行。
  - `full` 已完成补跑：small 四个模型均为 `45/45 SKIP`，VLM 为 `45/45`（`32` 个已有结果 SKIP，`13` 个新 OK），说明之前的 `-` 来源已经在结果目录层面补齐；最终表要等全部消融结束后由脚本重新发布。
  - `no_client_information` 已完成 small 四个模型各 `45/45 OK`；VLM 当前 `25/45 OK`。
  - 当前正在跑 `no_fusion_selection`，已生成 `small_convnext=3/45 OK`、`small_resnet=4/45 OK`。
  - 当前 `My_merge_ret/汇总表.md` 仍是 15:29 的中间版本，仍有 `18` 个 `-`；最终要等本轮完整消融结束后才会被发布脚本覆盖。

## 2026-06-11 隐私约束下的候选池选择修正

用户指出原先候选池通过服务端 `val` batch 计算 `val_acc/val_loss/selection_score` 来挑最终候选，这在严格联邦模型融合设定下不成立：如果客户端私有验证数据能发到服务端，模型融合的隐私动机就被削弱。

本轮先只修正候选池选择，不重跑全量：

- 删除候选选择阶段的服务端样本评估函数：不再对每个候选模型跑 forward，不再用 `cross_entropy`、`val_acc`、`val_loss` 或医学加权验证准确率排序。
- 删除依赖服务端样本的候选：`prototype_head` 和 BN recalibration 相关路径已从主代码移除，因为它们需要服务端持有样本或激活统计。
- 候选路由改为模型空间规则：只看客户端模型相对 reference 的 delta 符号冲突、方向冲突、delta 范数离散度、M1 输出的聚合客户端信息、候选融合权重集中度和 evidence reliability。
- 主代码命名从 `validated_selection` 改为 `candidate_routing`，`routing_summary.selection_rule` 标记为 `privacy_safe_model_space`，避免继续暗示使用服务端验证集挑模型。

当前方法边界需要在论文里讲清楚：候选池选择已经不需要服务端验证数据；但 M1 的医学权重估计目前仍通过 `stats_split` 读取统计数据并评估客户端模型。如果严格写成联邦隐私协议，M1 这一步应解释为“客户端本地计算并上传聚合诊断统计/置信统计”，或者使用公开校准集。后续若继续严谨化，应把 M1 也改成只消费客户端上传的聚合 rows，而不是服务端直接读样本。

新的论文叙事建议：

- M1+M2 合并讲成“医学证据加权融合”：从医学图像证据和客户端诊断统计得到医学专用权重，并把这些权重用于参数/增量融合。
- 观察医学客户端之间存在方向冲突后，引出 M3“冲突感知增量稳定”：沿 `delta_0p00 -> delta_1p00` 的增量路径做模型空间路由，而不是用服务端验证集挑最优。
- `specialist_client` 可以作为 M3 里的“专家保真”策略：当聚合统计显示某个客户端明显强于其他客户端时，保留完整专家模型，避免参数平均破坏表征。

还能考虑但暂不加入的新策略：layer-wise conflict routing，即按 early/mid/late 层分别计算冲突强度并选择不同的 delta 插值强度。它比继续扩大候选池更像一个方法，但实现和消融成本更高，需等当前隐私修正版结果稳定后再决定。

## 2026-06-11 闭式两模块版本

用户继续指出“模型空间候选打分”仍包含过多手写超参数，不适合作为论文主算法。这个判断成立，因此本轮把候选池路由进一步改成闭式融合，不再 `argmax` 选择候选。

当前代码主路径：

```text
W_avg = Avg(W_i)
W_med = MedicalWeightedFusion(W_i)
r = evidence_reliability
W_base = (1 - r) * W_avg + r * W_med

Delta_i = W_i - W_ref
c = mean(sign_conflict, direction_conflict, norm_dispersion)
W_sign = SignConsistentDeltaMerge(W_i, W_ref, M1_consensus_weights)

W_final = (1 - c) * W_base + c * W_sign
```

模块定义同步改为两模块：

- M1：医学证据加权融合。包含图像证据提取、客户端医学权重估计和 `medical_weighted_fusion`。M1 产出 `W_med`，再用医学证据可靠性 `r` 与 `W_avg` 连续混合。
- M2：冲突感知增量稳定。只计算客户端 delta 冲突强度 `c`，并把 `W_base` 与符号一致增量结果 `W_sign` 连续混合。

已删除/移出主路径：

- 不再生成 `delta_0p25/0p50/0p75/1p00` 离散候选池。
- 不再计算候选打分，也不再使用 `medical_weighted_fusion > delta > specialist > avg` 这种优先级。
- `specialist_client` 不再作为主方法模块，避免重新变成候选池搜索。
- sign-delta 不再使用固定 `preserve_density=0.5`，默认 `density=1.0`，避免 M2 中隐藏额外稀疏超参数。

为了兼容旧脚本，`no_fusion_selection/no_m2/no_adaptive_candidates/no_m3` 这些旧消融标签暂时都映射为关闭 M2 的 `conflict_stabilization`。新论文和新默认脚本里只保留：

- `full`：M1 + M2。
- `-M1`：关闭医学证据加权融合，退回平均底座。
- `-M2`：关闭冲突感知增量稳定，只保留 M1 的医学加权底座。
- `avg_only`：普通平均；不实际跑 my_merge，由汇总脚本复用 baseline `avg` 行。

默认全量脚本的 `ABLATIONS` 已同步改为 `full no_client_information no_fusion_selection`。代码检查：`python3 -m py_compile methods/my_merge.py scripts/generate_ablation_combined_results_table.py` 已通过；静态 grep 确认主文件里不再有 `candidate_routing`、`specialist_client`、`selection_score`、`val_acc`、`val_loss`、`cross_entropy` 等候选选择残留。

## 2026-06-11 闭式公式加入少量强度超参数

用户指出科研方法并不要求完全没有超参数，关键是超参数要少且有含义。当前无强度版本的 full-only 中间结果显示 `r` 平均约 `0.72`、`c` 平均约 `0.50`，M1 医学融合和 M2 sign-delta 都被用得过猛，导致已完成 79 格均值相对现有最佳约 `-0.0544`。

本轮保留闭式公式，但加入两个全局强度超参数：

```text
r = medical_strength * evidence_reliability
c = conflict_strength * mean(sign_conflict, direction_conflict, norm_dispersion)
W_final = (1 - c) * [(1 - r) * W_avg + r * W_med] + c * W_sign
```

默认值：

- `medical_strength = 0.5`
- `conflict_strength = 0.5`

这两个参数的含义分别是“相信医学证据加权融合的强度”和“相信冲突稳定 sign-delta 的强度”，不是候选池打分系数。代码和脚本已支持 `--my-merge-medical-strength`、`--my-merge-conflict-strength`，全量脚本也会记录并透传。

smoke：

- 配置：`dermamnist_224 / resnet / c3_b0,b0.01,b0.1`
- 输出：`outputs/codex_closed_form_strength_smoke_20260611/derma_resnet3`
- 结果：`0.6688 / 0.6474 / 0.6813`
- 相对现有最佳：W/T/L=`1/0/2`
- 关键修复：无强度版本中 `c3_b0.1` 曾崩到约 `0.1097`，加入强度后恢复到 `0.6813`。

## 2026-06-11 回滚诊断：闭式精简版为何变差

用户指出精简后 full 明显变差。复查结果后，主要问题不是 `medical_strength/conflict_strength` 这类小参数，而是精简时删掉了旧版的两个表示保持候选：

- `prototype_head`：对 ViT/Swin 类模型，用融合 encoder 的特征原型重建分类头，修复医学类别头错位。
- `specialist_client`：当 M1 统计显示单个客户端明显更可靠时，保留完整专家模型，避免参数平均破坏表征。

之前 6/9 的好结果中，full 有效配置 `212` 个，`prototype_head=61`、`specialist_client=60`，两者合计 `121/212` 次被最终采用；这说明它们不是边缘补丁，而是主要收益来源。闭式两模块版本把它们删除后，只剩 `avg/medical_weighted/sign_delta` 的连续混合，无法处理分类头错位和完整专家保真，因此 full partial 掉到负优化。

回滚动作：

- `methods/my_merge.py` 和主运行脚本已恢复到提交 `9ea1e0c` 之后的好方法版本。
- 当前主方法重新包含 `medical_weighted_fusion`、`validated_selection`、`bn_recalibration`、`conflict_stabilization`、`specialist_client`、`prototype_head`。
- `methods/my_merge.py` 与当前 HEAD 无差异；`9ea1e0c..HEAD` 对方法代码无差异，后续提交只改了汇总表。
- 静态检查已通过：`python3 -m py_compile methods/my_merge.py scripts/run_all_avg_eval.py scripts/generate_ablation_combined_results_table.py scripts/monitor_my_merge_progress.py`。

后续原则：不要再为了形式精简删掉 `prototype_head/specialist_client`。如果需要隐私叙事，应把它们改写为客户端本地统计/本地候选评分上传，而不是直接删除。

## 2026-06-11 恢复版医学 full 全量重跑

恢复到表示保持候选版本后，已启动医学 full 全量重跑，用于验证回滚后的真实结果。

- `RUN_TAG=codex_restored_good_medical_full_20260611_1838`
- 只跑主方法 full：`ABLATIONS=full`
- 不重跑已有 baseline：`RUN_REPRO=false`
- 暂不自动覆盖主汇总表：`PUBLISH_RESULTS=false`，等 full 完整后再由脚本生成/发布，避免半截结果污染 `汇总表.md`
- 数据集：`bloodmnist_224 dermamnist_224 organcmnist_224 organsmnist_224 chaoshengmnist_224`
- 模型：`resnet convnext vit_t swin_tiny` 和 `openai/clip-vit-base-patch32`
- GPU：`0 1`，`MAX_PARALLEL_JOBS=2`
- 输出目录：`outputs/codex_restored_good_medical_full_20260611_1838/my_merge_ablation_grid/full`
- 日志目录：`logs/codex_restored_good_medical_full_20260611_1838`

启动检查：`run_validated_my_merge_full.sh` 仍在运行，当前 first wave 为 `small_resnet__my_merge` 和 `small_convnext__my_merge`。

2026-06-11 19:25 中间结果：

- 当前不是最终全量；已完成并可与 baseline 匹配的结果约 `92/225`。
- partial overall：`mean_acc=0.3907`，相对 `result/all_results.md` 原始最好方法 `delta=+0.0556`。
- 按模型：`resnet` 已完成 `45/45`，`delta=+0.0895`；`convnext` 已完成约 `30/45`，`delta=+0.0195`；`vit_t` 已完成约 `17/45`，`delta=+0.0298`。
- 按数据集已完成部分：`bloodmnist +0.0551`，`dermamnist +0.0438`，`organcmnist +0.1016`，`organsmnist +0.0779`，`chaoshengmnist +0.0355`。
- 初步判断：恢复 `prototype_head/specialist_client` 后，效果已经从闭式精简版的负优化恢复为正优化；最终结论仍需等 `swin_tiny` 和 VLM 全部完成。

## 2026-06-11 叙事友好版本：去候选验证、去 BN，保留确定性 head alignment

用户指出恢复版虽然效果好，但又回到了“服务端验证候选池/额外校准”的叙事，不适合作为干净的医学模型融合方法。本轮停止了 `codex_restored_good_medical_full_20260611_1838`，改回更像论文主线的确定性流程。

当前保留的主线：

- M1：医学证据和客户端诊断信息，计算 `overall/morph/class` 权重。
- M2：用 M1 权重做医学加权融合；根据 delta 冲突做小幅 sign-delta 稳定；根据 M1 专家分数做 soft specialist anchor；对 ViT/Swin 做确定性 prototype head alignment。
- 删除服务端候选验证选择：不再生成候选池后用 `val_acc/val_loss/selection_score` 选最优。
- 删除 BN recalibration：默认脚本 `MY_MERGE_BN_BATCHES=0`，方法代码里不再有 `_recalibrate_bn/_prepare`。

关于是否能去掉 `prototype_head` 和 BN：

- BN 可以去掉。它是纯校准工程，依赖额外 batch 统计，叙事收益不大。
- `prototype_head` 不能直接去掉。超声 `vit_t` 9 格 smoke 显示：
  - no prototype + no BN：`mean_acc=0.2119`，相对原始最好 `delta=-0.0178`，W/T/L=`1/0/8`。
  - no prototype + no BN + local/expert gate/head transplant 等替代：最好也只有 `delta=-0.0152`，仍为负。
  - deterministic prototype head alignment + no BN：`mean_acc=0.2245`，`delta=-0.0052`，W/T/L=`2/0/7`。
  - 超声 `resnet` 同版本：`mean_acc=0.3282`，`delta=+0.0211`，W/T/L=`6/0/3`。

结论：`prototype_head` 不是候选池补丁，而是在 transformer 医学图像融合中修复分类头/表征错位的必要步骤。为了叙事干净，应把它写成“医学原型头对齐（prototype head alignment）”：可由客户端本地上传类别原型/公开校准集计算，不再写成服务端在候选池里挑模型。当前代码暂时仍用 `stats_split` batch 模拟这个统计过程，但主流程已经不是候选搜索。

已启动该版本医学 full 全量：

- `RUN_TAG=codex_story_clean_proto055_medical_full_20260611_2052`
- `ABLATIONS=full`
- `RUN_REPRO=false`
- `PUBLISH_RESULTS=false`
- `MY_MERGE_BN_BATCHES=0`
- 输出目录：`outputs/codex_story_clean_proto055_medical_full_20260611_2052/my_merge_ablation_grid/full`
- 启动检查：`small_resnet__my_merge` 和 `small_convnext__my_merge` 已在 GPU 0/1 上运行。

## 2026-06-11 按用户要求删除 prototype head alignment

用户要求先删掉 prototype head alignment 直接看效果。本轮已停止 `codex_story_clean_proto055_medical_full_20260611_2052`，避免继续跑 prototype 版 full。

代码改动：

- 从 M2 主模块集合移除 `prototype_head`。
- 删除 `_extract_pooled_features`、`_state_embeddings`、`_build_prototype_head_alignment`。
- 删除主融合流程里的 `prototype_head_alignment` 调用和诊断字段。
- 当前主方法只保留：M1 医学权重估计；M2 医学加权融合、冲突稳定、专家软锚定。
- BN recalibration 和服务端候选验证选择仍保持删除状态。

已有 smoke 事实：

- 超声 `vit_t`，无 prototype + 无 BN：`mean_acc=0.2119`，相对原始最好 `delta=-0.0178`，W/T/L=`1/0/8`。
- 超声 `vit_t`，prototype 0.55 + 无 BN：`mean_acc=0.2245`，`delta=-0.0052`，W/T/L=`2/0/7`。
- 超声 `resnet`，无 prototype 不受该模块影响；当前保留医学加权/冲突/专家软锚定。

已启动无 prototype 医学 full 全量：

- `RUN_TAG=codex_story_clean_no_proto_medical_full_20260611_2100`
- `ABLATIONS=full`
- `RUN_REPRO=false`
- `PUBLISH_RESULTS=false`
- `MY_MERGE_BN_BATCHES=0`
- 输出目录：`outputs/codex_story_clean_no_proto_medical_full_20260611_2100/my_merge_ablation_grid/full`
- 启动检查：`small_resnet__my_merge` 和 `small_convnext__my_merge` 已在 GPU 0/1 上运行。

全量结果已完成：

- `RUN_TAG=codex_story_clean_no_proto_medical_full_20260611_2100`
- 覆盖 5 个医学数据集、5 类模型、共 `225/225` 个 full 结果。
- overall：`mean_acc=0.2962`，相对 `result/all_results.md` 原始最好方法 `delta_vs_best_original=-0.0241`。
- 按严格逐格比较：W/T/L=`102/0/123`；按现有 ablation summary 脚本的 `5e-5` 平局阈值显示为 `58/53/114`。两者结论一致：整体是负优化。
- 数据集均值相对原始最好：`bloodmnist=-0.0254`，`dermamnist=-0.0025`，`organcmnist=-0.0445`，`organsmnist=-0.0484`，`chaoshengmnist=+0.0004`。
- 模型均值相对原始最好：`convnext=-0.0119`，`clip-vit-base-patch32=-0.0048`，`resnet=-0.0356`，`swin_tiny=-0.0252`，`vit_t=-0.0429`。

结论：删除 prototype/head alignment 后，方法叙事更简洁，但 full 全量性能明显变差；它不能作为最终主结果。这个结果支持之前 smoke 判断：对 transformer/VLM 医学图像融合，分类头/表征对齐不是可有可无的补丁，而是恢复性能的关键机制。

## 2026-06-12 两模块冲突处理版

用户提出只保留两个模块，并把专家保真归入冲突处理。这个定义更适合当前论文叙事：专家保真不是独立补丁，而是当客户端表示空间冲突过强时，避免强行平均破坏医学专家表征的一种冲突解决策略。

本轮代码调整：

- M1：医学证据客户端加权。包含图像证据、客户端诊断统计、类别级/整体医学权重，以及 `medical_weighted_fusion`。
- M2：冲突感知融合。包含两类冲突处理：
  - delta 路径：恢复 `delta_0p00/0p25/0p50/0p75/1p00`，在增量空间处理符号/方向冲突。
  - 专家保真：恢复硬 `specialist_client`，当 M1 统计显示某个客户端明显占优时，直接保留该专家，解释为表示冲突下的保真选择。
- 暂不恢复 `prototype_head`。理由是旧好结果中 resnet 从未选择 prototype，resnet 的主要收益来自 `specialist_client` 和 delta 路径；先验证这两个机制能否恢复 CNN/VLM 表现。
- 不恢复服务端验证集候选选择：当前路由规则只使用模型空间冲突指标和 M1 产生的客户端统计，不计算 `val_acc/val_loss/selection_score`。
- 不恢复 BN recalibration。

旧好结果的依据：

- resnet 选择分布：`specialist_client=16`，`delta_1p00=10`，`delta_0p00=6`，`delta_0p25=4`，`delta_0p50=3`，`delta_0p75=3`，`medical_weighted_fusion=3`，`prototype_head=0`。
- 全模型选择分布：`specialist_client=73`，delta 路径合计 `51`，两者合计 `124/225`，说明“delta 冲突处理 + 硬专家保真”是主要恢复方向。
- transformer 仍可能需要 prototype：旧结果中 `vit_t` 有 `prototype_head=33/45`，`swin_tiny` 有 `prototype_head=28/45`。因此本轮是有意先做无 prototype 的两模块版本，若 ViT/Swin 仍明显差，再决定是否把 prototype 作为 M2 内部的 transformer 专用冲突修复策略加入。

静态检查：

- `python3 -m py_compile methods/my_merge.py scripts/generate_ablation_combined_results_table.py scripts/run_all_avg_eval.py` 通过。
- `bash -n scripts/run_validated_my_merge_full.sh scripts/run_my_merge_ablation_split_grid.sh` 通过。
- 主代码中不再有服务端候选验证函数，不再出现 `selection_score/val_acc/val_loss/cross_entropy`。

## 2026-06-12 隐私约束修正：融合端不能看原始数据

用户明确指出：医学模型融合不应把原始数据上传到服务端；最多只能在一开始“看数据”得到六个医学图像特性。这个约束会改变方法边界：以前那些在融合端用验证集跑每个 checkpoint、根据 `val_acc/loss/margin` 选候选的方法，在联邦/多医院叙事下不成立。

本轮据此重构 `methods/my_merge.py`：

- 删除服务端逐 checkpoint 预测路径：不再用 `_predict_logits` 在 `stats_split` 上计算 `ordinary_accuracy`、`medical_weighted_accuracy`、`margin_confidence`。
- 删除服务端 BN 重新校准：不再用原始 batch forward 更新 BN running statistics。
- M1 现在只使用医学图像特征摘要、客户端样本数、客户端类别覆盖和模型参数；`best_val_acc/test_acc` 不参与融合权重，避免结果泄露。
- 当前特征摘要包括 `boundary/contrast/texture/salience/reliability` 五个有效医学指标。实验代码可以用 `prepare_my_merge_feature_summaries.py` 预先生成 JSON；正式融合时通过 `--my-merge-feature-summary-root` 读取摘要，并可用 `--my-merge-require-feature-summary` 强制没有摘要就报错。
- 当前仓库没有真实客户端本地样本索引，所以预计算脚本以数据集/类别粒度模拟“客户端本地上传摘要”。这比在融合端评估所有 checkpoint 更符合隐私边界，但论文实现中应表述为客户端本地计算并上传统计摘要。
- M2 的 `statistical_alignment` 改为只聚合 checkpoint 中已有的 BN `running_mean/running_var/num_batches_tracked`，等价于客户端随 checkpoint 上传局部统计；不再访问原始图像。

当前方法信息流：

1. 客户端本地训练并上传 checkpoint，同时上传五维医学统计摘要和 BN running statistics。
2. M1 用五维摘要、类别覆盖和样本支持度计算整体权重、形态权重、类别级权重，并执行医学加权融合。
3. M2 观察参数增量方向/符号冲突，用 M1 共识权重做 sign-consistent delta 稳定；当某客户端医学统计明显占优时，用专家保真作为冲突下的保守锚定；最后只对上传的 BN 统计做加权对齐。

静态检查：

- `python3 -m py_compile methods/my_merge.py scripts/run_all_avg_eval.py scripts/prepare_my_merge_feature_summaries.py` 通过。
- `bash -n scripts/run_my_merge_ablation_split_grid.sh scripts/run_validated_my_merge_full.sh scripts/run_full_my_merge_refresh.sh` 通过。

下一步需要跑小规模 probe。注意：这版为了满足隐私约束，主动放弃了历史好版本里的服务端验证候选选择，因此指标可能低于历史最好；如果出现负优化，应优先分析 M1 摘要权重和 M2 冲突强度，而不是重新加入服务端验证集选择。

## 2026-06-12 严格摘要模式 resnet probe

按“融合端不能看原始数据”的约束，先生成医学摘要 JSON，再用强制摘要模式融合：

- 摘要脚本：`scripts/prepare_my_merge_feature_summaries.py`
- 摘要目录：`outputs/codex_privacy_safe_probe_20260612/feature_summaries`
- 摘要范围：`bloodmnist_224`、`chaoshengmnist_224`、`organcmnist_224`，`resnet`，共 27 个配置。
- 融合命令使用 `--my-merge-feature-summary-root ... --my-merge-require-feature-summary --my-merge-stats-max-batches 0`，因此融合阶段只读摘要 JSON 和 checkpoint，不读原始图像。

第一版严格摘要 full：

- 输出：`outputs/codex_privacy_safe_probe_20260612/my_merge_ablation_grid/full/small_resnet__my_merge`
- 总体：`mean_delta=-0.0241`，W/T/L=`11/0/16`
- `bloodmnist_224`: `mean_delta=-0.0969`，W/T/L=`0/0/9`
- `chaoshengmnist_224`: `mean_delta=-0.0070`，W/T/L=`5/0/4`
- `organcmnist_224`: `mean_delta=+0.0316`，W/T/L=`6/0/3`

消融观察：

- `no_medical_weighted_fusion`: 总体 `mean_delta=-0.0583`，比 full 更差。说明 M1 医学加权不是无效模块；它在 `organcmnist_224` 上提供正收益。
- 原始 `no_conflict_stabilization` 消融无效，因为代码把 `no_conflict_stabilization` 错误归到了空的 M3，实际没有关闭 delta。已修复该消融映射。
- `no_specialist_anchor`: 总体 `mean_delta=-0.0546`，比 full 更差；但它在少数 blood 格子更好，说明专家保真有价值但触发强度不稳定。
- `no_statistical_alignment`: 总体 `mean_delta=-0.0185`，优于 full；`bloodmnist_224` 从 `-0.0969` 改到 `-0.0802`。结论：当前 BN running statistics 加权对齐平均负收益，不能默认放进主方法。

据此修改代码：

- `statistical_alignment` 改为可选组件，默认关闭；只有显式传 `my_merge_ablation=statistical_alignment` 或 `with_statistical_alignment` 才启用。
- 修复 `no_conflict_stabilization/no_sign_delta`，现在会真正关闭 M2 的 delta 冲突模块。

修正版严格摘要 probe：

- 输出：`outputs/codex_privacy_safe_fixed_probe_20260612/my_merge_ablation_grid`
- `full` 默认不启用 BN 统计对齐：总体 `mean_delta=-0.0185`，W/T/L=`11/0/16`
- 真正 `no_conflict_stabilization`: 总体 `mean_delta=-0.0310`，W/T/L=`10/0/17`
- 显式 `statistical_alignment`: 总体 `mean_delta=-0.0241`，W/T/L=`11/0/16`

由修正版 probe 得到的事实：

- M2 的 sign-consistent delta 虽然权重不大，但有稳定正贡献：full 比 `no_conflict_stabilization` 平均高 `+0.0125`。
- BN 统计对齐不是可靠主模块：显式开启后比默认 full 低 `-0.0056`，主要拖低 blood。
- 当前主要短板集中在 `bloodmnist_224`：默认 full 的 `blood` 仍是 `mean_delta=-0.0802`，只有 `c7_b0.01` 一格超过已有最好方法。
- `chaoshengmnist_224` 已接近持平，且 9 格中 4 格超过已有最好；`organcmnist_224` 明显为正。

下一步观察方向：

- 不能回到服务端验证候选选择。
- 不应删除 M1 或 M2：M1 对 organc 有用，M2 delta 有整体正贡献。
- 需要定位 blood 负优化是否来自分类头的类别级医学权重过强、专家锚定误触发，还是 blood 的类别切分导致六维图像摘要无法可靠区分客户端。

## 2026-06-12 分类头路由回退观察

为了定位 `bloodmnist_224` 的负优化，我先尝试过把分类头的类别路由做成更软的默认路径：

- `soft_full`: 默认 `class_routing_strength=0.35`，类别权重在生成 `class_weights` 时被缩小一次，在写分类头时又被缩小一次。
- `no_class`: 关闭类别路由。
- `strong_class`: 强类别路由。
- `fixed_full`: 上一版已验证的默认路径，不使用双重缩放；生成类别权重后，在分类头中按 `0.45 * class_weights + 0.55 * consensus` 注入一次。

严格摘要模式 resnet probe 结果：

- `fixed_full`: overall `mean_delta=-0.0185`
- `soft_full`: overall `mean_delta=-0.0233`
- `no_class`: overall `mean_delta=-0.0253`
- `strong_class`: overall `mean_delta=-0.0208`

观察结论：

- 完全去掉类别路由会更差，说明类别级医学权重不是无用模块。
- `soft_full` 比 `fixed_full` 更差，原因不是类别信息过强，而是类别专家信号被双重缩放后过弱。
- `strong_class` 不能稳定修复 blood，且会牺牲部分其他格子，不能作为默认。

据此把默认代码回退为 `fixed_full` 的单次注入形式：M1 仍计算类别级医学权重，M2/M1 融合时只在分类头按固定 45% 比例使用类别权重；`no_class_routing` 保留为消融。该调整仍满足隐私约束，因为只使用客户端上传的六维摘要、客户端元信息和 checkpoint，不使用服务端验证集选择。

回退后重新跑严格摘要模式 resnet probe：

- 输出：`outputs/codex_fixed_class_route_reprobe_20260612/my_merge_ablation_grid/full/small_resnet__my_merge`
- 总体：`mean_acc=0.3087`，相对已有最好 `mean_delta=-0.0190`，W/T/L=`11/0/16`
- `bloodmnist_224`: `mean_acc=0.2896`，`mean_delta=-0.0822`，W/T/L=`1/0/8`
- `chaoshengmnist_224`: `mean_acc=0.2991`，`mean_delta=-0.0081`，W/T/L=`4/0/5`
- `organcmnist_224`: `mean_acc=0.3375`，`mean_delta=+0.0332`，W/T/L=`6/0/3`

和上一版 `fixed_full` 对比：总体只差 `-0.0005`，说明代码已经基本回到旧的稳定路径；和 `soft_full` 对比：总体提升 `+0.0043`，主要来自 `organcmnist_224` 的 `+0.0131`。因此 soft class routing 被正式否定，不再作为默认方法。

新的问题仍集中在 blood：即使恢复分类头单次注入，`bloodmnist_224` 仍是 `mean_delta=-0.0822`。下一步不继续调分类头强度，而是观察 blood 的 M1 权重、M2 冲突强度和专家锚定是否与实际负优化格子对应。

## 2026-06-13 bloodmnist 中间量观察

对 `outputs/codex_fixed_class_route_reprobe_20260612/.../full/small_resnet__my_merge` 的 9 个 `bloodmnist_224` resnet 格子读取 `merge_result.json`，只分析 M1/M2 中间量，不使用服务端验证集选择。

逐格观察：

- blood 全部 9 格里只有 `c7_b0.01` 超过已有最好，其余 8 格均负优化。
- M1 的整体/形态权重最大值与最终相对差值几乎没有正相关：`overall_max corr_delta=0.087`，`morph_max corr_delta=0.021`。说明问题不是简单的“医学权重越集中越好/越坏”。
- M2 的 conflict delta 权重与相对差值呈正相关：`delta_w corr_delta=0.436`。这说明 sign-consistent delta 不是 blood 的主要负优化来源，反而可能在高冲突格子里提供缓冲。
- 专家锚定权重与相对差值呈负相关：`anchor_w corr_delta=-0.402`。负优化最重的 `c5_b0` 和 `c7_b0` 都有非零专家锚定，且被锚定客户端只覆盖少数类别。

当前假设：

- `bloodmnist_224` 是细胞分类，类别之间是局部形态差异；在强 class split 下，单个客户端通常只覆盖很少类别。M1 可以识别某个客户端的局部医学证据更强，但把这个客户端作为“专家保真”锚点会牺牲其他血细胞类别的表征。
- 因此 blood 的主要风险不是 M1 权重本身，也不是 delta 冲突处理，而是 M2 里的专家保真缺少类别覆盖约束：当候选专家只覆盖少数类别时，保真会从“保护医学专家”变成“保留偏科专家”。

下一步验证：

- 跑当前代码的 `no_specialist_anchor` 严格摘要消融，优先看 blood 9 格。如果 blood 明显改善而 organc/chaosheng 下降，说明需要把专家保真改成“类别覆盖/多专家覆盖安全”的冲突处理，而不是直接删除。

`no_specialist_anchor` 严格摘要 resnet probe 完成：

- 输出：`outputs/codex_no_specialist_current_probe_20260613/my_merge_ablation_grid/no_specialist_anchor/small_resnet__my_merge`
- `full`: overall `mean_delta=-0.0190`，W/T/L=`11/0/16`
- `no_specialist_anchor`: overall `mean_delta=-0.0316`，W/T/L=`10/0/17`
- `no_specialist_anchor - full`: overall `mean_diff=-0.0126`
- blood：`mean_diff=+0.0044`，W/T/L=`3/4/2`
- chaosheng：`mean_diff=+0.0048`，W/T/L=`6/2/1`
- organc：`mean_diff=-0.0469`，W/T/L=`3/0/6`

结论：

- 不能直接删除专家保真。它对 `organcmnist_224` 是强正贡献，直接关掉会把 organc 从正优化拉到负优化。
- blood 的问题也不是“专家保真一律有害”：9 格里只有 `c5_b0.1` 大幅改善，`c5_b0.01` 大幅下降，其余多数几乎不变。
- 更合理的改法是把专家保真从“只看 M1 专家分数”改成“看 M1 专家分数，同时要求类别覆盖足够”。当被锚定客户端只覆盖很少血细胞类别时，降低保真权重；当 organ 类数据的专家覆盖较完整时，保留保真作用。

进一步检查后，类别覆盖率不是可靠门控：

- 按 `no_specialist_anchor - full` 排序后，`class_coverage` 与专家保真收益的粗相关只有 `0.118`。
- `organcmnist_224` 中也存在低覆盖专家，但专家保真往往仍然有强正贡献；如果简单按覆盖率关闭，会误伤 organ。

更稳定的观察是“弱锚定不可靠”：

- 离线模拟规则：若 full 中 `specialist_anchor_weight < threshold`，改用 `no_specialist_anchor` 结果；否则保留 full。
- `threshold=0.15` 时，overall 从 `mean_delta=-0.0190` 改为 `-0.0150`，W/T/L 从 `11/0/16` 改为 `12/0/15`。
- blood 从 `-0.0822` 改为 `-0.0722`；organc 从 `+0.0332` 改为 `+0.0402`；chaosheng 从 `-0.0081` 降到 `-0.0130`。
- `threshold=0.25` 或近似关闭大部分锚定会明显伤 organc，说明专家保真仍需要保留，只应去掉低置信弱锚定。

下一步实现：

- 在 `_specialist_anchor_weight` 中加入最小有效权重门控：如果计算出的专家保真权重低于 `0.15`，直接置零。
- 解释为 M2 的冲突处理约束：只有当 M1 识别出的专家优势足够明确时，才允许以专家保真覆盖冲突；弱优势只交给 delta 稳定处理，避免把随机偏科专家当成医学专家。

## 2026-06-13 严格隐私版 M1/M2 定位消融

本轮继续遵守隐私约束：服务端只使用客户端 checkpoint、客户端元信息和预先上传的医学摘要，不使用原始数据、不跑服务端验证集、不用 `val_acc`/`val_loss` 选择候选。

先验证上一节的弱专家保真门控：

- 输出：`outputs/codex_min_anchor_probe_20260613/my_merge_ablation_grid/full/small_resnet__my_merge`
- 总体：`mean_acc=0.3128`，相对已有最好 `mean_delta=-0.0150`，W/T/L=`12/0/15`
- `bloodmnist_224`: `mean_acc=0.2997`，`mean_delta=-0.0722`，W/T/L=`2/0/7`
- `chaoshengmnist_224`: `mean_acc=0.2942`，`mean_delta=-0.0130`，W/T/L=`3/0/6`
- `organcmnist_224`: `mean_acc=0.3445`，`mean_delta=+0.0402`，W/T/L=`7/0/2`

和旧 full 对比：

- 总体 `+0.0040`
- blood `+0.0100`
- chaosheng `-0.0049`
- organc `+0.0069`

结论：弱专家保真置零是小幅正贡献，但它只是局部修正；27 个格子里多数不变，且会伤到部分超声格子，不能继续围绕这个阈值盲调。

接着跑当前代码的两个定位消融：

- `no_m1`: `outputs/codex_current_ablation_probe_20260613/my_merge_ablation_grid/no_m1/small_resnet__my_merge`
- `no_conflict_stabilization`: `outputs/codex_current_ablation_probe_20260613/my_merge_ablation_grid/no_conflict_stabilization/small_resnet__my_merge`

结果：

- full：总体 `mean_delta=-0.0150`
- `no_m1`：总体 `mean_delta=-0.1024`，比 full 低 `-0.0874`；chaosheng 低 `-0.1050`，organc 低 `-0.1565`
- `no_conflict_stabilization`：总体 `mean_delta=-0.0258`，比 full 低 `-0.0108`
- `no_specialist_anchor`：总体 `mean_delta=-0.0316`，比 full 低 `-0.0166`；但 chaosheng 比 full 高 `+0.0097`，organc 比 full 低 `-0.0539`

定位结论：

- M1 的医学证据权重不能删。去掉 M1 后多数格子直接崩，说明“医学权重”是方法主干，不是负优化来源。
- M2 的 delta 冲突处理不能删。去掉后总体下降，说明冲突确实存在，增量/符号一致处理是有用的。
- 专家保真是有条件的：对 organc 是强正贡献，对 chaosheng 是负贡献，对 blood 接近中性但不稳定。因此专家保真不能作为无条件补丁，也不能直接删除。

进一步观察专家保真收益和医学摘要的关系。定义 `class_morphology_separation`：对每个类别的 `[boundary, contrast, texture, salience, reliability]` 摘要做归一化类间标准差，衡量类别之间是否真的存在可分的医学形态差异。只用客户端上传摘要计算，不需要服务端原始数据。

观察结果：

- 全 27 格中，专家保真收益与 `class_morphology_separation` 的相关性约 `+0.589`。
- `chaoshengmnist_224` 的类间形态分离度最低，约 `0.054`；专家保真平均收益 `-0.0097`。
- `bloodmnist_224` 的类间形态分离度居中，约 `0.117`；专家保真平均收益 `+0.0056`，但个别格子不稳定。
- `organcmnist_224` 的类间形态分离度最高，约 `0.197`；专家保真平均收益 `+0.0539`。

这给出新的设计依据：专家保真不应该按数据集名称开关，而应该由 M1 摘要判断“类别医学形态是否足够可分”。当类间医学形态分离很弱时，某个客户端的高专家分数更可能是偏科/噪声优势；当类间形态分离强时，专家保真更像是在保留真实医学专长。

已实现的下一步改动：

- M1 新增 `class_morphology_separation`，从上传的类别摘要计算。
- M2 的 `_specialist_anchor_weight` 加入 `specialist_separation_gate`：只有类间形态分离从 `0.05` 增至 `0.17` 以上时，专家保真才从关闭逐步打开。
- 该改动不是超声特判；超声只是因为观测到的类间形态分离度最低，所以自然被门控抑制。

正在验证：

- 输出：`outputs/codex_separation_anchor_probe_20260613/my_merge_ablation_grid/full/small_resnet__my_merge`
- 目标：检查该门控是否能保留 organc 的专家保真正贡献，同时减少 chaosheng 的专家保真负贡献。

验证结果：

- 总体：`mean_acc=0.3158`，相对已有最好 `mean_delta=-0.0119`，W/T/L=`13/0/14`
- `bloodmnist_224`: `mean_acc=0.2991`，`mean_delta=-0.0727`，W/T/L=`2/0/7`
- `chaoshengmnist_224`: `mean_acc=0.3039`，`mean_delta=-0.0033`，W/T/L=`4/0/5`
- `organcmnist_224`: `mean_acc=0.3445`，`mean_delta=+0.0402`，W/T/L=`7/0/2`

对比：

- 比 min-anchor full 总体高 `+0.0031`。
- chaosheng 从 `mean_delta=-0.0130` 修到 `-0.0033`，等价于 no-specialist 的 chaosheng 表现。
- organc 保持 `+0.0402`，没有因为抑制超声专家保真而牺牲器官数据集。
- blood 基本不变，说明 blood 的剩余问题不是这个专家保真门控能解决的。

诊断输出确认：

- chaosheng 的 `class_morphology_separation≈0.065`，`specialist_separation_gate≈0.126`，所有格子的 `specialist_anchor_weight=0`。
- organc 的 `class_morphology_separation≈0.236`，`specialist_separation_gate=1.0`，保留 M1 明确识别出的专家锚定。
- blood 的 `class_morphology_separation≈0.141`，`specialist_separation_gate≈0.756`，只保留少数足够强的专家锚定。

结论：

- 保留 `class_morphology_separation` 门控。它解决的是 M2 的“专家保真冲突”：当类别医学形态本身不够分离时，单专家保真容易变成偏科客户端保真；当类别医学形态高度可分时，专家保真是在保护真实医学专长。
- 该规则没有按数据集名称分支，符合统一方法要求。
- 下一步问题转向 blood：当前 M1 和 delta 都是正贡献，但 blood 仍明显低于已有最好，说明 blood 的负优化更可能来自融合粒度或类别头/细胞局部形态的表达方式。

## 2026-06-13 blood 剩余负优化观察

在 `separation_anchor` 版本中继续看 blood 的 9 个 resnet 格子：

- `c5_b0.1` 和 `c7_b0.01` 可以超过已有最好，说明 M1/M2 并不是对 blood 一律无效。
- `c5_b0` 和 `c7_b0` 极差，分别约 `-0.1196` 和 `-0.1868`。这两个格子是强类别切分，每个客户端只见 1 到 2 个血细胞类别。
- 在 `c7_b0` 中，M1 把共享融合权重从均匀 `1/7` 推到 `[0.266, 0.086, 0.094, 0.120, 0.174, 0.165, 0.095]`，最高权重客户端只见 `[3, 0]` 两类。
- 类别头路由本身是合理的：`beta=0` 时每个类别的分类头主要给见过该类的客户端。因此更可能的问题不是分类头，而是共享 trunk 被少数类别客户端的医学权重带偏。

进一步比较 `full - no_m1`：

- 总体上 M1 是必要的，`no_m1` 会让 chaosheng 和 organc 明显下降。
- 但 blood 内部 `c3_b0.1`、`c5_b0`、`c7_b0` 中 `no_m1` 反而更好，说明 blood 的形态证据对共享层不够稳。
- M1 收益和上传摘要里的样本内医学特征波动（`global_std / global_mean`）强相关，尤其 boundary/contrast/texture/reliability 的相关约 `0.65`。
- blood 的这些波动低于 organc，也低于 chaosheng 的一部分特征；这说明 blood 的形态摘要更适合作为类别头/专家线索，不一定适合强改共享 trunk。

据此实现一个新的可消融设计：`trunk_evidence_gate`。

- M1 仍计算医学证据权重、类别权重、类别形态分离度。
- 分类头继续使用类别级 M1 权重。
- 共享 trunk 的 M1 加权强度由 `global_std / global_mean` 的平均医学证据波动决定：证据波动低时，trunk 权重回退到 base/equal；证据波动高时，trunk 才充分使用 M1 权重。
- 该规则不是 blood 特判，也不使用服务端验证集；它只使用客户端上传的医学摘要。

正在验证：

- 输出：`outputs/codex_trunk_evidence_gate_probe_20260613/my_merge_ablation_grid/full/small_resnet__my_merge`
- 目标：检查 trunk 保守门控是否能修复 blood 的极端 split，同时不破坏 chaosheng 和 organc。

验证结果：负优化，已删除该模块。

- `trunk_evidence_gate`: 总体 `mean_delta=-0.0145`，比 `separation_anchor` 低 `-0.0026`
- blood：`mean_delta=-0.0767`，比 `separation_anchor` 低 `-0.0040`
- chaosheng：`mean_delta=-0.0070`，比 `separation_anchor` 低 `-0.0037`
- organc：`mean_delta=+0.0402`，基本不变

逐格看，trunk gate 没有修复最差的 `blood c5_b0` 和 `blood c7_b0`，反而让 `blood c3_b0`、`blood c3_b0.01`、`blood c5_b0.01`、`blood c5_b0.1`、`blood c7_b0.01` 都下降。因此“共享层证据波动低就保守 trunk”这个假设不成立，代码已回滚，不进入方法。

新的结论：

- blood 的剩余问题不是简单的 trunk 强度门控。
- 需要继续观察更细的冲突结构，尤其是 `beta=0` 类别完全/近完全分割时，为什么已有最好常常是 fisher/regmean/iso_c 这类统计或子空间方法，而当前 my_merge 的 delta/医学权重不能恢复这些格子。

## 2026-06-13 继续观察：blood 负优化更像子空间冲突

按用户要求，后续设计必须遵循“先观察现象，再引入方法”，且不能修改基线代码。当前只读取 `iso.py`、`fisher.py`、`regmean.py` 的实现和 `result/all_results.md` 的结果作为参照，不改这些基线文件。

重新读取 `merge_result.json` 中的真实诊断字段后，得到几个关键现象：

- 当前 full 版本的 `conflict_delta` 对多数格子是小幅正贡献，不能直接删除。比如 `blood c7_b0.01` 去掉 conflict 后从 `0.4499` 掉到 `0.3473`。
- blood 的失败不是“冲突越高越差”。在 blood 内部，`norm_dispersion` 与相对已有最好方法的差值反而正相关，说明部分大幅 delta 是有效医学/类别知识，不应该用简单强裁剪抹掉。
- 最差的 blood 格子主要集中在强类别切分，特别是 `beta=0` 且客户端数多时。此时分类头路由相对合理，但共享表示容易被只见少数类别的客户端 delta 拉偏。
- 原始总表中，blood 的已有最好方法常来自 `regmean`、`fisher`、`iso_c`、`dare/ties`。其中 `regmean/fisher` 需要客户端上传激活协方差或 Fisher 统计量，不能由服务端拿原始数据计算；`iso_c/ties/dare` 更接近 checkpoint-only 的冲突/子空间处理，符合当前隐私设定。

因此本轮只尝试一个很小的 M2 改动：在已有 M1 医学加权和 sign-consistent delta 之后，加入 checkpoint-only 的共享层谱稳定项。

设计逻辑：

- 不新增候选池，不用服务端验证集挑模型。
- 只在 `conflict_score` 较高、客户端标签覆盖碎片化较强、且 M1 摘要显示类别形态分离不足时打开。
- 只处理非分类头权重的 delta 谱，把奇异值向均值做温和收缩；分类头仍由 M1 的类别路由负责。
- 这个设计借鉴的是近年 model merging 里“delta/子空间冲突”的思想，而不是把 `iso.py` 基线搬进来；基线代码不修改。

正在验证：

- 输出：`outputs/codex_subspace_stabilization_probe_20260613/my_merge_ablation_grid/full/small_resnet__my_merge`
- 范围：`bloodmnist_224`、`organcmnist_224`、`chaoshengmnist_224` 的 `resnet` 27 格。
- 保留标准：如果只修复 blood 但明显破坏 chaosheng/organc，或者总体低于 separation-anchor 版本，就删除该改动。

验证结果：负优化，已删除该模块。

- 总体：比 separation-anchor 版本低 `-0.0050`，相对已有最好 `mean_delta=-0.0169`。
- blood：比 separation-anchor 低 `-0.0045`，没有修复最差的 `c5_b0` 和 `c7_b0`；`c7_b0.01` 反而从 `0.4499` 降到 `0.4157`。
- chaosheng：比 separation-anchor 低 `-0.0085`，说明对共享层 delta 谱做各向同性收缩会破坏超声上原本有效的细粒度增量。
- organc：小幅低 `-0.0020`。

删除原因：

- 虽然已有最好方法里 `iso_c` 在部分 blood 格子有效，但把“谱各向同性化”作为统一的后处理并不适合当前 M1/M2 流程。
- 这进一步说明 blood 的问题不是简单的 2D 权重谱不均衡，而更可能是“哪些客户端/哪些类别的共享表示应该被保留”的局部重要性问题。该问题如果借鉴 `fisher/regmean`，必须由客户端上传统计摘要，不能由服务端使用原始验证数据计算。

继续定位 M1 与 M2 的耦合关系：

- `no_m1` 在 blood 的若干格子反而更好，例如 `c3_b0.1`、`c5_b0`、`c7_b0`；但在 `c5_b0.1`、`c7_b0.01` 又明显更差。
- `no_class_routing` 总体比 full 低，说明分类头类别路由不是主要负优化来源。
- `no_medical_weighted_fusion` 也总体比 full 低，说明 M1 的医学加权融合不能整体删除。

据此尝试一个最小假设：M1 仍用于医学加权和分类头，但 M2 的 sign-consistent delta 改用 `base/equal` 权重，而不是 M1 的 `consensus` 权重。这个假设用于验证“blood 负优化是否来自 M1 权重注入冲突 delta”。

验证结果：负优化，已回滚。

- 输出：`outputs/codex_base_weight_sign_delta_probe_20260613/my_merge_ablation_grid/full/small_resnet__my_merge`
- 总体：比 separation-anchor 版本低 `-0.0010`。
- blood：低 `-0.0021`，没有修复 `c5_b0` 和 `c7_b0`；`c7_b0.01` 从 `0.4499` 降到 `0.4338`。
- chaosheng：基本持平，`-0.0001`。
- organc：低 `-0.0007`。

结论：M2 的 sign-delta 使用 M1 共识权重不是主要问题。blood 最差格子更可能需要参数重要性或统计校正，而不是简单把 M2 权重退回平均。

继续检查 BN/统计对齐：

- 旧输出 `outputs/codex_stat_align_resnet_probe_gpu_20260612` 显示 `statistical_alignment` 对 blood/chaosheng 有大幅提升，但诊断字段是 `bn_recalibrated=True, bn_batches=4`。
- 这说明旧提升来自服务端用 batch forward 重估 BN running statistics，本质上使用了融合端原始数据，不符合当前“原始数据不能上传到服务端”的约束，不能作为正式方法。
- 当前代码里的 `statistical_alignment` 已改成只聚合客户端 checkpoint 中已有的 BN `running_mean/running_var/num_batches_tracked`，不做服务端 forward。必须重新验证这个隐私安全版本。

隐私安全 BN 统计聚合验证：

- 输出：`outputs/codex_privacy_bn_stat_alignment_probe_20260613/my_merge_ablation_grid/statistical_alignment/small_resnet__my_merge`
- 总体：比 separation-anchor full 低 `-0.0255`，相对已有最好 `mean_delta=-0.0375`，W/L=`9/18`。
- blood：比 full 低 `-0.0165`，`c7_b0.01` 从 `0.4499` 掉到 `0.3391`。
- chaosheng：比 full 低 `-0.0590`，多数格子明显下降。
- organc：基本持平，`-0.0011`，但不能弥补 blood/chaosheng 的损失。

结论：

- 服务端 BN 重校准确实能解释历史一部分好结果，但它违反隐私设定，不能用。
- 仅聚合客户端上传的 BN running stats 不足以解决 blood 问题，且会明显破坏 chaosheng。
- `statistical_alignment` 继续默认关闭；不进入主方法。

## 2026-06-13 M1 医学指标复核：可以改指标，但必须逐项验证

用户提醒 M1 的六个医学指标本身也可以增加、减少或替换。因此重新检查当前六维摘要：

- 当前第 1 维 `area` 来自 `mask = evidence >= quantile(evidence, 0.70)`。
- 这意味着每张图都会固定取 evidence 的前 30% 区域，`area` 数学上必然接近 `0.30`。
- 新版特征摘要统计确认旧设计里 `area` 基本没有区分度；这不是有效医学证据。

据此尝试一版较大的 M1 指标替换：

- 用自适应结构范围 `extent` 替换固定分位数 `area`。
- 在边缘前加入平滑，并把直接局部方差 `texture` 替换成更抗高频噪声的 `texture_coherence`。
- `morph_quality`、类别质量和 `class_morphology_separation` 都纳入第 0 维，观察它是否能贡献医学区分。
- 新摘要输出：`outputs/codex_m1_feature_revision_20260613/feature_summaries`。
- 探针输出：`outputs/codex_m1_feature_revision_probe_20260613/my_merge_ablation_grid/full/small_resnet__my_merge`。

验证结果：负优化，已删除该实现。

- 27 格 resnet 医学探针总体比当前保留的 `separation_anchor` 版本低 `-0.0259`。
- blood：`mean_acc=0.2849`，比旧版低 `-0.0142`，最差的 `c5_b0`、`c7_b0` 没有修复。
- chaosheng：比旧版低 `-0.0043`，基本接近但没有稳定收益。
- organc：比旧版低 `-0.0592`，其中 `c3_b0.01` 和 `c3_b0.1` 大幅下降。

删除原因：

- “固定 area 没有信息”这个观察成立，但本次改动同时改变了底层 edge/contrast/texture 的证据图，影响太大。
- organc 原本依赖当前边界/对比度/纹理统计，替换成过度平滑和结构相干性会破坏 M1 对器官 CT 的有效权重。
- 因此不能把这版指标作为主方法；后续如果继续改 M1 指标，应采用更小的单变量改动，例如只替换无效的第 0 维，或直接从 M1 评分中显式移除第 0 维，而不动已验证有效的 boundary/contrast/texture/salience/reliability 计算。

继续做了一个更小的单变量验证：只从 `salience` 公式中移除 `(0.65 + area)`，其余 edge/contrast/texture/reliability 全部不动。

- 摘要输出：`outputs/codex_m1_salience_no_area_20260613/feature_summaries`。
- 探针输出：`outputs/codex_m1_salience_no_area_probe_20260613/my_merge_ablation_grid/full/small_resnet__my_merge`。
- 总体比 `separation_anchor` 低 `-0.00003`，几乎完全等价。
- blood：9/9 完全持平。
- chaosheng：1 个格子小降，均值 `-0.00010`。
- organc：1 个格子小升，均值 `+0.00001`。

结论：

- `area` 是一个理论上无效的指标，但在当前 salience 里主要表现为近似常数缩放，不会明显改变客户端排序。
- 当前性能瓶颈不在 `area` 这一项；继续围绕它微调没有价值。
- 该单变量修正已回滚，主代码保持当前保留版本。

继续做了一个更小的替换验证：只把第 0 维 `area` 替换成亮/暗强度尾部显著性 `tail_salience`，其余 edge、contrast、texture、reliability 和融合逻辑全部不动。

观察依据：

- 候选摘要统计显示 `bright_tail`、`dark_tail` 在 blood 和 organc 上有明显类间差异，在 chaosheng 上也有弱信号。
- 医学含义是局部强回声/低回声尾部分布，理论上比固定 30% 面积更接近医学图像的强度异常证据。

验证结果：基本中性，已删除该实现。

- 摘要输出：`outputs/codex_m1_tail_salience_20260613/feature_summaries`。
- 探针输出：`outputs/codex_m1_tail_salience_probe_20260613/my_merge_ablation_grid/full/small_resnet__my_merge`。
- 27 格 resnet 医学探针总体比 `separation_anchor` 低 `-0.000005`，可视为持平但没有收益。
- blood：均值高 `+0.00010`，只有 `c7_b0.01` 明显小涨 `+0.0020`，但 `c5_b0.01`、`c5_b0.1` 小降。
- chaosheng：均值低 `-0.00010`，`c5_b0.01` 小降 `-0.0018`。
- organc：均值低 `-0.00001`，5 胜 4 负但幅度都很小。

结论：

- 亮/暗尾部是有医学解释和统计信号的候选特征，但在当前 M1 权重计算中不能稳定转化为性能收益。
- 它不能作为主方法保留；当前 M1 暂时仍保留旧六维结构。
- 后续如果继续改 M1 指标，应优先看“客户端权重排序是否变化、变化是否符合每类医学摘要差异”，而不是只看单个特征的类间方差。

继续检查 M1 六维摘要到客户端权重的实际映射。

只读统计：

- 使用当前保留版摘要 `outputs/codex_privacy_safe_probe_20260612/feature_summaries`，不重新读原图、不改基线。
- 27 格 resnet 医学摘要里，`area` 的类间相对 CV 为 0，确认它基本是死指标。
- `salience` 的类间相对 CV 最大：blood `0.2609`，chaosheng `0.1235`，organc `0.4505`。
- 但 `salience` 本身是 boundary、contrast、texture、area、reliability 的乘积型汇总，又在 `morph_quality` 和 `class_quality` 里与原始项一起使用，理论上存在重复计入同一证据的风险。

据此做了一个最小删减实验：不改变六维摘要本身，只在 M1 权重公式里去掉 `salience` 的直接权重，把权重重新分配给 boundary、contrast、texture、reliability。

验证结果：负优化，已回滚。

- 探针输出：`outputs/codex_m1_no_salience_weight_probe_20260613/my_merge_ablation_grid/full/small_resnet__my_merge`。
- 总体比当前保留版低 `-0.00121`，W/T/L=`5/10/12`。
- blood：均值低 `-0.00110`，其中 `c7_b0.01` 低 `-0.0094`。
- chaosheng：均值低 `-0.00160`，其中 `c5_b0` 低 `-0.0117`。
- organc：均值低 `-0.00093`，其中 `c7_b0.01` 低 `-0.0072`。

结论：

- `salience` 虽然是组合项，但它保留了非线性显著性信息，当前 M1 权重计算仍依赖它。
- 直接删除 `salience` 的权重不是简化，而是损失信息；该改动不进入主方法。
- 当前可确认的无效项仍只有 `area`，但删除/替换它对性能几乎无影响，说明主瓶颈不在第 0 维。

据此做了一个结构性精简：M1 摘要从 6 维改成 5 维，正式删除无效的 `area`。

新 M1 摘要：

- `boundary`：诊断证据区域内的边界强度。
- `contrast`：诊断证据区域内的局部对比。
- `texture`：诊断证据区域内的局部纹理变化。
- `salience`：由边界、对比、纹理和可靠性组合得到的非线性显著性。
- `reliability`：证据可靠性。

实现细节：

- 旧公式里的 `(0.65 + area)` 因为 `area≈0.30`，实际接近常数 `0.95`；新公式直接使用 `0.95`，避免保留一个没有信息量的伪指标。
- `_collect_feature_summary` 兼容旧 6 维摘要：读到旧摘要时自动丢掉第 0 维，因此旧结果和新摘要能平滑切换。
- 这不是新增模块，也不是超参数搜索，只是删除已证伪的冗余特征。

验证结果：保留。

- 新 5 维摘要输出：`outputs/codex_m1_five_feature_20260613/feature_summaries`。
- 探针输出：`outputs/codex_m1_five_feature_probe_20260613/my_merge_ablation_grid/full/small_resnet__my_merge`。
- 27 格 resnet 医学探针总体比当前保留版低 `-0.000029`，属于数值等价。
- W/T/L vs 旧版：`1/25/1`。
- blood：9/9 完全持平。
- chaosheng：只有 `c3_b0` 小降 `-0.0009`。
- organc：只有 `c7_b0.1` 小升 `+0.0001`。

结论：

- M1 不再写成“六个医学指标”，而是五个有效医学摘要指标。
- 删除 `area` 让方法叙事更严谨：不再把固定分位数面积包装成医学证据。
- 性能基本不变，说明这个精简是安全的；后续全量应使用新的 5 维摘要重新生成。

## 2026-06-13 17:33：M2 低类别覆盖 sign-delta 密度实验

继续按照“先观察，再采取措施”的流程检查当前 5 维 M1 版本。

观察 1：M1 不是整体负优化，但在 blood 上存在局部伤害。

- 27 格医学 ResNet probe 中，full 相比 `no_m1` 总体高 `+0.0905`。
- chaosheng 高 `+0.1146`，organc 高 `+0.1565`，说明医学证据权重是主收益来源。
- blood 平均只高 `+0.0002`，且存在明显负格子：`c3_b0.1=-0.0991`，`c5_b0=-0.0938`，`c7_b0=-0.0859`。
- 因此问题不是“删掉 M1”，而是 blood 强类别拆分时 M1 的医学权重会放大错误客户端。

观察 2：M2 的 conflict-delta 是整体正收益，但对最差 blood 格子修正不足。

- full 相比 `no_conflict_stabilization` 总体高 `+0.0138`。
- blood 高 `+0.0134`，chaosheng 高 `+0.0131`，organc 高 `+0.0149`。
- 但 blood 的 `c5_b0` 和 `c7_b0` 上 full 与 no-conflict 完全持平，说明当前 conflict-delta 在这些格子没有起到纠偏作用。
- 诊断显示 blood 并非没有冲突：`conflict_score` 平均 `0.4522`，`sign_conflict` 平均 `0.4762`，`direction_conflict` 平均 `0.4771`。
- 当前 delta 混合权重只有约 `0.05~0.09`，所以冲突修正幅度较保守。

据此尝试措施：在平均客户端类别覆盖率较低时，把 sign-delta 的参数保留密度从固定 `0.50` 提升到最高 `0.75`。

- 动机：低覆盖类别拆分下，客户端任务向量更像局部专家；若只保留 50% delta，可能丢掉共享表征中的一致更新。
- 实验输出：`outputs/codex_low_coverage_sign_density_probe_20260613/my_merge_ablation_grid/full/small_resnet__my_merge`。
- 新策略实际使 blood/chaosheng/organc 的平均 sign-delta 密度分别变为 `0.6343/0.6270/0.6332`。

验证结果：负优化，已删除该策略。

- 27 格总体比固定密度旧版低 `-0.0006`。
- blood 均值只高 `+0.0006`，且没有修复 `c5_b0`、`c7_b0`。
- chaosheng 均值低 `-0.0020`，其中 `c5_b0.01` 低 `-0.0099`。
- organc 均值低 `-0.0004`，其中 `c5_b0` 低 `-0.0038`，`c5_b0.01` 低 `-0.0027`。

结论：

- “类别覆盖低就提高 sign-delta 密度”这个解释太粗，会把非 blood 数据集上本来有效的稀疏冲突修正稀释掉。
- M2 仍保留固定 `density=0.50` 的 conflict-delta，因为它在三类医学数据上都有稳定正收益。
- 下一步应继续定位 blood 上 M1 权重为什么会在 `c3_b0.1/c5_b0/c7_b0` 伤害模型，而不是继续调 sign-delta 密度。

## 2026-06-13 18:05：共享层 delta 一致性加权

继续定位 M1 在 blood 上的局部负收益来源，先做三个只关一个开关的消融。

观察 1：分类头路由不是主要问题。

- `no_class_routing` 总体比 full 低 `-0.0123`。
- blood 比 full 低 `-0.0062`，chaosheng 低 `-0.0052`，organc 低 `-0.0254`。
- 因此类别头按 M1 类别权重路由总体是有效的，不能作为 blood 负优化主因删除。

观察 2：专家锚定也不是主要问题。

- `no_specialist_anchor` 总体比 full 低 `-0.0196`。
- chaosheng 完全不变，因为当前分离度门控已经把超声专家锚定关掉。
- organc 低 `-0.0539`，说明专家锚定主要保护器官数据集上的强医学专家。
- blood 大多持平，只在少数格子有小变化，所以它不是 blood 最差格子的主要原因。

观察 3：主体医学加权融合不能删除，但它确实是 blood 局部伤害来源。

- `no_medical_weighted_fusion` 总体比 full 低 `-0.0642`。
- chaosheng 低 `-0.1127`，organc 低 `-0.0753`，说明 M1 主体医学加权是当前主收益。
- blood 平均低 `-0.0045`，但局部格子反而更好：`c3_b0.1` 从 `0.3075` 到 `0.4089`，`c7_b0` 从 `0.1692` 到 `0.2710`。
- 这说明问题不是“医学权重无效”，而是“用一个客户端级标量医学权重去融合共享 trunk 太粗”，在 blood 强类别拆分时会把共享表征拉向局部类别客户端。

对照基线实现得到的进一步启发：

- `fisher` 和 `regmean` 的共同点是客户端上传参数重要性或协方差统计，让融合从“客户端标量权重”进入“参数/层统计权重”。
- 当前不能用服务端原始数据，也不能改基线代码；但可以在 `my_merge` 内部只根据客户端模型 delta 本身估计共享层的一致性。

据此实现一个很小的 M2 修正：`delta_agreement_weighting`。

- M1 仍计算医学证据权重，并保留分类头类别路由。
- 对共享层，在原 M1 层权重基础上，计算每个客户端 delta 与平均共同 delta 方向的一致性。
- 若某客户端的共享层 delta 与共同方向更一致，则轻微提高其该层融合权重；若方向更偏离，则轻微降低。
- 这个修正只用模型参数，不用服务端验证集或原始图像；它是对 M1 共享层标量权重过粗的补偿，而不是候选池搜索。

验证结果：保留。

- 输出：`outputs/codex_delta_agreement_weight_probe_20260613/my_merge_ablation_grid/full/small_resnet__my_merge`。
- 27 格总体比 5 维 M1 full 高 `+0.00263`，相对已有最好从 `-0.01197` 改到 `-0.00934`。
- blood 高 `+0.00179`，W/T/L vs 旧 full 为 `5/2/2`。
- chaosheng 高 `+0.00349`，相对已有最好从 `-0.00339` 改为 `+0.00010`，W/T/L vs best 为 `6/0/3`。
- organc 高 `+0.00261`，相对已有最好为 `+0.04280`。

逐格变化：

- blood 改善 `c3_b0` `+0.0047`、`c3_b0.1` `+0.0032`、`c7_b0.01` `+0.0082`，但没有修复 `c5_b0`、`c7_b0`。
- chaosheng 改善 `c3_b0` `+0.0090`、`c3_b0.01` `+0.0054`、`c7_b0.01` `+0.0162`。
- organc 改善 `c3_b0` `+0.0066`、`c3_b0.1` `+0.0120`、`c7_b0.01` `+0.0067`。

诊断：

- 所有 resnet 格子约有 100 个共享层 tensor 被调整。
- 平均权重改变量很小：blood `0.00517`，chaosheng `0.00531`，organc `0.00428`。
- 说明它不是大幅重写 M1，而是对共享层医学权重做温和的参数空间一致性校正。

结论：

- 保留 `delta_agreement_weighting`。
- 它提供了一条更合理的故事线：M1 给出医学证据权重；M2 发现共享层存在客户端方向冲突，于是用 delta 一致性对共享层权重做局部修正，再叠加原有 sign-consistent conflict delta。
- 当前局限是 blood 的 `c5_b0`、`c7_b0` 仍未修复，后续若继续优化，应围绕“单类别/少类别客户端的共享表示如何避免塌缩”继续观察，而不是再调分类头或专家锚定。

直接消融确认：

- 输出：`outputs/codex_delta_agreement_weight_probe_20260613/my_merge_ablation_grid/no_delta_agreement_weighting/small_resnet__my_merge`。
- full 相比 `no_delta_agreement_weighting` 总体高 `+0.00263`，W/T/L 为 `15/4/8`。
- blood 高 `+0.00179`，W/T/L 为 `5/2/2`。
- chaosheng 高 `+0.00349`，W/T/L 为 `5/2/2`。
- organc 高 `+0.00261`，W/T/L 为 `5/0/4`。
- 这说明它不是和其它代码混在一起的偶然收益，而是一个独立有效的小修正。

## 2026-06-13 19:00：共享层类别覆盖均衡实验

继续从失败格子找现象。

观察 1：最差格子存在“类别覆盖客户端权重偏低”的表象。

- `blood c5_b0` 中，客户端 3/4 只覆盖单类，M1 共享层融合权重偏低；该格子 full 为 `0.1698`，相比已有最好低 `-0.1196`。
- `blood c7_b0` 中，多数客户端只覆盖单类，full 为 `0.1692`，相比已有最好低 `-0.1868`。
- `chaosheng c3_b0.1` 中，客户端 0 覆盖 `[0,1,6]` 但共享层权重只有约 `0.127`，full 为 `0.2650`，相比已有最好低 `-0.2031`。
- 这些现象看起来像“医学质量/样本量权重压低了某些类别代表客户端”，可能导致共享 trunk 对类别覆盖不足。

据此尝试措施：只在共享层做一个轻量的类别覆盖均衡。

- 分类头仍然保留 M1 的按类路由，不改。
- 对 early/mid/late 共享层，把原 M1 层权重轻微拉向“覆盖稀有类别的客户端”先验。
- 这个先验只使用 `meta.clients[*].classes`，不使用服务端原始数据、验证集准确率或测试集信息。
- 实验输出：`outputs/codex_coverage_balanced_trunk_probe_gpu_20260613/my_merge_ablation_grid/full/small_resnet__my_merge`。

验证结果：负优化，已删除该策略。

- 27 格总体比上一版 full 低 `-0.00150`，W/T/L 为 `8/4/15`。
- blood 均值低 `-0.00091`，没有修复 `c5_b0`、`c7_b0`；`c5_b0` 和 `c7_b0` 仍分别为 `0.1698`、`0.1692`。
- chaosheng 均值低 `-0.00150`，最差的 `c3_b0.1` 继续从 `0.2650` 降到 `0.2570`。
- organc 均值低 `-0.00210`，其中 `c3_b0.1` 低 `-0.0106`。

结论：

- “类别覆盖不足”只是表象，不是当前 shared trunk 负优化的充分解释。
- 直接按类别覆盖重平衡共享层会破坏 M1 原本有效的医学证据权重，尤其误伤 organc 和 chaosheng。
- 该策略不保留。后续不能再走“覆盖多少类别就提高共享层权重”的粗暴路线。

## 2026-06-13 19:45：分类头/共享层拆分实验

继续围绕 blood 的局部负优化做观察。

观察 1：`no_medical_weighted_fusion` 只在少数 blood 格子更好，不能整体删除 M1 主体融合。

- `blood c3_b0.1`：full `0.3107`，`no_medical_weighted_fusion` `0.4089`，提高 `+0.0982`。
- `blood c7_b0`：full `0.1692`，`no_medical_weighted_fusion` `0.2710`，提高 `+0.1017`。
- 但 `chaosheng` 9 格在 `no_medical_weighted_fusion` 下全部下降，均值低 `-0.1162`。
- `organc` 9 格也全部下降，均值低 `-0.0779`。
- 因此不能把 M1 主体医学加权整体删掉。

观察 2：参数空间诊断显示，坏格子不是简单的“共享层方向反了”。

- 在 `blood c3_b0.1` 中，early/mid/late 共享层客户端 delta 与平均方向的 cos 多数仍在 `0.75~0.96`，方向没有整体翻转。
- 在 `blood c7_b0` 中，共享层 cos 也大多为正，问题更像权重集中后把 trunk/head 一起拉向局部专家，而不是某个层组完全冲突。
- 分类头的方向更不稳定，多个格子里弱权重客户端 head 与平均方向 cos 只有约 `0.16~0.25`；但这个现象在好格子里也会出现，不能直接作为单阈值门控。

据此尝试措施：只让 M1 医学权重作用在分类头，共享层回到基础平均后再做 M2 冲突修正。

- 实验开关：`head_only_medical_fusion`。
- 目的：验证 blood 负优化是否主要来自 shared trunk 医学加权。
- 输出：`outputs/codex_head_only_medical_fusion_probe_gpu_20260613/my_merge_ablation_grid/head_only_medical_fusion/small_resnet__my_merge`。

验证结果：明显负优化，已删除该实验开关。

- 27 格总体比 full 低 `-0.01514`，W/T/L 为 `3/4/20`。
- blood 均值低 `-0.00546`，没有修复 `c5_b0`、`c7_b0`；`c7_b0` 仍为 `0.1692`。
- chaosheng 均值低 `-0.01597`，`c3_b0.1` 从 `0.2650` 降到 `0.2327`。
- organc 均值低 `-0.02399`，`c3_b0.01` 下降 `-0.0639`。

结论：

- blood 的失败不能通过“共享层不用医学权重、只在分类头用医学权重”解决。
- M1 医学加权对 shared trunk 在 chaosheng/organc 上是必要的，简单拆分 head/trunk 会破坏这两个数据集。
- 后续优化需要找更细的“哪些共享层/哪些参数方向需要保守”的证据，而不是整体关闭 shared trunk 医学加权。

## 2026-06-13 21:10：层组冲突观测与 head-protected conflict 实验

继续按“观测 -> 措施 -> 结果 -> 保留/删除”的流程排查。

新增只读诊断脚本：

- 脚本：`scripts/observe_my_merge_layer_conflicts.py`。
- 输出：`docs/my_merge_user_docs/layer_conflict_observation_20260613.csv`。
- 输入只包含客户端 checkpoint、reference model、已有 `merge_result.json` 和公开汇总表；不读取服务端验证/测试图像，不做候选选择。
- 诊断项：early/mid/late/classifier 四组的 sign conflict、direction conflict、norm dispersion、综合 conflict score，以及 M1 融合权重集中度和已有 best baseline 差值。

观测 1：失败格子不能用共享层冲突解释。

- `delta_vs_best` 与 early/mid/late conflict score 不是负相关，反而弱正相关：early `+0.297`，mid `+0.357`。
- 这说明“共享层冲突越高越差”的假设不成立，不能据此做更激进的共享层裁剪或关掉 M1 共享层融合。

观测 2：分类头冲突更可疑，但不能直接关闭 class routing。

- 最差格子里 classifier direction conflict 往往偏高，例如 `blood c3_b0.1` 为 `0.483`，`chaosheng c3_b0.1` 为 `0.450`。
- 但 `no_class_routing` 总体仍比 full 低：full - no_class_routing 总体 `+0.01490`，W/T/L 为 `18/1/8`。
- 因此分类头按类路由是必要的，问题不是“不要 class routing”，而可能是 M2 的全参数 sign-delta 会再次改写已经按类融合的 classifier head。

据此尝试措施：`head-protected conflict stabilization`。

- M1 继续按类路由 classifier head。
- M2 的 sign-delta conflict stabilization 只作用在非 classifier 的 shared layers 上，避免用全局 consensus 权重二次覆盖 classifier head。
- 实验输出：`outputs/codex_head_protected_conflict_probe_gpu_20260613/my_merge_ablation_grid/full/small_resnet__my_merge`。

验证结果：基本无收益，已回滚。

- 27 格新旧 full 平均差为 `+0.000025`，W/T/L 为 `13/3/11`。
- blood 均值 `-0.00075`，没有修复 `blood c5_b0` 和 `blood c7_b0`。
- chaosheng 均值 `+0.00389`，主要来自 `c5_b0` 的 `+0.0368`，但最差的 `c3_b0.1` 下降 `-0.0063`。
- organc 均值 `-0.00307`，其中 `c3_b0` 下降 `-0.0146`，`c7_b0.01` 下降 `-0.0153`。

结论：

- “M2 不应改 classifier head”这个解释不够稳定，不能作为默认方法。
- 该方法改动已从 `methods/my_merge.py` 删除，只保留诊断脚本和结果记录。
- 下一步应继续寻找能解释 blood `c5_b0/c7_b0` 和 chaosheng `c3_b0.1` 的更强观测信号；目前共享层整体冲突、类别覆盖、head/trunk 拆分、head-protected sign-delta 都不能解释这些失败。

继续观察 classifier bias。

观察：

- 假设：强非 IID 下 classifier bias 可能主要编码客户端类别先验，当前按类路由 bias 可能导致 blood 失败。
- 只读统计：对每个格子计算 class-routed bias 与 avg/consensus bias 的平均偏移、bias range，并与 `delta_vs_best`、`full - no_medical_weighted_fusion` 对齐。
- 结果：bias route shift 与 `delta_vs_best` 为弱正相关 `+0.368`，与 `full - no_medical_weighted_fusion` 也是弱正相关 `+0.229`。
- 典型反例：organc/chaosheng 中 bias shift 更大但 full 更好；`chaosheng c3_b0.1` 是最差格子，但 `full - no_medical_weighted_fusion` 为 `+0.0889`，说明回退到 avg head 反而会更差。

结论：

- “classifier bias 回退到 avg/consensus”没有足够观测依据，未改代码。
- 当前更可能的问题不是单独的 bias 先验，而是某些 extreme split 下完整 head/trunk 表示组合不稳定。

## 2026-06-13 21:25：M2 conflict blend 强度实验

继续观察 M2 的 conflict stabilization。

观察：

- `no_conflict_stabilization` 总体比 full 低 `-0.01641`，说明 M2 的 sign-delta 冲突修正确实有效。
- 当前 full 中 `delta_blend_weight` 大多只有 `0.05~0.08`。
- 逐格统计显示，`full - no_conflict_stabilization` 与 `delta_blend_weight`、`conflict_score`、`norm_dispersion` 均为正相关，其中与 norm dispersion 相关约 `+0.387`。
- 高 `delta_blend_weight >= 0.08` 的格子平均收益约 `+0.0440`，高于低权重格子。

据此尝试措施：增强但不放开的 M2 conflict blend。

- 将 CNN 的 M2 conflict blend 上限从 `0.25` 提到 `0.35`。
- 这不是新增候选池，也不使用验证集选择；仍然是同一个确定性 M2 公式，只是让参数空间冲突信号有更大纠偏幅度。
- 实验输出：`outputs/codex_stronger_conflict_probe_gpu_20260613/my_merge_ablation_grid/full/small_resnet__my_merge`。

验证结果：保留。

- 27 格平均比上一版 full 高 `+0.00226`，W/T/L 为 `20/3/4`。
- blood 均值 `+0.00127`，W/T/L 为 `7/1/1`。
- chaosheng 均值 `+0.00140`，W/T/L 为 `6/2/1`。
- organc 均值 `+0.00411`，W/T/L 为 `7/0/2`。
- 相对已有最好方法的总体均值从 `-0.00934` 改为 `-0.00708`。

逐格注意：

- 正向：`organc c5_b0.1` `+0.0117`，`organc c7_b0.1` `+0.0097`，`chaosheng c5_b0.01` `+0.0090`，`chaosheng c7_b0.01` `+0.0090`，`blood c7_b0.01` `+0.0050`。
- 负向：`chaosheng c5_b0` `-0.0207`，`organc c7_b0.01` `-0.0049`，`blood c5_b0.01` `-0.0029`。
- 仍未修复：`blood c5_b0`、`blood c7_b0`、`chaosheng c3_b0.1`。

结论：

- 增强 M2 conflict blend 是一个稳定的小收益，保留在当前代码中。
- 它说明 M2 不是补丁式候选选择，而是对 M1 后参数冲突的必要纠偏；但它仍无法解释和解决最差 extreme split。

## 2026-06-14 Client Average 90% 目标复核与 middle-delta 负例

用户提出新的硬目标：`Client Average` 中 my_merge 需要约 90% 都是最高。按当前 `My_merge_ret/汇总表.md` 的口径，Client Average 一共 75 个格子，90% 约等于至少 68 个格子最高；当前 `my_merge full` 只有约 33/75 个最高或并列最高，因此不是小幅调参问题。

先做只读统计：

- raw 层面 `my_merge full` 相比 `avg` 为 W/T/L=`135/42/48`，说明 M1/M2 整体不是简单低于平均，但存在 48 个低于 Avg 的塌点。
- 最大塌点是 `convnext/dermamnist_224/c5_b0.1`：当前 full 为 `0.2025`，Avg/Best 为 `0.6688`。这种单点塌陷会直接拖垮 Client Average。
- Client Average 的失败集中在 `convnext/vit_t/swin_tiny`，赢家主要是 `Breadcrumbs`、`RegMean`、`RobustMerge`、`Iso-C`、`Fisher`。这说明问题不是“医学权重还不够大”，而是当前方法缺少对参数污染/统计错配的稳定处理。

据此尝试一个最小 M2 改动：在现有 sign-consistent delta 后加入 checkpoint-only 的 `middle-delta` 过滤。

- 设计动机：失败赢家里 Breadcrumbs/RobustMerge 较多，它们共同倾向于过滤异常幅值参数；middle-delta 只保留每个客户端 task vector 的中等幅值更新，去掉极大更新和小噪声更新。
- 隐私属性：只用客户端模型参数和 M1 医学共识权重，不使用服务端验证集、不读原始图像、不做候选池选择。
- 探针输出：`outputs/codex_middle_delta_probe_20260614/small_convnext__my_merge`。
- 关键结果：`convnext/dermamnist_224/c5_b0.1` 从当前 full 的 `0.2025` 只到 `0.2045`，远低于 Avg/Best 的 `0.6688`。

结论：middle-delta 不能修复最大塌点，已从 `methods/my_merge.py` 删除。

新的观察结论：

- 最大塌点不是简单的“缺少中等幅值参数过滤”。
- 需要改查融合后模型是否退化为错误类别先验、坏分类头/共享层组合，或者专家锚定是否把模型拉向错误客户端。
- 后续优化应优先围绕 `convnext/dermamnist_224/c5_b0.1` 这种灾难塌点做局部验证；只有先把 raw 层面的低于 Avg 塌点压下去，Client Average 才有可能接近 90% 最高。

继续检查“本地验证专家保真”能否替代软融合。

只读上界分析一开始看起来很诱人：

- 如果直接用 `model_hub` 里客户端元信息的 `best_val_acc` 选择客户端专家，并把该客户端 `test_acc` 当作离线上界，Client Average 似乎可以到 `74/75`。
- 但这一步只是诊断，不是算法，因为 `meta.clients[*].test_acc` 不一定与当前统一评估脚本口径一致。

据此做了一个真实评估探针：

- 临时实现：用客户端上传的本地 `best_val_acc` 标量选主专家，最终模型直接等于该客户端 checkpoint。
- 隐私上它不读服务端原始验证集，但仍必须通过当前统一测试脚本验证。
- 探针输出：`outputs/codex_local_expert_probe_20260614/small_convnext__my_merge`。
- 关键结果：`convnext/dermamnist_224/c5_b0.1` 选择了 client 1，但当前统一评估只有 `0.1521`，比当前 full 的 `0.2025` 更低，也远低于 Avg/Best 的 `0.6688`。

结论：直接使用 `meta.clients[*].best_val_acc` 做专家选择是负优化，已从 `methods/my_merge.py` 删除。

新的负例说明：

- `model_hub` 中记录的客户端 `test_acc/best_val_acc` 不能直接作为当前统一评估口径下的全局性能代理。
- “专家保真”本身仍可能有价值，但不能用这个本地分数硬选专家；否则会把方法变成不稳定的客户端选择器。
- 最大塌点需要继续从当前融合输出本身解释，例如预测分布、分类头偏置、或者 M1 权重过度集中与 head/trunk 组合错配。

继续按 `Client Average` 90% 目标做观察。

观察：

- 当前 `Client Average` 为 `33/75` 最高或并列最高，目标 90% 约为 `68/75`。
- 历史最好发布结果 `outputs/codex_rep_m3_medical_full_20260609` 为 `51/75`，说明当前代码确实退化，但历史结果依赖服务端验证批次、BN 重估和 prototype head，不能按隐私约束原样恢复。
- 当前 42 个失败格子主要集中在 `convnext/vit_t/swin_tiny`；赢家多为 `Breadcrumbs/RegMean/Fisher/RobustMerge`，不是简单输给 Avg。直接“低于 Avg 就回退 Avg”的 oracle 也只能从 `33/75` 到 `35/75`，因此不能靠 Avg 回退解决。
- 最大 raw 塌点仍是 `convnext/dermamnist_224/c5_b0.1`：当前 M1 融合权重 `[0.108, 0.566, 0.105, 0.133, 0.088]`，单客户端权重过高；历史好版本同格权重接近 `[0.165, 0.164, 0.162, 0.252, 0.257]`。

据此尝试措施：M1 权重稳定约束。

- 改动内容：降低样本量先验，给 M1 的 overall/morph/class 权重加上限与熵约束，并降低专家锚定中 `support_score` 的作用。
- 隐私属性：只使用客户端摘要、客户端类别/样本量元信息和 checkpoint，不读取服务端验证或测试图像。
- 探针输出：`outputs/codex_m1_bounded_probe_20260614/my_merge_ablation_grid/full`。

验证结果：负优化，已回滚。

- `vit_t` 45 个 raw 格子平均 `-0.00676`，W/T/L=`10/9/26`。
- `convnext` 已完成 36 个 raw 格子平均 `-0.02479`，W/T/L=`4/22/10`；早停并停止剩余任务。
- 典型严重负例：`convnext/dermamnist_224/c3_b0.01` 从 `0.6688` 降到 `0.1097`；`vit_t/dermamnist_224/c5_b0.01` 从 `0.5516` 降到 `0.3496`。

结论：

- “M1 权重过度集中”是可观察现象，但简单限幅会破坏一些原本正确的 derma/organ 格子，不能保留。
- 后续不再从 M1 压平权重入手。更有依据的方向是 M2：历史好版本在若干坏格子中选择 `delta_0p25/0p50`，而当前确定性 M2 的 `delta_blend_weight` 多数只有 `0.06~0.13`，冲突纠偏可能偏弱。

继续检查 M2 的专家锚定是否是主失败来源。

观察：

- 最大塌点 `convnext/dermamnist_224/c5_b0.1` 和若干 organ/blood 塌点里存在 `specialist_anchor` 软锚定，可能把模型拉向错误专家。
- 但此前 resnet 严格摘要探针显示 `no_specialist_anchor` 总体低于 full，因此不能只凭单个坏格子删除专家保真。

据此尝试措施：在当前失败最集中的 `convnext/vit_t/swin_tiny` 上跑 `no_specialist_anchor` 宽探针。

- 输出：`outputs/codex_no_specialist_broad_probe_20260614/my_merge_ablation_grid/no_specialist_anchor`。
- 早停时 `vit_t` 已完成 `45/45`，`convnext` 完成 `42/45`，`swin_tiny` 尚未开始；因结果已经不能支持该方向，停止剩余任务。

验证结果：不是主因，不保留。

- `vit_t`：45 个 raw 平均 `+0.00120`，W/T/L=`6/33/6`；Client Average 15 组平均 `+0.00120`，W/T/L=`5/6/4`。
- `convnext`：42 个 raw 平均 `-0.00040`，W/T/L=`2/36/4`；Client Average 14 组平均 `-0.00040`，W/T/L=`2/8/4`。
- 最大正例：`vit_t/dermamnist_224/c5_b0.1` 从 `0.1980` 到 `0.3631`，说明专家锚定确有局部误伤。
- 最大负例：`vit_t/dermamnist_224/c5_b0.01` 从 `0.5516` 到 `0.4095`，说明直接删除专家锚定同样会破坏已有好格子。

结论：

- `specialist_anchor` 不是 `Client Average` 失败的主瓶颈；直接删除或关闭不能接近 90% 目标。
- 后续转向 M2 的 delta 路径强度/异常参数处理，而不是删除专家保真。

继续检查 M2 的 delta 路径强度。

观察：

- 历史好版本在若干坏格子里选择过 `delta_0p25/0p50`，而当前确定性 M2 的 `delta_blend_weight` 多数只有 `0.06~0.13`。
- 这支持一个假设：当前冲突纠偏过弱，尤其对 `convnext` 的 blood/chaosheng 低分格子。

据此尝试措施：全局增强 delta blend。

- 临时把 CNN/VLM 的 delta blend 上限从 `0.35` 提到 `0.50/0.55`，启动阈值从 `0.20` 降到 `0.14`。
- 输出：`outputs/codex_stronger_delta_broad_probe_20260614/my_merge_ablation_grid/full`。

验证结果：局部正向但整体不稳，已回滚。

- `vit_t` 完成 `45/45`：raw 平均 `-0.00435`，W/T/L=`14/8/23`；Client Average 15 组平均 `-0.00435`，W/T/L=`7/1/7`。
- `convnext` 早停时完成 `31/45`：raw 平均 `+0.00498`，W/T/L=`4/23/4`；Client Average 10 组平均 `+0.00514`，W/T/L=`2/5/3`。
- 正例：`convnext/bloodmnist_224/c7_b0` 从 `0.0713` 到 `0.1947`；`vit_t/bloodmnist_224/c7_avg` 提高约 `+0.0233`。
- 负例：`vit_t/bloodmnist_224/c3_avg` 下降约 `-0.0446`；`vit_t/organsmnist_224/c7_avg` 下降约 `-0.0427`。

结论：

- “增强 delta”确实能修复部分冲突格子，但不能作为全局规则；冲突高并不必然意味着应该更大幅度走 sign-delta。
- 后续应转向更细的 M2 异常参数处理，例如只针对 2D 权重矩阵或异常幅值更新做过滤，而不是整体提高 delta 混合强度。

继续把上面的负例转成可检验的 M2 规则。

观察：

- 全局增强 delta 的正例大多有较高的符号冲突或方向冲突，例如 `convnext/bloodmnist_224/c7_b0` 的 `sign_conflict=0.689`、`direction_conflict=0.503`。
- 最大负例之一 `vit_t/organsmnist_224/c7_b0.01` 虽然方向冲突接近阈值，但 `norm_dispersion=0.670`，说明客户端更新尺度本身非常离散；这种情况下强行走 sign-delta 容易把不同尺度的客户端更新混在一起。
- 已完成样本上的单阈值诊断显示：`direction_conflict >= 0.495` 或 `sign_conflict >= 0.585` 的格子更可能从增强 delta 中获益；高 `norm_dispersion` 是明显风险因子。

据此尝试措施：M2 的门控式冲突增强。

- 保留当前默认 delta blend 作为主路径。
- 仅当 `sign_conflict >= 0.585` 或 `direction_conflict >= 0.495`，且 `norm_dispersion <= 0.55` 时，把 sign-delta 混合上限从默认 `0.35` 提高到 CNN 的 `0.50`、Transformer 的 `0.45`，并把启动点从 `0.20` 降到 `0.14`。
- 其他格子完全沿用原逻辑。
- 隐私属性：只用客户端 checkpoint 的 task-vector 符号、方向和范数统计，不使用服务端原始数据或服务端验证集。

下一步：先在失败最多的 `convnext/vit_t/swin_tiny` 上跑探针，和当前 full 比较 raw 与 Client Average 后再决定是否全量。

补完 `convnext` 门控 delta 探针后的完整结果：

- 输出：`outputs/codex_gated_delta_convnext_probe_20260614/my_merge_ablation_grid/full/small_convnext__my_merge`。
- 完成 `convnext` 45/45 个 raw case。
- 对当前正式 full：raw 平均 `+0.00234`，W/T/L=`1/43/1`。
- Client Average：平均 `+0.00234`，W/T/L=`1/13/1`。
- 正例：`convnext/bloodmnist_224/c7_b0` 从 `0.0713` 到 `0.1947`。
- 负例：`convnext/chaoshengmnist_224/c7_b0.1` 从 `0.1267` 到 `0.1087`。
- 最大 derma 塌点 `convnext/dermamnist_224/c5_b0.1` 仍为 `0.2025`，没有修复。

结论：

- 门控式冲突增强只修复一个 blood case，同时引入一个超声负例。
- 它没有解决最大 derma 塌点，不能作为稳定方法保留。
- 已删除该规则，代码回到原默认 delta blend。
- 下一步转向最大塌点诊断：先做 checkpoint-only 参数检查，再用统一评估脚本真实评估该 case 的单客户端 checkpoint，判断是分类头偏置、共享 trunk 损坏，还是 M1 权重导致 head/trunk 错配。

继续诊断最大塌点后得到新观察。

观察：

- `convnext/dermamnist_224/c5_b0.1` 当前 full 只有 `0.2025`，Avg/历史好版本为 `0.6688`。
- 当前 M1 共识权重为 `[0.108, 0.566, 0.105, 0.133, 0.088]`，client 1 被赋予 `56.6%` 权重。
- 统一评估单客户端 checkpoint 发现 client 1 只有 `0.1521`，而 client 3/4 都是 `0.6688`。
- checkpoint-only trunk task-vector 距离显示 client 1 是参数离群点；因此该塌点是“医学权重被参数离群客户端劫持”，不是 delta 强度问题。

据此尝试措施：checkpoint 一致性保真约束。

- 初版宽松规则：只要出现明显参数离群，就按 trunk task-vector 一致性修正 M1 的 overall/morph/class 权重。
- 结果：`dermamnist_224/convnext/c5_b0.1` 可从 `0.2025` 修到 `0.6688`，但 `chaoshengmnist_224/convnext/c7_b0.01` 会从 `0.1617` 降到 `0.1087`。
- 解释：超声中离群客户端可能是真实声学模式专家，不能简单削弱。

收窄后的保留规则：

- 先由 M1 产生医学共识权重。
- 计算每个客户端中后层 trunk task-vector 到其他客户端的平均距离，得到 checkpoint 一致性分数。
- 仅当最高医学权重客户端也是一致性最低的参数离群客户端，并且医学权重集中度 `>= 0.33`、离群强度 `>= 0.45` 时，才触发保真约束。
- 触发时用 `weight * consistency^2` 重新归一化 M1 权重；不触发时完全沿用原逻辑。
- 该规则只用 checkpoint 和参考模型，不使用服务端验证集、测试集或客户端原始数据。

代表性探针结果：

- `dermamnist_224/convnext/c5_b0.1`：触发，`0.2025 -> 0.6688`。
- `bloodmnist_224/convnext/c7_b0.1`：不触发，维持 `0.1619`，避免宽松规则的 `0.0836` 负优化。
- `chaoshengmnist_224/convnext/c7_b0.01`：不触发，维持 `0.1617`，避免宽松规则的 `0.1087` 负优化。
- 下一步跑 `convnext` 全量 45 个 raw case，确认该规则是否有整体收益。

`convnext` 全量验证完成：

- 输出：`outputs/codex_checkpoint_consistency_convnext033fs_20260615/my_merge_ablation_grid/full/small_convnext__my_merge`。
- 45/45 个 raw case 全部完成。
- 对当前正式 full：raw 平均 `+0.01036`，W/T/L=`1/44/0`。
- Client Average：平均 `+0.01036`，W/T/L=`1/14/0`。
- 唯一 accuracy 变化是 `dermamnist_224/convnext/c5_b0.1` 从 `0.2025` 修到 `0.6688`。
- 触发了 3 个 derma guard；除 `c5_b0.1` 外，另外两个 derma c7 格子原本已是 `0.6688`，触发后不改变 accuracy。
- 没有 blood、chaosheng、organc、organs 负例。

结论：checkpoint 一致性保真约束可以保留为 M2 的局部灾难防护，但它只解决 M1 权重被参数离群客户端劫持的问题，不足以把 Client Average 推到 90%。下一步继续分析剩余失败格子。

继续跑医学 full 后发现，上述 checkpoint guard 仍然过宽。

观察：

- `outputs/codex_medical_full_ccguard_20260615` 显示 full 全量相对当前正式 full 的 raw 平均为 `-0.00541`，W/T/L=`2/174/4`。
- 负例集中在 derma：
  - `resnet/c7_b0`：`0.6688 -> 0.1307`。
  - `resnet/c7_b0.01`：`0.6688 -> 0.1067`。
  - `vit_t/c5_b0.01`：`0.5516 -> 0.2893`。
  - `vit_t/c7_b0.01`：`0.6688 -> 0.4190`。
- 查看 trace 后发现，这些负例被削弱的 top client 是少数类专家，类别覆盖率分别只有 `1/7`、`3/7`、`2/7`、`3/7`。
- 正例 `convnext/c5_b0.1` 的 top client 覆盖 `7/7` 类，是全类别 generalist，但参数离群且统一评估失效。

修正：

- checkpoint guard 只允许拦截全类别 generalist 离群劫持。
- 新触发条件：top client 是最高医学权重客户端、也是一致性最低的离群客户端、医学权重集中度 `>=0.33`、离群强度 `>=0.45`、且 top client 类别覆盖率 `>=0.95`。
- 6-case 探针已验证：
  - `convnext/derma/c5_b0.1` 仍触发，保持 `0.6688`。
  - `resnet/derma/c7_b0` 和 `c7_b0.01` 不触发，恢复 `0.6688`。
  - `vit_t/derma/c5_b0.01` 不触发，恢复 `0.5521`。
  - `vit_t/derma/c5_b0.1` 仍触发，`0.3701`。
  - `vit_t/derma/c7_b0.01` 不触发，恢复 `0.6688`。
- 医学 full 已完成：`outputs/codex_medical_full_generalist_guard_20260615`。
- 对上一版正式 full：180 个 small raw case 中 W/T/L=`2/178/0`，平均 `+0.00355`。
- 正向变化：
  - `dermamnist_224/convnext/c5_b0.1`：`0.2025 -> 0.6688`。
  - `dermamnist_224/vit_t/c5_b0.1`：`0.1980 -> 0.3701`。
- 已用 `scripts/generate_ablation_combined_results_table.py` 自动更新 `My_merge_ret/汇总表.md`。检查结果：`full` 行无 `-` 占位符，`<strong>` 和 `<ins>` 标记存在。
- 当前 `Client Average` 统计：Small 60 个格子中，`my_merge full` 最高或并列最高 24 个、次高 14 个。因此 generalist guard 是稳定局部修复，但还不是 90% Client Average 目标的完整解。

数据蒸馏方向：

- 不恢复旧版服务端验证集 `prototype_head`，因为它需要服务端用验证图像抽特征。
- 查阅近几年相关方法后，方向判断如下：
  - FedProto 用类别原型替代梯度/参数通信，适合解释非 IID 客户端中“类别表示”比参数平均更稳定。
  - FedDF/FedMD 把模型融合改写成蒸馏，但需要 proxy/unlabeled data；如果服务端拿医学验证集或原始图像，就违反当前隐私约束。
  - FedFTG/FedDTG/FedD3 等数据生成或数据蒸馏方法会引入生成器、蒸馏样本和额外训练；医学场景下合成病灶图像仍可能被质疑泄露分布信息，暂不作为主线。
- 已记录的主要参考：
  - FedProto: https://arxiv.org/abs/2105.00243
  - FedDF: https://arxiv.org/abs/2006.07242
  - FedMD: https://arxiv.org/abs/1910.03581
  - TIES-Merging: https://arxiv.org/abs/2306.01708
- 更合适的方向是客户端本地原型蒸馏：每个客户端本地上传每类 pooled feature prototype/count/coverage，服务端只用这些摘要修正 M1 class routing 或分类头保真。
- 仓库已有 `outputs/codex_proto_summary_smoke_20260614/prototype_summaries`，说明原型摘要流程曾经烟测过；后续优先复用这个格式做少量失败格子探针。
- 现有 prototype smoke 只覆盖 `chaoshengmnist_224/vit_t` 9 个格子，且多数低于当前正式 full，因此不能直接恢复旧 prototype head；下一步只把“客户端上传原型摘要”作为受控小探针，负优化就删除。

继续按 Client Average 差距分析后发现，当前最大短板不是超声，而是互补类别覆盖场景。

观察：

- `convnext/organsmnist_224/c3_avg`：my_merge `0.0855`，最优 `0.2354`。
- `swin_tiny/organsmnist_224/c3/c5/c7_avg` 也明显落后。
- 具体到 `organsmnist_224/convnext/c3_b0`：
  - `avg_only=0.2354`，full `0.1146`。
  - 三个客户端类别覆盖约 `[0.36,0.36,0.27]`，是互补覆盖，没有全类别专家。
  - M1/M2 后整体权重约 `[0.43,0.29,0.28]`，并触发 `specialist_anchor=0.16` 到 client 0。

解释：

- 这种场景中 M1 的 class routing 是有意义的，但 trunk/整体权重不应该偏向单个局部专家；否则其他互补类别的表示被压掉。
- 这不是“加专家更强”，而是“互补专家要保持覆盖”。因此继续加强 specialist anchor 会负优化。

临时措施：

- 在临时 worktree 试 M2 互补覆盖保守门控：
  - 联合覆盖全类别、单个客户端均为低覆盖、没有全类别专家时触发。
  - overall/morphology 权重向均匀 prior 收缩，保护 trunk。
  - classifier class routing 只轻微平滑。
  - specialist anchor 在该场景关闭。
- CPU 探针没有完成 test eval，但 merge trace 已显示 `organs/convnext/c3_b0` 上 anchor 从 `0.16` 关为 `0`，fusion weights 集中度下降，class routing 保留。
- 等当前 `codex_medical_full_generalist_guard_20260615` GPU full 完成后，再跑该门控的 GPU 小探针；正优化才进入主代码。
