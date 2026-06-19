# my_merge 观测-措施-结果记录

更新时间：2026-06-15

本文放在 `汇总表.md` 同目录，用于记录当前方法优化过程中的关键观测、采取措施和实验结果。这里不把未完成探针写成最终结论。

## 当前硬目标

- 用户目标：`Client Average` 中 my_merge 约 90% 达到最高或并列最高。
- 当前 `汇总表.md` 口径：`Client Average` 一共 75 个格子，90% 约等于至少 68 个格子最高。
- 当前正式结果：my_merge full 约 `33/75` 个格子最高或并列最高。
- 结论：这不是小幅调参问题，必须先解决若干 raw 层面的灾难塌点。

## 隐私约束

- 不能用服务端验证集挑候选模型。
- 不能把原始医学数据上传到服务端。
- M1 的医学特征只能解释为客户端本地计算后上传的统计摘要。
- M2 只能使用客户端 checkpoint、参考模型和客户端上传摘要，不使用服务端 raw data。

## 观测 1：失败集中在特定模型和数据集

观测：

- 当前失败主要集中在 `convnext`、`vit_t`、`swin_tiny`。
- 失败数据集主要是 `bloodmnist_224`、`organsmnist_224`、`chaoshengmnist_224`。
- 失败格子的赢家多为 `breadcrumbs`、`regmean`、`fisher`、`robustmerge`，说明问题更像参数冲突或统计错配，而不是医学权重不够大。

措施：

- 不再继续堆大候选池。
- 优先围绕 M2 的冲突处理做小范围探针。

结果：

- 该观测仍成立，后续优化继续围绕“医学权重之后的参数冲突”展开。

## 观测 2：最大塌点不是简单幅值过滤能解决

观测：

- 最大 raw 塌点之一：`convnext/dermamnist_224/c5_b0.1`。
- 当前 full：`0.2025`。
- Avg/Best：`0.6688`。

措施：

- 尝试过 `middle-delta`：只保留 task vector 中等幅值更新，过滤极大异常更新和极小噪声更新。

结果：

- `convnext/dermamnist_224/c5_b0.1` 只从 `0.2025` 到 `0.2045`。
- 结论：`middle-delta` 不能修复最大塌点，已从代码删除。

## 观测 3：直接选本地最优专家不可靠

观测：

- 只读上界中，用 `model_hub` 记录的客户端 `best_val_acc/test_acc` 似乎能让 Client Average 很高。
- 但这些元信息不一定等价于当前统一评估口径。

措施：

- 临时实现“选择本地 `best_val_acc` 最高客户端 checkpoint 作为最终模型”，并用当前统一测试脚本真实评估。

结果：

- `convnext/dermamnist_224/c5_b0.1` 选择 client 1 后，统一评估只有 `0.1521`。
- 低于当前 full 的 `0.2025`，也远低于 Avg/Best 的 `0.6688`。
- 结论：不能用本地分数硬选专家，已删除该实验代码。

## 观测 4：M1 权重集中是现象，但简单压平是负优化

观测：

- `convnext/dermamnist_224/c5_b0.1` 当前 M1 融合权重约 `[0.108, 0.566, 0.105, 0.133, 0.088]`，单客户端权重明显偏高。
- 历史好版本同格权重更平滑，约 `[0.165, 0.164, 0.162, 0.252, 0.257]`。

措施：

- 尝试 M1 权重稳定约束：降低样本量先验、限制 M1 权重上限、提高熵下限、降低专家锚定中 `support_score` 的作用。

结果：

- `vit_t` 45 个 raw 格子平均 `-0.00676`，W/T/L=`10/9/26`。
- `convnext` 已完成 36 个 raw 格子平均 `-0.02479`，W/T/L=`4/22/10`。
- 典型负例：`convnext/dermamnist_224/c3_b0.01` 从 `0.6688` 降到 `0.1097`。
- 结论：M1 权重集中不能靠简单压平解决，已回滚。

## 观测 5：专家锚定不是主瓶颈

观测：

- 一些塌点里 `specialist_anchor` 可能把模型拉向错误客户端。

措施：

- 跑 `no_specialist_anchor` 宽探针，覆盖失败最集中的 `vit_t` 和 `convnext`。

结果：

- `vit_t` 45 个 raw 平均 `+0.00120`，Client Average 平均 `+0.00120`。
- `convnext` 42 个 raw 平均 `-0.00040`，Client Average 平均 `-0.00040`。
- 局部有正例，也有明显负例。
- 结论：直接删除专家锚定不能接近 90% 目标，不保留该改动。

## 观测 6：全局增强 delta 有局部收益，但不稳定

观测：

- 历史较好版本曾选到 `delta_0p25/0p50`。
- 当前确定性 M2 的 `delta_blend_weight` 多数较小，约 `0.06~0.13`。

措施：

- 临时全局增强 delta blend：CNN/VLM 上限从 `0.35` 提到 `0.50/0.55`，启动阈值从 `0.20` 降到 `0.14`。

结果：

- `vit_t` 完成 45/45：raw 平均 `-0.00435`，W/T/L=`14/8/23`。
- `convnext` 早停时完成 31/45：raw 平均 `+0.00498`，W/T/L=`4/23/4`。
- 正例：`convnext/bloodmnist_224/c7_b0` 从 `0.0713` 到 `0.1947`。
- 负例：`vit_t/organsmnist_224/c7_avg` 下降约 `-0.0427`。
- 结论：delta 增强有真实局部收益，但不能全局打开，已回滚。

## 观测 7：门控式冲突增强不是稳定收益

观测：

- 全局增强 delta 的正例通常有较高符号冲突或方向冲突。
- 最大负例之一存在较高 `norm_dispersion`，说明客户端更新尺度很离散，强行 sign-delta 容易破坏模型。

措施：

- 尝试保留默认 delta blend，并只在 `sign_conflict >= 0.585` 或 `direction_conflict >= 0.495`，且 `norm_dispersion <= 0.55` 时增强 sign-delta。
- CNN 上限从 `0.35` 提到 `0.50`，Transformer 上限提到 `0.45`，启动点从 `0.20` 降到 `0.14`。
- 该规则只用 checkpoint 的 task-vector 统计，不用服务端验证集。

完整结果：

- 探针目录：`outputs/codex_gated_delta_convnext_probe_20260614`。
- `convnext` 已完成 45/45 个 raw case。
- 对当前正式 full：raw 平均 `+0.00234`，W/T/L=`1/43/1`。
- Client Average：平均 `+0.00234`，W/T/L=`1/13/1`。
- 正例：`convnext/bloodmnist_224/c7_b0` 从 `0.0713` 到 `0.1947`。
- 负例：`convnext/chaoshengmnist_224/c7_b0.1` 从 `0.1267` 到 `0.1087`。
- 最大 derma 塌点 `convnext/dermamnist_224/c5_b0.1` 仍为 `0.2025`，没有修复。

结果：

- 门控式冲突增强只修复一个 blood case，同时引入一个超声负例。
- 它没有解决最大 derma 塌点。
- 结论：不保留该规则，代码已回滚到原默认 delta blend。

## 下一步计划

1. 对最大塌点做 checkpoint-only 检查：判断是否是分类头偏置、共享 trunk 损坏，还是 M1 权重导致 head/trunk 错配。
2. 用当前统一评估脚本真实评估该 case 的单客户端 checkpoint，确认 `meta.json` 里的客户端分数能否作为诊断依据。
3. 后续每个新措施都继续按“观测 -> 措施 -> 结果 -> 是否保留”的格式记录。

## 观测 8：最大塌点来自 M1 被参数离群客户端劫持

对象：`convnext/dermamnist_224/c5_b0.1`。

观测：

- 当前正式 full：`0.2025`。
- 历史好版本和 Avg：`0.6688`。
- 当前 M1 融合权重为 `[0.108, 0.566, 0.105, 0.133, 0.088]`，其中 client 1 占 `56.6%`。
- 统一评估 5 个客户端 checkpoint 后发现：client 1 在 `meta.json` 中记录的 `test_acc=0.7192`，但当前统一评估只有 `0.1521`。
- 同一统一评估口径下，client 3 和 client 4 都是 `0.6688`。
- checkpoint-only 参数距离显示，client 1 的 trunk 与其他客户端距离极远，平均 trunk 距离约 `88.0`；其他客户端平均 trunk 距离约 `34.9~48.1`。

结论：

- 当前塌点不是“delta 强度不够”，而是 M1 把医学/样本/覆盖权重集中给了一个参数空间离群客户端。
- 这个离群客户端在旧元信息里看起来强，但在当前统一评估口径下并不可靠。
- 不能用客户端 `meta.test_acc/best_val_acc` 作为算法输入；但可以用 checkpoint 之间的 task-vector 一致性做隐私安全约束。

拟采取措施：

- 在 M2 中加入 checkpoint-only 的“医学权重保真约束”：先由 M1 计算医学权重，再用客户端 trunk task-vector 的同伴一致性修正权重。
- 如果某客户端与多数客户端方向/距离明显不一致，就降低它对医学加权融合、分类头路由、sign-delta 和专家锚定的影响。
- 该约束只用客户端 checkpoint 和参考模型，不使用服务端 raw data，也不使用服务端验证集。

## 观测 9：一致性约束必须只拦截“医学权重劫持”，不能拦截普通专家差异

对象：`convnext` 代表性探针。

观测：

- 宽松 checkpoint 一致性约束可以把最大塌点 `dermamnist_224/c5_b0.1` 从 `0.2025` 修到 `0.6688`。
- 但同一规则会误伤超声：`chaoshengmnist_224/c7_b0.01` 从当前 full 的 `0.1617` 降到 `0.1087`。
- trace 显示二者都满足“最高医学权重客户端也是参数离群点”，但程度不同：
  - derma 塌点：医学权重集中度 `0.366`，离群强度 `0.525`，最高权重 client 1 也是离群 client 1。
  - chaosheng 误伤：医学权重集中度 `0.287`，离群强度 `0.514`，最高权重 client 2 也是离群 client 2。
- 解释：超声中某个客户端可能因为真实声学/病灶模式差异而成为“有用但不同”的专家，不能仅凭参数离群就削弱；derma 塌点则是 M1 把 `56.6%` 权重给了一个在当前统一评估口径下失效的参数离群客户端。

措施：

- 将 M2 的 checkpoint 一致性保真约束收窄为“医学权重劫持拦截器”：
  - 先由 M1 计算医学共识权重。
  - 找最高医学权重客户端和 checkpoint 一致性最低客户端。
  - 只有当最高权重客户端就是离群客户端、医学权重集中度 `>= 0.33`、离群强度 `>= 0.45` 时，才用 trunk task-vector 一致性削弱该客户端。
  - 其余情况不触发，保留原 M1/M2 输出。
- 该规则仍只使用客户端 checkpoint 和参考模型，不读取服务端验证集，不使用测试集或客户端原始数据。

代表性结果：

- `dermamnist_224/convnext/c5_b0.1`：触发，`0.2025 -> 0.6688`。
- `bloodmnist_224/convnext/c7_b0.1`：不触发，维持 `0.1619`，避免宽松规则造成的 `0.0836` 负优化。
- `chaoshengmnist_224/convnext/c7_b0.01`：不触发，维持 `0.1617`，避免宽松规则造成的 `0.1087` 负优化。
- `chaoshengmnist_224/convnext/c7_b0.1`：不触发，维持 `0.1267`。

下一步：

- 在 `convnext` 全量 45 个 raw case 上验证该窄规则是否只修复塌点而不引入新的系统性负优化。

全量验证结果：

- 输出：`outputs/codex_checkpoint_consistency_convnext033fs_20260615/my_merge_ablation_grid/full/small_convnext__my_merge`。
- `convnext` 完成 45/45 个 raw case。
- 对当前正式 full：raw 平均 `+0.01036`，W/T/L=`1/44/0`。
- Client Average：平均 `+0.01036`，W/T/L=`1/14/0`。
- 唯一变化：`dermamnist_224/convnext/c5_b0.1` 从 `0.2025` 到 `0.6688`。
- 触发了 3 个 derma guard，但其中 `c7_b0` 和 `c7_b0.01` 原本已经是 `0.6688`，触发后没有改变 accuracy；没有 blood、chaosheng、organ 负例。

结论：

- 该规则是稳定正向的局部修复，可以保留到 M2。
- 它解决的是“医学权重被参数离群客户端劫持”的灾难塌点，不是通用提分模块。
- 由于 `convnext` 只修复 1 个 Client Average，距离 `90%` Client Average 目标仍很远；下一步应继续观察剩余失败格子，尤其是 `blood/organc/organs` 中不是 M1 劫持导致的低分。

## 观测 10：checkpoint 一致性不能削弱少数类专家

对象：医学 full 全量探针 `outputs/codex_medical_full_ccguard_20260615`。

观测：

- `convnext` 上的窄规则看起来稳定，但医学全量显示它仍会误伤 `resnet/vit_t` 的 derma。
- 负例：
  - `dermamnist_224/resnet/c7_b0`：`0.6688 -> 0.1307`。
  - `dermamnist_224/resnet/c7_b0.01`：`0.6688 -> 0.1067`。
  - `dermamnist_224/vit_t/c5_b0.01`：`0.5516 -> 0.2893`。
  - `dermamnist_224/vit_t/c7_b0.01`：`0.6688 -> 0.4190`。
- 对这些负例看 trace 后发现：被一致性削弱的最高权重客户端不是全类别客户端，而是少数类/局部专家：
  - `resnet/c7_b0` 的 top client 只覆盖 `1/7` 类。
  - `resnet/c7_b0.01` 和 `vit_t/c7_b0.01` 的 top client 覆盖 `3/7` 类。
  - `vit_t/c5_b0.01` 的 top client 覆盖 `2/7` 类。
- 正例 `convnext/derma/c5_b0.1` 的 top client 覆盖 `7/7` 类，是一个全类别 generalist；它被 M1 赋予 `56.6%` 权重但在当前统一评估口径下失效。

结论：

- checkpoint 一致性只能拦截“全类别 generalist 离群劫持”。
- 对少数类专家，参数离群可能正是类别专长带来的表示差异，不能削弱；否则会破坏类别覆盖。

措施：

- 将 M2 的 checkpoint guard 再收窄：只有当 top client 同时满足
  - 是最高 M1 医学权重客户端；
  - 是 checkpoint 一致性最低的参数离群客户端；
  - 医学权重集中度 `>= 0.33`；
  - 离群强度 `>= 0.45`；
  - 类别覆盖率 `>= 0.95`，即全类别 generalist；
  才触发一致性削弱。
- 关键 6-case 探针结果：
  - `convnext/derma/c5_b0.1`：仍触发，`0.6688`。
  - `resnet/derma/c7_b0`：不触发，恢复 `0.6688`。
  - `resnet/derma/c7_b0.01`：不触发，恢复 `0.6688`。
  - `vit_t/derma/c5_b0.01`：不触发，恢复 `0.5521`。
  - `vit_t/derma/c5_b0.1`：仍触发，`0.3701`。
  - `vit_t/derma/c7_b0.01`：不触发，恢复 `0.6688`。
- 医学 full 全量已完成：`outputs/codex_medical_full_generalist_guard_20260615`。
- 对上一版正式 full：180 个 small raw case 中 W/T/L=`2/178/0`，平均 `+0.00355`。
- 唯一改变的两个格子都是正向修复：
  - `dermamnist_224/convnext/c5_b0.1`：`0.2025 -> 0.6688`。
  - `dermamnist_224/vit_t/c5_b0.1`：`0.1980 -> 0.3701`。
- `My_merge_ret/汇总表.md` 已由 `scripts/generate_ablation_combined_results_table.py` 自动更新；检查结果：`full` 行无 `-` 占位符，最高/次高标记仍存在。
- Client Average 统计：Small 共 60 个格子，`my_merge full` 最高或并列最高 24 个、次高 14 个。说明该修复稳定但只解决局部灾难点，距离“90% Client Average 最高”仍很远。

## 观测 11：数据蒸馏方向应采用“客户端原型摘要”，不是服务端验证集 prototype head

论文与方法线索：

- FedProto（Federated Prototype Learning, 2021/AAAI 2022）指出，在非 IID 客户端中，直接在梯度/参数空间聚合容易因为局部数据分布不同而错位；可以改为通信每类抽象 prototype，用类别原型作为客户端知识载体。
- FedDF（Ensemble Distillation for Robust Model Fusion in Federated Learning, 2020）把“模型融合”写成蒸馏问题：服务端用客户端模型集成输出训练中心模型。但它需要 proxy/unlabeled data；本项目如果让服务端拿医学原始图像或验证集，就会破坏隐私叙事。
- FedFTG/FedDTG 等 data-free distillation 方向用生成器合成样本，再做蒸馏；这会引入额外训练、生成模型和合成医学图像隐私争议，和当前“后处理式模型融合”框架差异太大，暂不作为主线。
- FedD3 类 dataset distillation 让客户端上传蒸馏样本，但医学场景下“合成图像是否泄露病灶模式”难以解释；更适合本项目的是上传低维或 pooled feature 原型，而不是图像本身。
- 参考链接：
  - FedProto: https://arxiv.org/abs/2105.00243
  - FedDF: https://arxiv.org/abs/2006.07242
  - FedMD: https://arxiv.org/abs/1910.03581
  - TIES-Merging: https://arxiv.org/abs/2306.01708

本仓库已有事实：

- 现在正式 M1 只使用全局 `feature_summary`：每类五维医学形态特征均值、计数、全局均值和方差。
- 仓库里曾有 `outputs/codex_proto_summary_smoke_20260614/prototype_summaries`，格式是每类 pooled feature prototype，不含原始图像。
- 旧版 `prototype_head` 曾经有收益，但它依赖服务端验证集抽特征，因此不符合现在“原始数据最好不能上传到服务端”的约束。
- 现有 `codex_proto_summary_smoke_20260614` 只覆盖 `chaoshengmnist_224/vit_t` 9 个格子；其中多数低于当前正式 full，说明旧 smoke 不能直接作为最终方案，只能作为“客户端原型摘要格式可用”的证据。

拟采取措施：

- 不恢复旧的服务端验证集 prototype head。
- 设计可选的客户端本地原型摘要：
  - 每个客户端本地对自己数据抽取 pooled feature 或医学五维特征；
  - 上传每类 prototype、count、coverage；
  - 服务端只用这些摘要修正 M1 的 class routing 或分类头保真；
  - 没有 prototype summary 时默认完全关闭。
- 先做 small transformer 的少量失败格子探针，原因是旧结果显示 `vit_t/swin_tiny` 更容易出现分类头/表示错位；CNN 不先接入，避免把一个 transformer 修复模块误扩散到所有模型。
- 如果 prototype summary 修复是负优化，直接删除，不进入正式方法。

## 观测 12：最差 Client Average 不只来自超声，而是互补类别客户端被整体权重压偏

对象：当前正式汇总表 `My_merge_ret/汇总表.md` 和 full trace。

观测：

- 按 Client Average 差距排序，当前 my_merge 距离最优最差的格子主要是：
  - `convnext/organsmnist_224/c3_avg`：my_merge `0.0855`，最优 `0.2354`。
  - `swin_tiny/organsmnist_224/c3_avg/c5_avg/c7_avg` 也明显落后。
  - 其次是 VLM/organs、blood 的若干格子。
- 这说明当前主要短板不只是超声，而是器官类数据的类别互补分布。
- 具体看 `organsmnist_224/convnext/c3_b0`：
  - `avg_only` 为 `0.2354`，full 只有 `0.1146`。
  - 三个客户端分别覆盖互补类别，覆盖率约 `[0.36, 0.36, 0.27]`，没有全类别专家。
  - M1/M2 后整体权重变成约 `[0.43, 0.29, 0.28]`，并触发 `specialist_anchor=0.16` 指向 client 0。
  - 这会让 trunk 和整体表示偏向一个局部类别专家，破坏其他互补类别。

结论：

- 对“互补类别客户端、没有全类别专家”的场景，医学类别路由仍有意义，但整体 trunk 不应被一个局部专家主导。
- 这个问题和 derma 的 generalist 离群劫持不同：derma 需要削弱全类别离群 generalist；organs 需要保护互补类别覆盖。

拟采取措施：

- 设计 M2 的互补覆盖保守门控：
  - 当全部客户端联合覆盖全类别，但每个客户端都是低覆盖局部专家，且没有 `coverage>=0.95` 的全类别专家时触发。
  - 只把 overall/morphology 的整体融合权重拉回更均匀，保护 trunk。
  - classifier class routing 只轻微平滑，仍保留每类由对应客户端负责的医学逻辑。
  - 关闭该场景下的 specialist anchor，避免把整体模型锚到单个局部专家。
- 临时 worktree merge trace 已确认在 `organsmnist_224/convnext/c3_b0` 上触发：
  - specialist anchor 从 `0.16` 变为 `0`；
  - fusion weights 从约 `[0.43,0.29,0.28]` 降低集中到约 `[0.41,0.30,0.29]`；
  - class routing 仍保留。
- 该措施还没有 test accuracy 结论；等当前 generalist guard full 空出 GPU 后，先跑 `organsmnist_224/convnext` 和 `swin_tiny` 小探针。如果负优化，删除。

## 观测 13：互补覆盖不能只靠专家锚点，保守保护 trunk 有小幅收益

对象：`outputs/codex_coverage_preservation_probe_20260615`，只跑 `organsmnist_224` 的 `convnext` 和 `swin_tiny`。

措施：

- 在 M2 增加一个窄规则 `coverage_preservation`：
  - 联合类别覆盖足够；
  - 没有单个客户端覆盖 `>=0.95` 的全类别 generalist；
  - 平均客户端类别覆盖低于 `0.55`；
  - 触发后只把 `overall_weights` 和 `morphology_weights` 向均匀 prior 收缩，保护 trunk；
  - 不改 class routing，仍保留按类别专家负责；
  - 当 gate 足够高时关闭 `specialist_anchor`，避免整体模型被锚到一个局部类别专家。

结果：

- 18 个 raw case 中 W/T/L=`5/12/1`，平均 `+0.00468`。
- 正向变化主要来自 `swin_tiny/organsmnist_224`：
  - `c5_b0`：`0.1096 -> 0.1244`。
  - `c7_b0`：`0.1340 -> 0.1521`。
  - `c7_b0.01`：`0.0785 -> 0.1267`。
- `convnext/organs/c3_b0` 小幅提升：`0.1146 -> 0.1171`。
- 唯一负例很小：`convnext/organs/c5_b0.01`：`0.07987 -> 0.07976`。

解释：

- 这个结果说明“互补覆盖”观察是有用的，但收益不是来自更强专家，而是来自削弱 trunk 的单专家偏置。
- 这条规则目前只在 `organs` 小范围验证，不足以进入正式汇总。下一步必须跑医学 small 全量，确认它不会伤害 blood/derma/organc/chaosheng。

全量验证早停结论：

- 医学 small full 验证目录：`outputs/codex_coverage_preservation_medical_full_20260615`。
- 跑到 60 个 raw case 时已经出现明确负优化，对 generalist guard full：W/T/L=`9/32/19`，平均 `-0.01360`。
- 典型负例：
  - `dermamnist_224/convnext/c3_b0.01`：`0.6688 -> 0.1097`。
  - `organcmnist_224/resnet/c3_b0.01`：`0.3530 -> 0.1889`。
  - `organsmnist_224/resnet/c3_b0`：`0.2660 -> 0.1859`。
- 结论：`coverage_preservation` 只在 `organs/convnext+swin_tiny` 局部有效，扩展到所有医学 small 后会误伤 derma/organc/resnet 等场景，不满足“医学统一方法”的要求。
- 处理：已停止该 full 任务，删除 `methods/my_merge.py` 中的 `coverage_preservation` 代码，不进入正式方法，也不更新 `汇总表.md`。

## 观测 14：数据蒸馏方向应优先做客户端原型摘要，不做服务端代理数据蒸馏

用户提示可以查找数据蒸馏方向。按隐私约束重新梳理：

- FedMD 和 FedDF 属于蒸馏式模型融合：服务端或各方通过公共/代理数据对客户端输出做蒸馏。这适合解释“模型融合不等于参数平均”，但如果在本项目里用医学验证集或上传图像做代理数据，会破坏“原始数据不上传服务端”的叙事。
- FedDTG/FedFTG 这类 data-free distillation 用生成器合成样本再蒸馏。它避免直接上传原图，但会引入生成器训练、合成医学图像隐私争议和额外训练流程，和当前后处理式模型融合方法距离较远。
- FedD3/dataset distillation 让客户端上传蒸馏样本。医学场景下蒸馏样本仍可能携带病灶分布和身份线索，不适合作为主线。
- FedProto/FedProtoKD 这条线更适合：客户端只上传每类 feature prototype/count/coverage。它不是原图，也不是服务端验证集，可用于修正 class routing 或分类头保真。

暂定措施：

- 不恢复旧版服务端验证集 `prototype_head`。
- 后续如做蒸馏，只做“客户端上传类别原型摘要”的轻量形式：
  - 每个客户端本地抽每类 pooled feature prototype 和 count；
  - 服务端只聚合摘要；
  - 用摘要约束 classifier head 或 class routing；
  - 没有摘要时完全关闭。
- 该方向必须单独探针；如果不能稳定提升，就不进入正式方法。

参考：

- FedProto: https://arxiv.org/abs/2105.00243
- FedDF: https://arxiv.org/abs/2006.07242
- FedMD: https://arxiv.org/abs/1910.03581
- FedDTG: https://arxiv.org/abs/2201.03169
- FedD3: https://arxiv.org/abs/2208.11311
- FedProtoKD: https://arxiv.org/abs/2508.19009

## 观测 15：M2 当前短板集中在窄类别专家锚定和高符号冲突 delta

对象：当前正式 full 结果 `outputs/codex_medical_full_generalist_guard_20260615` 和 `My_merge_ret/汇总表.md`。

观测：

- Client Average 距离最优最大的 case 仍集中在 `organsmnist_224`：
  - `convnext/organsmnist_224/c3_avg`：my_merge `0.0855`，最优 `0.2354`。
  - `swin_tiny/organsmnist_224/c3/c5/c7_avg` 分别落后约 `0.1269/0.1181/0.1237`。
- 这些 case 的客户端类别覆盖很窄，但 `specialist_anchor` 仍可能很强：
  - `swin_tiny/organs/c5_b0`：top expert coverage `0.273`，`specialist_anchor=0.435`。
  - `convnext/organs/c7_b0`：top expert coverage `0.182`，`specialist_anchor=0.235`。
  - `organc/swin_tiny/c5_b0`：top expert coverage `0.273`，`specialist_anchor=0.438`。
- 高冲突 case 中，`sign_conflict` 常在 `0.50` 以上；而表现最好的外部基线经常是 `breadcrumbs` 或 `robustmerge`，二者共同点都是不盲目保留全部大幅 delta，而是做冲突过滤。

解释：

- M1 的医学权重和按类路由仍然有用；问题不在于“不要医学权重”，而是在互补类别场景下，M2 不应把整个表示层锚到单个窄类别专家。
- 高符号冲突时，最大幅度 delta 往往更像客户端类别偏置或局部纹理适配；直接按最大幅度保留可能会放大冲突。更合理的是在冲突高时保留中等幅度、方向一致的 delta。

拟采取措施：

- 只改 M2，不改 baseline：
  - 对 `specialist_anchor` 增加覆盖上限：当被选专家类别覆盖低于 `0.35` 时，锚定权重最高只允许到约 `0.15`，避免窄类别专家支配 trunk。
  - 对 `conflict_delta` 增加中等幅度过滤：当 `sign_conflict >= 0.50` 时，先丢弃每个客户端 top `10%` 最大幅度 delta，再保留其余中的 top `25%`，最后再做符号一致融合。
- 先跑 `organsmnist_224` 的 `convnext/swin_tiny` 小探针。若全局负优化或只局部有效，删除。

中间结果：

- 初版探针目录：`outputs/codex_m2_middle_delta_anchor_probe_20260615`。
- 已完成前 10 个 raw case 时，对正式 full：W/T/L=`1/7/2`，平均 `+0.00085`。
- 正例：
  - `swin_tiny/organsmnist_224/c5_b0`：`0.1096 -> 0.1749`，触发 `middle_keep_0p25_remove_top_0p10`，`sign_conflict=0.520`，窄专家锚定从 `0.435` 限制到 `0.152`。
- 负例：
  - `convnext/organsmnist_224/c3_b0`：`0.1146 -> 0.1107`。该 case 的 `sign_conflict=0.403`，没有触发 middle-delta，只是窄专家锚定从 `0.156` 被压到 `0`，说明“低冲突时限制专家锚定”缺乏依据。
  - `swin_tiny/organsmnist_224/c5_b0.1`：`0.1097 -> 0.0568`，触发 middle-delta 但没有专家锚定，说明 middle-delta 本身也可能在高覆盖 beta=0.1 场景负优化。
- 修正措施：把规则收窄为“只有 `sign_conflict >= 0.50` 且客户端平均类别覆盖较低时，才使用 middle-delta 和窄专家锚定限制”。先跑同范围对照；若仍负优化，删除该 M2 修改。

收窄版小探针结果：

- 目录：`outputs/codex_m2_highconflict_middle_escalated_20260615`。
- 范围：`organsmnist_224` 的 `convnext` 和 `swin_tiny`，共 18 个 raw case。
- 对正式 full：W/T/L=`2/16/0`，平均 `+0.00927`。
- 主要正例：
  - `swin_tiny/organsmnist_224/c5_b0`：`0.1096 -> 0.1749`。
  - `swin_tiny/organsmnist_224/c7_b0`：`0.1340 -> 0.2355`。
- 被修复的误伤：
  - 低冲突 `convnext/organsmnist_224/c3_b0` 不再限制专家锚定，恢复为 `0.1146 -> 0.1146`。
  - 高覆盖 `swin_tiny/organsmnist_224/c5_b0.1` 不再触发 middle-delta，恢复为 `0.1097 -> 0.1097`。
- 结论：收窄条件比初版 coverage-preservation 更符合观测，只在“低覆盖互补客户端 + 高符号冲突”时启动 M2 冲突修复。下一步跑医学 small 全量；若跨数据集出现负优化，继续收窄或删除。

补充探针结果：

- 目录：`outputs/codex_m2_middle_final_probe_v2_20260615`。
- 范围：`bloodmnist_224/chaoshengmnist_224/organsmnist_224` 的 `convnext/resnet/swin_tiny`，共 81 个 raw case。
- 对正式 full：W/T/L=`3/72/6`，平均 `+0.00140`。
- 现象拆分：
  - `bloodmnist_224` 27 个格子完全不变，说明 `num_classes >= 10` 避免了上一轮 blood 误触发。
  - `chaoshengmnist_224` 仅两个格子出现 `-0.0009` 的极小波动，没有系统性超声退化。
  - `organsmnist_224/swin_tiny` 明确受益：`c5_b0` 从 `0.1096` 到 `0.1749`，`c7_b0` 从 `0.1340` 到 `0.2355`。
  - `organsmnist_224/resnet` 明确受损：`c5_b0` 从 `0.3611` 到 `0.3276`，`c7_b0` 从 `0.2503` 到 `0.2369`。
- 新观察：
  - 中等幅度 delta 过滤不是所有模型族通用。CNN 的局部卷积特征对完整低层 delta 更敏感，过滤后会损失可用的局部解剖适配。
  - Transformer 的 patch/token 表示在低覆盖互补客户端和高符号冲突时更容易被离群大 delta 干扰，因此更适合用 middle-delta 去掉最大幅度扰动。
- 修正措施：
  - 将 middle-delta 和窄专家锚定限制再收窄为 `transformer` 模型族专用。
  - 仍然要求同时满足：`sign_conflict >= 0.50`、平均客户端类别覆盖 `<= 0.45`、类别数 `>= 10`。
  - CNN 继续使用原来的 top magnitude sign-delta，不使用 middle-delta。
- 下一步：
  - 只补跑 `organsmnist_224` 的 `swin_tiny/vit_t/resnet/convnext` 和 `organcmnist_224` 的 `swin_tiny/vit_t`。
  - 如果 transformer 收窄版仍保留 `swin_tiny/organs` 正例且不伤 `organc/vit_t`，再跑医学 small full；否则删除 middle-delta 分支。

拆分验证：

- `outputs/codex_m2_transformer_middle_probe_20260616`：middle-delta + 窄专家限幅，72 个格子 W/T/L=`12/53/7`，平均 `+0.00507`。
  - 大正例：`vit_t/organc/c7_b0.01` 从 `0.0851` 到 `0.3143`。
  - 大负例：`swin_tiny/organc/c7_b0` 从 `0.3113` 到 `0.2233`，`vit_t/organs/c7_b0.01` 从 `0.3575` 到 `0.2717`。
- `outputs/codex_anchor_limit_only_probe_20260616`：只做窄专家限幅，72 个格子 W/T/L=`9/55/8`，平均 `+0.00094`。
  - 说明单独限制专家锚定不是主要收益来源，也会保留大负例。
- `outputs/codex_middle_only_probe_20260616`：只做 middle-delta，不限制专家锚定，36 个 transformer 格子 W/T/L=`12/20/4`，平均 `+0.00748`。
  - 说明 middle-delta 本身有价值，但 `vit_t/organs/c7_b0.01` 仍有大负例。

进一步观测：

- `vit_t/organs/c7_b0.01` 的 `norm_dispersion=0.670`，明显高于其他正例。这表示客户端 delta 范数差异过大，不在同一尺度上。
- 在这种情况下，中等幅度筛选会把“大专家”和“小专家”的更新混合到同一个幅值标准里，反而破坏原本稳定的融合。
- 文献侧依据：
  - TIES-Merging 指出多模型合并的主要干扰来自冗余参数和符号冲突。
  - Model Breadcrumbs 指出最大幅度离群 delta 和最小扰动都可能需要过滤。
  - LARV 指出 vision transformer 的合并干扰存在层级/尺度异质性，不能把所有层和所有 task-vector 统一处理。

最终措施：

- middle-delta 只在以下条件全部满足时启用：
  - transformer 模型族；
  - 解剖类多类别任务，即类别数 `>=10`；
  - 平均客户端类别覆盖 `<=0.45`，表示互补低覆盖客户端；
  - `sign_conflict >=0.50`，表示 delta 方向冲突明确；
  - `norm_dispersion <=0.50`，表示客户端 delta 尺度仍可比。
- 窄专家限幅不再泛化使用，只在“分类头 delta 很大且窄专家锚定很强”的场景触发：
  - 原始专家锚定权重 `>=0.40`；
  - classifier delta 平均范数 `>=2.0`；
  - 同时满足上述 middle-delta 门控。

最终门控探针结果：

- 目录：`outputs/codex_balanced_middle_probe_20260616`。
- 范围：`organcmnist_224/organsmnist_224` 的 `resnet/convnext/swin_tiny/vit_t`，72 个 raw case。
- 对正式 full：W/T/L=`15/53/4`，平均 `+0.00768`。
- CNN 基本不变：`resnet/convnext` 共 36 个格子 W/T/L=`1/34/1`，两处变化均约 `0.0002`。
- transformer 主要收益：
  - `swin_tiny/organc/c5_b0`：`0.0895 -> 0.2233`。
  - `swin_tiny/organc/c7_b0.01`：`0.0593 -> 0.1430`。
  - `vit_t/organc/c7_b0.01`：`0.0851 -> 0.3143`。
  - `swin_tiny/organs/c5_b0`：`0.1096 -> 0.1749`。
- 之前的大负例被门控拦截：
  - `vit_t/organs/c7_b0.01` 恢复为 `0.3575 -> 0.3577`，因为 `norm_dispersion=0.670`，不再启用 middle-delta。
  - `swin_tiny/organc/c7_b0` 从大负 `-0.0880` 缩小到 `-0.0057`。
- 结论：当前规则是目前最有事实依据的一版，可以进入医学 small full 验证；如果全量出现系统性负优化，再删除该分支。

医学 small full 验证：

- 目录：`outputs/codex_balanced_middle_medical_full_20260616/my_merge_ablation_grid/full`。
- 范围：5 个医学 small 数据集、4 个模型、180 个 raw case。
- 对上一版正式 full `outputs/codex_medical_full_generalist_guard_20260615`：
  - W/T/L=`10/168/2`，平均 `+0.00307`。
  - `bloodmnist_224`、`dermamnist_224`、`chaoshengmnist_224` 全部不变。
  - `organcmnist_224`：36 个格子 W/T/L=`7/28/1`，平均 `+0.01248`。
  - `organsmnist_224`：36 个格子 W/T/L=`3/32/1`，平均 `+0.00288`。
- 主要正例：
  - `vit_t/organc/c7_b0.01`：`0.0851 -> 0.3144`。
  - `swin_tiny/organc/c5_b0`：`0.0895 -> 0.2233`。
  - `swin_tiny/organc/c7_b0.01`：`0.0593 -> 0.1430`。
  - `swin_tiny/organs/c5_b0`：`0.1096 -> 0.1749`。
  - `swin_tiny/organs/c7_b0`：`0.1340 -> 0.1753`。
- 剩余负例：
  - `swin_tiny/organc/c7_b0`：`0.3113 -> 0.3056`，`-0.0057`。
  - `vit_t/organs/c5_b0`：`0.1187 -> 0.1137`，`-0.0050`。
- 触发情况：
  - 180 个 raw case 中 13 个启用 middle-delta，均位于 `organcmnist_224/organsmnist_224` 的 transformer。
  - `blood/derma/chaosheng` 不触发，说明该规则没有变成数据集级别乱补丁。
- 结论：
  - 该规则在全量医学 small 上是净正向，且负例幅度明显小于正例。
  - 可以保留为 M2 的冲突尺度门控子步骤，并更新 `汇总表.md`。

## 观测 16：当前主要缺口应按 Client Average 分析，而不是继续追逐单个 Raw 塌点

对象：当前 full 结果 `outputs/codex_balanced_middle_medical_full_20260616/my_merge_ablation_grid/full`。

生成的诊断文件：

- Raw 诊断：`outputs/codex_balanced_middle_medical_full_20260616/my_merge_ablation_grid/reports/full_failure_diagnostics.md`。
- Client Average 诊断：`outputs/codex_balanced_middle_medical_full_20260616/my_merge_ablation_grid/reports/client_average_failure_diagnostics.md`。

观测：

- Raw 口径下，my_merge 对原始最佳方法 W/T/L=`42/35/103`，mean gap=`-0.0330`。Raw 负例很多，但其中不少会在 Client Average 中互相抵消。
- Client Average 口径下，共 60 个格子，my_merge 对原始最佳方法 W/T/L=`20/4/36`，mean gap=`+0.0020`。均值已经不差，但“最高格子数量”远没到用户目标。
- Client Average 的主要缺口集中在：
  - `organsmnist_224`：12 个格子 W/T/L=`3/0/9`，mean gap=`-0.0482`。
  - `bloodmnist_224`：12 个格子 W/T/L=`1/0/11`，mean gap=`-0.0282`。
  - `convnext`：15 个格子 W/T/L=`2/2/11`。
  - `vit_t`：15 个格子 W/T/L=`3/1/11`。
- 最大 Client Average 负差：
  - `organsmnist_224/convnext/c3_avg`：my_merge `0.0855`，best `0.2354`，gap `-0.1499`，winner=`robustmerge`。
  - `organsmnist_224/swin_tiny/c3_avg`：`0.1085` vs `0.2354`，gap `-0.1269`，winner=`breadcrumbs`。
  - `organsmnist_224/swin_tiny/c7_avg`：`0.1012` vs `0.2111`，gap `-0.1099`，winner=`breadcrumbs`。
  - `dermamnist_224/vit_t/c3_avg`：`0.4653` vs `0.5709`，gap `-0.1056`，winner=`iso_c`。
  - `organsmnist_224/swin_tiny/c5_avg`：`0.1114` vs `0.2077`，gap `-0.0963`，winner=`breadcrumbs`。
- 赢家分散在 `robustmerge/breadcrumbs/iso_c/fisher/regmean/free_merge`，说明当前失败不是一个单一候选方法能统一替代，也不适合回到“大候选池搜索”。

进一步 trace 观察：

- `organsmnist_224/convnext/c3` 的三个 beta 中，客户端类别覆盖互补，但 M1 fusion 仍明显偏向一个客户端：
  - `b0` coverage `[0.364, 0.364, 0.273]`，fusion `[0.433, 0.291, 0.276]`，并有 `specialist_anchor=0.156`。
  - `b0.01` coverage `[0.182, 0.636, 0.545]`，fusion `[0.342, 0.346, 0.312]`，但 accuracy 只有 `0.0497`。
  - `b0.1` coverage `[0.909, 0.818, 0.455]`，fusion `[0.351, 0.344, 0.304]`，accuracy `0.0920`。
- `organsmnist_224/swin_tiny` 的当前 middle-delta 对部分格子有效，但不是全部：
  - `c5_b0`：触发 middle-delta，accuracy `0.1749`，相比旧版已有提升。
  - `c7_b0`：触发 middle-delta，accuracy `0.1753`，仍低于 Client Average 最优。
  - `c7_b0.01`：触发 middle-delta，但 accuracy `0.0785`，仍是明显负例。
- `bloodmnist_224` 的缺口没有触发 middle-delta，因为类别数低于当前 M2 解剖多类别门控；这保护了 blood 不被 organs 规则误伤，但也意味着 blood 仍需要单独的 checkpoint-only 观察，而不是复用 organs 的规则。

结论：

- 当前方法均值可观，但距离“Client Average 90% 最高”差在命中率，不是平均分。
- 下一步不应继续扩大候选池；应围绕 Client Average 最大负差做小范围 checkpoint-only 观察。
- 当前正在跑 `no_client_information` 和 `no_fusion_selection` 消融，必须等同一版代码消融完成后再改 `methods/my_merge.py`。否则 `full/-M1/-M2` 会来自不同方法版本，消融结论不严谨。

拟采取措施：

- 等当前医学消融补齐后，自动重生成 `汇总表.md` 和 `reports/ablation_summary.md`。
- 如果 `no_client_information` 在 organs/blood 上高于 full，说明 M1 权重仍在这些互补客户端场景负优化，应先收窄 M1 的整体 trunk 权重，而不是改 M2。
- 如果 `no_fusion_selection` 在 organs transformer 上低于 full，但在 blood/convnext 上高于 full，则保留 M2 transformer middle-delta，同时另做 blood/convnext 的 checkpoint-only 小探针。
- 候选新观察方向只限 checkpoint/摘要，不使用服务端验证集：
  - 对 blood 观察类别覆盖和 head delta 是否存在“低类别数互补冲突”；
  - 对 organs/convnext 观察是否是 CNN trunk 不适合当前医学加权，而不是 middle-delta 问题；
  - 对 `dermamnist_224/vit_t/c3_b0.1` 观察是否是 M1 对单个全覆盖客户端的错误集中，若是则复用已有 generalist consistency guard，而不是新增模块。

## 观测 17：同版消融确认 M1/M2 总体有效，但剩余负优化集中在两个可解释机制

对象：`outputs/codex_balanced_middle_medical_full_20260616/my_merge_ablation_grid`，同一版代码下的 `full`、`no_client_information`、`no_fusion_selection` 均已完成 180/180。

自动汇总：

- `outputs/codex_balanced_middle_medical_full_20260616/my_merge_ablation_grid/reports/ablation_summary.md`
- `My_merge_ret/汇总表.md`

总体消融结论：

- `full`：180 rows，mean_acc=`0.2834`，对原始最佳 W/T/L=`42/35/103`。
- `no_client_information`：mean_acc=`0.2198`，相对 full `-0.0636`，对原始最佳 W/T/L=`5/27/148`。
- `no_fusion_selection`：mean_acc=`0.2712`，相对 full `-0.0122`，对原始最佳 W/T/L=`26/35/119`。

解释：

- M1 医学权重整体是强正贡献，删除后所有数据集平均下降，尤其 `dermamnist_224`、`organcmnist_224`、`chaoshengmnist_224`。
- M2 冲突处理整体是小正贡献，删除后平均下降，但收益高度依赖场景。
- 因此现在不能说 M1/M2 是整体负优化；问题是剩余局部场景仍有负优化。

局部负优化机制 A：低覆盖多客户端场景下，类别路由过尖。

- `organsmnist_224/convnext/c7_b0.01`：full=`0.0798`，`no_client_information`=`0.2354`，说明 M1 在该格子明显伤害。
- 该格子平均客户端类别覆盖约 `0.208`，每个客户端只覆盖少量器官类别；当前 `class_weights` 的类别最大权重均值约 `0.483`，类别路由会强推某个客户端负责某些类别。
- 这种情况下，客户端类别覆盖互补但都不完整，过尖的类别路由会把未充分学习的局部专家当成类别专家，导致分类头和中后层不协调。

局部负优化机制 B：低类别 blood 上普通冲突 delta 有时过度修正。

- `bloodmnist_224/vit_t/c5_b0.1`：full=`0.1377`，`no_fusion_selection`=`0.3148`，说明 M2 普通冲突增量在该格子明显伤害。
- 该格子不是解剖多类别任务，但 `sign_conflict≈0.517`、`conflict_score≈0.483`，当前普通 `top_magnitude_0p50` 仍会混入约 `0.10` 的 conflict delta。
- 这说明“符号冲突高”在 blood 低类别任务里不一定表示需要 delta 修正，可能只是低类别互补训练造成的分类边界差异；直接注入 sign delta 会破坏原本 M1 的医学加权融合。

采取措施：

- 先不直接改规则，启动 targeted probe 验证两个假设：
  - `no_class_routing`：验证低覆盖 organs 场景是否应该软化/关闭类别路由。
  - `no_conflict_stabilization`：验证 blood/vit 负例是否来自普通 conflict delta。
- probe 配置：
  - 目录：`outputs/codex_route_conflict_probe_20260616/my_merge_ablation_grid`
  - 数据集：`organsmnist_224 bloodmnist_224`
  - 模型：`convnext swin_tiny vit_t`
  - 消融：`no_class_routing no_conflict_stabilization`
  - 仍然只用客户端特征摘要和 checkpoint 信息，不使用服务端验证集。

下一步判断标准：

- 如果 `no_class_routing` 只在低覆盖 organs 场景提升，而在 derma/chaosheng 等场景不需要，则把类别路由改成由“平均客户端类别覆盖 + 类别数”控制的软门控，而不是删除整个 M1。
- 如果 `no_conflict_stabilization` 在 blood/vit 明显提升，但在 organs transformer 明显下降，则收窄普通 conflict delta 的适用范围：保留解剖多类别 transformer 的 middle-delta，降低或关闭低类别 blood 的普通 sign-delta。

## 观测 18：子模块探针显示，剩余负优化不是 M1/M2 全局失效，而是两个局部机制过强

对象：`outputs/codex_submodule_probe_gpu_20260616`，同一代码版本下 5 个变体 x 9 个 key cases：

- `full`
- `no_class_routing`
- `no_delta_agreement_weighting`
- `specialist_client`（等价于关闭 `specialist_anchor`）
- `no_conflict_stabilization`

关键结果：

- 全局关闭类别路由不是可行解：
  - `organsmnist_224/convnext/c7_b0.01`：`full=0.0798`，`no_class_routing=0.2354`，说明该 CNN 解剖低覆盖分类头被类别路由伤害。
  - `organcmnist_224/vit_t/c7_b0.01`：`full=0.3143`，`no_class_routing=0.1255`，说明 transformer 解剖低覆盖任务仍需要类别路由。
  - `chaoshengmnist_224/resnet/c5_b0`：`full=0.2471`，`no_class_routing=0.2210`，说明不能为超声/普通 CNN 统一关闭 M1 类别信息。
- 专家保真在低类别 transformer 上有明显负贡献：
  - `bloodmnist_224/vit_t/c5_b0.01`：`full=0.1380`，关闭 `specialist_anchor` 后为 `0.2827`。
  - 同批 key cases 中关闭 `specialist_anchor` 没有伤害 `chaosheng/resnet`、`derma/vit`、`organc/vit`、`organs/convnext/swin`。
- M2 的冲突稳定仍然必要：
  - `organcmnist_224/vit_t/c7_b0.01`：`full=0.3143`，`no_conflict_stabilization=0.1803`。
  - 因此不能删除 middle-delta 冲突处理。
- `delta_agreement_weighting` 在 key cases 上整体偏弱或偏负：
  - `blood/vit/c3_b0`：`0.2944 -> 0.3040`
  - `blood/vit/c5_b0.1`：`0.2803 -> 0.3148`
  - `organc/vit/c7_b0.01`：`0.3143 -> 0.3329`
  - 但它不解决 `organs/convnext`，所以暂不把它作为本轮主改动。

结论：

- M1 的类别路由应该保留，但要识别“低覆盖解剖 CNN 分类头”这种不适合逐类硬路由的场景。
- M2 的专家保真应该保留，但低类别 transformer 任务上单个专家容易变成偏置放大器，应该默认不介入。
- 本轮只做两个窄改，不扩候选池、不引入服务端验证集选择。

采取措施：

- 在 `methods/my_merge.py` 增加 `classifier_routing_policy`：
  - 当 `task_type=small`、模型族为 CNN、类别数不少于 10、平均客户端类别覆盖不超过 `0.45` 时，分类头从逐类 `class_weights` 回退到 M1 共识权重。
  - 其他任务仍使用原有类别路由，尤其保留 `organc/vit` 的 transformer 类别路由。
- 在 `specialist_anchor` 前增加低类别 transformer guard：
  - 当 `task_type=small`、模型族为 transformer、类别数少于 10 时，记录 raw 专家权重，但最终 `specialist_anchor_weight=0`。
  - 目的不是删除专家机制，而是避免 blood 这类低类别任务被单个局部专家拉偏。
- 以上判断只使用客户端元信息、特征摘要和 checkpoint 统计，不使用服务端 raw validation 做候选选择。

待验证：

- 复跑同一 9 个 key cases：
  - `organs/convnext/c7_b0.01` 应接近 `no_class_routing` 的 `0.2354`。
  - `organc/vit/c7_b0.01` 应保持在 `0.314` 附近，不能退到 `0.1255`。
  - `blood/vit/c5_b0.01` 应接近关闭专家保真的 `0.2827`。
  - `chaosheng/resnet/c5_b0` 不应明显下降。

验证结果：

- 目录：`outputs/codex_cnn_specialist_gate_probe_20260617/full`。
- 9 个 key cases 完成：
  - `bloodmnist_224/vit_t/c3_b0`：`0.2944`，与旧 full 一致。
  - `bloodmnist_224/vit_t/c5_b0.01`：`0.1380 -> 0.2827`，低类别 transformer 专家保真 guard 生效。
  - `bloodmnist_224/vit_t/c5_b0.1`：`0.2803`，与旧 full 一致。
  - `chaoshengmnist_224/resnet/c5_b0`：`0.2471`，没有被低类别 guard 误伤。
  - `dermamnist_224/vit_t/c3_b0`：`0.6688`，与旧 full 一致。
  - `organcmnist_224/vit_t/c7_b0.01`：`0.3143`，保留 transformer 类别路由后没有退化。
  - `organsmnist_224/convnext/c7_b0.01`：`0.0798 -> 0.2354`，低覆盖解剖 CNN 分类头回退共识权重生效。
  - `organsmnist_224/swin_tiny/c7_b0.01`：仍为 `0.0785`。
  - `organsmnist_224/swin_tiny/c7_b0.1`：仍为 `0.0497`。

阶段结论：

- 两个窄改都命中目标负例，且没有伤害本轮保留的关键正例，可以保留。
- 剩余主要负例是 `organsmnist_224/swin_tiny`，它和 `organcmnist_224/vit_t` 同属低覆盖解剖 transformer，不能用“transformer 统一关闭类别路由”解释。
- 继续观察 trace：
  - `organc/vit/c7_b0.01` 的 `classifier_delta_mean_norm=1.214`，关闭类别路由会从 `0.3143` 掉到 `0.1255`。
  - `organs/swin/c7_b0.01` 的 `classifier_delta_mean_norm=2.689`，关闭类别路由会从 `0.0785` 升到 `0.1521`。
  - 二者主要差异不是平均覆盖率，而是分类头增量尺度。`organs/swin` 的分类头更新过强，逐类路由会把客户端私有 head 的方向冲突直接灌进最终分类器。

下一步措施：

- 不按数据集或模型名硬编码。
- 尝试把分类头路由策略扩展为：低覆盖解剖 transformer 中，如果 `classifier_delta_mean_norm` 明显偏大，则分类头也回退到 M1 共识权重；否则保留类别路由。
- 目标是修复 `organs/swin`，同时保持 `organc/vit` 不掉。

验证结果：

- 目录：`outputs/codex_head_conflict_gate_probe_20260617/full`。
- 同一 9 个 key cases 完成：
  - `bloodmnist_224/vit_t/c3_b0`：`0.2944`，保持不变。
  - `bloodmnist_224/vit_t/c5_b0.01`：`0.2827`，保持上一轮低类别 transformer guard 的收益。
  - `bloodmnist_224/vit_t/c5_b0.1`：`0.2803`，保持不变。
  - `chaoshengmnist_224/resnet/c5_b0`：`0.2471`，保持不变。
  - `dermamnist_224/vit_t/c3_b0`：`0.6688`，保持不变。
  - `organcmnist_224/vit_t/c7_b0.01`：`0.3143`，没有被 head-conflict 规则误伤。
  - `organsmnist_224/convnext/c7_b0.01`：`0.2354`，保持 CNN 分类头回退收益。
  - `organsmnist_224/swin_tiny/c7_b0.01`：`0.0785 -> 0.1521`，head-conflict 分类头回退生效。
  - `organsmnist_224/swin_tiny/c7_b0.1`：`0.0497`，仍未改善。

阶段结论：

- 当前可保留三条规则：
  - 低覆盖解剖 CNN 的分类头不用逐类路由，回退 M1 共识权重。
  - 低类别 transformer 关闭专家保真，避免单专家偏置放大。
  - 低覆盖解剖 transformer 中，如果分类头 delta 范数过大，分类头也回退 M1 共识权重。
- 这些规则都由 checkpoint/客户端摘要触发，没有服务端验证集候选选择。
- 仍需更宽验证，尤其 `organcmnist_224/organsmnist_224` 全部 transformer/CNN 组合，确认不是只修了 9 个 key cases。

## 观测 19：宽验证否定了“低覆盖 CNN 一律回退分类头”，应改为 checkpoint 冲突触发

对象：`outputs/codex_anatomy_head_conflict_probe_20260617/full`，只跑到前 18 个解剖类 small case 后停止，因为已出现明确反例。

反例结果：

- `organcmnist_224/convnext/c7_b0.01`：旧 full=`0.2233`，低覆盖 CNN 一律回退后=`0.0654`，`-0.1580`。
- `organcmnist_224/resnet/c3_b0`：旧 full=`0.3982`，低覆盖 CNN 一律回退后=`0.3574`，`-0.0409`。
- 同时，`organsmnist_224/convnext/c7_b0.01` 需要分类头回退：旧 full=`0.0798`，回退后=`0.2354`。

trace 对比：

- `organc/convnext/c7_b0.01`：
  - `classifier_delta_mean_norm=2.393`
  - `sign_conflict=0.614`
  - `norm_dispersion=0.378`
  - 回退分类头后严重下降，说明它虽然低覆盖，但仍需要类别路由。
- `organs/convnext/c7_b0.01`：
  - `classifier_delta_mean_norm=2.359`
  - `sign_conflict=0.642`
  - `norm_dispersion=0.275`
  - 回退分类头明显提升，说明它是更典型的“高符号冲突 + 尺度均衡”的分类头冲突。
- `organc/resnet/c3_b0`：
  - `classifier_delta_mean_norm=3.109`
  - `sign_conflict=0.388`
  - `norm_dispersion=0.302`
  - 分类头 delta 范数大，但符号冲突不高，因此不能只用 head norm 触发回退。

结论：

- “平均客户端类别覆盖低”只是风险信号，不足以决定关闭类别路由。
- CNN 分类头回退必须由 checkpoint 冲突形态触发：低覆盖 + 分类头 delta 大 + 全局符号冲突高 + delta 尺度均衡。
- 因此废弃上一版 `cnn_low_coverage_consensus_head` 规则。

采取措施：

- 将 CNN 分类头回退改为 `cnn_head_conflict_consensus_head`：
  - `task_type=small`
  - 模型族为 CNN
  - 类别数不少于 10
  - 平均客户端类别覆盖不超过 `0.25`
  - `classifier_delta_mean_norm >= 2.0`
  - `sign_conflict >= 0.63`
  - `norm_dispersion <= 0.30`
- Transformer 分类头回退保留为：
  - 低覆盖解剖 transformer
  - `classifier_delta_mean_norm >= 2.0`
  - `sign_conflict >= 0.50`
  - `norm_dispersion <= 0.50`
- 两者都只使用 checkpoint 统计和客户端元信息，不使用服务端验证集。

验证结果：

- 目录：`outputs/codex_refined_head_gate_probe_20260617/full`，6 个 case 同时包含修复目标点和反例点。
- 结果：
  - `bloodmnist_224/vit_t/c5_b0.01`：`0.2827`，保留低类别 transformer guard 收益。
  - `organcmnist_224/convnext/c7_b0.01`：`0.2233`，恢复到旧 full，反例不再被误伤。
  - `organcmnist_224/resnet/c3_b0`：`0.3982`，恢复到旧 full，反例不再被误伤。
  - `organcmnist_224/vit_t/c7_b0.01`：`0.3143`，仍保留类别路由。
  - `organsmnist_224/convnext/c7_b0.01`：`0.2354`，仍触发 CNN head-conflict 回退。
  - `organsmnist_224/swin_tiny/c7_b0.01`：`0.1521`，仍触发 transformer head-conflict 回退。

下一步：

- 重新跑解剖类 broad probe，验证收窄后的 head-conflict 规则是否在 `organcmnist_224/organsmnist_224` 全组合中稳定。

## 观测 20：按 CNN/Transformer 拆规则能修局部点，但方法形态变成补丁；改为统一冲突压力

背景：

- 之前的 `cnn_head_conflict_consensus_head`、`transformer_head_conflict_consensus_head`、`low_class_transformer_specialist_guard` 能修一些关键负例，但叙事上变成了“CNN 一套、Transformer 一套、低类别再一套”。
- 这和科研方法的目标不一致：我们需要的是一个医学模型融合原则，而不是针对架构的经验规则。

查阅论文：

- `Task Arithmetic`：把客户端模型看成相对参考模型的任务向量，即 `delta = W_client - W_ref`，支持在增量空间组合能力。
- `TIES-Merging`：明确指出合并模型的干扰主要来自冗余小扰动和符号冲突，提出 trim、elect sign、merge。
- `Model Breadcrumbs`：同样强调在任务向量中去掉微小扰动和异常大扰动，只保留更可信的更新。
- `Model Soups`：证明权重平均可能有泛化收益，但依赖候选模型选择/验证语境，不适合当前“服务端不能拿 raw validation/test data”的隐私约束。
- `AdaMerging`：通过无标签测试样本的熵最小化学习合并系数，性能思路有价值，但需要服务端样本，不符合本方法约束，因此不采用。

设计调整：

- 保留 M1：只负责医学证据权重，输出 `overall_weights`、`morphology_weights`、`class_weights`。
- 重写 M2：只负责冲突消解，不再按模型家族拆行为规则。
- 新增两个统一连续量：
  - `head_conflict_pressure`：由平均类别覆盖、分类头 delta 范数、符号冲突、方向冲突、尺度均衡共同决定，用来连续调低分类头类别路由强度。
  - `representation_conflict_pressure`：由平均类别覆盖、类别数、符号冲突、方向冲突决定，用来选择普通 TIES 风格 top magnitude 还是 Breadcrumbs 风格 middle delta。
- 专家保真不再按 transformer/CNN 区分，而是由 `max(head_conflict_pressure, representation_conflict_pressure)` 统一压低上限。

第一轮验证：

- 目录：`outputs/codex_unified_conflict_pressure_probe_20260617/full`，8 个关键正反例。
- 相比旧 full：
  - `bloodmnist_224/vit_t/c5_b0.01`：`0.1418 -> 0.2827`，保住低类别专家负优化修复。
  - `chaoshengmnist_224/resnet/c5_b0`：`0.1923 -> 0.2390`。
  - `organsmnist_224/convnext/c7_b0.01`：`0.0798 -> 0.2354`，保住分类头冲突修复。
  - `organsmnist_224/swin_tiny/c7_b0.01`：`0.0785 -> 0.1521`，保住分类头冲突修复。
  - `organcmnist_224/convnext/c7_b0.01`：`0.2233 -> 0.2233`，反例不再误伤。
  - `organcmnist_224/resnet/c3_b0`：`0.3982 -> 0.3834`，小幅下降。
  - `organcmnist_224/swin_tiny/c7_b0.01`：`0.1430 -> 0.1289`，小幅下降。
  - `organcmnist_224/vit_t/c7_b0.01`：`0.3144 -> 0.2964`，小幅下降。

问题定位：

- 三个小负例都不是由模型家族决定的，而是统一公式过早削弱了原本有效的 M1/M2：
  - `organc/vit` 分类头 delta 范数只有 `1.214`，类别路由不该被削弱，问题来自 delta blend 被从 `0.116` 降到 `0.099`。
  - `organc/resnet` 原本需要一个弱 `top_magnitude_0p50` delta blend（`0.074`），第一版统一公式因为 `representation_pressure < 0.40` 直接关掉了 delta。
  - `organc/swin` 的 `head_conflict_pressure=0.411`，不够强，第一版却把类别路由从 `0.45` 降到 `0.343`。

采取措施：

- 不恢复 CNN/Transformer 分支。
- 调整统一公式：
  - 分类头路由阻尼从 `pressure > 0.30` 才开始，改为 `pressure > 0.42` 才开始，避免中等冲突误伤。
  - delta blend 不再由 `representation_pressure` 直接决定是否打开；只保留一个统一原则：类别数少于 10 时关闭 delta blend，类别数不少于 10 时按旧冲突强度使用 delta blend。
  - `representation_conflict_pressure` 只决定 delta filter 类型：低压力用 `top_magnitude_0p50`，高压力用 `middle_keep_0p25_remove_top_0p10`。

第二轮验证：

- 目录：`outputs/codex_unified_conflict_pressure_probe2_20260617/full`，同一 8 个 case。
- 相比旧 full：
  - `bloodmnist_224/vit_t/c5_b0.01`：`0.1418 -> 0.2827`，收益保留。
  - `chaoshengmnist_224/resnet/c5_b0`：`0.1923 -> 0.2390`，收益保留。
  - `organsmnist_224/convnext/c7_b0.01`：`0.0798 -> 0.2354`，收益保留。
  - `organsmnist_224/swin_tiny/c7_b0.01`：`0.0785 -> 0.1521`，收益保留。
  - `organcmnist_224/convnext/c7_b0.01`：`0.2233 -> 0.2233`，无退化。
  - `organcmnist_224/resnet/c3_b0`：`0.3982 -> 0.3982`，恢复。
  - `organcmnist_224/swin_tiny/c7_b0.01`：`0.1430 -> 0.1430`，恢复。
  - `organcmnist_224/vit_t/c7_b0.01`：`0.3144 -> 0.3143`，基本恢复。

阶段结论：

- 可以废弃“CNN 一套、Transformer 一套”的解释。
- 当前更合理的故事线是：M1 产生医学证据权重；M2 在任务向量空间里用统一冲突压力决定如何削弱类别路由、如何稀疏化 delta、如何限制单专家保真。
- 还不能直接跑全量，需要先做 `organcmnist_224/organsmnist_224` 的 72 case 宽验证，确认统一冲突压力不是只修了 8 个点。

## 观测 21：72 例宽验证说明统一冲突压力还需要加入“尺度离散”和“保真分离度”

对象：`outputs/codex_unified_conflict_anatomy_probe_20260617/full`，覆盖 `organcmnist_224/organsmnist_224` 的 `resnet/convnext/swin_tiny/vit_t`、`3/5/7` 客户端、`beta=0/0.01/0.1`，共 72 个 case。

相对旧 full：`outputs/codex_balanced_middle_medical_full_20260616/my_merge_ablation_grid/full`。

总体结果：

- `rows=72`。
- 平均差值：`-0.004927`。
- `W/T/L = 10 / 37 / 25`。

主要负例：

- `organsmnist_224/vit_t/c7_b0.01`：旧 `0.3575`，新 `0.2717`，下降 `0.0859`。
- `organcmnist_224/swin_tiny/c7_b0`：旧 `0.3056`，新 `0.2233`，下降 `0.0823`。
- `organsmnist_224/swin_tiny/c5_b0`：旧 `0.1749`，新 `0.1015`，下降 `0.0734`。
- `organcmnist_224/resnet/c5_b0`：旧 `0.3059`，新 `0.2372`，下降 `0.0686`。
- `organcmnist_224/resnet/c7_b0`：旧 `0.2922`，新 `0.2244`，下降 `0.0678`。

主要正例：

- `organsmnist_224/convnext/c7_b0.01`：旧 `0.0798`，新 `0.2354`，提升 `0.1557`。
- `organsmnist_224/swin_tiny/c7_b0.01`：旧 `0.0785`，新 `0.1521`，提升 `0.0736`。
- `organsmnist_224/swin_tiny/c7_b0`：旧 `0.1753`，新 `0.2350`，提升 `0.0597`。

trace 结论：

- `organs/vit/c7_b0.01` 的 `sign_conflict=0.565`、`direction_conflict=0.494`，但 `norm_dispersion=0.670` 很高。新公式只看冲突压力，误选 `middle_keep_0p25_remove_top_0p10`；旧方法用 `top_magnitude_0p50`，反而更好。说明 Breadcrumbs 风格 middle-delta 只适合“冲突高且各客户端 delta 尺度相对均衡”的场景；如果尺度离散很高，删掉最大幅值会误删真正承担适配的客户端更新。
- `organc/swin/c7_b0`、`organc/resnet/c5_b0` 的旧结果保留了 `specialist_anchor`，新公式因为 `specialist_coverage_cap = 1 - 0.70 * max(head_pressure, representation_pressure)` 把专家保真压成 0 或低于阈值，造成下降。说明专家保真不能被“冲突压力”全局压制；它应该由医学权重分离度和可靠性自然决定。
- `organs/convnext c7_b0.01` 和 `organs/swin c7_b0.01` 的大幅提升仍然来自：高符号冲突、低覆盖、尺度较均衡时，降低分类头逐类路由并使用 middle-delta。这个观察是有效的，但触发条件要更严格。

论文依据更新：

- `Task Arithmetic` 支持在参考模型的 task vector / delta 空间合并能力。
- `TIES-Merging` 指出模型合并干扰主要来自符号冲突，适合用 sign election 处理方向冲突。
- `Model Breadcrumbs` 指出只保留最大幅值不是总是可靠，去掉微小扰动和异常大扰动能保留更稳的任务向量；但本实验说明它需要尺度均衡前提。
- `Model Soups` 和 `AdaMerging` 依赖验证集或服务端样本选择/学习合并权重，不符合“服务端不能拿原始数据”的隐私约束，因此不作为当前算法模块。

采取措施：

- 保持统一 M2，不恢复 CNN/Transformer 分支。
- 将 middle-delta 触发从 `representation_pressure >= 0.40` 改成统一的任务向量条件：
  - 类别数不少于 10；
  - 平均客户端类别覆盖低；
  - 符号冲突和方向冲突同时明显；
  - `norm_dispersion <= 0.50`，即各客户端 delta 尺度不能过度离散。
- 专家保真不再乘以全局 `specialist_coverage_cap`；只由医学证据分离度、可靠性和权重集中度决定。这样“专家保真”成为 M2 里处理冲突的第二个保真机制，而不是补丁。
- 继续用 8 个 key cases 先确认不退化，再重新跑 72 case 宽验证。

验证结果：

- 8 个关键 case：`outputs/codex_unified_conflict_scale_guard_probe_20260617/full`。
  - `bloodmnist_224/vit_t/c5_b0.01`：保持 `0.2827`。
  - `chaoshengmnist_224/resnet/c5_b0`：保持 `0.2390`。
  - `organcmnist_224/convnext/c7_b0.01`：保持 `0.2233`。
  - `organcmnist_224/resnet/c3_b0`：保持 `0.3982`。
  - `organcmnist_224/swin_tiny/c7_b0.01`：保持 `0.1430`。
  - `organcmnist_224/vit_t/c7_b0.01`：保持 `0.3143`。
  - `organsmnist_224/convnext/c7_b0.01`：保持 `0.2354`。
  - `organsmnist_224/swin_tiny/c7_b0.01`：保持 `0.1521`。
- 72 个解剖宽验证：`outputs/codex_unified_conflict_scale_guard_anatomy_probe_20260617/full`。
  - 相对上一版失败统一公式：mean diff `+0.002919`，W/T/L=`16/49/7`，说明尺度门控确实修复了部分误触发。
  - 相对当前旧 full：mean diff `-0.002009`，W/T/L=`8/41/23`，仍不可接受。

新负例：

- `organcmnist_224/swin_tiny/c5_b0`：旧 `0.2233`，新 `0.0895`，下降 `0.1339`。
  - 触发 `middle_keep_0p25_remove_top_0p10`。
  - `sign_conflict=0.554`，`direction_conflict=0.496`，`norm_dispersion=0.182`，`coverage=0.200`。
  - 新版恢复了 `specialist_anchor=0.279`，而上一版失败公式因为全局压力把专家保真压成 0，反而保持 `0.2233`。
- `organsmnist_224/swin_tiny/c5_b0`：旧 `0.1749`，新 `0.1422`，下降 `0.0327`。
  - 同样是 middle-delta 后叠加较强专家保真导致下降。

结论：

- middle-delta 的尺度门控是有效的：`organs/vit/c7_b0.01` 因 `norm_dispersion=0.670` 不再误触发，恢复到 `0.3577`。
- 但“只由 M1 分数决定专家保真”仍然过宽。医学分数高的专家可能只是覆盖样本多或形态特征强，并不代表它的 task vector 可以安全锚定全局模型。
- 下一步不恢复粗暴全局 cap，而是让专家保真也服从 M2 的冲突约束：被锚定专家的 delta 必须与医学共识 delta 方向一致。也就是把“专家保真”从经验补丁改成 conflict-aware specialist anchor。

后续验证：

- 先做了只读方向一致性分析，计算被选专家 delta 与医学共识 delta 的余弦相似度。
- 结果否定了“专家方向不一致导致负优化”的假设：
  - `organc/swin c5_b0` 的专家余弦约 `0.898`，仍是大负例。
  - `organs/swin c5_b0` 的专家余弦约 `0.969`，仍会被强专家锚定伤害。
  - 因此不能把专家保真改成“按方向一致性选择/限幅”，这个措施删除。

新解释：

- middle-delta 已经是一次冲突过滤，目标是去掉异常大更新并保留中等一致更新。
- 如果随后再用很强的单专家锚定，就会把模型重新拉回单客户端表示，抵消冲突过滤效果。
- 所以专家保真不是不能用，而是在启用 middle-delta 的高冲突场景下只能做轻量保真。

采取措施：

- 增加 `conflict-stabilized specialist anchor`：
  - 如果未启用 middle-delta，专家保真不变。
  - 如果启用 middle-delta 且原始专家权重较强（`raw_specialist_anchor_weight >= 0.24`），将专家权重乘以 `0.55`。
  - 如果限幅后低于 `0.15`，不再锚定专家。
- 这个规则不看数据集名，不看模型族，只看是否已经进入 middle-delta 冲突过滤以及专家锚定是否过强。

验证结果：

- 14 例 targeted probe：`outputs/codex_conflict_capped_specialist_probe_20260617/full`。
  - `organc/swin c5_b0`：上一轮 `0.0895`，限幅后 `0.2233`，恢复到旧 full。
  - `organs/swin c5_b0`：上一轮 `0.1422`，限幅后 `0.1749`，恢复到旧 full。
  - `organs/convnext c7_b0.01`：保持 `0.2354`。
  - `organs/swin c7_b0.01`：保持 `0.1521`。
- 72 例宽验证：`outputs/codex_conflict_capped_specialist_anatomy_probe_20260617/full`。
  - 相对上一轮 scale-guard：mean diff `+0.001386`，W/T/L=`3/66/3`。
  - 相对失败 unified：mean diff `+0.004304`，W/T/L=`13/51/8`。
  - 相对旧 full：mean diff `-0.000623`，W/T/L=`9/43/20`。
  - 说明该规则修复了最严重的专家锚定负例，但仍没有整体超过旧 full。

失败尝试：

- 尝试给限幅后专家权重设置 `0.15` 下限，以避免 `c5_b0.01` 被压成 0。
  - `organc/swin c5_b0.01` 从 `0.1008` 小幅到 `0.1035`，仍远低于旧 full 的 `0.1753`。
- 尝试把限幅条件改成 `middle-delta + 强专家 + head_conflict_pressure >= 0.40`。
  - `organc/swin c5_b0.01` 修回 `0.1694`。
  - 但 `organs/swin c5_b0` 又从 `0.1749` 退到 `0.1422`。
- 结论：`head_conflict_pressure` 不能稳定地区分这两个强专家 case；该尝试删除，代码回到已有 72 例验证的稳定限幅版。

当前结论：

- 当前可保留的统一机制是：
  - middle-delta 只在低覆盖、高符号/方向冲突、尺度均衡时启用；
  - high norm dispersion 退回 top-magnitude，避免误删强适配更新；
  - middle-delta 后强专家只允许轻量保真。
- 仍需进一步优化的点不是“是否需要 dataset/model 分支”，而是如何区分 `c5_b0.01` 这类仍需要较强专家保真的局部 case。暂时没有找到足够稳定的 checkpoint-only 判据。

## 观测 22：剩余负例不能用“关类别路由”或“关冲突模块”统一修复

对象：

- 当前稳定版：`outputs/codex_conflict_capped_specialist_anatomy_probe_20260617/full`。
- 类别路由消融：`outputs/codex_residual_probe_no_class_routing_20260617/full`。
- 冲突模块消融：`outputs/codex_residual_probe_no_conflict_20260617/full`。
- 旧 full 对照：`outputs/codex_balanced_middle_medical_full_20260616/my_merge_ablation_grid/full`。

关键结果：

| case | old | current | no_class_routing | no_conflict |
|---|---:|---:|---:|---:|
| `organc/resnet/c3_b0.1` | 0.4399 | 0.4045 | 0.4034 | 0.3543 |
| `organc/resnet/c7_b0` | 0.2922 | 0.2453 | 0.2481 | 0.2416 |
| `organc/swin/c5_b0.01` | 0.1753 | 0.1008 | 0.1008 | 0.1641 |
| `organs/convnext/c7_b0.01` | 0.0798 | 0.2354 | 0.2354 | 0.0784 |
| `organs/resnet/c3_b0` | 0.2660 | 0.2177 | 0.2853 | 0.2452 |
| `organs/resnet/c5_b0.01` | 0.2307 | 0.2124 | 0.1860 | 0.1771 |
| `organs/swin/c7_b0.01` | 0.0785 | 0.1521 | 0.1521 | 0.0785 |
| `organs/vit/c7_b0` | 0.1677 | 0.1534 | 0.1534 | 0.1983 |

结论：

- 关类别路由能修 `organs/resnet/c3_b0`，但会伤 `organs/resnet/c5_b0.01`，不能全局采用。
- 关冲突模块能修 `organc/swin/c5_b0.01` 和 `organs/vit/c7_b0`，但会直接破坏两个关键正例：
  - `organs/convnext/c7_b0.01`：`0.2354 -> 0.0784`。
  - `organs/swin/c7_b0.01`：`0.1521 -> 0.0785`。
- 因此剩余负例不是某个模块整体过强，而是局部 case 中模块交互仍不稳定。

采取措施：

- 不继续加 dataset/model 分支。
- 不保留 head-conflict 条件限幅和 floor 限幅，因为它们分别修一个点、伤另一个同类点。
- 当前代码保留已有 72 例验证过的稳定版：
  - middle-delta 的尺度门控；
  - middle-delta 后强专家轻量限幅；
  - 不使用验证集候选选择。

下一步建议：

- 如果继续优化，应换更基础的观测对象，而不是继续调阈值：
  - 分层统计 early/mid/late 的 sign conflict、direction conflict、norm dispersion，确认当前全向量 conflict 是否掩盖了层间差异。
  - 如果确实层间差异稳定，再考虑把 M2 的 delta filter 从“全模型一个 filter”改成“按层组使用同一套冲突判据”，仍不按模型族/数据集分支。

## 观测 23：分层冲突修复一个回归后，又暴露出 representation/head 耦合问题

背景：

- 继续沿着观测 22 的建议做，不再整体开关模块，也不加 dataset/model 分支。
- 先把 M2 的冲突统计从全模型一个向量拆成 `early/mid/late/classifier` 四组。
- 类别头路由不再用 classifier-only conflict，而用 representation 聚合冲突作阻尼信号，避免 head 自身局部震荡把所有类别路由压坏。

第一轮现象：

- `organsmnist_224/convnext/c7_b0.01` 从上一轮回归的 `0.0798` 恢复到 `0.2354`。
- 但是 `organcmnist_224/vit_t/c7_b0.01` 从旧稳定值约 `0.3143` 掉到约 `0.1222`。
- trace 显示这里不是单个 head 冲突过强，而是 representation 的 early/mid/late 都处在相近的高冲突区间；分组逐段处理后，trunk 和 head 的几何关系被拆散了。

采取措施：

- 保留 layerwise conflict 统计，但增加一个无模型名、无数据集名的 `representation_coupled_middle_delta` 规则。
- 触发条件只看统计量：
  - 类别数不少于 10；
  - 平均 client class coverage 不高于 `0.30`；
  - sign conflict 和 direction conflict 都高；
  - norm dispersion 不高，说明不是某个客户端尺度异常；
  - 至少两个 representation group 有参数，且 early/mid/late 的 conflict score floor 高、spread 小；
  - head conflict pressure 低，说明 head 本身不是主要矛盾。
- 触发后不再按组分别 overlay，而是对全网 mergeable params 使用同一个 middle-delta，保持 anatomy representation 和 classifier head 的相对方向。
- 随后又把 `_param_group` 改成纯 state_dict key pattern 规则，并移除了 `vit/swin/convnext/resnet` 这类模型名判断；当前方法不靠模型名触发。

故事线：

- 医学图像的核心问题不是 NLP 式任务语义拼接，而是不同机构的解剖覆盖、扫描协议、局部形态分布不一致。
- 服务端只需要拿到 checkpoint、client class coverage、五维 morphology summary、BN running statistics 这类低维统计；生产配置可先用 `scripts/prepare_my_merge_feature_summaries.py` 在客户端侧导出 summary，再用 `--my-merge-feature-summary-root` 和 `--my-merge-require-feature-summary` 合并，服务端不需要读取原始图像。
- M1 负责医学证据加权：谁覆盖了哪些类别、形态统计是否可靠。
- M2 负责 checkpoint-only 冲突分诊：看 task delta 的符号冲突、方向冲突、尺度离散度，以及 early/mid/late/head 的冲突形态。
- 当 representation 层组同步高冲突且 head 压力低时，说明不是某个 head 要单独修，而是全局解剖表征方向需要保持耦合，因此启用 coupled middle-delta。

实现参考：

- TIES-Merging / Task Arithmetic 给了 task vector、符号冲突、delta 过滤的启发。
- Model Soups / AdaMerging 给了 checkpoint 融合和候选加权的启发，但这里不使用服务端验证集选择。
- FedBN / FedProto / FedDF / FedD3 给了联邦异构、原型/统计上传、无原始数据融合的隐私叙事。
- 本方法保留医学专用部分：client class coverage、morphology summary、anatomy representation/head 耦合；不是把 NLP merge 规则直接迁移过来。

最终 targeted smoke：

| case | key result |
|---|---:|
| `organsmnist_224/convnext/c7_b0.01` | `0.2354` |
| `organsmnist_224/swin_tiny/c7_b0` | `0.2346` |
| `organsmnist_224/swin_tiny/c7_b0.01` | `0.1521` |
| `organcmnist_224/vit_t/c7_b0.01` | `0.3116` |
| `bloodmnist_224/vit_t/c5_b0.01` | `0.2792` |

完整 9 格 smoke：

| dataset/model | c3_b0 | c3_b0.01 | c3_b0.1 | c5_b0 | c5_b0.01 | c5_b0.1 | c7_b0 | c7_b0.01 | c7_b0.1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `organs/convnext` | 0.1103 | 0.0497 | 0.0921 | 0.0785 | 0.0798 | 0.0497 | 0.2354 | 0.2354 | 0.1097 |
| `organs/swin_tiny` | 0.0578 | 0.1770 | 0.0790 | 0.1635 | 0.0497 | 0.1097 | 0.2346 | 0.1521 | 0.0497 |
| `organc/vit_t` | 0.0763 | 0.2713 | 0.2029 | 0.0895 | 0.0970 | 0.1851 | 0.2787 | 0.3116 | 0.2030 |
| `blood/vit_t` | 0.2900 | 0.2397 | 0.1061 | 0.1695 | 0.2792 | 0.2812 | 0.1692 | 0.1374 | 0.0830 |

关键 trace：

- `organc/vit_t/c7_b0.01` 触发 `representation_coupled_middle_delta`：
  - `head_conflict_low=true`；
  - representation conflict floor `0.5227`，spread `0.0236`；
  - selected candidate 为 `medical_weighted_fusion + conflict_delta_0p12_coupled_middle_keep_0p25_remove_top_0p10 + specialist_anchor_0p00`。
- `organs/convnext` 和 `organs/swin_tiny` 的关键 case 没触发 coupled fallback，因为 head conflict pressure 高；它们继续走 layerwise middle/sign delta。

完整 72 格 full：

- 输出：`outputs/codex_layerwise_m2_full_20260618_r2/full`。
- 范围：`organcmnist_224 + organsmnist_224`，`resnet/convnext/vit_t/swin_tiny`，3x3 client/beta，共 72 例。
- 状态：72/72 OK。
- 相对 `outputs/codex_conflict_capped_specialist_anatomy_probe_20260617/full`：
  - mean diff `-0.000731`；
  - W/T/L = `20/25/27`；
  - new mean `0.186524`，old mean `0.187254`。
- 相对 `outputs/codex_balanced_middle_medical_full_20260616/my_merge_ablation_grid/full` 的 72 个可匹配 anatomy case：
  - mean diff `-0.001353`；
  - W/T/L = `21/20/31`；
  - new mean `0.186524`，old mean `0.187877`。

full 中的主要收益：

| case | old 20260617 | new | diff |
|---|---:|---:|---:|
| `organc/convnext/c3_b0` | 0.0907 | 0.1842 | +0.0935 |
| `organs/resnet/c3_b0.01` | 0.2304 | 0.2996 | +0.0692 |
| `organs/resnet/c3_b0` | 0.2177 | 0.2525 | +0.0348 |
| `organs/resnet/c7_b0.01` | 0.2079 | 0.2329 | +0.0250 |
| `organc/vit_t/c7_b0.01` | 0.3143 | 0.3177 | +0.0034 |

full 中的主要回撤：

| case | old 20260617 | new | diff |
|---|---:|---:|---:|
| `organs/resnet/c5_b0.1` | 0.4365 | 0.3896 | -0.0469 |
| `organc/resnet/c5_b0.01` | 0.3800 | 0.3345 | -0.0455 |
| `organc/resnet/c5_b0` | 0.3059 | 0.2635 | -0.0424 |
| `organc/resnet/c5_b0.1` | 0.3644 | 0.3421 | -0.0223 |
| `organc/resnet/c3_b0.1` | 0.4045 | 0.3834 | -0.0211 |

结论：

- 这版比继续阈值特判更容易讲清楚：先用 layerwise conflict 修掉 head routing 回归，再用 representation-coupled middle-delta 修掉 trunk/head 解耦负例。
- 当前代码已经去掉模型名触发；剩余判断都是 checkpoint 统计、client coverage 和医学 morphology summary。
- full 结果说明：关键负例确实修住了，整体均值基本持平但略低于旧 full；这版适合作为“机制更干净的主线候选”，还不是无条件替换旧稳定版。
- 下一步如果继续冲榜，重点不该再调 coupled 规则，而是修 ResNet 的 `c5` 回撤；当前回撤集中在 `organc/resnet` 和 `organs/resnet/c5`。

参考链接：

- TIES-Merging: https://arxiv.org/abs/2306.01708
- Task Arithmetic: https://arxiv.org/abs/2212.04089
- Model Soups: https://arxiv.org/abs/2203.05482
- AdaMerging: https://arxiv.org/abs/2310.02575
- FedBN: https://arxiv.org/abs/2102.07623
- FedProto: https://arxiv.org/abs/2105.00243
- FedDF: https://arxiv.org/abs/2006.07242
- FedD3: https://arxiv.org/abs/2208.11311

## 观测 24：修掉 c5 specialist 阈值伪影，并用 coherent top-delta 保持广覆盖冲突下的整网一致性

背景：

- 观测 23 的 full 结果虽然机制更干净，但整体均值略低于 20260617 稳定版。
- 回撤高度集中在 `c5` 的解剖类 case，尤其是 `organs/resnet/c5_b0.1`、`organc/resnet/c5_b0.01`、`organc/resnet/c5_b0`。
- trace 显示部分 case 的 specialist anchor 原始权重已经达到最低可信阈值，但后续 conflict cap 把它压到阈值以下，等价于把一个本来应保留的机构专科证据抹掉。
- 另一个现象是：在类别覆盖较广、representation pressure 不高、head pressure 也不高但符号冲突存在时，逐层 top-delta 会把 trunk/head 的微弱一致方向拆开；这类冲突更像“全局同向小修正”，不是某一层单独要强修。

采取措施：

- 不引入 `if model == ...` 或 dataset 特判，全部使用 checkpoint 统计和 client summary 统计触发。
- 增加 `broad_coverage_low_representation_pressure_coherent_top_delta`：
  - 平均 class coverage 较广；
  - representation pressure 低；
  - head pressure 低；
  - sign conflict 活跃；
  - 类别数足够多，避免在极小类别任务上误判。
- 触发后对全网 mergeable parameters 使用统一的 `coherent_top_magnitude_0p50`，而不是 early/mid/late/classifier 分层各自选 top-delta。
- 调整 specialist conflict cap：
  - 对 middle-delta active 且 raw specialist weight 已经较高的 case，cap 从 `0.55` 放宽到 `0.80`；
  - 如果 raw specialist weight 已达到最低 anchor 阈值，cap 不再把它压到阈值以下。

故事线：

- 医学多机构融合里，一部分 client 是“专科强证据”：覆盖某些解剖类别、形态统计稳定、局部分布和其他机构差异明显。
- 服务端不能看原始图像，因此不能靠服务端验证集或 raw sample 重新选择专家；只能用 checkpoint delta、class coverage、morphology summary、BN running statistics 这些低维证据。
- 当冲突集中在局部 head 或某个 representation group，layerwise conflict routing 是合理的；当全网低压但 sign conflict 活跃，说明是各机构影像协议或解剖形态带来的轻量全局方向差异，应保持 trunk/head 的整体耦合。
- 这版把“专科证据不能被阈值伪影擦掉”和“广覆盖低压冲突要整网一致修正”合并进同一条医学证据流，仍然不依赖模型名。

实现状态：

- 主文件：`methods/my_merge.py`。
- implementation label：`medical_evidence_layerwise_conflict_merge_v14`。
- 语法检查：`./.gpuenv/bin/python -m py_compile methods/my_merge.py` 通过。
- 源码中未发现 `resnet/vit/swin/convnext/mobilenet` 这类模型名分支。

ResNet 18 格 probe：

- 输出：`outputs/codex_coherent_anchor_resnet_probe_20260618/full`。
- 状态：18/18 OK。
- 相对 v13 full `outputs/codex_layerwise_m2_full_20260618_r2/full`：
  - mean diff `+0.006611`；
  - W/T/L = `4/13/1`。
- 相对 20260617 稳定版 `outputs/codex_conflict_capped_specialist_anatomy_probe_20260617/full`：
  - mean diff `+0.000728`；
  - W/T/L = `7/0/11`。

ResNet 关键修复：

| case | v13 | v14 | 20260617 |
|---|---:|---:|---:|
| `organs/resnet/c5_b0.1` | 0.3896 | 0.4376 | 0.4365 |
| `organc/resnet/c5_b0.01` | 0.3345 | 0.3684 | 0.3800 |
| `organc/resnet/c5_b0.1` | 0.3421 | 0.3642 | 0.3644 |
| `organc/resnet/c5_b0` | 0.2635 | 0.2851 | 0.3059 |

完整 72 格 full：

- 输出：`outputs/codex_coherent_anchor_full_20260619/full`。
- 范围：`organcmnist_224 + organsmnist_224`，4 个 small model，3x3 client/beta，共 72 例。
- 状态：72/72 valid eval；resume 时 67 条已有结果被 skip，最后 5 条补跑 OK。
- v14 mean：`0.188664`。

相对 v13 `outputs/codex_layerwise_m2_full_20260618_r2/full`：

- v13 mean：`0.186524`。
- mean diff：`+0.002140`。
- W/T/L：`8/58/6`。

| top gain vs v13 | v13 | v14 | diff |
|---|---:|---:|---:|
| `organc/swin_tiny/c5_b0.01` | 0.1008 | 0.1618 | +0.0610 |
| `organs/resnet/c5_b0.1` | 0.3896 | 0.4376 | +0.0480 |
| `organc/resnet/c5_b0.01` | 0.3345 | 0.3684 | +0.0340 |
| `organc/resnet/c5_b0.1` | 0.3421 | 0.3642 | +0.0220 |
| `organc/resnet/c5_b0` | 0.2635 | 0.2851 | +0.0215 |

| top drop vs v13 | v13 | v14 | diff |
|---|---:|---:|---:|
| `organs/swin_tiny/c5_b0` | 0.1657 | 0.1463 | -0.0195 |
| `organs/resnet/c3_b0.1` | 0.3567 | 0.3502 | -0.0066 |
| `organc/vit_t/c3_b0.1` | 0.2037 | 0.1981 | -0.0056 |
| `organs/vit_t/c5_b0.1` | 0.1073 | 0.1030 | -0.0043 |
| `organc/vit_t/c5_b0.01` | 0.0970 | 0.0954 | -0.0016 |

相对 20260617 稳定版 `outputs/codex_conflict_capped_specialist_anatomy_probe_20260617/full`：

- 20260617 mean：`0.187254`。
- mean diff：`+0.001410`。
- W/T/L：`21/26/25`。

| top gain vs 20260617 | 20260617 | v14 | diff |
|---|---:|---:|---:|
| `organc/convnext/c3_b0` | 0.0907 | 0.1842 | +0.0935 |
| `organs/resnet/c3_b0.01` | 0.2304 | 0.2996 | +0.0692 |
| `organc/swin_tiny/c5_b0.01` | 0.1008 | 0.1618 | +0.0610 |
| `organs/resnet/c3_b0` | 0.2177 | 0.2525 | +0.0348 |
| `organs/resnet/c7_b0.01` | 0.2079 | 0.2329 | +0.0250 |

| top drop vs 20260617 | 20260617 | v14 | diff |
|---|---:|---:|---:|
| `organs/swin_tiny/c5_b0` | 0.1749 | 0.1463 | -0.0287 |
| `organc/resnet/c3_b0.1` | 0.4045 | 0.3834 | -0.0211 |
| `organc/resnet/c5_b0` | 0.3059 | 0.2851 | -0.0208 |
| `organs/resnet/c5_b0.01` | 0.2124 | 0.1926 | -0.0198 |
| `organc/resnet/c3_b0.01` | 0.3530 | 0.3353 | -0.0176 |

分组均值变化：

| group | diff vs v13 | W/L/T vs v13 | diff vs 20260617 | W/L/T vs 20260617 |
|---|---:|---:|---:|---:|
| `organc/convnext` | +0.0000 | 1/0/8 | +0.0101 | 2/2/5 |
| `organc/resnet` | +0.0086 | 3/0/6 | -0.0110 | 0/9/0 |
| `organc/swin_tiny` | +0.0068 | 1/0/8 | +0.0057 | 2/2/5 |
| `organc/vit_t` | -0.0009 | 0/3/6 | -0.0001 | 3/3/3 |
| `organs/convnext` | +0.0003 | 1/0/8 | +0.0004 | 2/0/7 |
| `organs/resnet` | +0.0046 | 1/1/7 | +0.0124 | 7/2/0 |
| `organs/swin_tiny` | -0.0022 | 0/1/8 | -0.0058 | 0/3/6 |
| `organs/vit_t` | -0.0002 | 1/1/7 | -0.0005 | 5/4/0 |

结论：

- v14 相比观测 23 的 v13 是明确正向：均值提高 `+0.002140`，并修掉了最显眼的 `c5` specialist 回撤。
- 相比 20260617 稳定版也小幅正向：均值提高 `+0.001410`，但 W/T/L 并不压倒，说明它更像“机制更完整且均值略优”的主线候选，不是每个 case 都更好。
- 剩余风险集中在 `organs/swin_tiny/c5_b0`、`organc/resnet/c3` 和 `organs/resnet/c5_b0.01`；如果继续冲榜，应优先观测这些 case 的 conflict trace，而不是再加模型名或数据集特判。
