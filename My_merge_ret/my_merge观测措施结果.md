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
