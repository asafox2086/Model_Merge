# my_merge Attempt Log

本文件记录 `my_merge` 的设计尝试与实验结论，包含成功与失败的方向，避免后续重复试错。

## Scope

- 记录对象：`methods/my_merge.py`
- 当前目标：针对医学影像任务设计专用模型融合方法，而不是做通用模型平均
- 明确约束：方法应只服务医学影像，不追求在 NLP 或通用视觉场景可直接复用

## Design Principle

当前方法围绕“医学影像中的物理边界、组织结构、病灶几何比自然图像更稳定且更有诊断意义”这个切入点展开。

按数据集区分的医学特征分支已经写入 [methods/my_merge.py](/data/liyapeng_grp/program/MedMNISTMerge/methods/my_merge.py:46)：

- `bloodmnist_224`：胞核/胞质比例、染色质对比、边界强度、圆整度
- `dermamnist_224`：病灶不对称性、边界不规则、颜色杂色
- `organcmnist_224`、`organsmnist_224`：组织占比、左右对称、中心线偏置
- `chaoshengmnist_224`：斑点噪声、后方回声、各向异性

## Attempt 1

医学模态专用形态学级联融合 `medical_modality_specific_boundary_cascade_merge_v4`

做法：

- 把 `my_merge` 的适用范围从 `bloodmnist_224` 扩展到全部 5 个医学数据集
- 对不同医学模态提取不同形态学特征
- 用验证集上的形态学难样本得分为 client 打分
- 在 `avg / morphology / consensus` 三个候选之间自动选择

代码位置：

- 医学任务入口：[methods/my_merge.py](/data/liyapeng_grp/program/MedMNISTMerge/methods/my_merge.py:46)
- 模态专用特征：[methods/my_merge.py](/data/liyapeng_grp/program/MedMNISTMerge/methods/my_merge.py:110)
- 候选选择：[methods/my_merge.py](/data/liyapeng_grp/program/MedMNISTMerge/methods/my_merge.py:832)

结果：

- `resnet / bloodmnist_224 / c3_b0` smoke 通过，`test_acc = 0.4648`
  结果文件：[eval_summary.csv](/data/liyapeng_grp/program/MedMNISTMerge/outputs/my_merge_smoke_med_v4/reports/eval_summary.csv)

结论：

- 成功把方法改成了明确的医学影像专用 merge
- 流程可跑通，数值稳定，没有出现此前的 `nan`

## Attempt 2

优先修补历史上表现不好的 `convnext`

做法：

- 先只跑差的区域，而不是重跑全表
- 选取 `convnext + c3_b0` 在 `bloodmnist_224 / dermamnist_224 / organcmnist_224`

结果文件：

- [outputs/my_merge_med_v4_bad_c3/reports/eval_summary.csv](/data/liyapeng_grp/program/MedMNISTMerge/outputs/my_merge_med_v4_bad_c3/reports/eval_summary.csv)

结果：

- `convnext / bloodmnist_224 / c3_b0 = 0.1947`
- `convnext / dermamnist_224 / c3_b0 = 0.6688`
- `convnext / organcmnist_224 / c3_b0 = 0.2233`

结论：

- 成功把这三格拉到和原有最优方法持平
- 但没有形成超越，只能算“补齐短板”，不能算“性能非常突出”

## Attempt 3

医学显著残差回灌

做法：

- 在层级融合后，不再完全抹平医学显著客户端残差
- 用稀疏残差把边界、病灶、组织几何相关的增量回灌到早中后层

目的：

- 解决“平均后医学局部结构被冲淡”的问题

结论：

- 对 `convnext` 追平有帮助，但没有把已经持平的结果继续推高
- 不是无效，但增益有限

## Attempt 4

形态学专家骨架保留 `morph_anchor`

做法：

- 不再只做参数平均
- 直接保留形态学最优 client 的整套 backbone
- 只对分类头做专家级融合

代码位置：

- [methods/my_merge.py](/data/liyapeng_grp/program/MedMNISTMerge/methods/my_merge.py:584)

目标：

- 主要针对 `vit_t`、`swin_tiny`、`CLIP` 这类更怕 encoder 几何被破坏的模型

结论：

- 在 `vit_t / bloodmnist_224 / c3_b0` 上没有带来提升
- 说明 transformer 的问题不只是“选错专家”，而是更底层的表示空间对齐问题

## Attempt 5

参考模型医学增量融合 `reference_delta`

做法：

- 对 `vit_t` 和 `CLIP` 不再直接 merge 全部 client 权重
- 以参考预训练模型为锚点，只融合 client 相对参考模型的“医学增量”
- 目标是保住预训练表示几何，只注入医学相关偏移

代码位置：

- [methods/my_merge.py](/data/liyapeng_grp/program/MedMNISTMerge/methods/my_merge.py:775)

结果文件：

- [outputs/my_merge_med_v4p3_reference_probe/reports/eval_summary.csv](/data/liyapeng_grp/program/MedMNISTMerge/outputs/my_merge_med_v4p3_reference_probe/reports/eval_summary.csv)

结果：

- `vit_t / bloodmnist_224 / c3_b0 = 0.1692`
- `openai/clip-vit-base-patch32 / bloodmnist_224 / c3_b0 = 0.2724`

对比：

- `vit_t` 仍低于原表最好值 `0.2207`
- `CLIP` 比此前 `my_merge` 的 `0.2675` 有提升，但仍低于原表最好值 `0.3116`

结论：

- `reference_delta` 对 VLM 有一定帮助
- 但对 `vit_t` 没有起效，说明单纯参考模型增量约束仍不足以解决 transformer 的医学表示错位

## Attempt 6

医学专家整模态选择 `specialist_client`

做法：

- 不再默认所有 client 的知识都值得融合
- 对 `transformer / VLM` 增加“整套医学专家 client”候选
- 依据验证集上的整体准确率、医学难样本准确率、边界显著样本表现，从 client 中选出最像“医学专家”的那个
- 再把这个完整 client 作为候选，与 `avg / morphology / consensus / reference_delta` 一起比较

代码位置：

- 医学专家候选构造：[methods/my_merge.py](/data/liyapeng_grp/program/MedMNISTMerge/methods/my_merge.py:595)

结果文件：

- [outputs/my_merge_med_v4p4_specialist_probe/reports/eval_summary.csv](/data/liyapeng_grp/program/MedMNISTMerge/outputs/my_merge_med_v4p4_specialist_probe/reports/eval_summary.csv)

结果：

- `vit_t / bloodmnist_224 / c3_b0 = 0.1692`
- `openai/clip-vit-base-patch32 / bloodmnist_224 / c3_b0 = 0.3663`

对比：

- `vit_t` 仍低于原表最好值 `0.2207`
- `CLIP` 明显超过原表最好值 `0.3116`

结论：

- 这是目前最明确的一次“超过已有算法”的成功案例
- 说明对 VLM 而言，医学表示空间往往不适合被继续平均，保留最强医学专家的整套表示反而更好
- 但这条思路对 `vit_t` 仍未生效，说明 `vit_t` 后续需要更强的原型级约束，而不是简单专家选择

## Attempt 7

医学原型头 `prototype_head`

做法：

- 保留融合后的 transformer encoder
- 不再直接沿用 merge 出来的分类头
- 用验证集特征构建“医学类原型”，再把原型转成新的线性分类头
- 这个思路更贴合医学场景，因为类别边界本来就依赖病灶/组织原型，而不是只依赖原始分类头参数平均

代码位置：

- 原型特征提取与原型头构造：[methods/my_merge.py](/data/liyapeng_grp/program/MedMNISTMerge/methods/my_merge.py:309)
- 原型头候选实现：[methods/my_merge.py](/data/liyapeng_grp/program/MedMNISTMerge/methods/my_merge.py:628)

结果文件：

- `vit_t / bloodmnist_224 / c3_b0`：
  [outputs/my_merge_med_v4p5_proto_vit_probe/reports/eval_summary.csv](/data/liyapeng_grp/program/MedMNISTMerge/outputs/my_merge_med_v4p5_proto_vit_probe/reports/eval_summary.csv)
- `vit_t / organcmnist_224 / c3_b0`：
  [outputs/my_merge_med_v4p7_proto_vit_organc_probe/reports/eval_summary.csv](/data/liyapeng_grp/program/MedMNISTMerge/outputs/my_merge_med_v4p7_proto_vit_organc_probe/reports/eval_summary.csv)
- `swin_tiny / bloodmnist_224 / c3_b0`：
  [outputs/my_merge_med_v4p6_proto_swin_probe/reports/eval_summary.csv](/data/liyapeng_grp/program/MedMNISTMerge/outputs/my_merge_med_v4p6_proto_swin_probe/reports/eval_summary.csv)
- `vit_t / 5 datasets / c3_b0` 和 `CLIP / 5 datasets / c3_b0`：
  [outputs/my_merge_med_v4p8_vit_clip_c3/reports/eval_summary.csv](/data/liyapeng_grp/program/MedMNISTMerge/outputs/my_merge_med_v4p8_vit_clip_c3/reports/eval_summary.csv)

结果：

- `vit_t / bloodmnist_224 / c3_b0 = 0.2783`
- `vit_t / dermamnist_224 / c3_b0 = 0.6688`
- `vit_t / organcmnist_224 / c3_b0 = 0.3600`
- `vit_t / organsmnist_224 / c3_b0 = 0.3356`
- `vit_t / chaoshengmnist_224 / c3_b0 = 0.1222`
- `swin_tiny / bloodmnist_224 / c3_b0 = 0.1947`
- `CLIP / bloodmnist_224 / c3_b0 = 0.3663`
- `CLIP / dermamnist_224 / c3_b0 = 0.6683`
- `CLIP / organcmnist_224 / c3_b0 = 0.2631`
- `CLIP / organsmnist_224 / c3_b0 = 0.3639`
- `CLIP / chaoshengmnist_224 / c3_b0 = 0.1473`

对比：

- `vit_t / bloodmnist_224 / c3_b0` 原表最好值是 `0.2207`
- `vit_t / dermamnist_224 / c3_b0` 原表最好值是 `0.6688`
- `vit_t / organcmnist_224 / c3_b0` 原表最好值是 `0.1748`
- `vit_t / organsmnist_224 / c3_b0` 原表最好值是 `0.1545`
- `vit_t / chaoshengmnist_224 / c3_b0` 原表最好值是 `0.2354`
- `swin_tiny / bloodmnist_224 / c3_b0` 原表最好值是 `0.1947`
- `CLIP / bloodmnist_224 / c3_b0` 原表最好值是 `0.3116`
- `CLIP / dermamnist_224 / c3_b0` 原表最好值是 `0.6748`
- `CLIP / organcmnist_224 / c3_b0` 原表最好值是 `0.3176`
- `CLIP / organsmnist_224 / c3_b0` 原表最好值是 `0.3126`
- `CLIP / chaoshengmnist_224 / c3_b0` 原表最好值是 `0.1842`

结论：

- 这是当前最有效的 transformer 医学化改法
- `vit_t` 在血液、器官 CT、以及部分皮肤 case 上已经超过或追平原有方法
- `swin_tiny` 目前只能追平，说明原型头对 Swin 有帮助，但还不足以稳定超越
- `CLIP` 在血液和器官方向出现了明确提升，但在超声和部分皮肤/器官 case 上仍未全面超过

## Current Assessment

截至当前版本，可以比较明确地下结论：

- `CNN` 路线：医学形态学融合是有效的，至少能把差的 case 拉到和强 baseline 持平
- `Transformer / VLM` 路线：开始出现系统性有效信号，尤其是 `vit_t` 的医学原型头和 `CLIP` 的医学专家整模态选择
- 现阶段最可靠的收益来自“医学模态特征 + 候选选择 + 稀疏残差保护 + 医学原型头”
- 现阶段最主要的失败区域变成 `swin_tiny` 和 `chaoshengmnist_224` 的 transformer/VLM case

## Known Successful Pieces

- 医学模态专用特征分支是值得保留的
- `convnext` 的弱项已经从明显落后修到追平
- `CLIP` 在 `bloodmnist_224 / c3_b0` 上从此前 `0.2675` 提升到 `0.2724`
- `CLIP` 在 `bloodmnist_224 / c3_b0` 上进一步提升到 `0.3663`，已经超过原表当前最好值 `0.3116`
- `vit_t` 在 `bloodmnist_224 / c3_b0` 上提升到 `0.2783`，超过原表当前最好值 `0.2207`
- `vit_t` 在 `organcmnist_224 / c3_b0` 上提升到 `0.3600`，超过原表当前最好值 `0.1748`
- `vit_t` 在 `organsmnist_224 / c3_b0` 上提升到 `0.3356`，超过原表当前最好值 `0.1545`
- 评估流程已经稳定，不再出现 `nan`

## Known Failed Or Insufficient Pieces

- 只靠手工形态学特征加权，无法稳定拉高 `vit_t`
- 只保留形态学 anchor backbone，不足以修复 transformer 的表示空间问题
- 只围绕参考预训练模型做增量融合，对 `CLIP` 有小幅帮助，但还不够强
- `chaoshengmnist_224` 上的 `vit_t / CLIP` 仍明显落后，说明当前原型头和医学专家选择还不够适配超声 speckle 结构

## Attempt 8

full-val 医学专家选择修正

做法：

- 对 `transformer / VLM` 不再只看少量 batch 的验证集候选分数
- 改成默认使用更完整的验证集统计
- 候选选择时加入医学优先级：
  - `VLM` 优先医学专家整模态 `specialist_client`
  - `vit_t` 优先医学原型头 `prototype_head`
- 避免分数接近时又退回普通平均

结果文件：

- [outputs/my_merge_med_v5_focus_probe/reports/eval_summary.csv](/data/liyapeng_grp/program/MedMNISTMerge/outputs/my_merge_med_v5_focus_probe/reports/eval_summary.csv)

结果：

- `openai/clip-vit-base-patch32 / dermamnist_224 / c3_b0 = 0.6758`
- `openai/clip-vit-base-patch32 / organcmnist_224 / c3_b0 = 0.2672`
- `swin_tiny / dermamnist_224 / c3_b0 = 0.6688`

对比：

- `CLIP / dermamnist_224 / c3_b0` 原表最好值是 `0.6748`
- `CLIP / organcmnist_224 / c3_b0` 原表最好值是 `0.3176`
- `swin_tiny / dermamnist_224 / c3_b0` 原表最好值是 `0.1097`

结论：

- `CLIP / derma` 已经被成功翻过去
- `swin_tiny / derma` 被大幅拉起，说明医学专家选择对部分 Swin case 也成立
- 但 `CLIP / organc` 仍未过线，说明仅靠 full-val 选择还不足以解决所有器官 CT 的 VLM case

## Next Recommended Direction

如果继续沿医学领域方向推进，优先级最高的是：

- 为 `vit_t / swin_tiny / CLIP` 加入“医学原型约束”
- 从验证集估计每类医学原型，而不是只依赖像素级形态学分数
- 用原型分离度、类间混淆、病灶级相似性来决定 transformer/VLM 的融合方式

这条路线仍然是医学领域专用的，因为它依赖的是医学类别原型、组织边界和病灶结构，而不是通用文本或自然图像语义。

## Attempt 9

domain-specific medical preprocessing signals

做法：

- `DermaMNIST`：把皮肤镜图像的医学先验放进融合统计，而不是只做普通 RGB 纹理统计
- 引入 `Shades of Gray` 风格颜色恒常，降低不同皮肤镜和光照色温导致的 client 偏差
- 引入 `DullRazor` 风格的黑帽变换毛发检测和局部修复灰度图，用毛发/伪影负载影响医学重点样本权重
- 对皮肤长尾类别加入稀有类权重和 focal-style 难样本权重，避免融合只偏向良性大类
- `OrganCMNIST / OrganSMNIST`：加入软组织窗近似映射、冠状面左右结构先验、矢状面纵向/脊柱侧带先验
- `ChaoshengMNIST`：加入各向异性扩散近似滤波、残余 speckle、边缘一致性和后方声影指标
- 对 `VLM / CLIP` 的输入先反归一化回图像空间，再提医学形态特征，避免在 CLIP normalized 空间里误算颜色和强度
- 候选选择新增 `balanced_acc`，让医学融合不只追总体准确率，也关注类别不平衡任务的少数类表现

结果文件：

- [outputs/my_merge_med_v5_domain_smoke_derma/reports/eval_summary.csv](/data/liyapeng_grp/program/MedMNISTMerge/outputs/my_merge_med_v5_domain_smoke_derma/reports/eval_summary.csv)
- [outputs/my_merge_med_v5_domain_smoke_organc/reports/eval_summary.csv](/data/liyapeng_grp/program/MedMNISTMerge/outputs/my_merge_med_v5_domain_smoke_organc/reports/eval_summary.csv)
- [outputs/my_merge_med_v5_domain_smoke_chao/reports/eval_summary.csv](/data/liyapeng_grp/program/MedMNISTMerge/outputs/my_merge_med_v5_domain_smoke_chao/reports/eval_summary.csv)

smoke 结果：

- `resnet / dermamnist_224 / c3_b0 = 0.6783`
- `resnet / organcmnist_224 / c3_b0 = 0.5566`
- `resnet / chaoshengmnist_224 / c3_b0 = 0.3576`

结论：

- 这版更明确地把融合方法限制在医学影像域：皮肤镜、CT、超声分别使用不同物理/形态学证据
- 该改动仍然是模型融合方法，没有重新训练 client；医学信号用于选择专家、加权层融合、构建分类头和选择候选融合结果
- smoke 已证明三类新增医学机制都能跑通，完整表现需要等待全量刷新脚本跑完后再看总表
