# MedMNISTMerge 用户与交接文档

这个目录与 `program` 同级，用来存放给用户、论文写作和交接人看的材料，不再和程序输出目录混在一起。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `summary_table.md` | 当前汇总表副本，来源于仓库里的 `My_merge_ret/汇总表.md`。 |
| `handoff_guide.md` | 详细交接指南，包含用户要求复述、代码状态、已跑结果和下一步建议。 |
| `method_design.md` | 方法设计说明，讲清楚 `my_merge` 是什么、为什么不是联邦学习、为什么不能直接迁移到 NLP。 |
| `current_progress.md` | 当前进度说明，方便接手人快速判断哪些已完成、哪些还不稳定。 |
| `presentation/method_logic.pptx` | 方法逻辑 PPT 副本。 |
| `presentation/speech_draft.md` | 演讲稿副本。 |
| `reference_reports/` | 重要中间报告副本，只用于阅读参考。 |

## 当前总判断

`my_merge` 的方向已经明确：它应当是医学图像 checkpoint 模型融合方法，而不是联邦学习。当前代码正在收敛到两个模块：

- M1：用验证集估计医学图像相关的客户端信息。
- M2：用验证集选择保守的 checkpoint 融合候选。

但当前主代码仍未达到“可以直接放心跑全量”的状态。超声数据集已经有一些 smoke 结果较好，但另一些 probe 暴露出候选触发逻辑不稳定。因此下一步应先修稳 M2，再小跑超声，不应直接大规模全量运行。
