# 医学模型融合经验研究结果汇总

## 工作流完成状态

本轮工作严格按 `experiment_workflow.md` 执行，且每完成一组实验后回看了对应工作流章节。

- 实验一：已有融合方法的大规模基线测试与逐样本塌缩诊断，已完成。
- 实验二：客户端能力与融合损失分析，已完成。
- 实验三：同类型公开数据替代源域验证集实验，已完成。
- BN recalibration 子实验，已完成。

所有新增脚本和结果均放在 `medmerge_empirical_study/`，没有混入原始 `program/MedMNISTMerge` 结果目录。

## 实验一结论

正式 accuracy 大表覆盖：

- 7 个医学数据集
- 12 个已有融合方法
- 567 个 setting
- 6804 条 accuracy 记录

正式表显示，没有单一方法跨 setting 稳定领先。按 mean accuracy 排序，FROM、TIES、Fisher、Avg 较靠前，但多个方法 median accuracy 很低，且 low-performance rate 较高。

逐样本诊断子集覆盖 DermaMNIST / ResNet / 3 clients / beta = 0, 0.01, 0.1，方法包括 Avg、TIES、Fisher、RegMean、FROM，共 15 条记录，全部成功。

关键发现：

- DermaMNIST test set 多数类先验为 0.6688。
- 多个方法 accuracy 接近 0.67，但 balanced accuracy 只有约 0.14-0.22。
- Avg、Fisher、FROM 在 beta=0 时几乎全部预测同一类，`majority_prediction_ratio` 接近或等于 1.0。
- TIES / RegMean 有时 balanced accuracy 更高，但 accuracy 排名不一定更高。

支撑结论：

> 医学模型融合不能只看 accuracy。类别不均衡数据集上，高 accuracy 可能只是多数类塌缩。

相关文件：

- `results/experiment1/experiment1_formal_accuracy_summary.md`
- `results/experiment1/experiment1_prediction_diagnostic.md`
- `results/experiment1/prediction_metrics.csv`

## 实验二结论

实验二覆盖 DermaMNIST / ResNet / 3 clients / beta = 0, 0.01, 0.1。

评估对象：

- 每个 single client
- Best single client
- Mean single client
- Prob ensemble
- Logit ensemble
- Oracle any-correct client
- Oracle min-loss client

关键结果：

| baseline | mean acc | mean balanced acc | mean macro F1 |
|---|---:|---:|---:|
| best_single_client | 0.6323 | 0.2525 | 0.1859 |
| prob_ensemble | 0.5328 | 0.2524 | 0.1923 |
| oracle_any_correct | 0.7505 | 0.5834 | 0.4325 |

融合模型相对客户端池上界的平均 gap：

| comparison | acc gap | balanced acc gap | macro F1 gap |
|---|---:|---:|---:|
| merged - best_single_client | +0.0018 | -0.0770 | -0.0367 |
| merged - prob_ensemble | +0.1012 | -0.0769 | -0.0430 |
| merged - oracle_any_correct | -0.1164 | -0.4079 | -0.2833 |

支撑结论：

> 客户端池中存在可用知识，融合失败不能简单归因于客户端都弱。更准确的说法是，权重空间融合没有稳定恢复客户端池中的互补知识。

相关文件：

- `results/experiment2/client_pool_metrics.csv`
- `results/experiment2/merged_vs_client_pool_gaps.csv`
- `results/experiment2/experiment2_client_pool_summary.md`

## 实验三设置

实验三覆盖：

- 数据集：BloodMNIST、DermaMNIST
- Backbone：ResNet
- 客户端数量：3
- beta：0, 0.01, 0.1
- 方法：Fisher、RegMean、AdaMerging、BN recalibration
- 记录数：84 / 84 成功

公开数据不是弱对照，而是使用仓库里已经准备好的高质量同类型公开数据：

- BloodMNIST public：`Falah/Blood_8_classes_Dataset`，8 类一一映射，去重后 1024 张。
- DermaMNIST public：`flwrlabs/fed-isic2019`，映射到 DermaMNIST 7 类，丢弃不能映射的 squamous cell carcinoma，去重后 1024 张。

校准来源：

| source | 含义 |
|---|---|
| `source_val_512` | 原始源域验证集抽取 512 张 |
| `public_labeled_val_512` | 同类型公开医学数据 512 张 |
| `public_labeled_reweighted_512` | 公开数据按源域验证集类别比例重采样 |
| `no_image_stats` / `no_recalibration` | 不使用图像统计 |

测试集始终固定为原始 `Med_data` 的 test split；checkpoint、merge 超参数和样本数量固定。

## 实验三结论

整体平均结果：

| method | source | mean acc | mean balanced acc | mean macro F1 |
|---|---|---:|---:|---:|
| BN recalibration | source_val_512 | 0.6330 | 0.3719 | 0.3481 |
| BN recalibration | public_labeled_val_512 | 0.5061 | 0.2366 | 0.1872 |
| BN recalibration | public_labeled_reweighted_512 | 0.5114 | 0.2480 | 0.1941 |
| Fisher | source_val_512 | 0.5015 | 0.2421 | 0.1878 |
| Fisher | public_labeled_val_512 | 0.4944 | 0.2871 | 0.2290 |
| RegMean | source_val_512 | 0.4518 | 0.2586 | 0.2008 |
| RegMean | public_labeled_val_512 | 0.4468 | 0.2861 | 0.2234 |
| AdaMerging | source_val_512 | 0.4869 | 0.2299 | 0.1653 |
| AdaMerging | public_labeled_val_512 | 0.4650 | 0.2097 | 0.1481 |

最强证据来自 BN recalibration：

- source validation 在 6/6 个 setting 上的 balanced accuracy 都高于 public 和 public-reweighted。
- source validation 在 6/6 个 setting 上的 macro F1 都高于 public 和 public-reweighted。
- 平均 balanced accuracy：source 0.3719，public 0.2366，public-reweighted 0.2480。
- 平均 macro F1：source 0.3481，public 0.1872，public-reweighted 0.1941。

BloodMNIST 上 BN recalibration 差异尤其明显：

- source balanced accuracy：0.5540
- public balanced accuracy：0.3284
- public-reweighted balanced accuracy：0.3454

DermaMNIST 上 BN recalibration 仍然 source 更高：

- source balanced accuracy：0.1897
- public balanced accuracy：0.1447
- public-reweighted balanced accuracy：0.1506

Fisher、RegMean、AdaMerging 的结果更复杂：

- Public data 并不总是更差；在 Fisher / RegMean 的 balanced accuracy 上，public 有时高于 source。
- 这说明公开同类数据确实有信息，不是一个低质量 negative control。
- 但 public 的收益不稳定，且不同方法、不同指标上方向不一致。
- 因此不能把 public data 当成可靠、可预期的源域统计替代。

支撑结论应写成：

> Reliable medical model merging requires source-distribution-matched statistics, especially for restoring model state such as BatchNorm running statistics. High-quality same-task public medical data can provide useful signals, but it is not a stable drop-in replacement for source-domain validation images.

不建议写成：

> 所有统计型融合方法在所有 setting 上都必须使用原图，否则一定更差。

相关文件：

- `results/experiment3/public_stats_substitution_metrics.csv`
- `results/experiment3/experiment3_public_stats_summary.md`
- `results/experiment3/calibration_roots/manifest.json`

## 论文故事线

本文不讲 MyMerge，也不把失败方法包装成贡献。主线应是 empirical study：

1. 现有后训练融合方法在医学异质客户端上不稳定。
2. Accuracy 会严重掩盖类别塌缩，必须报告 balanced accuracy、macro F1、per-class recall 和预测分布。
3. 客户端池中存在可用知识，oracle 和 prediction ensemble 证明问题不是所有客户端都弱。
4. 同类型公开医学数据不是无效数据，但它不能稳定替代源域分布匹配的图像统计。
5. BN recalibration 是最清晰证据：融合后模型状态统计明显依赖源域图像。

推荐摘要句：

> Our empirical study shows that post-hoc medical model merging is not merely a checkpoint-level problem. In heterogeneous medical clients, reliable merging requires source-distribution-matched image statistics. Same-task public medical data, even when carefully mapped, cleaned, and reweighted, is not a stable substitute for the original source-domain validation images.

中文对应：

> 医学模型融合不只是权重合并问题。在异质医学客户端中，可靠融合需要与源域分布匹配的图像统计。即使公开医学数据属于相同任务类型，并经过类别映射、去重和比例重采样，也不能稳定替代原始源域验证图像。

## 当前限制

- 实验一正式 accuracy 覆盖较广，但逐样本预测级指标目前只补了 DermaMNIST / ResNet / c3 子集。
- 实验二目前覆盖 DermaMNIST / ResNet / c3 / 三个 beta。
- 实验三覆盖 BloodMNIST 与 DermaMNIST 的 ResNet / c3 / 三个 beta；后续可扩展到 ConvNeXt 或更多 client 数。
- Public-reweighted 在部分类别上使用 replacement，因为公开集每类样本数有限；这已记录在 calibration manifest 中。
- Fisher / RegMean 的结果不能支持“source 在所有方法上恒优”的强结论，论文应如实表述为“不稳定替代”和“源域统计对模型状态恢复尤其关键”。
