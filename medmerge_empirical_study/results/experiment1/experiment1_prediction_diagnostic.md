# 实验一：逐样本预测级诊断

## 工作流核对

对应 `experiment_workflow.md` 第 4 节。本部分补齐正式 accuracy 表缺少的逐样本指标：

- Accuracy
- Balanced accuracy
- Macro F1
- Test loss / NLL
- Per-class recall
- 预测类别分布
- `majority_prediction_ratio`
- `effective_predicted_classes`

正式结果表已经覆盖 7 个医学数据集、12 个融合方法、567 个 setting；本诊断子集选择 DermaMNIST / ResNet / 3 clients / beta = 0, 0.01, 0.1，原因是 DermaMNIST 类别不均衡明显，最适合检验 accuracy 是否掩盖类别塌缩。

## 运行范围

- 数据集：DermaMNIST
- Backbone：ResNet
- 客户端数量：3
- Dirichlet beta：0, 0.01, 0.1
- 方法：Avg, TIES, Fisher, RegMean, FROM
- 评估 split：test
- 成功记录：15 / 15

原始结果：

- `prediction_metrics.csv`
- `prediction_metrics_summary.md`

## 方法级摘要

| method | cases | mean acc | mean balanced acc | mean macro F1 | mean collapse ratio | median effective classes |
|---|---:|---:|---:|---:|---:|---:|
| avg | 3 | 0.6291 | 0.1737 | 0.1363 | 0.8926 | 1.14 |
| fisher | 3 | 0.6687 | 0.1613 | 0.1441 | 0.9716 | 1.17 |
| from | 3 | 0.6713 | 0.1526 | 0.1311 | 0.9909 | 1.03 |
| regmean | 3 | 0.6018 | 0.1891 | 0.1628 | 0.7930 | 1.43 |
| ties | 3 | 0.5995 | 0.2009 | 0.1719 | 0.8015 | 1.82 |

## 关键观察

1. 单看 accuracy 会产生误导。DermaMNIST test set 多数类先验是 0.6688，因此 Avg、Fisher、FROM 在若干 setting 中达到约 0.67 accuracy，并不代表模型学会了多类别识别。
2. Balanced accuracy 和 Macro F1 暴露了类别塌缩。多个高 accuracy 结果的 balanced accuracy 只有 0.1429 左右，接近 7 类任务中只预测一个类别时的水平。
3. 预测类别分布显示真实塌缩。Avg 在 beta=0 时 `majority_prediction_ratio = 1.0`，`effective_predicted_classes = 1.0`，即 2005 个 test 样本全部预测为同一类。
4. 不同融合方法没有稳定优势。TIES 的平均 balanced accuracy 较高，但 accuracy 不最高；FROM 的平均 accuracy 最高，但 collapse ratio 也最高。这说明 accuracy 排名和稳健识别能力并不一致。
5. 图像统计型方法也不是天然稳定。Fisher 在该子集的 mean accuracy 高，但 mean balanced accuracy 只有 0.1613，说明仅加入某种统计型融合机制仍不能保证可靠多类别预测。

## 可写入论文的结论

在类别不均衡医学任务上，后训练模型融合方法可能通过多数类预测取得看似可接受的 accuracy。逐样本诊断显示，这些结果往往伴随极低 balanced accuracy、macro F1 和有效预测类别数。因此，医学模型融合的评价不能只依赖 accuracy；需要报告类别敏感指标和预测塌缩指标。

该实验支撑工作流第 4.6 节结论：

> 现有权重空间融合方法直接用于医学客户端模型时，并不可靠。医学模型融合不能只用 accuracy 或单一 setting 判断有效性。
