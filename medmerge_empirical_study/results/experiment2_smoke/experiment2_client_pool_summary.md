# 实验二：客户端能力与融合损失分析

## 工作流核对

对应 `experiment_workflow.md` 第 5 节：评估 single client、best single client、mean single client、prediction ensemble 和 oracle per-sample client。

## Baseline 摘要

| baseline | cases | mean acc | mean balanced acc | mean macro F1 | mean collapse ratio |
|---|---:|---:|---:|---:|---:|
| best_single_client | 1 | 0.5771 | 0.2100 | 0.1227 | 0.7885 |
| logit_ensemble | 1 | 0.4554 | 0.2174 | 0.1454 | 0.5681 |
| mean_single_client | 1 | 0.2216 | 0.1765 | 0.0600 | 0.7734 |
| oracle_any_correct | 1 | 0.6648 | 0.5294 | 0.3185 | 0.5731 |
| oracle_min_loss | 1 | 0.6648 | 0.5294 | 0.3185 | 0.5731 |
| prob_ensemble | 1 | 0.4888 | 0.2475 | 0.1577 | 0.6185 |
| single_client_0 | 1 | 0.0214 | 0.1592 | 0.0265 | 0.8953 |
| single_client_1 | 1 | 0.0663 | 0.1601 | 0.0307 | 0.6364 |
| single_client_2 | 1 | 0.5771 | 0.2100 | 0.1227 | 0.7885 |

## 融合模型相对客户端池上界的 Gap

- merged - best_single_client: mean acc gap +0.0645, mean balanced acc gap -0.0447, mean macro F1 gap +0.0146
- merged - prob_ensemble: mean acc gap +0.1528, mean balanced acc gap -0.0822, mean macro F1 gap -0.0204
- merged - logit_ensemble: mean acc gap +0.1862, mean balanced acc gap -0.0521, mean macro F1 gap -0.0080
- merged - oracle_any_correct: mean acc gap -0.0232, mean balanced acc gap -0.3641, mean macro F1 gap -0.1812
- merged - oracle_min_loss: mean acc gap -0.0232, mean balanced acc gap -0.3641, mean macro F1 gap -0.1812

## 结论

如果 oracle 或 prediction ensemble 明显高于权重融合，说明客户端池中存在可利用信息；融合失败不能简单归因于所有客户端都弱，而应归因于权重空间融合在医学异质客户端中无法稳定恢复这些信息。
