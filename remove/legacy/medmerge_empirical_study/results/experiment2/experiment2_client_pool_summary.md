# 实验二：客户端能力与融合损失分析

## 工作流核对

对应 `experiment_workflow.md` 第 5 节：评估 single client、best single client、mean single client、prediction ensemble 和 oracle per-sample client。

## Baseline 摘要

| baseline | cases | mean acc | mean balanced acc | mean macro F1 | mean collapse ratio |
|---|---:|---:|---:|---:|---:|
| best_single_client | 3 | 0.6323 | 0.2525 | 0.1859 | 0.8160 |
| logit_ensemble | 3 | 0.4658 | 0.2328 | 0.1698 | 0.6444 |
| mean_single_client | 3 | 0.2966 | 0.2170 | 0.1007 | 0.7499 |
| oracle_any_correct | 3 | 0.7505 | 0.5834 | 0.4325 | 0.6258 |
| oracle_min_loss | 3 | 0.7498 | 0.5792 | 0.4316 | 0.6256 |
| prob_ensemble | 3 | 0.5328 | 0.2524 | 0.1923 | 0.6195 |
| single_client_0 | 3 | 0.1907 | 0.2220 | 0.0861 | 0.6485 |
| single_client_1 | 3 | 0.4620 | 0.2359 | 0.1553 | 0.7653 |
| single_client_2 | 3 | 0.2371 | 0.1931 | 0.0606 | 0.8359 |

## 融合模型相对客户端池上界的 Gap

- merged - best_single_client: mean acc gap +0.0018, mean balanced acc gap -0.0770, mean macro F1 gap -0.0367
- merged - prob_ensemble: mean acc gap +0.1012, mean balanced acc gap -0.0769, mean macro F1 gap -0.0430
- merged - logit_ensemble: mean acc gap +0.1682, mean balanced acc gap -0.0573, mean macro F1 gap -0.0206
- merged - oracle_any_correct: mean acc gap -0.1164, mean balanced acc gap -0.4079, mean macro F1 gap -0.2833
- merged - oracle_min_loss: mean acc gap -0.1157, mean balanced acc gap -0.4037, mean macro F1 gap -0.2823

## 结论

如果 oracle 或 prediction ensemble 明显高于权重融合，说明客户端池中存在可利用信息；融合失败不能简单归因于所有客户端都弱，而应归因于权重空间融合在医学异质客户端中无法稳定恢复这些信息。
