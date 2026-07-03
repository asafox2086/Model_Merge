# 实验四：公开验证集选择最强客户端

## 工作流核对

固定已训练客户端 checkpoint，不做权重融合。对每个 setting，先用 public validation set 给所有客户端打分，选择分数最高的客户端，再在源域 test set 上评估被选中的客户端。

## 选择规则摘要

| selection rule | cases | mean source-test acc | mean source-test bacc | mean source-test macro F1 | mean selected public acc | mean selected source-val acc |
|---|---:|---:|---:|---:|---:|---:|
| oracle_test_accuracy_best | 6 | 0.4978 | 0.2841 | 0.2108 | 0.2015 | 0.4935 |
| public_val_accuracy_best | 6 | 0.3395 | 0.2748 | 0.1670 | 0.2415 | 0.3316 |
| public_val_balanced_accuracy_best | 6 | 0.3395 | 0.2748 | 0.1670 | 0.2415 | 0.3316 |
| source_val_accuracy_best | 6 | 0.4978 | 0.2841 | 0.2108 | 0.2015 | 0.4935 |
| source_val_balanced_accuracy_best | 6 | 0.4670 | 0.3017 | 0.2224 | 0.2174 | 0.4598 |

## Public 选择相对 Source-Val 选择的 regret

- cases: 6
- mean regret: +0.1583
- max regret: +0.6294
- public 与 source-val 选中同一客户端: 3/6
- public 与 oracle-test 选中同一客户端: 3/6

## 排名相关性

| dataset | model | clients | beta | public acc vs test acc | source-val acc vs test acc |
|---|---|---:|---:|---:|---:|
| bloodmnist_224 | resnet | 3 | 0 | 1.0000 | 1.0000 |
| bloodmnist_224 | resnet | 3 | 0.01 | -1.0000 | 1.0000 |
| bloodmnist_224 | resnet | 3 | 0.1 | 0.5000 | 1.0000 |
| dermamnist_224 | resnet | 3 | 0 | 0.5000 | 1.0000 |
| dermamnist_224 | resnet | 3 | 0.01 | -0.5000 | 1.0000 |
| dermamnist_224 | resnet | 3 | 0.1 | -0.5000 | 1.0000 |

## 结论读取方式

如果 `public_val_accuracy_best` 的 source-test accuracy 接近 `source_val_accuracy_best`，说明公开数据可以作为客户端选择代理；如果 regret 明显为正，说明 public 排名不能稳定替代源域验证集。
