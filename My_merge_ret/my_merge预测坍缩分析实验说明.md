# LAMP-Merge 预测坍缩分析实验说明

## 1. 实验目的

本实验用于证明 `LAMP-Merge` 的核心现象和方法动机：常规模型融合在医学图像任务中容易出现预测坍缩，而 `LAMP-Merge` 通过类别级医学原型恢复多类别诊断输出。

这里的“预测坍缩”不是单纯准确率下降，而是融合模型把大量样本预测到一个或少数几个类别。医学分类任务通常有明确诊断类别，若融合后模型实际只使用少数类别，即使 accuracy 在强类别不平衡数据集上不低，也不能说明它具备稳定诊断能力。

## 2. 覆盖范围

本次诊断覆盖正式小表中的 5 个医学数据集：

```text
bloodmnist_224
chaoshengmnist_224
dermamnist_224
organcmnist_224
organsmnist_224
```

每个数据集覆盖 4 个 backbone：

```text
resnet
convnext
vit_t
swin_tiny
```

固定实验设置：

```text
clients = 3
beta = 0.01
seed = 42
split = test
```

统计对象包括：

```text
client_mean
client_best
avg
ties
dare_linear
dare_ties
regmean
fisher
breadcrumbs
model_stock
from
iso_c
free_merge
robustmerge
my_merge
```

因此，本轮不是单点案例分析，而是 5 个数据集分开统计、每个数据集对 4 个 backbone 取平均。

## 3. 分析脚本

新增脚本：

```text
scripts/collect_prediction_diagnostics.py
```

该脚本会重新评估 individual clients 和各融合方法，统计预测分布、balanced accuracy、macro F1、collapse ratio 等指标，并为每个数据集生成两类图：

1. 指标柱状图：accuracy、balanced accuracy、macro F1、collapse ratio、effective predicted classes、pred-true TV。
2. 预测类别分布热图：显示 true test distribution、client、baseline 和 `LAMP-Merge` 的预测类别比例。

本次正式诊断的命令为：

```bash
python scripts/collect_prediction_diagnostics.py \
  --datasets bloodmnist_224 dermamnist_224 organcmnist_224 organsmnist_224 chaoshengmnist_224 \
  --small-models resnet convnext vit_t swin_tiny \
  --num-clients 3 \
  --betas 0.01 \
  --methods avg ties dare_linear dare_ties regmean fisher breadcrumbs model_stock from iso_c free_merge robustmerge LAMP-Merge \
  --metrics-csv My_merge_ret/reports/prediction_diagnostics_4models_c3_b001.csv \
  --summary-dir My_merge_ret/reports/prediction_diagnostics_4models_c3_b001 \
  --figure-dir My_merge_ret/figures/prediction_diagnostics_4models_c3_b001 \
  --output-root outputs/prediction_diagnostics_4models_c3_b001 \
  --device cuda:0 \
  --batch-size 256 \
  --num-workers 4
```

实际运行时为了加速，按数据集拆分到多张 GPU 并行执行；最终 clean 结果已合并。

## 4. 指标定义

设测试集有 $C$ 个类别，模型 hard prediction 的类别计数为：

```text
n_pred = [n_1, n_2, ..., n_C]
```

预测分布为：

```text
p_pred,c = n_c / sum_j n_j
```

### 4.1 collapse ratio

```text
collapse_ratio = max_c p_pred,c
```

该指标表示测试样本中被预测到最多类别的比例。数值越大，说明模型越接近单类预测器。

### 4.2 balanced accuracy

```text
balanced_accuracy = mean_c recall_c
recall_c = TP_c / (TP_c + FN_c)
```

该指标对每个真实类别的召回率取平均，可以避免强多数类数据集上的 accuracy 虚高。

### 4.3 macro F1

```text
macro_f1 = mean_c F1_c
F1_c = 2 * precision_c * recall_c / (precision_c + recall_c)
```

该指标衡量模型是否同时保留多个诊断类别，而不是只在多数类上表现好。

### 4.4 effective predicted classes

```text
effective_predicted_classes = exp(H(p_pred))
H(p_pred) = - sum_c p_pred,c log p_pred,c
```

该指标可以理解为模型实际使用了多少个预测类别。若接近 1，说明模型几乎只预测一个类别。

### 4.5 pred-true TV

令测试集真实标签分布为 $p_true$，则：

```text
pred_true_tv = 0.5 * sum_c |p_pred,c - p_true,c|
```

该指标衡量预测类别分布与真实测试类别分布的距离。数值越低，说明输出类别比例越合理。

## 5. 输出文件

总报告：

```text
My_merge_ret/reports/prediction_diagnostics_4models_c3_b001/prediction_diagnostics_all_datasets_summary.md
```

clean CSV：

```text
My_merge_ret/reports/prediction_diagnostics_4models_c3_b001/prediction_diagnostics_all_datasets_clean.csv
My_merge_ret/reports/prediction_diagnostics_4models_c3_b001/prediction_diagnostics_with_client_aggregates_clean.csv
My_merge_ret/reports/prediction_diagnostics_4models_c3_b001/prediction_diagnostics_dataset_summary_clean.csv
My_merge_ret/reports/prediction_diagnostics_4models_c3_b001/prediction_diagnostics_overall_summary_clean.csv
My_merge_ret/reports/prediction_diagnostics_4models_c3_b001/prediction_diagnostics_wins_clean.csv
```

按数据集分开的图：

```text
My_merge_ret/figures/prediction_diagnostics_4models_c3_b001/by_dataset/<dataset>/<dataset>_prediction_diagnostics.png
My_merge_ret/figures/prediction_diagnostics_4models_c3_b001/by_dataset/<dataset>/<dataset>_prediction_distribution.png
```

补跑前的 OOM 失败行已归档在：

```text
My_merge_ret/reports/prediction_diagnostics_4models_c3_b001/prediction_diagnostics_failures_archived.csv
```

这些失败行不参与最终统计；对应的 `fisher` 结果已用小 batch 补跑成功。

## 6. 总体结果

20 个 case = 5 个数据集 × 4 个 backbone。总体均值如下：

| method | cases | acc | balanced acc | macro F1 | collapse ratio | effective classes | pred-true TV |
|---|---:|---:|---:|---:|---:|---:|---:|
| client_mean | 20 | 0.1804 | 0.1515 | 0.0638 | 0.8207 | 1.7370 | 0.7644 |
| client_best | 20 | 0.3156 | 0.1810 | 0.1123 | 0.7619 | 2.0201 | 0.6168 |
| avg | 20 | 0.2304 | 0.1543 | 0.0751 | 0.8028 | 1.7666 | 0.7082 |
| fisher | 20 | 0.2585 | 0.1604 | 0.0886 | 0.8228 | 1.9324 | 0.6671 |
| robustmerge | 20 | 0.1486 | 0.1425 | 0.0502 | 0.8225 | 1.6559 | 0.8035 |
| LAMP-Merge | 20 | 0.5885 | 0.5747 | 0.5352 | 0.2294 | 8.0390 | 0.1510 |

`LAMP-Merge` 的单 case 最优或并列最优次数：

| metric | LAMP-Merge best/tied cases |
|---|---:|
| accuracy | 16/20 |
| balanced_accuracy | 20/20 |
| macro_f1 | 20/20 |
| collapse_ratio | 20/20 |
| pred_true_tv | 18/20 |

这说明 `LAMP-Merge` 不是只在某一个数据集或某一个 backbone 上缓解坍缩，而是在全 5 个医学数据集上稳定降低 collapse ratio，并显著提高 balanced accuracy 与 macro F1。

## 7. 按数据集结论

### bloodmnist_224

`LAMP-Merge` 的平均 accuracy 为 0.8190，balanced accuracy 为 0.8071，macro F1 为 0.8022，collapse ratio 为 0.1943。其他方法的 collapse ratio 普遍在 0.78 到 0.93 附近，说明常规融合在血细胞多分类任务上严重集中到少数类别。

图：

```text
My_merge_ret/figures/prediction_diagnostics_4models_c3_b001/by_dataset/bloodmnist_224/bloodmnist_224_prediction_diagnostics.png
My_merge_ret/figures/prediction_diagnostics_4models_c3_b001/by_dataset/bloodmnist_224/bloodmnist_224_prediction_distribution.png
```

### chaoshengmnist_224

`LAMP-Merge` 的平均 accuracy 为 0.4625，balanced accuracy 为 0.4536，macro F1 为 0.4378，collapse ratio 为 0.2035。client 和常规融合方法普遍 collapse ratio 很高，说明超声任务中也存在明显预测坍缩。

图：

```text
My_merge_ret/figures/prediction_diagnostics_4models_c3_b001/by_dataset/chaoshengmnist_224/chaoshengmnist_224_prediction_diagnostics.png
My_merge_ret/figures/prediction_diagnostics_4models_c3_b001/by_dataset/chaoshengmnist_224/chaoshengmnist_224_prediction_distribution.png
```

### dermamnist_224

`LAMP-Merge` 的 average accuracy 为 0.4627，低于 `client_best` 的 0.6726；但 `LAMP-Merge` 的 balanced accuracy 为 0.4492，macro F1 为 0.2934，均为最高。`client_best`、`fisher` 和部分融合方法在 DermaMNIST 上的高 accuracy 主要来自多数类预测，其 collapse ratio 接近 1。

因此 DermaMNIST 说明：accuracy 可能被类别不平衡放大，balanced accuracy 和 macro F1 更能体现医学多类别诊断能力。

图：

```text
My_merge_ret/figures/prediction_diagnostics_4models_c3_b001/by_dataset/dermamnist_224/dermamnist_224_prediction_diagnostics.png
My_merge_ret/figures/prediction_diagnostics_4models_c3_b001/by_dataset/dermamnist_224/dermamnist_224_prediction_distribution.png
```

### organcmnist_224

`LAMP-Merge` 的 average accuracy 为 0.6249，balanced accuracy 为 0.6175，macro F1 为 0.6030，collapse ratio 为 0.1772。相比其他方法 0.59 到 0.90 的 collapse ratio，`LAMP-Merge` 明显恢复了多器官类别输出。

图：

```text
My_merge_ret/figures/prediction_diagnostics_4models_c3_b001/by_dataset/organcmnist_224/organcmnist_224_prediction_diagnostics.png
My_merge_ret/figures/prediction_diagnostics_4models_c3_b001/by_dataset/organcmnist_224/organcmnist_224_prediction_distribution.png
```

### organsmnist_224

`LAMP-Merge` 的 average accuracy 为 0.5734，balanced accuracy 为 0.5460，macro F1 为 0.5397，collapse ratio 为 0.2021。其他方法的 effective predicted classes 普遍只有 1.67 到 2.93，而 `LAMP-Merge` 达到 9.5033。

图：

```text
My_merge_ret/figures/prediction_diagnostics_4models_c3_b001/by_dataset/organsmnist_224/organsmnist_224_prediction_diagnostics.png
My_merge_ret/figures/prediction_diagnostics_4models_c3_b001/by_dataset/organsmnist_224/organsmnist_224_prediction_distribution.png
```

## 8. 论文解释

本实验支持以下论文叙事：

1. 常规模型融合与单个 client 在医学 non-IID 分类任务中容易退化为少数类预测器。
2. 单看 accuracy 会掩盖这种问题，尤其在 DermaMNIST 这类强类别不平衡任务上，多数类坍缩可以产生看似较高的 accuracy。
3. `LAMP-Merge` 显著降低 collapse ratio，同时提高 balanced accuracy 和 macro F1，说明它恢复的是多类别医学诊断能力，而不是简单选择多数类。
4. 预测类别分布热图提供了可视化证据：baseline 的预测分布通常集中在一列或少数列，`LAMP-Merge` 的分布更接近真实测试类别分布。
