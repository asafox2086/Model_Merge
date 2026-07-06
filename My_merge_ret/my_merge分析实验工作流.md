# LAMP-Merge 分析实验工作流

本文档固定 `LAMP-Merge` 的论文分析实验流程。当前重点不是继续调参，而是系统证明三个命题：

1. 常规模型融合在医学图像任务上存在预测坍缩。
2. `LAMP-Merge` 能显著降低坍缩，并恢复多类别诊断能力。
3. 在强类别不平衡数据集上，accuracy 可能被多数类坍缩虚高，因此必须同时报告 balanced accuracy、macro F1 和预测分布指标。

## 1. 标准诊断脚本

统一使用：

```text
scripts/collect_prediction_diagnostics.py
```

该脚本负责三件事：

1. 评估 individual clients、通用模型融合 baseline 和 `LAMP-Merge`。
2. 统计 accuracy、balanced accuracy、macro F1、collapse ratio、effective predicted classes、pred-true TV 等指标。
3. 为每个数据集生成指标柱状图和预测类别分布热图。

## 2. 标准运行设置

正式小表诊断使用 5 个医学数据集：

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

默认诊断格子：

```text
clients = 3
beta = 0.01
seed = 42
split = test
```

该设置对应 20 个 case。每个 case 评估 13 个融合方法和 3 个 individual clients，因此 clean raw 结果为 320 条 OK 评估。

## 3. 推荐命令

串行命令如下：

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

实际运行时可以按数据集拆分到多张 GPU。每个数据集写独立 CSV、独立 summary 和独立 figure directory，最后再合并 clean CSV。

## 4. 指标解释

### collapse ratio

```text
collapse_ratio = max_c p_pred,c
```

其中 `p_pred,c` 是模型预测为类别 `c` 的样本比例。该指标越高，说明模型越接近单类坍缩。

### balanced accuracy

```text
balanced_accuracy = mean_c recall_c
```

该指标对每个真实类别的 recall 取平均，避免强多数类任务上 accuracy 虚高。

### macro F1

```text
macro_f1 = mean_c F1_c
```

该指标衡量模型是否保留多类别诊断能力。

### effective predicted classes

```text
effective_predicted_classes = exp(H(p_pred))
```

若该值接近 1，说明模型几乎只使用一个预测类别。

### pred-true TV

```text
pred_true_tv = 0.5 * sum_c |p_pred,c - p_true,c|
```

该指标衡量预测类别分布与测试集真实类别分布的距离。

## 5. 当前已完成结果

当前已完成 5 个数据集 × 4 个 backbone 的诊断，结果位于：

```text
My_merge_ret/reports/prediction_diagnostics_4models_c3_b001/prediction_diagnostics_all_datasets_summary.md
```

clean 数据文件：

```text
My_merge_ret/reports/prediction_diagnostics_4models_c3_b001/prediction_diagnostics_all_datasets_clean.csv
My_merge_ret/reports/prediction_diagnostics_4models_c3_b001/prediction_diagnostics_with_client_aggregates_clean.csv
My_merge_ret/reports/prediction_diagnostics_4models_c3_b001/prediction_diagnostics_dataset_summary_clean.csv
My_merge_ret/reports/prediction_diagnostics_4models_c3_b001/prediction_diagnostics_overall_summary_clean.csv
My_merge_ret/reports/prediction_diagnostics_4models_c3_b001/prediction_diagnostics_wins_clean.csv
```

按数据集分开的图位于：

```text
My_merge_ret/figures/prediction_diagnostics_4models_c3_b001/by_dataset/
```

每个数据集有两张图：

```text
<dataset>_prediction_diagnostics.png
<dataset>_prediction_distribution.png
```

## 6. 当前关键结论

在 20 个 case 上，`LAMP-Merge` 的单 case 最优或并列最优次数为：

| metric | LAMP-Merge best/tied cases |
|---|---:|
| accuracy | 16/20 |
| balanced_accuracy | 20/20 |
| macro_f1 | 20/20 |
| collapse_ratio | 20/20 |
| pred_true_tv | 18/20 |

总体均值：

| method | cases | acc | balanced acc | macro F1 | collapse ratio | effective classes | pred-true TV |
|---|---:|---:|---:|---:|---:|---:|---:|
| client_mean | 20 | 0.1804 | 0.1515 | 0.0638 | 0.8207 | 1.7370 | 0.7644 |
| client_best | 20 | 0.3156 | 0.1810 | 0.1123 | 0.7619 | 2.0201 | 0.6168 |
| avg | 20 | 0.2304 | 0.1543 | 0.0751 | 0.8028 | 1.7666 | 0.7082 |
| fisher | 20 | 0.2585 | 0.1604 | 0.0886 | 0.8228 | 1.9324 | 0.6671 |
| LAMP-Merge | 20 | 0.5885 | 0.5747 | 0.5352 | 0.2294 | 8.0390 | 0.1510 |

结论：

1. client 和常规融合方法的 collapse ratio 普遍很高，说明模型经常只使用少数类别。
2. `LAMP-Merge` 的 collapse ratio 显著更低，effective predicted classes 显著更高。
3. `LAMP-Merge` 在 balanced accuracy 与 macro F1 上 20/20 最优或并列最优，说明它恢复的是多类别医学诊断能力。
4. DermaMNIST 上 `client_best` 的 accuracy 更高，但 collapse ratio 接近 1，balanced accuracy 和 macro F1 明显低于 `LAMP-Merge`。这证明单看 accuracy 会掩盖多数类坍缩。

## 7. 论文中应展示的图表

主文建议展示：

1. 5 个数据集的 `collapse_ratio` 对比柱状图。
2. 5 个数据集的 `balanced_accuracy` 和 `macro_f1` 对比柱状图。
3. 代表性数据集的预测类别分布热图，例如 BloodMNIST、DermaMNIST、OrganCMNIST。

附录建议完整放入：

1. `prediction_diagnostics_all_datasets_summary.md` 中的每个数据集表格。
2. 每个数据集的两张图。
3. clean raw CSV 的字段说明。

## 8. 写作口径

推荐表述：

```text
We observe that post-hoc merging baselines often collapse their predictions to one or a few diagnostic categories on medical image tasks. This behavior is not fully reflected by accuracy, especially under strong class imbalance. We therefore report collapse ratio, balanced accuracy, macro F1, effective predicted classes, and prediction-to-label distribution distance. Across five medical datasets and four backbones, LAMP-Merge consistently reduces prediction collapse and improves class-balanced diagnostic performance.
```

中文对应：

```text
我们观察到，常规事后模型融合方法在医学图像任务中经常将预测集中到一个或少数几个诊断类别。该问题不能只用 accuracy 衡量，尤其在类别不平衡任务上，多数类坍缩会产生虚高 accuracy。因此，我们同时报告 collapse ratio、balanced accuracy、macro F1、有效预测类别数和预测分布距离。结果显示，LAMP-Merge 在 5 个医学数据集和 4 个 backbone 上稳定降低预测坍缩，并提升类别均衡的诊断性能。
```
