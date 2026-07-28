# BloodMNIST Non-Fisher Baseline Experiment

## Setting

All methods use BloodMNIST (`224x224`), ResNet, seven clients, `beta=0`, seed
42, equal client merge weights, and the test split (`n=3,421`). The evaluated
formal non-Fisher baseline set is: Avg, Avg-Head, TIES, DARE-Linear,
DARE-TIES, RegMean, Breadcrumbs, Model Stock, From, Iso-C, FreeMerge, and
RobustMerge. Fisher is deliberately excluded. The proposed method is the
formal `lamp_merge` configuration with the same data split and client set.

## Result

| Method | Test Acc |
| --- | ---: |
| Avg | 0.1944 |
| Avg-Head | 0.1795 |
| TIES | **0.0941** |
| DARE-Linear | 0.1757 |
| DARE-TIES | 0.2087 |
| RegMean | 0.2736 |
| Breadcrumbs | 0.1795 |
| Model Stock | 0.1725 |
| From | 0.1903 |
| Iso-C | 0.1865 |
| FreeMerge | 0.1944 |
| RobustMerge | 0.1847 |
| LAMP-Merge | **0.8004** |

The strongest unmerged client reaches `0.2406` Acc. Eleven of twelve
non-Fisher baselines are below that value; RegMean reaches `0.2736`, a small
`3.30` percentage-point increase, but remains `52.67` percentage points below
LAMP-Merge. Thus the conventional non-Fisher merging methods do not provide a
material fused-model improvement in this class-partitioned BloodMNIST setting.

TIES is the lowest baseline (`0.0941`) and is compared directly with
LAMP-Merge (`0.8004`) in
[`ties_vs_lamp_merge_acc.png`](ties_vs_lamp_merge_acc.png). The full exact
values are in [`blood_nonfisher_baseline_acc.csv`](blood_nonfisher_baseline_acc.csv).

## Reproduction

```bash
FORMAL_TASK_TYPES=small \
FORMAL_METHODS="avg avg_head ties dare_linear dare_ties regmean breadcrumbs model_stock from iso_c free_merge robustmerge" \
DATASETS=bloodmnist_224 SMALL_MODELS=resnet \
FORMAL_EXTRA_ARGS="--num-clients 7 --betas 0" \
bash run_compare_multi_gpu.sh

/data2/liyapeng_grp/.conda/envs/MM/bin/python \
  results/blood_nonfisher_baselines/plot_worst_baseline_acc.py
```
