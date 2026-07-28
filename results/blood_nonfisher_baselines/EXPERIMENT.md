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

## Prefix-Wise Accuracy

To compare client-arrival behavior directly, TIES is re-merged after every
prefix of the same seven client checkpoints (`0 -> 1 -> ... -> 6`). This uses
FP32 test evaluation, matching the BloodMNIST asynchronous LAMP-Merge curve.
TIES obtains `0.1736, 0.0710, 0.0737, 0.1660, 0.0994, 0.1172, 0.0938` from
`k=1` to `7`: its accuracy decreases three times and ends below its first-step
score. LAMP-Merge instead increases at every step from `0.2102` to `0.8004`.

[`ties_vs_lamp_merge_acc_k1_to_k7.png`](ties_vs_lamp_merge_acc_k1_to_k7.png)
shows both methods on the same Acc axis. Its exact plotted values are in
[`ties_vs_lamp_merge_k1_to_k7.csv`](ties_vs_lamp_merge_k1_to_k7.csv).

## Four-Backbone Prefix Comparison

The same prefix-wise TIES evaluation is run for all BloodMNIST backbones used
by the asynchronous study: ResNet, ConvNeXt, ViT-T, and Swin-T. Each run uses
the same seven ordered client checkpoints, `beta=0`, seed 42, equal merge
weights, and FP32 test evaluation as the LAMP-Merge asynchronous reports.

The comparison deliberately uses separate method panels rather than overlaying
TIES and LAMP-Merge: the left column is TIES and the right column is
LAMP-Merge, while all panels share the same `k=1..7` and `0..1` metric axes.
The backbone encoding is consistent in every panel. TIES does not show stable
improvement: ResNet falls three times, ConvNeXt plateaus after `k=4`, ViT-T
falls sharply from `k=1` and then plateaus, and Swin-T falls twice.
LAMP-Merge increases strictly at every arrival for both Acc and macro-F1 on
all four backbones.

- Acc: [`ties_vs_lamp_merge_blood_acc_k1_to_k7.png`](ties_vs_lamp_merge_blood_acc_k1_to_k7.png)
- F1: [`ties_vs_lamp_merge_blood_macro_f1_k1_to_k7.png`](ties_vs_lamp_merge_blood_macro_f1_k1_to_k7.png)
- Combined four-panel view: [`ties_vs_lamp_merge_blood_acc_macro_f1_k1_to_k7.png`](ties_vs_lamp_merge_blood_acc_macro_f1_k1_to_k7.png)
- Exact values: [`ties_vs_lamp_merge_backbones_k1_to_k7.csv`](ties_vs_lamp_merge_backbones_k1_to_k7.csv)

## Reproduction

```bash
FORMAL_TASK_TYPES=small \
FORMAL_METHODS="avg avg_head ties dare_linear dare_ties regmean breadcrumbs model_stock from iso_c free_merge robustmerge" \
DATASETS=bloodmnist_224 SMALL_MODELS=resnet \
FORMAL_EXTRA_ARGS="--num-clients 7 --betas 0" \
bash run_compare_multi_gpu.sh

/data2/liyapeng_grp/.conda/envs/MM/bin/python \
  results/blood_nonfisher_baselines/plot_worst_baseline_acc.py

/data2/liyapeng_grp/.conda/envs/MM/bin/python \
  results/blood_nonfisher_baselines/run_ties_prefix_experiment.py \
  --framework-root /data2/liyapeng_grp/program/MedMNISTMerge \
  --model-hub-root /data2/liyapeng_grp/program/MedMNISTMerge/model_hub \
  --data-root /data2/liyapeng_grp/program/MedMNISTMerge/Med_data

/data2/liyapeng_grp/.conda/envs/MM/bin/python \
  results/blood_nonfisher_baselines/plot_ties_vs_lamp_merge_acc.py

for backbone in resnet convnext vit_t swin_tiny; do
  /data2/liyapeng_grp/.conda/envs/MM/bin/python \
    results/blood_nonfisher_baselines/run_ties_prefix_experiment.py \
    --framework-root /data2/liyapeng_grp/program/MedMNISTMerge \
    --model-hub-root /data2/liyapeng_grp/program/MedMNISTMerge/model_hub \
    --data-root /data2/liyapeng_grp/program/MedMNISTMerge/Med_data \
    --backbone "$backbone" \
    --result-root "results/blood_nonfisher_baselines/ties_prefixes/$backbone"
done

/data2/liyapeng_grp/.conda/envs/MM/bin/python \
  results/blood_nonfisher_baselines/plot_ties_vs_lamp_merge_backbones.py
```
