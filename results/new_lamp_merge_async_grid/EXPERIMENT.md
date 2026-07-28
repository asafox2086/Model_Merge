# Strict Asynchronous `new_lamp_merge` Grid

This experiment evaluates incremental `new_lamp_merge` on five medical image
datasets and four backbones. For every dataset/backbone pair, seven client
uploads arrive in the fixed order `0 -> 1 -> ... -> 6`. The server immediately
materializes and evaluates a deployable checkpoint after each arrival, yielding
the 140 Acc and macro-F1 observations in
[`async_grid_k1_to_k7.csv`](async_grid_k1_to_k7.csv). The figure plots only the
four BloodMNIST backbone trajectories, with a distinct solid/dashed line style
per backbone; its data are not smoothed.

## Asynchronous and Equivalence Checks

Each persisted state contains only the immutable reference state, running
prototype numerator, evidence totals, support/prevalence totals, and received
client IDs. It contains no per-client prototype list. The runner opens only the
new `client_<id>.pt` upload in each step; earlier payloads are not reread.

[`strict_async_audit.json`](strict_async_audit.json) verifies all 20 runs:

- every run contains the consecutive seven arrival steps and the expected IDs;
- every persisted state has only the permitted sufficient-statistic fields;
- each final asynchronous checkpoint equals the previous one-shot
  `lamp_merge` checkpoint within absolute tolerance `1e-6`.

The observed maximum tensor error is `9.536743e-7`, which is floating-point
accumulation noise and remains within the stated tolerance. Therefore the
asynchronous implementation changes server delivery semantics, not the final
fusion rule.

Metric precision is held to the historical evaluation setting. BloodMNIST is
evaluated in FP32; other grid entries use their recorded AMP setting. The
previous 0.63 percentage-point Ultrasound discrepancy was traced to AMP
argmax changes on 7 of 1,113 samples, rather than a merge-state difference.

## K=7 Results

The table below reports the final checkpoint after all seven uploads. Full
`k=1..7` trajectories and the exact plotted values are in
[`async_grid_k1_to_k7.csv`](async_grid_k1_to_k7.csv).

| Dataset | Backbone | Acc | Macro-F1 |
| --- | --- | ---: | ---: |
| BloodMNIST | ResNet | 0.8004 | 0.7873 |
| BloodMNIST | ConvNeXt | 0.8483 | **0.8327** |
| BloodMNIST | ViT-T | **0.8515** | 0.8291 |
| BloodMNIST | Swin-T | 0.7740 | 0.7573 |
| DermaMNIST | ResNet | **0.6758** | 0.1497 |
| DermaMNIST | ConvNeXt | 0.5980 | 0.3388 |
| DermaMNIST | ViT-T | 0.5950 | **0.3683** |
| DermaMNIST | Swin-T | 0.6334 | 0.2826 |
| OrganCMNIST | ResNet | 0.6005 | 0.5310 |
| OrganCMNIST | ConvNeXt | 0.6563 | 0.6340 |
| OrganCMNIST | ViT-T | 0.5864 | 0.5726 |
| OrganCMNIST | Swin-T | **0.6842** | **0.6570** |
| OrganSMNIST | ResNet | 0.5275 | 0.4230 |
| OrganSMNIST | ConvNeXt | 0.6058 | 0.5613 |
| OrganSMNIST | ViT-T | 0.5446 | 0.5227 |
| OrganSMNIST | Swin-T | **0.6277** | **0.5844** |
| UltrasoundMNIST | ResNet | 0.4753 | 0.4536 |
| UltrasoundMNIST | ConvNeXt | 0.4304 | 0.4100 |
| UltrasoundMNIST | ViT-T | 0.4627 | 0.4252 |
| UltrasoundMNIST | Swin-T | **0.4834** | **0.4650** |

The selected configuration is BloodMNIST with ConvNeXt at `k=7`: it has the
largest macro-F1 (`0.8327`) of all 20 final configurations. Macro-F1 is the
selection metric; the highest raw accuracy belongs to BloodMNIST with ViT-T
(`0.8515`).

## Reproduction

```bash
/data2/liyapeng_grp/.conda/envs/MM/bin/python scripts/run_new_lamp_merge_async_grid.py \
  --models convnext vit_t swin_tiny

/data2/liyapeng_grp/.conda/envs/MM/bin/python results/new_lamp_merge_async_grid/plot_async_grid.py \
  --outputs-root outputs --result-root results/new_lamp_merge_async_grid

/data2/liyapeng_grp/.conda/envs/MM/bin/python results/new_lamp_merge_async_grid/audit_async_grid.py \
  --outputs-root outputs --result-root results/new_lamp_merge_async_grid
```
