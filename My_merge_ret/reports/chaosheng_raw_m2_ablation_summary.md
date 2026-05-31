# my_merge Ablation Summary

- `delta_vs_full` compares each ablation against the `full` my_merge run on exactly matched task keys.
- `W/T/L` compares each ablation against the best original method in `result/all_results.md`.
- A useful medical component should usually lower performance when disabled, especially on its matched modality.

## Overall

| ablation | rows | mean_acc | delta_vs_full | delta_vs_best_original | W/T/L |
| --- | --- | --- | --- | --- | --- |
| avg_only | 45 | 0.1551 | 0.0152 | -0.0661 | 1/5/39 |
| full | 45 | 0.1399 | 0.0000 | -0.0813 | 4/1/40 |
| no_client_information | 45 | 0.1568 | 0.0169 | -0.0644 | 2/6/37 |
| no_fusion_selection | 45 | 0.1551 | 0.0152 | -0.0661 | 1/5/39 |

## Dataset Breakdown

| ablation | dataset | rows | delta_vs_full | W/T/L |
| --- | --- | --- | --- | --- |
| avg_only | chaoshengmnist | 45 | 0.0152 | 1/5/39 |
| full | chaoshengmnist | 45 | 0.0000 | 4/1/40 |
| no_client_information | chaoshengmnist | 45 | 0.0169 | 2/6/37 |
| no_fusion_selection | chaoshengmnist | 45 | 0.0152 | 1/5/39 |
