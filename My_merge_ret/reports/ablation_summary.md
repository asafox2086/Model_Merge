# my_merge Ablation Summary

- `delta_vs_full` compares each ablation against the `full` my_merge run on exactly matched task keys.
- `W/T/L` compares each ablation against the best original method in `result/all_results.md`.
- A useful medical component should usually lower performance when disabled, especially on its matched modality.

## Overall

| ablation | rows | mean_acc | delta_vs_full | delta_vs_best_original | W/T/L |
| --- | --- | --- | --- | --- | --- |
| avg_only | 225 | 0.2227 | -0.1197 | -0.0976 | 4/33/188 |
| full | 225 | 0.3424 | 0.0000 | 0.0220 | 106/45/74 |
| no_client_information | 225 | 0.3217 | -0.0206 | 0.0014 | 82/44/99 |
| no_fusion_selection | 225 | 0.2227 | -0.1197 | -0.0976 | 4/33/188 |

## Dataset Breakdown

| ablation | dataset | rows | delta_vs_full | W/T/L |
| --- | --- | --- | --- | --- |
| avg_only | bloodmnist | 45 | -0.1439 | 1/5/39 |
| avg_only | dermamnist | 45 | -0.1726 | 1/19/25 |
| avg_only | organcmnist | 45 | -0.1609 | 0/2/43 |
| avg_only | organsmnist | 45 | -0.1346 | 1/2/42 |
| avg_only | chaoshengmnist | 45 | 0.0137 | 1/5/39 |
| full | bloodmnist | 45 | 0.0000 | 29/7/9 |
| full | dermamnist | 45 | 0.0000 | 16/25/4 |
| full | organcmnist | 45 | 0.0000 | 28/6/11 |
| full | organsmnist | 45 | 0.0000 | 27/7/11 |
| full | chaoshengmnist | 45 | 0.0000 | 6/0/39 |
| no_client_information | bloodmnist | 45 | -0.0326 | 21/5/19 |
| no_client_information | dermamnist | 45 | -0.0652 | 5/25/15 |
| no_client_information | organcmnist | 45 | -0.0112 | 28/5/12 |
| no_client_information | organsmnist | 45 | -0.0097 | 24/6/15 |
| no_client_information | chaoshengmnist | 45 | 0.0157 | 4/3/38 |
| no_fusion_selection | bloodmnist | 45 | -0.1439 | 1/5/39 |
| no_fusion_selection | dermamnist | 45 | -0.1726 | 1/19/25 |
| no_fusion_selection | organcmnist | 45 | -0.1609 | 0/2/43 |
| no_fusion_selection | organsmnist | 45 | -0.1346 | 1/2/42 |
| no_fusion_selection | chaoshengmnist | 45 | 0.0137 | 1/5/39 |
