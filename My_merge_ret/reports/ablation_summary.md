# my_merge Ablation Summary

- `delta_vs_full` compares each ablation against the `full` my_merge run on exactly matched task keys.
- `W/T/L` compares each ablation against the best original method in `result/all_results.md`.
- A useful medical component should usually lower performance when disabled, especially on its matched modality.

## Overall

| ablation | rows | mean_acc | delta_vs_full | delta_vs_best_original | W/T/L |
| --- | --- | --- | --- | --- | --- |
| avg_only | 225 | 0.2273 | -0.0851 | -0.0930 | 0/35/190 |
| full | 225 | 0.3124 | 0.0000 | -0.0079 | 54/56/115 |
| no_client_information | 225 | 0.2731 | -0.0393 | -0.0472 | 35/39/151 |
| no_fusion_selection | 225 | 0.2227 | -0.0897 | -0.0976 | 4/33/188 |

## Dataset Breakdown

| ablation | dataset | rows | delta_vs_full | W/T/L |
| --- | --- | --- | --- | --- |
| avg_only | bloodmnist | 45 | -0.0804 | 0/6/39 |
| avg_only | dermamnist | 45 | -0.1160 | 0/19/26 |
| avg_only | organcmnist | 45 | -0.0872 | 0/2/43 |
| avg_only | organsmnist | 45 | -0.0682 | 0/2/43 |
| avg_only | chaoshengmnist | 45 | -0.0737 | 0/6/39 |
| full | bloodmnist | 45 | 0.0000 | 8/10/27 |
| full | dermamnist | 45 | 0.0000 | 10/24/11 |
| full | organcmnist | 45 | 0.0000 | 11/6/28 |
| full | organsmnist | 45 | 0.0000 | 11/9/25 |
| full | chaoshengmnist | 45 | 0.0000 | 14/7/24 |
| no_client_information | bloodmnist | 45 | -0.0245 | 6/6/33 |
| no_client_information | dermamnist | 45 | -0.0743 | 7/20/18 |
| no_client_information | organcmnist | 45 | -0.0353 | 7/2/36 |
| no_client_information | organsmnist | 45 | -0.0391 | 8/2/35 |
| no_client_information | chaoshengmnist | 45 | -0.0234 | 7/9/29 |
| no_fusion_selection | bloodmnist | 45 | -0.0777 | 1/5/39 |
| no_fusion_selection | dermamnist | 45 | -0.1406 | 1/19/25 |
| no_fusion_selection | organcmnist | 45 | -0.0899 | 0/2/43 |
| no_fusion_selection | organsmnist | 45 | -0.0701 | 1/2/42 |
| no_fusion_selection | chaoshengmnist | 45 | -0.0702 | 1/5/39 |
