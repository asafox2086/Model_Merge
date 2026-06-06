# my_merge Ablation Summary

- `delta_vs_full` compares each ablation against the `full` my_merge run on exactly matched task keys.
- `W/T/L` compares each ablation against the best original method in `result/all_results.md`.
- A useful medical component should usually lower performance when disabled, especially on its matched modality.

## Overall

| ablation | rows | mean_acc | delta_vs_full | delta_vs_best_original | W/T/L |
| --- | --- | --- | --- | --- | --- |
| full | 225 | 0.3191 | 0.0000 | -0.0012 | 85/38/102 |
| no_adaptive_candidates | 225 | 0.2734 | -0.0457 | -0.0470 | 40/43/142 |
| no_client_information | 225 | 0.3096 | -0.0095 | -0.0107 | 74/41/110 |
| no_fusion_selection | 225 | 0.2227 | -0.0964 | -0.0976 | 4/33/188 |

## Dataset Breakdown

| ablation | dataset | rows | delta_vs_full | W/T/L |
| --- | --- | --- | --- | --- |
| full | bloodmnist | 45 | 0.0000 | 22/9/14 |
| full | dermamnist | 45 | 0.0000 | 13/25/7 |
| full | organcmnist | 45 | 0.0000 | 24/0/21 |
| full | organsmnist | 45 | 0.0000 | 25/4/16 |
| full | chaoshengmnist | 45 | 0.0000 | 1/0/44 |
| no_adaptive_candidates | bloodmnist | 45 | -0.0669 | 10/8/27 |
| no_adaptive_candidates | dermamnist | 45 | -0.0574 | 9/22/14 |
| no_adaptive_candidates | organcmnist | 45 | -0.0631 | 7/2/36 |
| no_adaptive_candidates | organsmnist | 45 | -0.0963 | 9/5/31 |
| no_adaptive_candidates | chaoshengmnist | 45 | 0.0551 | 5/6/34 |
| no_client_information | bloodmnist | 45 | -0.0190 | 17/11/17 |
| no_client_information | dermamnist | 45 | -0.0234 | 10/26/9 |
| no_client_information | organcmnist | 45 | 0.0106 | 22/0/23 |
| no_client_information | organsmnist | 45 | -0.0157 | 24/4/17 |
| no_client_information | chaoshengmnist | 45 | 0.0000 | 1/0/44 |
| no_fusion_selection | bloodmnist | 45 | -0.1163 | 1/5/39 |
| no_fusion_selection | dermamnist | 45 | -0.1682 | 1/19/25 |
| no_fusion_selection | organcmnist | 45 | -0.0995 | 0/2/43 |
| no_fusion_selection | organsmnist | 45 | -0.1361 | 1/2/42 |
| no_fusion_selection | chaoshengmnist | 45 | 0.0381 | 1/5/39 |
