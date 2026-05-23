# my_merge Ablation Summary

- `delta_vs_full` compares each ablation against the `full` my_merge run on exactly matched task keys.
- `W/T/L` compares each ablation against the best original method in `result/all_results.md`.
- A useful medical component should usually lower performance when disabled, especially on its matched modality.

## Overall

| ablation | rows | mean_acc | delta_vs_full | delta_vs_best_original | W/T/L |
| --- | --- | --- | --- | --- | --- |
| avg_only | 225 | 0.2227 | -0.0967 | -0.0976 | 4/33/188 |
| full | 225 | 0.3194 | 0.0000 | -0.0010 | 74/43/108 |
| no_client_information | 225 | 0.3034 | -0.0160 | -0.0170 | 60/45/120 |
| no_fusion_selection | 225 | 0.2423 | -0.0771 | -0.0780 | 21/32/172 |
| no_medical_prior | 225 | 0.3193 | -0.0001 | -0.0010 | 70/44/111 |

## Dataset Breakdown

| ablation | dataset | rows | delta_vs_full | W/T/L |
| --- | --- | --- | --- | --- |
| avg_only | bloodmnist | 45 | -0.1226 | 1/5/39 |
| avg_only | dermamnist | 45 | -0.1672 | 1/19/25 |
| avg_only | organcmnist | 45 | -0.0814 | 0/2/43 |
| avg_only | organsmnist | 45 | -0.1297 | 1/2/42 |
| avg_only | chaoshengmnist | 45 | 0.0177 | 1/5/39 |
| full | bloodmnist | 45 | 0.0000 | 21/8/16 |
| full | dermamnist | 45 | 0.0000 | 10/27/8 |
| full | organcmnist | 45 | 0.0000 | 17/1/27 |
| full | organsmnist | 45 | 0.0000 | 22/7/16 |
| full | chaoshengmnist | 45 | 0.0000 | 4/0/41 |
| no_client_information | bloodmnist | 45 | -0.0309 | 16/7/22 |
| no_client_information | dermamnist | 45 | -0.0599 | 6/27/12 |
| no_client_information | organcmnist | 45 | 0.0113 | 17/2/26 |
| no_client_information | organsmnist | 45 | -0.0203 | 18/4/23 |
| no_client_information | chaoshengmnist | 45 | 0.0198 | 3/5/37 |
| no_fusion_selection | bloodmnist | 45 | -0.0931 | 6/5/34 |
| no_fusion_selection | dermamnist | 45 | -0.1656 | 3/18/24 |
| no_fusion_selection | organcmnist | 45 | -0.0447 | 5/2/38 |
| no_fusion_selection | organsmnist | 45 | -0.1112 | 4/2/39 |
| no_fusion_selection | chaoshengmnist | 45 | 0.0293 | 3/5/37 |
| no_medical_prior | bloodmnist | 45 | 0.0004 | 21/9/15 |
| no_medical_prior | dermamnist | 45 | 0.0000 | 10/27/8 |
| no_medical_prior | organcmnist | 45 | 0.0034 | 15/1/29 |
| no_medical_prior | organsmnist | 45 | -0.0048 | 20/7/18 |
| no_medical_prior | chaoshengmnist | 45 | 0.0005 | 4/0/41 |
