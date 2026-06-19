# my_merge Ablation Summary

- `delta_vs_full` compares each ablation against the `full` my_merge run on exactly matched task keys.
- `W/T/L` compares each ablation against the best original method in `result/all_results.md`.
- A useful medical component should usually lower performance when disabled, especially on its matched modality.

## Overall

| ablation | rows | mean_acc | delta_vs_full | delta_vs_best_original | W/T/L |
| --- | --- | --- | --- | --- | --- |
| full | 180 | 0.2834 | 0.0000 | -0.0330 | 42/35/103 |

## Dataset Breakdown

| ablation | dataset | rows | delta_vs_full | W/T/L |
| --- | --- | --- | --- | --- |
| full | bloodmnist | 36 | 0.0000 | 5/3/28 |
| full | dermamnist | 36 | 0.0000 | 5/22/9 |
| full | organcmnist | 36 | 0.0000 | 15/4/17 |
| full | organsmnist | 36 | 0.0000 | 7/1/28 |
| full | chaoshengmnist | 36 | 0.0000 | 10/5/21 |
