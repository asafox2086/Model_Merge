# my_merge Ablation Summary

- `delta_vs_full` compares each ablation against the `full` my_merge run on exactly matched task keys.
- `W/T/L` compares each ablation against the best original method in `result/all_results.md`.
- A useful medical component should usually lower performance when disabled, especially on its matched modality.

## Overall

| ablation | rows | mean_acc | delta_vs_full | delta_vs_best_original | W/T/L |
| --- | --- | --- | --- | --- | --- |
| full | 225 | 0.2848 | 0.0000 | -0.0355 | 57/38/130 |

## Dataset Breakdown

| ablation | dataset | rows | delta_vs_full | W/T/L |
| --- | --- | --- | --- | --- |
| full | bloodmnist | 45 | 0.0000 | 8/3/34 |
| full | dermamnist | 45 | 0.0000 | 9/24/12 |
| full | organcmnist | 45 | 0.0000 | 18/5/22 |
| full | organsmnist | 45 | 0.0000 | 10/1/34 |
| full | chaoshengmnist | 45 | 0.0000 | 12/5/28 |
