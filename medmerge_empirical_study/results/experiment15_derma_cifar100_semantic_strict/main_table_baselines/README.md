# Main-Table Baselines Under the Strict Domain Control

This run applies every formal main-table baseline to the identical three client checkpoints used by the strict DermaMNIST/CIFAR-100 control. No client was retrained, no image was resampled, and the merge weight is equal for all methods. `fisher` and `regmean` use the main-table validation-statistics budget: one 32-image validation batch per client.

| Method | Medical collapse ratio | Medical effective predictions | Natural collapse ratio | Natural effective predictions |
| --- | ---: | ---: | ---: | ---: |
| avg | 1.000000 | 1 | 0.792020 | 4 |
| ties | 0.558603 | 2 | 1.000000 | 1 |
| dare_linear | 1.000000 | 1 | 0.728678 | 4 |
| dare_ties | 1.000000 | 1 | 0.637905 | 4 |
| regmean | 0.999501 | 2 | 0.530673 | 7 |
| fisher | 1.000000 | 1 | 0.760100 | 5 |
| breadcrumbs | 0.508728 | 7 | 0.892269 | 6 |
| model_stock | 1.000000 | 1 | 0.783541 | 5 |
| from | 1.000000 | 1 | 0.912718 | 3 |
| iso_c | 1.000000 | 1 | 0.714214 | 7 |
| free_merge | 1.000000 | 1 | 0.792020 | 4 |
| robustmerge | 1.000000 | 1 | 0.799002 | 6 |

`regmean` is the only main-table method in this run that preserves the intended medical prediction collapse while making the natural control materially less concentrated: medical predictions are `[0, 0, 0, 0, 1, 2004, 0]`, while natural predictions are `[47, 87, 373, 7, 426, 1064, 1]` over the same 2,005-image test support.

`dare_ties` produces a non-finite cross-entropy loss on the medical test set, despite producing finite class predictions; it is therefore not a reliable alternative to `regmean` here.

The raw outputs are retained separately for [medical](medical/main_table_baseline_collapse_metrics.csv) and [natural](natural/main_table_baseline_collapse_metrics.csv). The runner uses the existing formal baseline implementations from `/data2/liyapeng_grp/program/MedMNISTMerge`, because this experiment worktree does not contain the baseline modules `avg`, `fisher`, `regmean`, `breadcrumbs`, `model_stock`, `from`, `iso`, `free_merge`, and `robustmerge`.

## RegMean Prediction Distributions

The RegMean figures are separate for the two domains and use all 2,005 matched test images:

- [Medical prediction distribution](regmean_diagnostics/dermamnist_regmean_prediction_distribution.png)
- [Natural prediction distribution](regmean_diagnostics/cifar100_semantic7_regmean_prediction_distribution.png)

In each chart, the colored bars are RegMean predictions and the black line is the true-label distribution of the same test images. The plotting output includes both count series, plot settings, 350-DPI PNG, and vector PDF.

These metrics measure output/prediction collapse. They do not establish feature collapse.
