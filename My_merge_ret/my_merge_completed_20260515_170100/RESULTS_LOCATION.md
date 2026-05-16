# my_merge Result Locations

Updated at: `2026-05-17T04:49:54`.

## Source Output Roots

- Ablation source: `/data/liyapeng_grp/program/MedMNISTMerge/outputs/my_merge_ablation_three_module_full_20260515_170100`
- Hyperparameter source: `/data/liyapeng_grp/program/MedMNISTMerge/outputs/my_merge_hparam_sensitivity_20260515_170100`

## Preserved Result Folder

- Collected results: `My_merge_ret/my_merge_completed_20260515_170100`

## Main Files

- Method paper-style description: `My_merge_ret/my_merge_paper_method.md`
- Combined baseline + ablation table: `My_merge_ret/my_merge_completed_20260515_170100/ablation_grid_reports/all_results_ablation_combined.md`
- Ablation summary: `My_merge_ret/my_merge_completed_20260515_170100/ablation_grid_reports/ablation_summary.md`
- Ablation grid config: `My_merge_ret/my_merge_completed_20260515_170100/ablation_grid_reports/ablation_grid_config.txt`
- Hyperparameter sensitivity report: `My_merge_ret/my_merge_completed_20260515_170100/hparam_reports/hparam_plots/hparam_sensitivity.md`

## Per-Ablation Diagnostics

- Per-ablation eval/merge summaries: `My_merge_ret/my_merge_completed_20260515_170100/ablation_runs/<ablation>/reports/`
- Client diagnostic weight tables: `My_merge_ret/my_merge_completed_20260515_170100/ablation_runs/<ablation>/reports/my_merge_client_diagnostic_weights.md`
- Weight + PCA visualization tables: `My_merge_ret/my_merge_completed_20260515_170100/ablation_runs/<ablation>/reports/my_merge_weight_visualization_table.md`
- PCA figure folder: `My_merge_ret/my_merge_completed_20260515_170100/ablation_runs/<ablation>/reports/diagnostic_figures/`

## Notes

- `outputs/` remains ignored and is not pushed; this folder only preserves the report artifacts needed for reading and comparison.
- PCA images are generated before merged checkpoints are deleted, so the repository can keep visual evidence without storing large model checkpoints.
- Hyperparameter plots are intentionally separate from the main table; they are for sensitivity analysis rather than leaderboard comparison.
