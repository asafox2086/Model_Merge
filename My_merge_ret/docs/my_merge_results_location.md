# my_merge Result Locations

## Main Files

- Root shortcut table: `My_merge_ret/汇总表.md`
- Root reports folder: `My_merge_ret/reports/`
- Root figures folder: `My_merge_ret/figures/`
- Method description: `My_merge_ret/docs/my_merge_paper_method.md`
- Ablation design: `My_merge_ret/docs/my_merge_ablation_design.md`

## Current Running Entry

```bash
bash scripts/run_validated_my_merge_full.sh
```

The script first reproduces selected original baseline methods from `result/all_results.md`. Only after that check passes, it runs the two-module `my_merge` grid and writes the combined table to `My_merge_ret/汇总表.md`.

## Notes

- Plots are disabled by default in the current run scripts.
- Hyperparameter sensitivity is disabled by default.
- Historical collected folders remain under `My_merge_ret/my_merge_completed_*` and probe folders for traceability.
