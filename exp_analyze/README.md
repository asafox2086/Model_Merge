# LAMP-Merge Experiment Analysis Code

This directory contains the experiment-side code for LAMP-Merge. The release method implementation remains in `methods/lamp_merge.py`; this directory is only for full-scope evaluation, ablation, diagnostics, hyperparameter analysis, and result validation.

## Full-Scope Runners

Run missing full-scope analysis experiments in parallel:

```bash
bash exp_analyze/run_full_experiments_parallel.sh
```

The parallel runner validates every requested output group first. Completed
full outputs are skipped. If a matching full experiment is already running, the
runner waits for that process and validates again before launching anything
new. By default, GPU groups are split across long-running full experiments; the
assignment can be overridden explicitly:

```bash
GPU_IDS="0 1 2 3" bash exp_analyze/run_full_experiments_parallel.sh
PREDICTION_GPU_IDS="0 1" HPARAM_GPU_IDS="2 3" bash exp_analyze/run_full_experiments_parallel.sh
STEPS="prediction_diagnostics hparam_full" bash exp_analyze/run_full_experiments_parallel.sh
DRY_RUN=1 bash exp_analyze/run_full_experiments_parallel.sh
```

Canonical report paths under `My_merge_ret/reports/` and
`My_merge_ret/figures/` remain the paths used by the paper and by validation.
Before a full script rewrites one of these canonical outputs, any existing file
or directory is copied to:

```text
My_merge_ret/archive/full_outputs/<RUN_TAG>/
```

This keeps previous full tables and figures available while still leaving a
single latest version at the canonical path.

Run all full-scope analysis experiments sequentially:

```bash
bash exp_analyze/run_full_experiments_sequential.sh
```

The runner executes the following steps in order:

1. `main_table`: validate `My_merge_ret/汇总表.md`.
2. `prototype_geometry`: validate or run prototype geometry and t-SNE analysis.
3. `internal_ablation`: validate or run the full module-internal ablation.
4. `prediction_diagnostics`: validate or run full collapse diagnostics.
5. `hparam_full`: validate or run the full hyperparameter scan.

Each step first calls `validate_full_outputs.py`. If the expected full result already exists and passes validation, the step is skipped. If a matching full experiment is already running, the script waits for it to finish and validates again before deciding whether to start a new run.

Useful options:

```bash
DRY_RUN=1 bash exp_analyze/run_full_experiments_sequential.sh
FORCE=1 bash exp_analyze/run_full_experiments_sequential.sh
STEPS="prediction_diagnostics hparam_full" bash exp_analyze/run_full_experiments_sequential.sh
CHECK_INTERVAL=1800 bash exp_analyze/run_full_experiments_sequential.sh
GPU_IDS="0 1 2 3" bash exp_analyze/run_full_experiments_sequential.sh
```

## Validation

Validate all registered full-scope outputs:

```bash
/data2/liyapeng_grp/.conda/envs/MM/bin/python exp_analyze/validate_full_outputs.py
```

Validate a single output group:

```bash
/data2/liyapeng_grp/.conda/envs/MM/bin/python exp_analyze/validate_full_outputs.py --step internal_ablation
```

Current full-scope validation targets:

- `main_table`: main accuracy table with the formal LAMP-Merge row.
- `internal_ablation`: 13 module/internal variants, 60 client-average cells.
- `prototype_geometry`: full prototype geometry table and per-dataset figures.
- `prediction_diagnostics`: full prediction-collapse diagnostics over all datasets, backbones, client counts, beta values, baselines, LAMP variants, and client models.
- `hparam_full`: full hyperparameter scan over prototype-head scale and long-tail calibration strength.

## Compatibility Note

The old `scripts/` copies are intentionally left in place while previously started background jobs are still running. New full-scope analysis should be launched from `exp_analyze/`. After all active jobs finish, the old analysis entrypoints can be replaced by wrappers or removed.

## Method Separation

- `methods/lamp_merge.py` is the frozen release implementation.
- `methods/lamp_merge_analysis.py` contains ablation and analysis branches.
- `exp_analyze/` contains experiment orchestration, diagnostics, summarization, and validation code.

This separation prevents ablation code from changing the release method used by `汇总表.md`.
