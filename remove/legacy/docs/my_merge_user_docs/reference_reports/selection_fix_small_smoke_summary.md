# Current my_merge Small Smoke Summary

- Scope: `small / resnet / clients=3 / beta=0 / seed=42`.
- Source: generated from `outputs/selection_fix_*_smoke`; no manual table filling.
- Baseline: best same-cell formal result from `result/all_results.md`.
- CSV: `My_merge_ret/reports/selection_fix_small_smoke_summary.csv`.
- Quick check against formal existing methods: `5` wins, `0` ties, `0` losses.

## Main Smoke Cases

| dataset | acc | selected | client weights | best existing | delta | output |
|---|---:|---|---|---:|---:|---|
| bloodmnist_224 | 0.4680 | consensus | 0.509/0.391/0.100 | regmean 0.3826 | 0.0854 | `outputs/selection_fix_blood_smoke` |
| dermamnist_224 | 0.6793 | evidence_routed | 0.133/0.133/0.734 | fisher 0.6718 | 0.0075 | `outputs/selection_fix_derma_smoke` |
| organcmnist_224 | 0.4877 | consensus | 0.336/0.464/0.201 | robustmerge 0.3824 | 0.1053 | `outputs/selection_fix_organc_smoke` |
| organsmnist_224 | 0.4945 | avg | 0.672/0.125/0.203 | regmean 0.2189 | 0.2756 | `outputs/selection_fix_organs_smoke` |
| chaoshengmnist_224 | 0.3890 | consensus | 0.235/0.444/0.321 | robustmerge 0.2659 | 0.1231 | `outputs/selection_fix_chaosheng_smoke` |

## Ultrasound Ablation Smoke

| ablation | acc | delta vs full | selected | client weights |
|---|---:|---:|---|---|
| full | 0.3890 | 0.0000 | consensus | 0.235/0.444/0.321 |
| no_client_information | 0.3720 | -0.0171 | avg | 0.333/0.333/0.333 |
| no_fusion_selection | 0.2552 | -0.1339 | avg | 0.235/0.444/0.321 |
| avg_only | 0.2552 | -0.1339 | avg | 0.333/0.333/0.333 |

## Reading

- The current smoke subset is above the best formal existing method in every checked dataset.
- On `chaoshengmnist_224`, `full` is higher than both `-M1` and `-M2`, so the two modules are visible in this small check.
- This is still a smoke subset, not the full grid.
