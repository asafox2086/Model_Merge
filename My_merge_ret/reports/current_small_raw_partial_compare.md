# Reproduction Check

- Baseline: `result/all_results.md`
- Output roots: `outputs/my_merge_small_raw_full_safe_20260530_161700/small_resnet__my_merge`
- Tolerance: `5e-05`

## Status

| status | rows |
| --- | --- |
| missing_baseline | 3 |

## By Method

| method | status | rows |
| --- | --- | --- |
| my_merge | missing_baseline | 3 |

## Non-OK Rows (first 20)

| status | method | task | dataset | model | c | beta | actual | expected | delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| missing_baseline | my_merge | small | bloodmnist_224 | resnet | 3 | 0 | 0.58783981292 |  |  |
| missing_baseline | my_merge | small | bloodmnist_224 | resnet | 3 | 0.01 | 0.545162233265 |  |  |
| missing_baseline | my_merge | small | bloodmnist_224 | resnet | 3 | 0.1 | 0.931598947676 |  |  |
