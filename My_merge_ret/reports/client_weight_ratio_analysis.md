# Client Weight Ratio Analysis

`ratio = alpha / pi`. `1.0` means the final weight is equal to the prior average weight.
Values above `1.0` mean the client/checkpoint is up-weighted; values below `1.0` mean it is down-weighted.

## Ablation Ratio Summary

| ablation | rows | morph_ratio_p10 | morph_ratio_mean | morph_ratio_p90 | mean_case_max/min | max_case_max/min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `full` | 1125 | 0.417 | 1.000 | 1.759 | 4.072 | 14.177 |
| `no_client_information` | 1125 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `no_fusion_selection` | 1125 | 0.417 | 1.000 | 1.759 | 4.072 | 14.177 |

## Interpretation

- `no_client_information` stays at ratio `1.0`, so removing M1 makes all clients collapse back to prior weights.
- `full` has a non-trivial ratio range and max/min spread, so the method is distinguishing clients/checkpoints.
- `no_fusion_selection` keeps almost the same ratios as `full`, which means M1 still distinguishes clients, but the accuracy drop shows M2 is needed to use that distinction effectively.
