## New M2 Smoke Check

- This is a single-case smoke result for the rewritten M2, not a full rerun.
- Case: `small / bloodmnist_224 / resnet / c=3 / beta=0 / seed=42`.
- Old rows are automatically read from the previous full-run output; the new row is read from `outputs/m2_routed_smoke2`.

| source | variant | test_acc | selected_candidate | fusion_rule |
| --- | --- | ---: | --- | --- |
| old full-run | `full` | 0.4607 | `-` | `-` |
| old full-run | `no_client_information` | 0.4607 | `-` | `-` |
| old full-run | `no_fusion_selection` | 0.3017 | `-` | `-` |
| old full-run | `avg_only` | 0.3017 | `-` | `-` |
| new M2 smoke | `full_routed_m2` | 0.3704 | `evidence_routed` | `direct_evidence_routed_fusion` |

