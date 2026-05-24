# Client Weight Analysis

- Source: generated from `merge_result.json`; no manual table filling.
- `pi` is the prior merge weight.
- `alpha_all` is the M1 overall diagnostic weight.
- `alpha_morph` is the M2 morphology/medical-evidence weight.
- CSV files in this folder contain the full detail for filtering and plotting.

## Files

- `client_weight_detail.csv`: one row per ablation/case/client.
- `client_weight_case_summary.csv`: one row per ablation/case.
- `client_weight_by_ablation.csv`: whether each ablation really changes weights.
- `client_weight_by_dataset.csv`: which datasets create larger medical reweighting.
- `client_weight_by_model.csv`: which backbone families are more sensitive.

## Ablation Summary

| ablation | cases | mean_acc | mean_abs_delta_all | mean_abs_delta_morph | mean_max_alpha_all | mean_entropy_alpha_all |
| --- | --- | --- | --- | --- | --- | --- |
| avg_only | 225 | 0.2227 | - | - | - | - |
| full | 225 | 0.3424 | 0.0825 | 0.0946 | 0.3683 | 0.9229 |
| no_client_information | 225 | 0.3217 | 0.0000 | 0.0000 | 0.2254 | 1.0000 |
| no_fusion_selection | 225 | 0.2227 | 0.0825 | 0.0946 | 0.3683 | 0.9229 |

## Dataset Summary

| ablation | dataset | cases | mean_acc | mean_abs_delta_all | mean_abs_delta_morph | mean_max_alpha_all |
| --- | --- | --- | --- | --- | --- | --- |
| avg_only | bloodmnist_224 | 45 | 0.1726 | - | - | - |
| avg_only | chaoshengmnist_224 | 45 | 0.1551 | - | - | - |
| avg_only | dermamnist_224 | 45 | 0.5058 | - | - | - |
| avg_only | organcmnist_224 | 45 | 0.1331 | - | - | - |
| avg_only | organsmnist_224 | 45 | 0.1469 | - | - | - |
| full | bloodmnist_224 | 45 | 0.3164 | 0.0411 | 0.0457 | 0.2871 |
| full | chaoshengmnist_224 | 45 | 0.1414 | 0.1372 | 0.1592 | 0.4717 |
| full | dermamnist_224 | 45 | 0.6785 | 0.1412 | 0.1366 | 0.4717 |
| full | organcmnist_224 | 45 | 0.2940 | 0.0485 | 0.0682 | 0.3091 |
| full | organsmnist_224 | 45 | 0.2815 | 0.0446 | 0.0634 | 0.3020 |
| no_client_information | bloodmnist_224 | 45 | 0.2838 | 0.0000 | 0.0000 | 0.2254 |
| no_client_information | chaoshengmnist_224 | 45 | 0.1571 | 0.0000 | 0.0000 | 0.2254 |
| no_client_information | dermamnist_224 | 45 | 0.6133 | 0.0000 | 0.0000 | 0.2254 |
| no_client_information | organcmnist_224 | 45 | 0.2828 | 0.0000 | 0.0000 | 0.2254 |
| no_client_information | organsmnist_224 | 45 | 0.2717 | 0.0000 | 0.0000 | 0.2254 |
| no_fusion_selection | bloodmnist_224 | 45 | 0.1726 | 0.0411 | 0.0457 | 0.2871 |
| no_fusion_selection | chaoshengmnist_224 | 45 | 0.1551 | 0.1372 | 0.1592 | 0.4717 |
| no_fusion_selection | dermamnist_224 | 45 | 0.5058 | 0.1412 | 0.1366 | 0.4717 |
| no_fusion_selection | organcmnist_224 | 45 | 0.1331 | 0.0485 | 0.0682 | 0.3091 |
| no_fusion_selection | organsmnist_224 | 45 | 0.1469 | 0.0446 | 0.0634 | 0.3020 |

## Model Summary

| ablation | task_type | model | cases | mean_acc | mean_abs_delta_all | mean_abs_delta_morph | mean_entropy_alpha_all |
| --- | --- | --- | --- | --- | --- | --- | --- |
| avg_only | small | convnext | 45 | 0.1985 | - | - | - |
| avg_only | small | resnet | 45 | 0.2933 | - | - | - |
| avg_only | small | swin_tiny | 45 | 0.2031 | - | - | - |
| avg_only | small | vit_t | 45 | 0.1867 | - | - | - |
| avg_only | vlm | clip-vit-base-patch32 | 45 | 0.2319 | - | - | - |
| full | small | convnext | 45 | 0.2748 | 0.0774 | 0.0911 | 0.9329 |
| full | small | resnet | 45 | 0.4320 | 0.0860 | 0.0953 | 0.9246 |
| full | small | swin_tiny | 45 | 0.2895 | 0.0805 | 0.0920 | 0.9218 |
| full | small | vit_t | 45 | 0.3175 | 0.0813 | 0.0957 | 0.9268 |
| full | vlm | clip-vit-base-patch32 | 45 | 0.3980 | 0.0873 | 0.0991 | 0.9085 |
| no_client_information | small | convnext | 45 | 0.2385 | 0.0000 | 0.0000 | 1.0000 |
| no_client_information | small | resnet | 45 | 0.4258 | 0.0000 | 0.0000 | 1.0000 |
| no_client_information | small | swin_tiny | 45 | 0.2719 | 0.0000 | 0.0000 | 1.0000 |
| no_client_information | small | vit_t | 45 | 0.3205 | 0.0000 | 0.0000 | 1.0000 |
| no_client_information | vlm | clip-vit-base-patch32 | 45 | 0.3520 | 0.0000 | 0.0000 | 1.0000 |
| no_fusion_selection | small | convnext | 45 | 0.1985 | 0.0774 | 0.0911 | 0.9329 |
| no_fusion_selection | small | resnet | 45 | 0.2933 | 0.0860 | 0.0953 | 0.9246 |
| no_fusion_selection | small | swin_tiny | 45 | 0.2031 | 0.0805 | 0.0920 | 0.9218 |
| no_fusion_selection | small | vit_t | 45 | 0.1867 | 0.0813 | 0.0957 | 0.9268 |
| no_fusion_selection | vlm | clip-vit-base-patch32 | 45 | 0.2319 | 0.0873 | 0.0991 | 0.9085 |

## Cases With Largest Weight Shift

| ablation | task_type | dataset | model | num_clients | beta | selected_candidate | test_acc | mean_abs_delta_all | mean_abs_delta_morph |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | vlm | chaoshengmnist_224 | clip-vit-base-patch32 | 3 | 0.0100 | specialist_client | 0.1087 | 0.3224 | 0.3493 |
| no_fusion_selection | vlm | chaoshengmnist_224 | clip-vit-base-patch32 | 3 | 0.0100 | avg | 0.1617 | 0.3224 | 0.3493 |
| full | small | chaoshengmnist_224 | convnext | 3 | 0.0000 | morphology | 0.1087 | 0.2511 | 0.2986 |
| no_fusion_selection | small | chaoshengmnist_224 | convnext | 3 | 0.0000 | avg | 0.1051 | 0.2511 | 0.2986 |
| full | small | chaoshengmnist_224 | swin_tiny | 3 | 0.0000 | specialist_client | 0.1087 | 0.2481 | 0.2981 |
| no_fusion_selection | small | chaoshengmnist_224 | swin_tiny | 3 | 0.0000 | avg | 0.1087 | 0.2481 | 0.2981 |
| full | small | chaoshengmnist_224 | vit_t | 3 | 0.0000 | prototype_head | 0.1186 | 0.2392 | 0.2853 |
| no_fusion_selection | small | chaoshengmnist_224 | vit_t | 3 | 0.0000 | avg | 0.1339 | 0.2392 | 0.2853 |
| full | vlm | chaoshengmnist_224 | clip-vit-base-patch32 | 3 | 0.0000 | specialist_client | 0.1851 | 0.2533 | 0.2820 |
| no_fusion_selection | vlm | chaoshengmnist_224 | clip-vit-base-patch32 | 3 | 0.0000 | avg | 0.1734 | 0.2533 | 0.2820 |
| full | small | dermamnist_224 | vit_t | 3 | 0.0000 | prototype_head | 0.6688 | 0.2767 | 0.2741 |
| no_fusion_selection | small | dermamnist_224 | vit_t | 3 | 0.0000 | avg | 0.6688 | 0.2767 | 0.2741 |
| full | small | chaoshengmnist_224 | resnet | 3 | 0.0100 | morphology | 0.1635 | 0.2455 | 0.2710 |
| no_fusion_selection | small | chaoshengmnist_224 | resnet | 3 | 0.0100 | avg | 0.3010 | 0.2455 | 0.2710 |
| full | small | chaoshengmnist_224 | convnext | 3 | 0.0100 | morphology | 0.1087 | 0.2304 | 0.2653 |
| no_fusion_selection | small | chaoshengmnist_224 | convnext | 3 | 0.0100 | avg | 0.1734 | 0.2304 | 0.2653 |
| full | small | chaoshengmnist_224 | swin_tiny | 3 | 0.0100 | prototype_head | 0.1087 | 0.2258 | 0.2609 |
| no_fusion_selection | small | chaoshengmnist_224 | swin_tiny | 3 | 0.0100 | avg | 0.1734 | 0.2258 | 0.2609 |
| full | small | chaoshengmnist_224 | vit_t | 3 | 0.0100 | prototype_head | 0.1087 | 0.2352 | 0.2604 |
| no_fusion_selection | small | chaoshengmnist_224 | vit_t | 3 | 0.0100 | avg | 0.1743 | 0.2352 | 0.2604 |
| full | small | dermamnist_224 | swin_tiny | 3 | 0.0000 | prototype_head | 0.6688 | 0.2629 | 0.2486 |
| no_fusion_selection | small | dermamnist_224 | swin_tiny | 3 | 0.0000 | avg | 0.0329 | 0.2629 | 0.2486 |
| full | vlm | dermamnist_224 | clip-vit-base-patch32 | 3 | 0.0000 | specialist_client | 0.6758 | 0.2490 | 0.2416 |
| no_fusion_selection | vlm | dermamnist_224 | clip-vit-base-patch32 | 3 | 0.0000 | avg | 0.6683 | 0.2490 | 0.2416 |
| full | small | dermamnist_224 | convnext | 3 | 0.0000 | morphology | 0.6688 | 0.2515 | 0.2347 |
| no_fusion_selection | small | dermamnist_224 | convnext | 3 | 0.0000 | avg | 0.1112 | 0.2515 | 0.2347 |
| full | vlm | chaoshengmnist_224 | clip-vit-base-patch32 | 5 | 0.0000 | specialist_client | 0.1087 | 0.2008 | 0.2300 |
| no_fusion_selection | vlm | chaoshengmnist_224 | clip-vit-base-patch32 | 5 | 0.0000 | avg | 0.1222 | 0.2008 | 0.2300 |
| full | vlm | chaoshengmnist_224 | clip-vit-base-patch32 | 5 | 0.1000 | specialist_client | 0.1087 | 0.1989 | 0.2277 |
| no_fusion_selection | vlm | chaoshengmnist_224 | clip-vit-base-patch32 | 5 | 0.1000 | avg | 0.1258 | 0.1989 | 0.2277 |

## Cases With Most Concentrated Diagnostic Weight

| ablation | task_type | dataset | model | num_clients | beta | selected_candidate | test_acc | max_alpha_all | entropy_alpha_all |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | vlm | chaoshengmnist_224 | clip-vit-base-patch32 | 3 | 0.0100 | specialist_client | 0.1087 | 0.8169 | 0.5482 |
| no_fusion_selection | vlm | chaoshengmnist_224 | clip-vit-base-patch32 | 3 | 0.0100 | avg | 0.1617 | 0.8169 | 0.5482 |
| full | small | dermamnist_224 | vit_t | 3 | 0.0000 | prototype_head | 0.6688 | 0.7484 | 0.6663 |
| no_fusion_selection | small | dermamnist_224 | vit_t | 3 | 0.0000 | avg | 0.6688 | 0.7484 | 0.6663 |
| full | small | dermamnist_224 | swin_tiny | 3 | 0.0000 | prototype_head | 0.6688 | 0.7277 | 0.7033 |
| no_fusion_selection | small | dermamnist_224 | swin_tiny | 3 | 0.0000 | avg | 0.0329 | 0.7277 | 0.7033 |
| full | vlm | chaoshengmnist_224 | clip-vit-base-patch32 | 3 | 0.0000 | specialist_client | 0.1851 | 0.7133 | 0.7253 |
| no_fusion_selection | vlm | chaoshengmnist_224 | clip-vit-base-patch32 | 3 | 0.0000 | avg | 0.1734 | 0.7133 | 0.7253 |
| full | small | dermamnist_224 | convnext | 3 | 0.0000 | morphology | 0.6688 | 0.7106 | 0.7266 |
| no_fusion_selection | small | dermamnist_224 | convnext | 3 | 0.0000 | avg | 0.1112 | 0.7106 | 0.7266 |
| full | small | chaoshengmnist_224 | convnext | 3 | 0.0000 | morphology | 0.1087 | 0.7099 | 0.7302 |
| no_fusion_selection | small | chaoshengmnist_224 | convnext | 3 | 0.0000 | avg | 0.1051 | 0.7099 | 0.7302 |
| full | vlm | dermamnist_224 | clip-vit-base-patch32 | 3 | 0.0000 | specialist_client | 0.6758 | 0.7068 | 0.7349 |
| no_fusion_selection | vlm | dermamnist_224 | clip-vit-base-patch32 | 3 | 0.0000 | avg | 0.6683 | 0.7068 | 0.7349 |
| full | small | chaoshengmnist_224 | swin_tiny | 3 | 0.0000 | specialist_client | 0.1087 | 0.7056 | 0.7362 |
| no_fusion_selection | small | chaoshengmnist_224 | swin_tiny | 3 | 0.0000 | avg | 0.1087 | 0.7056 | 0.7362 |
| full | vlm | chaoshengmnist_224 | clip-vit-base-patch32 | 5 | 0.0000 | specialist_client | 0.1087 | 0.7021 | 0.6350 |
| no_fusion_selection | vlm | chaoshengmnist_224 | clip-vit-base-patch32 | 5 | 0.0000 | avg | 0.1222 | 0.7021 | 0.6350 |
| full | small | chaoshengmnist_224 | resnet | 3 | 0.0100 | morphology | 0.1635 | 0.7015 | 0.7236 |
| no_fusion_selection | small | chaoshengmnist_224 | resnet | 3 | 0.0100 | avg | 0.3010 | 0.7015 | 0.7236 |
| full | vlm | chaoshengmnist_224 | clip-vit-base-patch32 | 5 | 0.1000 | specialist_client | 0.1087 | 0.6973 | 0.6411 |
| no_fusion_selection | vlm | chaoshengmnist_224 | clip-vit-base-patch32 | 5 | 0.1000 | avg | 0.1258 | 0.6973 | 0.6411 |
| full | small | dermamnist_224 | resnet | 3 | 0.0100 | morphology | 0.7042 | 0.6963 | 0.7500 |
| no_fusion_selection | small | dermamnist_224 | resnet | 3 | 0.0100 | avg | 0.5461 | 0.6963 | 0.7500 |
| full | small | dermamnist_224 | vit_t | 3 | 0.0100 | prototype_head | 0.6688 | 0.6940 | 0.7530 |
| no_fusion_selection | small | dermamnist_224 | vit_t | 3 | 0.0100 | avg | 0.6688 | 0.6940 | 0.7530 |
| full | small | chaoshengmnist_224 | vit_t | 3 | 0.0000 | prototype_head | 0.1186 | 0.6921 | 0.7562 |
| no_fusion_selection | small | chaoshengmnist_224 | vit_t | 3 | 0.0000 | avg | 0.1339 | 0.6921 | 0.7562 |
| full | small | dermamnist_224 | swin_tiny | 3 | 0.0100 | prototype_head | 0.6688 | 0.6920 | 0.7556 |
| no_fusion_selection | small | dermamnist_224 | swin_tiny | 3 | 0.0100 | avg | 0.6688 | 0.6920 | 0.7556 |
