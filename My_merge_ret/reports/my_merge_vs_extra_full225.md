# my_merge vs my_merge_extra Full Comparison

- my_merge root: `outputs/my_merge_full_weights_20260523_171624/my_merge_ablation_grid/full`
- my_merge_extra root: `outputs/full225_my_merge_extra_20260528_200221`
- my_merge rows: 225
- my_merge_extra rows: 225
- matched rows: 225
- exactly same accuracy rows: 28/225

## Group Summary

| task | dataset | model | rows | mean my_merge | mean extra | mean extra-my | same rows |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| small | bloodmnist_224 | convnext | 9 | 0.1905 | 0.1329 | -0.0576 | 2 |
| small | bloodmnist_224 | resnet | 9 | 0.4671 | 0.2627 | -0.2044 | 0 |
| small | bloodmnist_224 | swin_tiny | 9 | 0.2135 | 0.1362 | -0.0774 | 1 |
| small | bloodmnist_224 | vit_t | 9 | 0.2724 | 0.1301 | -0.1423 | 0 |
| small | chaoshengmnist_224 | convnext | 9 | 0.1087 | 0.1449 | 0.0361 | 2 |
| small | chaoshengmnist_224 | resnet | 9 | 0.2305 | 0.1946 | -0.0359 | 0 |
| small | chaoshengmnist_224 | swin_tiny | 9 | 0.1103 | 0.1466 | 0.0362 | 2 |
| small | chaoshengmnist_224 | vit_t | 9 | 0.1103 | 0.1457 | 0.0353 | 0 |
| small | dermamnist_224 | convnext | 9 | 0.6688 | 0.4828 | -0.1860 | 6 |
| small | dermamnist_224 | resnet | 9 | 0.7030 | 0.5807 | -0.1223 | 0 |
| small | dermamnist_224 | swin_tiny | 9 | 0.6688 | 0.5079 | -0.1609 | 6 |
| small | dermamnist_224 | vit_t | 9 | 0.6679 | 0.4606 | -0.2074 | 2 |
| small | organcmnist_224 | convnext | 9 | 0.2055 | 0.1195 | -0.0860 | 2 |
| small | organcmnist_224 | resnet | 9 | 0.4069 | 0.2144 | -0.1925 | 0 |
| small | organcmnist_224 | swin_tiny | 9 | 0.2232 | 0.1049 | -0.1183 | 0 |
| small | organcmnist_224 | vit_t | 9 | 0.2683 | 0.1332 | -0.1350 | 0 |
| small | organsmnist_224 | convnext | 9 | 0.2003 | 0.1423 | -0.0580 | 2 |
| small | organsmnist_224 | resnet | 9 | 0.3527 | 0.2588 | -0.0939 | 0 |
| small | organsmnist_224 | swin_tiny | 9 | 0.2316 | 0.0913 | -0.1403 | 0 |
| small | organsmnist_224 | vit_t | 9 | 0.2686 | 0.1216 | -0.1470 | 1 |
| vlm | bloodmnist_224 | clip-vit-base-patch32 | 9 | 0.4387 | 0.1779 | -0.2608 | 0 |
| vlm | chaoshengmnist_224 | clip-vit-base-patch32 | 9 | 0.1472 | 0.1508 | 0.0036 | 0 |
| vlm | dermamnist_224 | clip-vit-base-patch32 | 9 | 0.6838 | 0.5291 | -0.1547 | 2 |
| vlm | organcmnist_224 | clip-vit-base-patch32 | 9 | 0.3661 | 0.1368 | -0.2292 | 0 |
| vlm | organsmnist_224 | clip-vit-base-patch32 | 9 | 0.3541 | 0.1775 | -0.1766 | 0 |

## Largest Absolute Differences

| task | dataset | model | clients | beta | my_merge | extra | extra-my | same |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| vlm | dermamnist_224 | clip-vit-base-patch32 | 5 | 0.01 | 0.6958 | 0.0534 | -0.6424 | False |
| small | dermamnist_224 | swin_tiny | 3 | 0.0 | 0.6688 | 0.0329 | -0.6359 | False |
| small | bloodmnist_224 | resnet | 3 | 0.1 | 0.8530 | 0.2306 | -0.6223 | False |
| small | dermamnist_224 | vit_t | 3 | 0.1 | 0.6688 | 0.0529 | -0.6160 | False |
| small | dermamnist_224 | convnext | 3 | 0.01 | 0.6688 | 0.1097 | -0.5591 | False |
| small | dermamnist_224 | convnext | 3 | 0.0 | 0.6688 | 0.1112 | -0.5576 | False |
| small | dermamnist_224 | convnext | 5 | 0.01 | 0.6688 | 0.1112 | -0.5576 | False |
| small | dermamnist_224 | swin_tiny | 5 | 0.1 | 0.6688 | 0.1112 | -0.5576 | False |
| small | dermamnist_224 | vit_t | 7 | 0.1 | 0.6688 | 0.1112 | -0.5576 | False |
| vlm | bloodmnist_224 | clip-vit-base-patch32 | 5 | 0.1 | 0.6402 | 0.0830 | -0.5571 | False |
| vlm | dermamnist_224 | clip-vit-base-patch32 | 5 | 0.1 | 0.7197 | 0.1965 | -0.5232 | False |
| vlm | organcmnist_224 | clip-vit-base-patch32 | 3 | 0.1 | 0.6164 | 0.1112 | -0.5051 | False |
| small | dermamnist_224 | vit_t | 7 | 0.01 | 0.6688 | 0.1746 | -0.4943 | False |
| vlm | organcmnist_224 | clip-vit-base-patch32 | 3 | 0.01 | 0.5527 | 0.0909 | -0.4618 | False |
| vlm | bloodmnist_224 | clip-vit-base-patch32 | 7 | 0.1 | 0.6106 | 0.1692 | -0.4414 | False |
| small | organcmnist_224 | resnet | 3 | 0.1 | 0.6626 | 0.2258 | -0.4368 | False |
| vlm | organsmnist_224 | clip-vit-base-patch32 | 7 | 0.1 | 0.5177 | 0.1097 | -0.4081 | False |
| small | organcmnist_224 | resnet | 3 | 0.01 | 0.5775 | 0.1877 | -0.3898 | False |
| small | bloodmnist_224 | resnet | 5 | 0.1 | 0.5981 | 0.2274 | -0.3707 | False |
| vlm | bloodmnist_224 | clip-vit-base-patch32 | 5 | 0.01 | 0.4087 | 0.0830 | -0.3256 | False |
| small | organsmnist_224 | vit_t | 3 | 0.01 | 0.3775 | 0.0583 | -0.3191 | False |
| small | bloodmnist_224 | resnet | 7 | 0.01 | 0.4540 | 0.1374 | -0.3166 | False |
| small | organsmnist_224 | resnet | 3 | 0.0 | 0.4953 | 0.1875 | -0.3078 | False |
| small | chaoshengmnist_224 | resnet | 3 | 0.1 | 0.5094 | 0.2022 | -0.3073 | False |
| vlm | organcmnist_224 | clip-vit-base-patch32 | 5 | 0.0 | 0.3737 | 0.0679 | -0.3057 | False |
| small | dermamnist_224 | resnet | 5 | 0.0 | 0.6594 | 0.3536 | -0.3057 | False |
| vlm | organcmnist_224 | clip-vit-base-patch32 | 5 | 0.1 | 0.5417 | 0.2371 | -0.3046 | False |
| vlm | organsmnist_224 | clip-vit-base-patch32 | 5 | 0.0 | 0.3537 | 0.0497 | -0.3040 | False |
| small | organsmnist_224 | vit_t | 5 | 0.0 | 0.3364 | 0.0469 | -0.2895 | False |
| vlm | bloodmnist_224 | clip-vit-base-patch32 | 3 | 0.1 | 0.5878 | 0.3037 | -0.2841 | False |
