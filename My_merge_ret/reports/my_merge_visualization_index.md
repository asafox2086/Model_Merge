# my_merge Classification PCA Visualizations

Each row identifies the dataset/model/case and shows the corresponding PCA visualization. Points are circles colored by true class; red outlines mark misclassified samples.

| task_type | dataset | model | clip_model | num_clients | beta | seed | selected_candidate | split | embedding_source | num_samples | plot_path | status | note | image |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small | bloodmnist_224 | convnext |  | 3 | 0.0 | 42 | avg | test |  | 0 |  | checkpoint_missing | merged checkpoint not found: outputs/my_merge_ablation_three_module_full_20260515_170100/full/merged/small/bloodmnist_224/convnext/clients_3/beta_0/seed_42/my_merge/merged.pt |  |
| small | bloodmnist_224 | convnext |  | 3 | 0.01 | 42 | avg | test |  | 0 |  | checkpoint_missing | merged checkpoint not found: outputs/my_merge_ablation_three_module_full_20260515_170100/full/merged/small/bloodmnist_224/convnext/clients_3/beta_0p01/seed_42/my_merge/merged.pt |  |
| small | bloodmnist_224 | convnext |  | 3 | 0.1 | 42 | morphology | test |  | 0 |  | checkpoint_missing | merged checkpoint not found: outputs/my_merge_ablation_three_module_full_20260515_170100/full/merged/small/bloodmnist_224/convnext/clients_3/beta_0p1/seed_42/my_merge/merged.pt |  |
| small | bloodmnist_224 | convnext |  | 5 | 0.0 | 42 | specialist_client | test |  | 0 |  | checkpoint_missing | merged checkpoint not found: outputs/my_merge_ablation_three_module_full_20260515_170100/full/merged/small/bloodmnist_224/convnext/clients_5/beta_0/seed_42/my_merge/merged.pt |  |
| small | bloodmnist_224 | convnext |  | 5 | 0.01 | 42 | morphology | test |  | 0 |  | checkpoint_missing | merged checkpoint not found: outputs/my_merge_ablation_three_module_full_20260515_170100/full/merged/small/bloodmnist_224/convnext/clients_5/beta_0p01/seed_42/my_merge/merged.pt |  |
| small | bloodmnist_224 | convnext |  | 5 | 0.1 | 42 | specialist_client | test | model_pre_logits | 128 | ../figures/small__bloodmnist_224__convnext__c5__b0.1__s42__specialist_client.png | ok |  | ![](../figures/small__bloodmnist_224__convnext__c5__b0.1__s42__specialist_client.png) |
| small | bloodmnist_224 | convnext |  | 7 | 0.0 | 42 | specialist_client | test | model_pre_logits | 128 | ../figures/small__bloodmnist_224__convnext__c7__b0.0__s42__specialist_client.png | ok |  | ![](../figures/small__bloodmnist_224__convnext__c7__b0.0__s42__specialist_client.png) |
| small | bloodmnist_224 | convnext |  | 7 | 0.01 | 42 | avg | test | model_pre_logits | 128 | ../figures/small__bloodmnist_224__convnext__c7__b0.01__s42__avg.png | ok |  | ![](../figures/small__bloodmnist_224__convnext__c7__b0.01__s42__avg.png) |
| small | bloodmnist_224 | convnext |  | 7 | 0.1 | 42 | specialist_client | test | model_pre_logits | 128 | ../figures/small__bloodmnist_224__convnext__c7__b0.1__s42__specialist_client.png | ok |  | ![](../figures/small__bloodmnist_224__convnext__c7__b0.1__s42__specialist_client.png) |
| small | bloodmnist_224 | resnet |  | 3 | 0.0 | 42 | consensus | test | model_pre_logits | 128 | ../figures/small__bloodmnist_224__resnet__c3__b0.0__s42__consensus.png | ok |  | ![](../figures/small__bloodmnist_224__resnet__c3__b0.0__s42__consensus.png) |
| small | bloodmnist_224 | resnet |  | 3 | 0.01 | 42 | consensus | test | model_pre_logits | 128 | ../figures/small__bloodmnist_224__resnet__c3__b0.01__s42__consensus.png | ok |  | ![](../figures/small__bloodmnist_224__resnet__c3__b0.01__s42__consensus.png) |
| small | bloodmnist_224 | resnet |  | 3 | 0.1 | 42 | consensus | test | model_pre_logits | 128 | ../figures/small__bloodmnist_224__resnet__c3__b0.1__s42__consensus.png | ok |  | ![](../figures/small__bloodmnist_224__resnet__c3__b0.1__s42__consensus.png) |
| small | bloodmnist_224 | resnet |  | 5 | 0.0 | 42 | avg | test | model_pre_logits | 128 | ../figures/small__bloodmnist_224__resnet__c5__b0.0__s42__avg.png | ok |  | ![](../figures/small__bloodmnist_224__resnet__c5__b0.0__s42__avg.png) |
| small | bloodmnist_224 | resnet |  | 5 | 0.01 | 42 | morphology | test | model_pre_logits | 128 | ../figures/small__bloodmnist_224__resnet__c5__b0.01__s42__morphology.png | ok |  | ![](../figures/small__bloodmnist_224__resnet__c5__b0.01__s42__morphology.png) |
| small | bloodmnist_224 | resnet |  | 5 | 0.1 | 42 | specialist_client | test | model_pre_logits | 128 | ../figures/small__bloodmnist_224__resnet__c5__b0.1__s42__specialist_client.png | ok |  | ![](../figures/small__bloodmnist_224__resnet__c5__b0.1__s42__specialist_client.png) |
| small | bloodmnist_224 | resnet |  | 7 | 0.0 | 42 | consensus | test | model_pre_logits | 128 | ../figures/small__bloodmnist_224__resnet__c7__b0.0__s42__consensus.png | ok |  | ![](../figures/small__bloodmnist_224__resnet__c7__b0.0__s42__consensus.png) |
| small | bloodmnist_224 | resnet |  | 7 | 0.01 | 42 | consensus | test | model_pre_logits | 128 | ../figures/small__bloodmnist_224__resnet__c7__b0.01__s42__consensus.png | ok |  | ![](../figures/small__bloodmnist_224__resnet__c7__b0.01__s42__consensus.png) |
| small | bloodmnist_224 | resnet |  | 7 | 0.1 | 42 | specialist_client | test | model_pre_logits | 128 | ../figures/small__bloodmnist_224__resnet__c7__b0.1__s42__specialist_client.png | ok |  | ![](../figures/small__bloodmnist_224__resnet__c7__b0.1__s42__specialist_client.png) |
| small | bloodmnist_224 | swin_tiny |  | 3 | 0.0 | 42 | prototype_head | test | model_pre_logits | 128 | ../figures/small__bloodmnist_224__swin_tiny__c3__b0.0__s42__prototype_head.png | ok |  | ![](../figures/small__bloodmnist_224__swin_tiny__c3__b0.0__s42__prototype_head.png) |
| small | bloodmnist_224 | swin_tiny |  | 3 | 0.01 | 42 | morph_anchor | test | model_pre_logits | 128 | ../figures/small__bloodmnist_224__swin_tiny__c3__b0.01__s42__morph_anchor.png | ok |  | ![](../figures/small__bloodmnist_224__swin_tiny__c3__b0.01__s42__morph_anchor.png) |
| small | bloodmnist_224 | swin_tiny |  | 3 | 0.1 | 42 | specialist_client | test | model_pre_logits | 128 | ../figures/small__bloodmnist_224__swin_tiny__c3__b0.1__s42__specialist_client.png | ok |  | ![](../figures/small__bloodmnist_224__swin_tiny__c3__b0.1__s42__specialist_client.png) |
| small | bloodmnist_224 | swin_tiny |  | 5 | 0.0 | 42 | prototype_head | test | model_pre_logits | 128 | ../figures/small__bloodmnist_224__swin_tiny__c5__b0.0__s42__prototype_head.png | ok |  | ![](../figures/small__bloodmnist_224__swin_tiny__c5__b0.0__s42__prototype_head.png) |
| small | bloodmnist_224 | swin_tiny |  | 5 | 0.01 | 42 | prototype_head | test | model_pre_logits | 128 | ../figures/small__bloodmnist_224__swin_tiny__c5__b0.01__s42__prototype_head.png | ok |  | ![](../figures/small__bloodmnist_224__swin_tiny__c5__b0.01__s42__prototype_head.png) |
| small | bloodmnist_224 | swin_tiny |  | 5 | 0.1 | 42 | prototype_head | test | model_pre_logits | 128 | ../figures/small__bloodmnist_224__swin_tiny__c5__b0.1__s42__prototype_head.png | ok |  | ![](../figures/small__bloodmnist_224__swin_tiny__c5__b0.1__s42__prototype_head.png) |
| small | bloodmnist_224 | swin_tiny |  | 7 | 0.0 | 42 | prototype_head | test | model_pre_logits | 128 | ../figures/small__bloodmnist_224__swin_tiny__c7__b0.0__s42__prototype_head.png | ok |  | ![](../figures/small__bloodmnist_224__swin_tiny__c7__b0.0__s42__prototype_head.png) |
| small | bloodmnist_224 | swin_tiny |  | 7 | 0.01 | 42 | prototype_head | test | model_pre_logits | 128 | ../figures/small__bloodmnist_224__swin_tiny__c7__b0.01__s42__prototype_head.png | ok |  | ![](../figures/small__bloodmnist_224__swin_tiny__c7__b0.01__s42__prototype_head.png) |
| small | bloodmnist_224 | swin_tiny |  | 7 | 0.1 | 42 | specialist_client | test | model_pre_logits | 128 | ../figures/small__bloodmnist_224__swin_tiny__c7__b0.1__s42__specialist_client.png | ok |  | ![](../figures/small__bloodmnist_224__swin_tiny__c7__b0.1__s42__specialist_client.png) |
| small | bloodmnist_224 | vit_t |  | 3 | 0.0 | 42 | prototype_head | test | model_pre_logits | 128 | ../figures/small__bloodmnist_224__vit_t__c3__b0.0__s42__prototype_head.png | ok |  | ![](../figures/small__bloodmnist_224__vit_t__c3__b0.0__s42__prototype_head.png) |
| small | bloodmnist_224 | vit_t |  | 3 | 0.01 | 42 | prototype_head | test | model_pre_logits | 128 | ../figures/small__bloodmnist_224__vit_t__c3__b0.01__s42__prototype_head.png | ok |  | ![](../figures/small__bloodmnist_224__vit_t__c3__b0.01__s42__prototype_head.png) |
| small | bloodmnist_224 | vit_t |  | 3 | 0.1 | 42 | prototype_head | test | model_pre_logits | 128 | ../figures/small__bloodmnist_224__vit_t__c3__b0.1__s42__prototype_head.png | ok |  | ![](../figures/small__bloodmnist_224__vit_t__c3__b0.1__s42__prototype_head.png) |
| small | bloodmnist_224 | vit_t |  | 5 | 0.0 | 42 | prototype_head | test | model_pre_logits | 128 | ../figures/small__bloodmnist_224__vit_t__c5__b0.0__s42__prototype_head.png | ok |  | ![](../figures/small__bloodmnist_224__vit_t__c5__b0.0__s42__prototype_head.png) |
| small | bloodmnist_224 | vit_t |  | 5 | 0.01 | 42 | prototype_head | test | model_pre_logits | 128 | ../figures/small__bloodmnist_224__vit_t__c5__b0.01__s42__prototype_head.png | ok |  | ![](../figures/small__bloodmnist_224__vit_t__c5__b0.01__s42__prototype_head.png) |
| small | bloodmnist_224 | vit_t |  | 5 | 0.1 | 42 | prototype_head | test | model_pre_logits | 128 | ../figures/small__bloodmnist_224__vit_t__c5__b0.1__s42__prototype_head.png | ok |  | ![](../figures/small__bloodmnist_224__vit_t__c5__b0.1__s42__prototype_head.png) |
| small | bloodmnist_224 | vit_t |  | 7 | 0.0 | 42 | prototype_head | test | model_pre_logits | 128 | ../figures/small__bloodmnist_224__vit_t__c7__b0.0__s42__prototype_head.png | ok |  | ![](../figures/small__bloodmnist_224__vit_t__c7__b0.0__s42__prototype_head.png) |
| small | bloodmnist_224 | vit_t |  | 7 | 0.01 | 42 | consensus | test | model_pre_logits | 128 | ../figures/small__bloodmnist_224__vit_t__c7__b0.01__s42__consensus.png | ok |  | ![](../figures/small__bloodmnist_224__vit_t__c7__b0.01__s42__consensus.png) |
| small | bloodmnist_224 | vit_t |  | 7 | 0.1 | 42 | consensus | test | model_pre_logits | 128 | ../figures/small__bloodmnist_224__vit_t__c7__b0.1__s42__consensus.png | ok |  | ![](../figures/small__bloodmnist_224__vit_t__c7__b0.1__s42__consensus.png) |
| small | chaoshengmnist_224 | convnext |  | 3 | 0.0 | 42 | morphology | test | model_pre_logits | 128 | ../figures/small__chaoshengmnist_224__convnext__c3__b0.0__s42__morphology.png | ok |  | ![](../figures/small__chaoshengmnist_224__convnext__c3__b0.0__s42__morphology.png) |
| small | chaoshengmnist_224 | convnext |  | 3 | 0.01 | 42 | morphology | test | model_pre_logits | 128 | ../figures/small__chaoshengmnist_224__convnext__c3__b0.01__s42__morphology.png | ok |  | ![](../figures/small__chaoshengmnist_224__convnext__c3__b0.01__s42__morphology.png) |
| small | chaoshengmnist_224 | convnext |  | 3 | 0.1 | 42 | morphology | test | model_pre_logits | 128 | ../figures/small__chaoshengmnist_224__convnext__c3__b0.1__s42__morphology.png | ok |  | ![](../figures/small__chaoshengmnist_224__convnext__c3__b0.1__s42__morphology.png) |
| small | chaoshengmnist_224 | convnext |  | 5 | 0.0 | 42 | morphology | test | model_pre_logits | 128 | ../figures/small__chaoshengmnist_224__convnext__c5__b0.0__s42__morphology.png | ok |  | ![](../figures/small__chaoshengmnist_224__convnext__c5__b0.0__s42__morphology.png) |
| small | chaoshengmnist_224 | convnext |  | 5 | 0.01 | 42 | morphology | test | model_pre_logits | 128 | ../figures/small__chaoshengmnist_224__convnext__c5__b0.01__s42__morphology.png | ok |  | ![](../figures/small__chaoshengmnist_224__convnext__c5__b0.01__s42__morphology.png) |
| small | chaoshengmnist_224 | convnext |  | 5 | 0.1 | 42 | morphology | test | model_pre_logits | 128 | ../figures/small__chaoshengmnist_224__convnext__c5__b0.1__s42__morphology.png | ok |  | ![](../figures/small__chaoshengmnist_224__convnext__c5__b0.1__s42__morphology.png) |
| small | chaoshengmnist_224 | convnext |  | 7 | 0.0 | 42 | avg | test | model_pre_logits | 128 | ../figures/small__chaoshengmnist_224__convnext__c7__b0.0__s42__avg.png | ok |  | ![](../figures/small__chaoshengmnist_224__convnext__c7__b0.0__s42__avg.png) |
| small | chaoshengmnist_224 | convnext |  | 7 | 0.01 | 42 | morphology | test | model_pre_logits | 128 | ../figures/small__chaoshengmnist_224__convnext__c7__b0.01__s42__morphology.png | ok |  | ![](../figures/small__chaoshengmnist_224__convnext__c7__b0.01__s42__morphology.png) |
| small | chaoshengmnist_224 | convnext |  | 7 | 0.1 | 42 | morphology | test | model_pre_logits | 128 | ../figures/small__chaoshengmnist_224__convnext__c7__b0.1__s42__morphology.png | ok |  | ![](../figures/small__chaoshengmnist_224__convnext__c7__b0.1__s42__morphology.png) |
| small | chaoshengmnist_224 | resnet |  | 3 | 0.0 | 42 | morphology | test | model_pre_logits | 128 | ../figures/small__chaoshengmnist_224__resnet__c3__b0.0__s42__morphology.png | ok |  | ![](../figures/small__chaoshengmnist_224__resnet__c3__b0.0__s42__morphology.png) |
| small | chaoshengmnist_224 | resnet |  | 3 | 0.01 | 42 | morphology | test | model_pre_logits | 128 | ../figures/small__chaoshengmnist_224__resnet__c3__b0.01__s42__morphology.png | ok |  | ![](../figures/small__chaoshengmnist_224__resnet__c3__b0.01__s42__morphology.png) |
| small | chaoshengmnist_224 | resnet |  | 3 | 0.1 | 42 | morph_anchor | test | model_pre_logits | 128 | ../figures/small__chaoshengmnist_224__resnet__c3__b0.1__s42__morph_anchor.png | ok |  | ![](../figures/small__chaoshengmnist_224__resnet__c3__b0.1__s42__morph_anchor.png) |
| small | chaoshengmnist_224 | resnet |  | 5 | 0.0 | 42 | morphology | test | model_pre_logits | 128 | ../figures/small__chaoshengmnist_224__resnet__c5__b0.0__s42__morphology.png | ok |  | ![](../figures/small__chaoshengmnist_224__resnet__c5__b0.0__s42__morphology.png) |
| small | chaoshengmnist_224 | resnet |  | 5 | 0.01 | 42 | morphology | test | model_pre_logits | 128 | ../figures/small__chaoshengmnist_224__resnet__c5__b0.01__s42__morphology.png | ok |  | ![](../figures/small__chaoshengmnist_224__resnet__c5__b0.01__s42__morphology.png) |
| small | chaoshengmnist_224 | resnet |  | 5 | 0.1 | 42 | morphology | test | model_pre_logits | 128 | ../figures/small__chaoshengmnist_224__resnet__c5__b0.1__s42__morphology.png | ok |  | ![](../figures/small__chaoshengmnist_224__resnet__c5__b0.1__s42__morphology.png) |
| small | chaoshengmnist_224 | resnet |  | 7 | 0.0 | 42 | morphology | test | model_pre_logits | 128 | ../figures/small__chaoshengmnist_224__resnet__c7__b0.0__s42__morphology.png | ok |  | ![](../figures/small__chaoshengmnist_224__resnet__c7__b0.0__s42__morphology.png) |
| small | chaoshengmnist_224 | resnet |  | 7 | 0.01 | 42 | morphology | test | model_pre_logits | 128 | ../figures/small__chaoshengmnist_224__resnet__c7__b0.01__s42__morphology.png | ok |  | ![](../figures/small__chaoshengmnist_224__resnet__c7__b0.01__s42__morphology.png) |
| small | chaoshengmnist_224 | resnet |  | 7 | 0.1 | 42 | morphology | test | model_pre_logits | 128 | ../figures/small__chaoshengmnist_224__resnet__c7__b0.1__s42__morphology.png | ok |  | ![](../figures/small__chaoshengmnist_224__resnet__c7__b0.1__s42__morphology.png) |
| small | chaoshengmnist_224 | swin_tiny |  | 3 | 0.0 | 42 | prototype_head | test | model_pre_logits | 128 | ../figures/small__chaoshengmnist_224__swin_tiny__c3__b0.0__s42__prototype_head.png | ok |  | ![](../figures/small__chaoshengmnist_224__swin_tiny__c3__b0.0__s42__prototype_head.png) |
| small | chaoshengmnist_224 | swin_tiny |  | 3 | 0.01 | 42 | prototype_head | test | model_pre_logits | 128 | ../figures/small__chaoshengmnist_224__swin_tiny__c3__b0.01__s42__prototype_head.png | ok |  | ![](../figures/small__chaoshengmnist_224__swin_tiny__c3__b0.01__s42__prototype_head.png) |
| small | chaoshengmnist_224 | swin_tiny |  | 3 | 0.1 | 42 | prototype_head | test | model_pre_logits | 128 | ../figures/small__chaoshengmnist_224__swin_tiny__c3__b0.1__s42__prototype_head.png | ok |  | ![](../figures/small__chaoshengmnist_224__swin_tiny__c3__b0.1__s42__prototype_head.png) |
| small | chaoshengmnist_224 | swin_tiny |  | 5 | 0.0 | 42 | prototype_head | test | model_pre_logits | 128 | ../figures/small__chaoshengmnist_224__swin_tiny__c5__b0.0__s42__prototype_head.png | ok |  | ![](../figures/small__chaoshengmnist_224__swin_tiny__c5__b0.0__s42__prototype_head.png) |
| small | chaoshengmnist_224 | swin_tiny |  | 5 | 0.01 | 42 | prototype_head | test | model_pre_logits | 128 | ../figures/small__chaoshengmnist_224__swin_tiny__c5__b0.01__s42__prototype_head.png | ok |  | ![](../figures/small__chaoshengmnist_224__swin_tiny__c5__b0.01__s42__prototype_head.png) |
| small | chaoshengmnist_224 | swin_tiny |  | 5 | 0.1 | 42 | prototype_head | test | model_pre_logits | 128 | ../figures/small__chaoshengmnist_224__swin_tiny__c5__b0.1__s42__prototype_head.png | ok |  | ![](../figures/small__chaoshengmnist_224__swin_tiny__c5__b0.1__s42__prototype_head.png) |
| small | chaoshengmnist_224 | swin_tiny |  | 7 | 0.0 | 42 | prototype_head | test | model_pre_logits | 128 | ../figures/small__chaoshengmnist_224__swin_tiny__c7__b0.0__s42__prototype_head.png | ok |  | ![](../figures/small__chaoshengmnist_224__swin_tiny__c7__b0.0__s42__prototype_head.png) |
| small | chaoshengmnist_224 | swin_tiny |  | 7 | 0.01 | 42 | prototype_head | test | model_pre_logits | 128 | ../figures/small__chaoshengmnist_224__swin_tiny__c7__b0.01__s42__prototype_head.png | ok |  | ![](../figures/small__chaoshengmnist_224__swin_tiny__c7__b0.01__s42__prototype_head.png) |
| small | chaoshengmnist_224 | swin_tiny |  | 7 | 0.1 | 42 | prototype_head | test | model_pre_logits | 128 | ../figures/small__chaoshengmnist_224__swin_tiny__c7__b0.1__s42__prototype_head.png | ok |  | ![](../figures/small__chaoshengmnist_224__swin_tiny__c7__b0.1__s42__prototype_head.png) |
| small | chaoshengmnist_224 | vit_t |  | 3 | 0.0 | 42 | prototype_head | test | model_pre_logits | 128 | ../figures/small__chaoshengmnist_224__vit_t__c3__b0.0__s42__prototype_head.png | ok |  | ![](../figures/small__chaoshengmnist_224__vit_t__c3__b0.0__s42__prototype_head.png) |
| small | chaoshengmnist_224 | vit_t |  | 3 | 0.01 | 42 | prototype_head | test | model_pre_logits | 128 | ../figures/small__chaoshengmnist_224__vit_t__c3__b0.01__s42__prototype_head.png | ok |  | ![](../figures/small__chaoshengmnist_224__vit_t__c3__b0.01__s42__prototype_head.png) |
| small | chaoshengmnist_224 | vit_t |  | 3 | 0.1 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | chaoshengmnist_224 | vit_t |  | 5 | 0.0 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | chaoshengmnist_224 | vit_t |  | 5 | 0.01 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | chaoshengmnist_224 | vit_t |  | 5 | 0.1 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | chaoshengmnist_224 | vit_t |  | 7 | 0.0 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | chaoshengmnist_224 | vit_t |  | 7 | 0.01 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | chaoshengmnist_224 | vit_t |  | 7 | 0.1 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | dermamnist_224 | convnext |  | 3 | 0.0 | 42 | morphology | test |  | 0 |  | plot_limit_skipped |  |  |
| small | dermamnist_224 | convnext |  | 3 | 0.01 | 42 | morphology | test |  | 0 |  | plot_limit_skipped |  |  |
| small | dermamnist_224 | convnext |  | 3 | 0.1 | 42 | avg | test |  | 0 |  | plot_limit_skipped |  |  |
| small | dermamnist_224 | convnext |  | 5 | 0.0 | 42 | avg | test |  | 0 |  | plot_limit_skipped |  |  |
| small | dermamnist_224 | convnext |  | 5 | 0.01 | 42 | morphology | test |  | 0 |  | plot_limit_skipped |  |  |
| small | dermamnist_224 | convnext |  | 5 | 0.1 | 42 | avg | test |  | 0 |  | plot_limit_skipped |  |  |
| small | dermamnist_224 | convnext |  | 7 | 0.0 | 42 | avg | test |  | 0 |  | plot_limit_skipped |  |  |
| small | dermamnist_224 | convnext |  | 7 | 0.01 | 42 | avg | test |  | 0 |  | plot_limit_skipped |  |  |
| small | dermamnist_224 | convnext |  | 7 | 0.1 | 42 | avg | test |  | 0 |  | plot_limit_skipped |  |  |
| small | dermamnist_224 | resnet |  | 3 | 0.0 | 42 | avg | test |  | 0 |  | plot_limit_skipped |  |  |
| small | dermamnist_224 | resnet |  | 3 | 0.01 | 42 | morphology | test |  | 0 |  | plot_limit_skipped |  |  |
| small | dermamnist_224 | resnet |  | 3 | 0.1 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| small | dermamnist_224 | resnet |  | 5 | 0.0 | 42 | morphology | test |  | 0 |  | plot_limit_skipped |  |  |
| small | dermamnist_224 | resnet |  | 5 | 0.01 | 42 | avg | test |  | 0 |  | plot_limit_skipped |  |  |
| small | dermamnist_224 | resnet |  | 5 | 0.1 | 42 | morph_anchor | test |  | 0 |  | plot_limit_skipped |  |  |
| small | dermamnist_224 | resnet |  | 7 | 0.0 | 42 | avg | test |  | 0 |  | plot_limit_skipped |  |  |
| small | dermamnist_224 | resnet |  | 7 | 0.01 | 42 | avg | test |  | 0 |  | plot_limit_skipped |  |  |
| small | dermamnist_224 | resnet |  | 7 | 0.1 | 42 | morphology | test |  | 0 |  | plot_limit_skipped |  |  |
| small | dermamnist_224 | swin_tiny |  | 3 | 0.0 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | dermamnist_224 | swin_tiny |  | 3 | 0.01 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | dermamnist_224 | swin_tiny |  | 3 | 0.1 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | dermamnist_224 | swin_tiny |  | 5 | 0.0 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | dermamnist_224 | swin_tiny |  | 5 | 0.01 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | dermamnist_224 | swin_tiny |  | 5 | 0.1 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | dermamnist_224 | swin_tiny |  | 7 | 0.0 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | dermamnist_224 | swin_tiny |  | 7 | 0.01 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | dermamnist_224 | swin_tiny |  | 7 | 0.1 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | dermamnist_224 | vit_t |  | 3 | 0.0 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | dermamnist_224 | vit_t |  | 3 | 0.01 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | dermamnist_224 | vit_t |  | 3 | 0.1 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | dermamnist_224 | vit_t |  | 5 | 0.0 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | dermamnist_224 | vit_t |  | 5 | 0.01 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | dermamnist_224 | vit_t |  | 5 | 0.1 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | dermamnist_224 | vit_t |  | 7 | 0.0 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | dermamnist_224 | vit_t |  | 7 | 0.01 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | dermamnist_224 | vit_t |  | 7 | 0.1 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organcmnist_224 | convnext |  | 3 | 0.0 | 42 | morphology | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organcmnist_224 | convnext |  | 3 | 0.01 | 42 | morphology | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organcmnist_224 | convnext |  | 3 | 0.1 | 42 | morph_anchor | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organcmnist_224 | convnext |  | 5 | 0.0 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organcmnist_224 | convnext |  | 5 | 0.01 | 42 | morphology | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organcmnist_224 | convnext |  | 5 | 0.1 | 42 | morphology | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organcmnist_224 | convnext |  | 7 | 0.0 | 42 | morphology | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organcmnist_224 | convnext |  | 7 | 0.01 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organcmnist_224 | convnext |  | 7 | 0.1 | 42 | avg | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organcmnist_224 | resnet |  | 3 | 0.0 | 42 | consensus | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organcmnist_224 | resnet |  | 3 | 0.01 | 42 | avg | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organcmnist_224 | resnet |  | 3 | 0.1 | 42 | avg | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organcmnist_224 | resnet |  | 5 | 0.0 | 42 | morphology | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organcmnist_224 | resnet |  | 5 | 0.01 | 42 | morphology | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organcmnist_224 | resnet |  | 5 | 0.1 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organcmnist_224 | resnet |  | 7 | 0.0 | 42 | morphology | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organcmnist_224 | resnet |  | 7 | 0.01 | 42 | morphology | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organcmnist_224 | resnet |  | 7 | 0.1 | 42 | morph_anchor | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organcmnist_224 | swin_tiny |  | 3 | 0.0 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organcmnist_224 | swin_tiny |  | 3 | 0.01 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organcmnist_224 | swin_tiny |  | 3 | 0.1 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organcmnist_224 | swin_tiny |  | 5 | 0.0 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organcmnist_224 | swin_tiny |  | 5 | 0.01 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organcmnist_224 | swin_tiny |  | 5 | 0.1 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organcmnist_224 | swin_tiny |  | 7 | 0.0 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organcmnist_224 | swin_tiny |  | 7 | 0.01 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organcmnist_224 | swin_tiny |  | 7 | 0.1 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organcmnist_224 | vit_t |  | 3 | 0.0 | 42 | morph_anchor | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organcmnist_224 | vit_t |  | 3 | 0.01 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organcmnist_224 | vit_t |  | 3 | 0.1 | 42 | consensus | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organcmnist_224 | vit_t |  | 5 | 0.0 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organcmnist_224 | vit_t |  | 5 | 0.01 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organcmnist_224 | vit_t |  | 5 | 0.1 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organcmnist_224 | vit_t |  | 7 | 0.0 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organcmnist_224 | vit_t |  | 7 | 0.01 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organcmnist_224 | vit_t |  | 7 | 0.1 | 42 | reference_delta | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organsmnist_224 | convnext |  | 3 | 0.0 | 42 | avg | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organsmnist_224 | convnext |  | 3 | 0.01 | 42 | morphology | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organsmnist_224 | convnext |  | 3 | 0.1 | 42 | morph_anchor | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organsmnist_224 | convnext |  | 5 | 0.0 | 42 | morphology | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organsmnist_224 | convnext |  | 5 | 0.01 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organsmnist_224 | convnext |  | 5 | 0.1 | 42 | morphology | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organsmnist_224 | convnext |  | 7 | 0.0 | 42 | morphology | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organsmnist_224 | convnext |  | 7 | 0.01 | 42 | avg | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organsmnist_224 | convnext |  | 7 | 0.1 | 42 | morphology | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organsmnist_224 | resnet |  | 3 | 0.0 | 42 | morphology | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organsmnist_224 | resnet |  | 3 | 0.01 | 42 | morph_anchor | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organsmnist_224 | resnet |  | 3 | 0.1 | 42 | morphology | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organsmnist_224 | resnet |  | 5 | 0.0 | 42 | avg | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organsmnist_224 | resnet |  | 5 | 0.01 | 42 | morph_anchor | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organsmnist_224 | resnet |  | 5 | 0.1 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organsmnist_224 | resnet |  | 7 | 0.0 | 42 | morphology | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organsmnist_224 | resnet |  | 7 | 0.01 | 42 | consensus | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organsmnist_224 | resnet |  | 7 | 0.1 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organsmnist_224 | swin_tiny |  | 3 | 0.0 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organsmnist_224 | swin_tiny |  | 3 | 0.01 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organsmnist_224 | swin_tiny |  | 3 | 0.1 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organsmnist_224 | swin_tiny |  | 5 | 0.0 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organsmnist_224 | swin_tiny |  | 5 | 0.01 | 42 | morph_anchor | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organsmnist_224 | swin_tiny |  | 5 | 0.1 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organsmnist_224 | swin_tiny |  | 7 | 0.0 | 42 | morphology | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organsmnist_224 | swin_tiny |  | 7 | 0.01 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organsmnist_224 | swin_tiny |  | 7 | 0.1 | 42 | consensus | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organsmnist_224 | vit_t |  | 3 | 0.0 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organsmnist_224 | vit_t |  | 3 | 0.01 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organsmnist_224 | vit_t |  | 3 | 0.1 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organsmnist_224 | vit_t |  | 5 | 0.0 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organsmnist_224 | vit_t |  | 5 | 0.01 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organsmnist_224 | vit_t |  | 5 | 0.1 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organsmnist_224 | vit_t |  | 7 | 0.0 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organsmnist_224 | vit_t |  | 7 | 0.01 | 42 | prototype_head | test |  | 0 |  | plot_limit_skipped |  |  |
| small | organsmnist_224 | vit_t |  | 7 | 0.1 | 42 | reference_delta | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | bloodmnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 3 | 0.0 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | bloodmnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 3 | 0.01 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | bloodmnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 3 | 0.1 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | bloodmnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 5 | 0.0 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | bloodmnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 5 | 0.01 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | bloodmnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 5 | 0.1 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | bloodmnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 7 | 0.0 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | bloodmnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 7 | 0.01 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | bloodmnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 7 | 0.1 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | chaoshengmnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 3 | 0.0 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | chaoshengmnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 3 | 0.01 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | chaoshengmnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 3 | 0.1 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | chaoshengmnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 5 | 0.0 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | chaoshengmnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 5 | 0.01 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | chaoshengmnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 5 | 0.1 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | chaoshengmnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 7 | 0.0 | 42 | avg | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | chaoshengmnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 7 | 0.01 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | chaoshengmnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 7 | 0.1 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | dermamnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 3 | 0.0 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | dermamnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 3 | 0.01 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | dermamnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 3 | 0.1 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | dermamnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 5 | 0.0 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | dermamnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 5 | 0.01 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | dermamnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 5 | 0.1 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | dermamnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 7 | 0.0 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | dermamnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 7 | 0.01 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | dermamnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 7 | 0.1 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | organcmnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 3 | 0.0 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | organcmnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 3 | 0.01 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | organcmnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 3 | 0.1 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | organcmnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 5 | 0.0 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | organcmnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 5 | 0.01 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | organcmnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 5 | 0.1 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | organcmnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 7 | 0.0 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | organcmnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 7 | 0.01 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | organcmnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 7 | 0.1 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | organsmnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 3 | 0.0 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | organsmnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 3 | 0.01 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | organsmnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 3 | 0.1 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | organsmnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 5 | 0.0 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | organsmnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 5 | 0.01 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | organsmnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 5 | 0.1 | 42 | reference_delta | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | organsmnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 7 | 0.0 | 42 | reference_delta | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | organsmnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 7 | 0.01 | 42 | specialist_client | test |  | 0 |  | plot_limit_skipped |  |  |
| vlm | organsmnist_224 | clip-vit-base-patch32 | openai/clip-vit-base-patch32 | 7 | 0.1 | 42 | reference_delta | test |  | 0 |  | plot_limit_skipped |  |  |
