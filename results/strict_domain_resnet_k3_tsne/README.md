# Strict Medical/Natural Domain Control t-SNE

This result compares ResNet-18 feature geometry for a medical image task and a natural-image task under a strictly matched three-client setting. Each figure uses the complete 2,005-image test split and colors points by true class.

- [Medical DermaMNIST t-SNE](medical_dermamnist_resnet_k3_avg_feature_tsne.png)
- [Natural CIFAR t-SNE](natural_cifar_resnet_k3_avg_feature_tsne.png)
- [Coordinates](strict_domain_resnet_k3_avg_feature_tsne_coordinates.csv)
- [Machine-readable settings and input checksums](strict_domain_resnet_k3_avg_feature_tsne_summary.json)

## Controlled Variables

| Variable | Value |
| --- | --- |
| Dataset domain | DermaMNIST medical dermoscopy / CIFAR-10 natural images |
| Classes | 7 |
| Clients | 3 |
| Client classes | `0: [3,4,0]`, `1: [2,1]`, `2: [6,5]` |
| Client class counts | `0: 3:80,4:779,0:228`; `1: 2:769,1:359`; `2: 6:99,5:4693` |
| Train/validation/test class counts | `[228,359,769,80,779,4693,99]` / `[33,52,110,12,111,671,14]` / `[66,103,220,23,223,1341,29]` |
| Backbone and training | pretrained ResNet-18, 224px, 50 epochs, batch size 64, learning rate 0.001, weight decay 0.0001 |
| Merge | uniform weight averaging across the three client checkpoints |
| t-SNE | classifier-input features (512D), PCA initialization, perplexity 30, seed 1701 |

The natural-image dataset is `cifar7_derma_strict`, a seven-class CIFAR-10 subset constructed to match every split and client count above. Its class 5 requires sampling with replacement because CIFAR-10 does not contain 4,693 distinct train images for a single class. This is a dataset-construction limitation; the image domain is the intended experimental difference.

The two t-SNE embeddings are fitted separately, so their axes are not shared coordinates. They use identical extraction and t-SNE settings, and are intended to show within-domain class geometry.

## Reproduction

The released repository does not bundle the private training checkpoints or the large `.npz` files. With the paired assets available, rerun:

```bash
python scripts/plot_strict_domain_tsne.py \
  --medical-meta /path/to/model_hub/small/dermamnist_224/resnet/clients_3/beta_0/seed_42/meta.json \
  --medical-hub-dir /path/to/model_hub/small/dermamnist_224/resnet/clients_3/beta_0/seed_42 \
  --medical-data-root /path/to/Med_data \
  --natural-meta /path/to/natural_probe_hub/small/cifar7_derma_strict/resnet/clients_3/partial_label/seed_42/meta.json \
  --natural-hub-dir /path/to/natural_probe_hub/small/cifar7_derma_strict/resnet/clients_3/partial_label/seed_42 \
  --natural-data-root /path/to/Med_data \
  --output-dir results/strict_domain_resnet_k3_tsne \
  --device cuda:0
```

The script validates all controlled variables before it loads models or writes results.
