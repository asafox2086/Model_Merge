# Matched-Domain Feature t-SNE: Ground Truth and AVG Prediction

- [Four-panel figure](strict_domain_resnet_k3_feature_prediction_tsne.png)
- [Vector PDF](strict_domain_resnet_k3_feature_prediction_tsne.pdf)
- [Per-sample coordinates, GT, and predictions](strict_domain_resnet_k3_feature_prediction_tsne_coordinates.csv)
- [Settings, control checks, and checkpoint hashes](strict_domain_resnet_k3_feature_prediction_tsne_summary.json)

Each domain uses its complete 2,005-image test split. The left and right panel within a domain reuse exactly the same classifier-input feature t-SNE coordinates: the left panel is colored by ground truth and the right panel by the uniform three-client AVG prediction. The medical and natural t-SNE fits are separate, so their two-dimensional axes must not be compared across domains.

## Reading the Figure

- DermaMNIST GT colors are broadly mixed in panel (a), and the corresponding panel (b) is entirely Class 5: all 2,005 test images receive the same AVG prediction.
- CIFAR has more localized GT regions in panel (c). In its matched prediction view (d), 1,818 images predict Class 5 and 187 predict Class 4; the Class 4 predictions occupy a localized lower-right region.
- This is a visualization of this matched Derma/CIFAR experiment, not a universal claim about every medical or natural image dataset. It shows that the stronger medical prediction collapse occurs despite non-identical penultimate features; t-SNE alone does not establish a causal mechanism.

## Controlled Setup

| Variable | Value |
| --- | --- |
| Medical/natural data | DermaMNIST dermoscopy / matched seven-class CIFAR-10 subset |
| Test images | 2,005 in each domain; class counts `[66, 103, 220, 23, 223, 1341, 29]` |
| Model | pretrained ResNet-18, three clients, 50 local epochs |
| Client label partitions | `0: [3, 4, 0]`, `1: [2, 1]`, `2: [6, 5]` |
| Merge | uniform average of the three client checkpoints |
| Feature and t-SNE | 512D classifier input; PCA initialization; perplexity 30; seed 1701 |

The natural subset matches the original strict-control counts and partitions, but its Class 5 training examples include sampling with replacement because CIFAR-10 has fewer than 4,693 images in one class. This is the same controlled pair used by the associated output diagnostics.

## Reproduction

```bash
python scripts/plot_strict_domain_prediction_tsne.py \
  --medical-meta /path/to/model_hub/small/dermamnist_224/resnet/clients_3/beta_0/seed_42/meta.json \
  --medical-hub-dir /path/to/model_hub/small/dermamnist_224/resnet/clients_3/beta_0/seed_42 \
  --medical-data-root /path/to/Med_data \
  --natural-meta /path/to/natural_probe_hub/small/cifar7_derma_strict/resnet/clients_3/partial_label/seed_42/meta.json \
  --natural-hub-dir /path/to/natural_probe_hub/small/cifar7_derma_strict/resnet/clients_3/partial_label/seed_42 \
  --natural-data-root /path/to/Med_data \
  --output-dir medmerge_empirical_study/results/experiment14_strict_domain_control/tsne_prediction_overlay \
  --device cuda:0
```
