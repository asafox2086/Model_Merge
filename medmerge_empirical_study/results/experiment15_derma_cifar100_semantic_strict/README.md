# Strict DermaMNIST / CIFAR-100 Semantic Seven-Class Control

This experiment is a no-replacement control for comparing prediction collapse between a medical-image task and a natural-image task. It uses a pretrained ResNet with `K=3` partial-label clients and uniform checkpoint averaging.

## Result

The AVG model predicts a single class for every DermaMNIST test image, whereas the natural-image control makes predictions in four classes.

| Domain | Most frequent prediction | Collapse ratio | Nonzero predicted classes | AVG test accuracy |
| --- | --- | --- | --- | --- |
| Medical DermaMNIST | Class 5: 2,005 / 2,005 | 1.000 | 1 | 0.668828 |
| Natural CIFAR-100 semantic 7-class control | Class 4: 1,588 / 2,005 | 0.792020 | 4 | 0.212469 |

The separate figures are deliberately not combined because their prediction distributions differ:

- [Medical prediction distribution](diagnostics/dermamnist_avg_prediction_distribution.png)
- [Natural prediction distribution](diagnostics/cifar100_semantic7_avg_prediction_distribution.png)
- [Medical prediction-colored centered-logit t-SNE](diagnostics/dermamnist_avg_centered_logit_prediction_tsne.png)
- [Natural prediction-colored centered-logit t-SNE](diagnostics/cifar100_semantic7_avg_centered_logit_prediction_tsne.png)

The bar charts are the primary evidence for output/prediction collapse. The t-SNE plots embed centered logits and color points by the predicted class: all medical points are predicted as Class 5, while the natural control has predictions in Classes 0, 2, 4, and 5. Neither plot is evidence of classifier-feature collapse; a feature t-SNE would be needed for that different claim.

## Strict Matching

Both domains use exactly the same seven target labels, split-level supports, client labels, client samples, architecture, initialization, optimizer, epochs, image size, seed, and merge rule.

| Split | Per-class counts (Class 0 through Class 6) | Total |
| --- | --- | --- |
| Train | `[228, 359, 769, 80, 779, 4693, 99]` | 7,007 |
| Validation | `[33, 52, 110, 12, 111, 671, 14]` | 1,003 |
| Test | `[66, 103, 220, 23, 223, 1341, 29]` | 2,005 |

| Client | Visible target labels | Per-class train counts | Total |
| --- | --- | --- | --- |
| 0 | `[3, 4, 0]` | `[80, 779, 228]` | 1,087 |
| 1 | `[2, 1]` | `[769, 359]` | 1,128 |
| 2 | `[6, 5]` | `[99, 4693]` | 4,792 |

Shared training configuration: pretrained ResNet, `K=3`, 7 output classes, 50 epochs, batch size 64, Adam learning rate `1e-3`, weight decay `1e-4`, seed 42, image size 224, mixed precision, and uniform averaging.

The natural control is sampled without replacement from CIFAR-100. Its target-class groups are disjoint: `bowl=[10]`, `trees=[47,52]`, `fruit=[0,53,57]`, `house=[37]`, `vehicles=[8,13,48]`, `animals=[1,3,4,6,7,14,15,18,19,21,24,26,27,30]`, and `plain=[60]`. The 14 fine labels in target Class 5 are required to supply the 4,693 training images present in DermaMNIST Class 5. The exact sampled source indices, counts, split seed, and no-replacement declaration are in [the manifest](cifar100_semantic_strict_manifest.json); the dataset can be rebuilt with [`build_derma_cifar100_semantic_strict_data.py`](../../scripts/build_derma_cifar100_semantic_strict_data.py).

This controls label counts and client exposure exactly, but the two domains are not semantically identical: most importantly, natural target Class 5 pools multiple CIFAR-100 fine labels. Therefore this is evidence for a sharply different outcome under a matched client-label protocol, not proof that the medical versus natural domain alone causes the effect.

## Reproducibility

The training metrics, per-client checkpoints, predictions, plotted source CSVs, t-SNE coordinates, and plot settings are retained in this directory. The plot script uses the project `plot_templates` style layer and exports 350-DPI PNG plus vector PDF.

```bash
/data2/liyapeng_grp/.conda/envs/MM/bin/python scripts/plot_derma_cifar100_semantic_strict.py \
  --medical-metrics medmerge_empirical_study/results/experiment15_derma_cifar100_semantic_strict/medical/natural_collapse_metrics.csv \
  --natural-metrics medmerge_empirical_study/results/experiment15_derma_cifar100_semantic_strict/natural/natural_collapse_metrics.csv \
  --medical-meta medmerge_empirical_study/results/experiment15_derma_cifar100_semantic_strict/medical/natural_probe_hub/small/dermamnist_224/resnet/clients_3/partial_label/seed_42/meta.json \
  --natural-meta medmerge_empirical_study/results/experiment15_derma_cifar100_semantic_strict/natural/natural_probe_hub/small/cifar100_semantic7_derma_strict/resnet/clients_3/partial_label/seed_42/meta.json \
  --medical-hub-dir medmerge_empirical_study/results/experiment15_derma_cifar100_semantic_strict/medical/natural_probe_hub/small/dermamnist_224/resnet/clients_3/partial_label/seed_42 \
  --natural-hub-dir medmerge_empirical_study/results/experiment15_derma_cifar100_semantic_strict/natural/natural_probe_hub/small/cifar100_semantic7_derma_strict/resnet/clients_3/partial_label/seed_42 \
  --medical-data-root /data2/liyapeng_grp/program/MedMNISTMerge/Med_data \
  --natural-data-root /data2/liyapeng_grp/program/MedMNISTMerge/Med_data \
  --output-dir medmerge_empirical_study/results/experiment15_derma_cifar100_semantic_strict/diagnostics \
  --device cuda:0 --batch-size 128 --seed 1701 --perplexity 50
```
