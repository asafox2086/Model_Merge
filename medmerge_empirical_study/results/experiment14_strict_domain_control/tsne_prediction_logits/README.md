# Prediction-Colored Centered-Logit t-SNE

These figures show only the AVG-model prediction view. For each image, the seven- or eight-dimensional logit vector is centered by subtracting its mean before t-SNE; this removes the common score offset while retaining every pairwise class margin and the predicted class. Unlike the earlier feature t-SNE, this is a view of the final decision geometry.

## Figures

- [DermaMNIST / CIFAR](derma_cifar/strict_domain_resnet_k3_centered_logit_prediction_tsne.png): Derma predicts Class 5 for all 2,005 test images. CIFAR predicts Class 5 for 1,818 images and Class 4 for 187 images.
- [BloodMNIST / SVHN](blood_svhn/strict_domain_resnet_k3_centered_logit_prediction_tsne.png): Blood predicts Class 3 for 3,256 of 3,421 test images. SVHN predicts Classes 0, 3, 5, and 6, with Class 6 dominant (2,586 images).

Each panel is fitted separately, so coordinates are only interpretable within a panel. Colors represent predicted, rather than ground-truth, classes. PNG, PDF, per-sample coordinates, and a machine-readable settings summary sit beside each figure.

## Interpretation Boundary

The Blood/SVHN comparison gives the clearer contrast: both domains have dominant-class bias after averaging, but Blood has two predicted classes with a 95.1% dominant class while SVHN retains four predicted classes with a 75.6% dominant class. The Derma/CIFAR pair has a weaker contrast because CIFAR is also heavily biased. These plots visualize that observed outcome; they do not by themselves establish why a domain collapses.

The Blood/SVHN pair uses eight classes and matching no-replacement samples. The Derma/CIFAR pair reuses the original strict-control assets, whose CIFAR Class 5 training set includes replacement because its target count exceeds the source-class population.

## Reproduction

Use [the plotting script](../../../../scripts/plot_strict_domain_logit_tsne.py) with the paired metadata, checkpoint directories, and `.npz` data roots. The JSON files beside each figure record the exact command-relevant configuration, source checkpoint hashes, sample counts, prediction counts, and t-SNE seed.
