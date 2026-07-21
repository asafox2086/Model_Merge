# Proportional no-replacement strict-domain control

This experiment repeats the ResNet, 3-client strict-domain control with sampling **without replacement** in both domains.  The medical source is DermaMNIST-224 and the natural-image source is CIFAR-10; both are restricted to the same seven labels.

The original per-class split sizes were scaled by the largest common feasible factor, `43/61` (limited by CIFAR validation class 5), then rounded down independently for every split and class.  The resulting train counts are `[160, 253, 542, 56, 549, 3308, 69]`; validation counts are `[23, 36, 77, 8, 78, 473, 9]`; and test counts are `[46, 72, 155, 16, 157, 945, 20]`.

Client label sets remain `[3, 4, 0]`, `[2, 1]`, and `[6, 5]`, with respective training sizes `765`, `795`, and `3377`.  Both domains use the same ResNet initialization, optimizer settings, 50 local epochs, uniform three-client averaging, and t-SNE settings.  The source indices, split counts, scaling factor, and no-duplicate sampling checks are recorded in `proportional_noreplace_manifest.json`.

The previous `../tsne/` result used sampling with replacement and is retained only for traceability; this directory is the strict comparison without replacement.  The regenerated publication PNG/PDF figures and t-SNE coordinates are in `tsne/`.
