# Joint medical/natural domain t-SNE diagnostics

This dashboard uses all existing samples from the no-replacement DermaMNIST/CIFAR control: 7,052 medical and 7,052 natural images from the train, validation, and test splits.  No image is removed, duplicated, or synthesized.

Every panel fits one t-SNE jointly over both domains, so the two colors share a coordinate system inside that panel.  Blue points are medical images and orange points are natural images.

* **(a)** Both domains encoded by the medical AVG model before its classifier head.
* **(b)** Both domains encoded by the natural AVG model before its classifier head.
* **(c)** Logits from each domain's own AVG model, concatenated in the shared seven-class output coordinate system.

All three variants show a strong domain partition under the matched client, class-count, optimizer, initialization, and merge settings.  The feature panels demonstrate domain separation inside a common encoder space; the logit panel demonstrates that the two models occupy different regions of the shared output space.  This is visual evidence of domain shift, not a quantitative distance estimate or a causal proof by itself.
