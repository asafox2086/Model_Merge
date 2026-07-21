# Balanced GT feature t-SNE

This diagnostic visualizes the classifier-input features of the uniform three-client AVG model from the no-replacement DermaMNIST/CIFAR control.  Points are colored by ground-truth class, not predicted class.

It uses only the test split and selects up to 16 distinct images per class without replacement, because class 3 has only 16 test samples.  Each domain therefore contributes 112 points, with exactly 16 points for every class.  The embedding answers whether pre-classifier features retain local class structure after class-balanced sampling; it is not a visualization of the collapsed output predictions.

The matching medical AVG model predicts all 1,411 test images as class 5.  That prediction collapse occurs after these features enter the merged classifier head, so it should be assessed with predicted-label distributions or output-logit analyses rather than inferred from this figure alone.
