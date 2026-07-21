# Full GT feature t-SNE

This visualization uses every existing sample from the no-replacement DermaMNIST/CIFAR control: all train, validation, and test splits, for 7,052 samples per domain.  It does not balance, duplicate, remove, or synthesize samples.

Points are classifier-input features from the uniform three-client AVG model and are colored by ground-truth class.  The medical model is the same AVG model whose 1,411 test predictions all select class 5.  This figure visualizes pre-classifier feature geometry, so it must not be interpreted as a direct visualization of output-prediction collapse.

The t-SNE settings are PCA initialization, perplexity 30, and seed 1701.  Coordinates and machine-readable settings are stored beside the PNG/PDF figures.
