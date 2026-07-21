# Full no-replacement BloodMNIST/SVHN strict-domain control

This experiment compares all eight BloodMNIST classes with eight SVHN digit classes under a matched, no-replacement three-client setting.  BloodMNIST is used in full; SVHN classes `0` through `7` are sampled without replacement to match every BloodMNIST class count in every split.

The shared class counts are train `[852, 2181, 1085, 2026, 849, 993, 2330, 1643]`, validation `[122, 312, 155, 290, 122, 143, 333, 235]`, and test `[244, 624, 311, 579, 243, 284, 666, 470]`.  The three clients see classes `[3, 4, 0]`, `[2, 1, 7]`, and `[6, 5]`, with training sizes `3727`, `4909`, and `3323`.

Both domains use a pretrained ResNet, 50 local epochs, batch size 64, learning rate 0.001, weight decay 0.0001, and uniform averaging across the three client checkpoints.  The t-SNE plot uses all three splits (`11959 + 1712 + 3421 = 17092` samples per domain), with identical feature extraction and embedding settings.

The source-index manifest records the no-duplicate sampling checks.  Figures and coordinates are in `tsne/`.
