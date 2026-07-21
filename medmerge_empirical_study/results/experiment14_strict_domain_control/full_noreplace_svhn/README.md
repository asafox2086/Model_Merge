# Full no-replacement DermaMNIST/SVHN strict-domain control

This is the full-size strict comparison between DermaMNIST-224 medical images and natural SVHN images.  Both datasets contain seven remapped classes and use sampling **without replacement**.

The original strict-control class counts are retained exactly: train `[228, 359, 769, 80, 779, 4693, 99]`, validation `[33, 52, 110, 12, 111, 671, 14]`, and test `[66, 103, 220, 23, 223, 1341, 29]`.  SVHN source digits are mapped to target classes as `[0, 2, 3, 4, 5, 1, 6]`, placing the largest target class on the sufficiently large SVHN digit-1 pool.

Client label sets and train counts remain `[3, 4, 0]` / `[80, 779, 228]`, `[2, 1]` / `[769, 359]`, and `[6, 5]` / `[99, 4693]`, for client totals `1087`, `1128`, and `4792`.  Both domains use the same pretrained ResNet initialization, 50 local epochs, batch size 64, learning rate 0.001, weight decay 0.0001, and uniform three-client averaging.

The t-SNE visualization uses all three splits (`7007 + 1003 + 2005 = 10015` samples per domain) with identical feature extraction and t-SNE settings.  The manifest records all source indices and no-duplicate checks; figures and coordinates are in `tsne/`.
