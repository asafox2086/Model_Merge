# LAMP-Merge v2 Revision Workflow

## 2026-07-15 Four-Backbone Full-Scope Synchronization

This revision uses the completed interaction sensitivity scan and the existing full-scope reports as the only numerical sources. The formal scope contains five medical datasets, four visual backbones, three client counts, and three Dirichlet skew levels: 180 raw cells and 60 client-average cells per complete setting.

1. [x] Verify that the interaction scan contains all 50 parameter points and 9,000 raw evaluations.
2. [x] Repair the sensitivity-plot mathematical labels and regenerate the 5x5 M1 and M2 curves.
3. [x] Regenerate the module-ablation, predicted-distribution, and prototype-geometry figures from four-backbone full-scope reports.
4. [x] Replace the paper figures and update all numerical diagnostics, scope descriptions, and hyperparameter conclusions to the same four-backbone protocol.
5. [x] Compile and inspect the PDF, then commit and push the synchronized manuscript and analysis artifacts.

## 2026-07-15 Denser Hyperparameter Curves

The formal method and its default settings remain fixed. To make the sensitivity curves smoother without replacing any existing measurement, the analysis adds five interleaved horizontal samples for every fixed-value curve. The existing and added runs together yield ten points per curve.

1. [x] Retain the completed 5x5 scans for both modules as the first five horizontal samples.
2. [x] Add a separate 5x5 continuation script with $s\in\{13.75,16.25,18.75,21.25,23.75\}$ at every fixed $\gamma$, and $\lambda\in\{3.75,4.25,4.75,5.25,5.75\}$ at every fixed $\tau$.
3. [ ] Run all 50 added full-scope points: 180 raw cells and 60 client-average cells per point.
4. [ ] Merge the prior and continuation results, regenerate the ten-point-per-curve plot, update the manuscript, compile, inspect, and push.
