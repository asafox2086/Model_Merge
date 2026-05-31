# Source Image Analysis

This report is generated from NPZ source images, not from model predictions.

## Files

- metrics_csv: `/data/liyapeng_grp/program/MedMNISTMerge/My_merge_ret/reports/source_image_analysis/image_source_metrics.csv`
- summary_json: `/data/liyapeng_grp/program/MedMNISTMerge/My_merge_ret/reports/source_image_analysis/image_source_summary.json`
- sample_montage: `/data/liyapeng_grp/program/MedMNISTMerge/My_merge_ret/figures/source_image_analysis/source_samples.png`
- metric_bars: `/data/liyapeng_grp/program/MedMNISTMerge/My_merge_ret/figures/source_image_analysis/source_metric_bars.png`
- chaosheng_evidence_maps: `/data/liyapeng_grp/program/MedMNISTMerge/My_merge_ret/figures/source_image_analysis/chaosheng_evidence_maps.png`
- chaosheng_class_samples: `/data/liyapeng_grp/program/MedMNISTMerge/My_merge_ret/figures/source_image_analysis/chaosheng_class_samples.png`

## Dataset Structure

| dataset | train shape | val shape | test shape | classes | analyzed samples |
| --- | --- | --- | --- | ---: | ---: |
| `bloodmnist_224` | `[11959, 224, 224, 3]` | `[1712, 224, 224, 3]` | `[3421, 224, 224, 3]` | 8 | 480 |
| `dermamnist_224` | `[7007, 224, 224, 3]` | `[1003, 224, 224, 3]` | `[2005, 224, 224, 3]` | 7 | 480 |
| `organcmnist_224` | `[12975, 224, 224]` | `[2392, 224, 224]` | `[8216, 224, 224]` | 11 | 480 |
| `organsmnist_224` | `[13932, 224, 224]` | `[2452, 224, 224]` | `[8827, 224, 224]` | 11 | 480 |
| `chaoshengmnist_224` | `[3869, 224, 224, 3]` | `[549, 224, 224, 3]` | `[1113, 224, 224, 3]` | 8 | 480 |

## Metric Summary

| dataset | RGB channel difference | dark pixel fraction | border evidence fraction | p95-p05 dynamic range | Sobel edge mean | residual noise | speckle ratio | edge-noise correlation | M1 evidence reliability | salience between-sample CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bloodmnist_224` | 0.1455 | 0.0000 | 0.4288 | 0.5464 | 0.2478 | 0.0182 | 0.0493 | 0.7504 | 0.7255 | 0.1715 |
| `dermamnist_224` | 0.1605 | 0.0000 | 0.4920 | 0.3524 | 0.2254 | 0.0269 | 0.0809 | 0.6911 | 0.5679 | 0.2456 |
| `organcmnist_224` | 0.0000 | 0.1213 | 0.4168 | 0.6764 | 0.2680 | 0.0197 | 0.0479 | 0.6999 | 0.6799 | 0.3820 |
| `organsmnist_224` | 0.0000 | 0.1088 | 0.4163 | 0.6571 | 0.2632 | 0.0186 | 0.0464 | 0.6965 | 0.6584 | 0.3864 |
| `chaoshengmnist_224` | 0.0099 | 0.4271 | 0.2340 | 0.5626 | 0.2482 | 0.0356 | 0.0987 | 0.7396 | 0.6405 | 0.2604 |

## Main Findings

- `chaoshengmnist_224` is stored as RGB but has near-zero channel difference (0.0099); it is effectively grayscale replicated into 3 channels.
- Its edge-noise correlation is 0.7396, compared with other-dataset average 0.7095. High correlation means Sobel edges are strongly coupled with high-frequency residuals.
- Its speckle ratio is 0.0987, compared with other-dataset average 0.0561. This makes an edge-heavy M1 feature likely to score acquisition noise as diagnostic evidence.
- Its dark-pixel fraction is 0.4271, and border evidence fraction is 0.2340. Ultrasound frames and machine overlays can therefore receive non-trivial evidence mass.
- Its M1 evidence reliability is 0.6405, compared with other-dataset average 0.6579. Lower reliability should gate client-specific weighting more aggressively.
- Its salience between-sample CV is 0.2604. If this is high while evidence is noisy, M1 can over-amplify arbitrary ultrasound texture differences between clients.

## Optimization Direction

1. Treat ultrasound as a low-reliability evidence domain unless the source-image reliability score is high.
2. Reduce the role of Sobel edge strength when edge-noise correlation or speckle ratio is high.
3. Use structure-preserving smoothing before evidence extraction; median/SRAD-like despeckling is more appropriate than raw edge maps.
4. Prefer robust texture and coarse ROI statistics over hard foreground masks for ultrasound.
5. In M2, do not anchor strongly to one client when M1 evidence reliability is low; keep routed weights close to average/consensus.

## Suggested Code-Level Rule

For each dataset or validation split, compute `speckle_ratio`, `edge_residual_corr`, and `evidence_reliability` from source images. If `speckle_ratio` and `edge_residual_corr` are high, apply an ultrasound-safe profile: lower edge weight, higher smoothing, lower M1 gate, and no hard anchor.
