# LAMP-Merge Prototype Geometry Analysis

This report analyzes uploaded class-level prototype statistics on the formal full grid. The numerical rows aggregate five medical datasets, four backbones, three client counts, and three Dirichlet skew levels unless explicitly filtered.

Expected cases per mode: `180`.

## Dataset-Level Geometry

### bloodmnist_224

| mode | cases | pairwise dist | nearest-class dist | consistency | proto-client align | evidence entropy |
|---|---:|---:|---:|---:|---:|---:|
| LAMP-Merge | 36 | 0.0705 | 0.0324 | 0.9409 | 0.9746 | 0.2406 |
| M1 only | 36 | 0.0705 | 0.0324 | 0.9409 | 0.9746 | 0.2406 |
| avg+M2 | 36 | - | - | - | - | - |
| Classifier-head aggregation | 36 | 0.9617 | 0.8105 | 0.0008 | 0.5048 | 0.2406 |
| Global-feature mean | 36 | 0.0000 | -0.0000 | 1.0000 | 1.0000 | 0.2406 |
| Support-only synthetic head | 36 | 0.9874 | 0.9191 | 1.0000 | 1.0000 | 0.2406 |
| Shuffled-label prototype | 36 | 0.0705 | 0.0324 | 0.9409 | 0.9339 | 0.2406 |
| Uniform client weight | 36 | 0.0813 | 0.0414 | 0.9409 | 0.9810 | 0.4167 |
| Binary support only | 36 | 0.0813 | 0.0414 | 0.9409 | 0.9810 | 0.4167 |
| Global client-size weight | 36 | 0.0812 | 0.0405 | 0.9409 | 0.9787 | 0.3420 |
| Uniform prevalence prior | 36 | 0.0705 | 0.0324 | 0.9409 | 0.9746 | 0.2406 |
| Smoothed prevalence prior | 36 | 0.0705 | 0.0324 | 0.9409 | 0.9746 | 0.2406 |

### chaoshengmnist_224

| mode | cases | pairwise dist | nearest-class dist | consistency | proto-client align | evidence entropy |
|---|---:|---:|---:|---:|---:|---:|
| LAMP-Merge | 36 | 0.0447 | 0.0227 | 0.8847 | 0.9487 | 0.2549 |
| M1 only | 36 | 0.0447 | 0.0227 | 0.8847 | 0.9487 | 0.2549 |
| avg+M2 | 36 | - | - | - | - | - |
| Classifier-head aggregation | 36 | 0.9689 | 0.8470 | 0.0034 | 0.5274 | 0.2549 |
| Global-feature mean | 36 | -0.0000 | -0.0000 | 1.0000 | 1.0000 | 0.2549 |
| Support-only synthetic head | 36 | 0.9874 | 0.9191 | 1.0000 | 1.0000 | 0.2549 |
| Shuffled-label prototype | 36 | 0.0447 | 0.0227 | 0.8847 | 0.9093 | 0.2549 |
| Uniform client weight | 36 | 0.0693 | 0.0370 | 0.8847 | 0.9594 | 0.3750 |
| Binary support only | 36 | 0.0693 | 0.0370 | 0.8847 | 0.9594 | 0.3750 |
| Global client-size weight | 36 | 0.0614 | 0.0328 | 0.8847 | 0.9536 | 0.3069 |
| Uniform prevalence prior | 36 | 0.0447 | 0.0227 | 0.8847 | 0.9487 | 0.2549 |
| Smoothed prevalence prior | 36 | 0.0447 | 0.0227 | 0.8847 | 0.9487 | 0.2549 |

### dermamnist_224

| mode | cases | pairwise dist | nearest-class dist | consistency | proto-client align | evidence entropy |
|---|---:|---:|---:|---:|---:|---:|
| LAMP-Merge | 36 | 0.0550 | 0.0298 | 0.8575 | 0.9393 | 0.2822 |
| M1 only | 36 | 0.0550 | 0.0298 | 0.8575 | 0.9393 | 0.2822 |
| avg+M2 | 36 | - | - | - | - | - |
| Classifier-head aggregation | 36 | 0.9822 | 0.8101 | 0.0066 | 0.5379 | 0.2822 |
| Global-feature mean | 36 | -0.0000 | -0.0000 | 1.0000 | 1.0000 | 0.2822 |
| Support-only synthetic head | 36 | 0.9847 | 0.9175 | 1.0000 | 1.0000 | 0.2822 |
| Shuffled-label prototype | 36 | 0.0550 | 0.0298 | 0.8575 | 0.8846 | 0.2822 |
| Uniform client weight | 36 | 0.0862 | 0.0497 | 0.8575 | 0.9507 | 0.4127 |
| Binary support only | 36 | 0.0862 | 0.0497 | 0.8575 | 0.9507 | 0.4127 |
| Global client-size weight | 36 | 0.0965 | 0.0529 | 0.8575 | 0.9304 | 0.2528 |
| Uniform prevalence prior | 36 | 0.0550 | 0.0298 | 0.8575 | 0.9393 | 0.2822 |
| Smoothed prevalence prior | 36 | 0.0550 | 0.0298 | 0.8575 | 0.9393 | 0.2822 |

### organcmnist_224

| mode | cases | pairwise dist | nearest-class dist | consistency | proto-client align | evidence entropy |
|---|---:|---:|---:|---:|---:|---:|
| LAMP-Merge | 36 | 0.1964 | 0.0567 | 0.8467 | 0.9325 | 0.2505 |
| M1 only | 36 | 0.1964 | 0.0567 | 0.8467 | 0.9325 | 0.2505 |
| avg+M2 | 36 | - | - | - | - | - |
| Classifier-head aggregation | 36 | 0.9703 | 0.7823 | -0.0095 | 0.4904 | 0.2505 |
| Global-feature mean | 36 | 0.0000 | -0.0000 | 1.0000 | 1.0000 | 0.2505 |
| Support-only synthetic head | 36 | 0.9879 | 0.9074 | 1.0000 | 1.0000 | 0.2505 |
| Shuffled-label prototype | 36 | 0.1964 | 0.0567 | 0.8467 | 0.7795 | 0.2505 |
| Uniform client weight | 36 | 0.2229 | 0.0832 | 0.8467 | 0.9482 | 0.4343 |
| Binary support only | 36 | 0.2229 | 0.0832 | 0.8467 | 0.9482 | 0.4343 |
| Global client-size weight | 36 | 0.2204 | 0.0801 | 0.8467 | 0.9466 | 0.4080 |
| Uniform prevalence prior | 36 | 0.1964 | 0.0567 | 0.8467 | 0.9325 | 0.2505 |
| Smoothed prevalence prior | 36 | 0.1964 | 0.0567 | 0.8467 | 0.9325 | 0.2505 |

### organsmnist_224

| mode | cases | pairwise dist | nearest-class dist | consistency | proto-client align | evidence entropy |
|---|---:|---:|---:|---:|---:|---:|
| LAMP-Merge | 36 | 0.1944 | 0.0387 | 0.8459 | 0.9280 | 0.2244 |
| M1 only | 36 | 0.1944 | 0.0387 | 0.8459 | 0.9280 | 0.2244 |
| avg+M2 | 36 | - | - | - | - | - |
| Classifier-head aggregation | 36 | 0.9840 | 0.7707 | -0.0117 | 0.4789 | 0.2244 |
| Global-feature mean | 36 | 0.0000 | -0.0000 | 1.0000 | 1.0000 | 0.2244 |
| Support-only synthetic head | 36 | 0.9879 | 0.9074 | 1.0000 | 1.0000 | 0.2244 |
| Shuffled-label prototype | 36 | 0.1944 | 0.0387 | 0.8459 | 0.7698 | 0.2244 |
| Uniform client weight | 36 | 0.2229 | 0.0671 | 0.8459 | 0.9462 | 0.4242 |
| Binary support only | 36 | 0.2229 | 0.0671 | 0.8459 | 0.9462 | 0.4242 |
| Global client-size weight | 36 | 0.2251 | 0.0691 | 0.8459 | 0.9416 | 0.3899 |
| Uniform prevalence prior | 36 | 0.1944 | 0.0387 | 0.8459 | 0.9280 | 0.2244 |
| Smoothed prevalence prior | 36 | 0.1944 | 0.0387 | 0.8459 | 0.9280 | 0.2244 |

## Interpretation

Prototype separation measures whether class directions occupy distinct positions in the shared reference feature space. Prototype consistency measures whether clients that contain the same diagnosis class agree on its reference-space direction. Evidence entropy measures whether the server assigns class-specific evidence to several clients or concentrates it on one reliable client. These quantities are diagnostic analyses of the uploaded aggregate statistics; they do not require the server to read raw client images.

Rows without a prototype head, such as avg+M2, are retained for scope consistency but have no prototype geometry values. Prevalence-only variants share the same prototype geometry as LAMP-Merge because they modify the score bias rather than the class prototype construction.
