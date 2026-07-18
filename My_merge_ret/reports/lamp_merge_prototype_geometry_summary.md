# LAMP-Merge Prototype Geometry Analysis

This report analyzes uploaded class-level prototype statistics on the formal full grid. The numerical rows aggregate five medical datasets, four backbones, three client counts, and three Dirichlet skew levels unless explicitly filtered.

Expected cases per mode: `180`.

## Overall Geometry

| mode | cases | pairwise dist | nearest-class dist | consistency | proto-client align | evidence entropy |
|---|---:|---:|---:|---:|---:|---:|
| LAMP-Merge | 180 | 0.1115 | 0.0355 | 0.8751 | 0.9419 | 0.2124 |
| M1 only | 180 | 0.1115 | 0.0355 | 0.8751 | 0.9419 | 0.2124 |
| Classifier-head aggregation | 180 | 0.9733 | 0.8043 | -0.0021 | 0.4910 | 0.2124 |
| Global-feature mean | 180 | 0.0000 | -0.0000 | 1.0000 | 1.0000 | 0.2124 |
| Support-only synthetic head | 180 | 0.9870 | 0.9141 | 1.0000 | 1.0000 | 0.2124 |
| Shuffled-label prototype | 180 | 0.1115 | 0.0355 | 0.8751 | 0.8553 | 0.2124 |
| Uniform client weight | 180 | 0.1365 | 0.0557 | 0.8751 | 0.9571 | 0.4126 |
| Binary support only | 180 | 0.1365 | 0.0557 | 0.8751 | 0.9571 | 0.4126 |
| Global client-size weight | 180 | 0.1369 | 0.0551 | 0.8751 | 0.9502 | 0.3399 |
| Uniform prevalence prior | 180 | 0.1115 | 0.0355 | 0.8751 | 0.9419 | 0.2124 |
| Smoothed prevalence prior | 180 | 0.1115 | 0.0355 | 0.8751 | 0.9419 | 0.2124 |

## Dataset-Level Geometry

### bloodmnist_224

| mode | cases | pairwise dist | nearest-class dist | consistency | proto-client align | evidence entropy |
|---|---:|---:|---:|---:|---:|---:|
| LAMP-Merge | 36 | 0.0703 | 0.0323 | 0.9409 | 0.9736 | 0.2031 |
| M1 only | 36 | 0.0703 | 0.0323 | 0.9409 | 0.9736 | 0.2031 |
| Classifier-head aggregation | 36 | 0.9613 | 0.8106 | 0.0008 | 0.4881 | 0.2031 |
| Global-feature mean | 36 | 0.0000 | -0.0000 | 1.0000 | 1.0000 | 0.2031 |
| Support-only synthetic head | 36 | 0.9874 | 0.9191 | 1.0000 | 1.0000 | 0.2031 |
| Shuffled-label prototype | 36 | 0.0703 | 0.0323 | 0.9409 | 0.9337 | 0.2031 |
| Uniform client weight | 36 | 0.0813 | 0.0414 | 0.9409 | 0.9810 | 0.4167 |
| Binary support only | 36 | 0.0813 | 0.0414 | 0.9409 | 0.9810 | 0.4167 |
| Global client-size weight | 36 | 0.0812 | 0.0405 | 0.9409 | 0.9787 | 0.3420 |
| Uniform prevalence prior | 36 | 0.0703 | 0.0323 | 0.9409 | 0.9736 | 0.2031 |
| Smoothed prevalence prior | 36 | 0.0703 | 0.0323 | 0.9409 | 0.9736 | 0.2031 |

### chaoshengmnist_224

| mode | cases | pairwise dist | nearest-class dist | consistency | proto-client align | evidence entropy |
|---|---:|---:|---:|---:|---:|---:|
| LAMP-Merge | 36 | 0.0439 | 0.0222 | 0.8847 | 0.9460 | 0.2235 |
| M1 only | 36 | 0.0439 | 0.0222 | 0.8847 | 0.9460 | 0.2235 |
| Classifier-head aggregation | 36 | 0.9685 | 0.8465 | 0.0034 | 0.5110 | 0.2235 |
| Global-feature mean | 36 | -0.0000 | -0.0000 | 1.0000 | 1.0000 | 0.2235 |
| Support-only synthetic head | 36 | 0.9874 | 0.9191 | 1.0000 | 1.0000 | 0.2235 |
| Shuffled-label prototype | 36 | 0.0439 | 0.0222 | 0.8847 | 0.9087 | 0.2235 |
| Uniform client weight | 36 | 0.0693 | 0.0370 | 0.8847 | 0.9594 | 0.3750 |
| Binary support only | 36 | 0.0693 | 0.0370 | 0.8847 | 0.9594 | 0.3750 |
| Global client-size weight | 36 | 0.0614 | 0.0328 | 0.8847 | 0.9536 | 0.3069 |
| Uniform prevalence prior | 36 | 0.0439 | 0.0222 | 0.8847 | 0.9460 | 0.2235 |
| Smoothed prevalence prior | 36 | 0.0439 | 0.0222 | 0.8847 | 0.9460 | 0.2235 |

### dermamnist_224

| mode | cases | pairwise dist | nearest-class dist | consistency | proto-client align | evidence entropy |
|---|---:|---:|---:|---:|---:|---:|
| LAMP-Merge | 36 | 0.0536 | 0.0289 | 0.8575 | 0.9360 | 0.2455 |
| M1 only | 36 | 0.0536 | 0.0289 | 0.8575 | 0.9360 | 0.2455 |
| Classifier-head aggregation | 36 | 0.9819 | 0.8089 | 0.0066 | 0.5209 | 0.2455 |
| Global-feature mean | 36 | -0.0000 | -0.0000 | 1.0000 | 1.0000 | 0.2455 |
| Support-only synthetic head | 36 | 0.9847 | 0.9175 | 1.0000 | 1.0000 | 0.2455 |
| Shuffled-label prototype | 36 | 0.0536 | 0.0289 | 0.8575 | 0.8849 | 0.2455 |
| Uniform client weight | 36 | 0.0862 | 0.0497 | 0.8575 | 0.9507 | 0.4127 |
| Binary support only | 36 | 0.0862 | 0.0497 | 0.8575 | 0.9507 | 0.4127 |
| Global client-size weight | 36 | 0.0965 | 0.0529 | 0.8575 | 0.9304 | 0.2528 |
| Uniform prevalence prior | 36 | 0.0536 | 0.0289 | 0.8575 | 0.9360 | 0.2455 |
| Smoothed prevalence prior | 36 | 0.0536 | 0.0289 | 0.8575 | 0.9360 | 0.2455 |

### organcmnist_224

| mode | cases | pairwise dist | nearest-class dist | consistency | proto-client align | evidence entropy |
|---|---:|---:|---:|---:|---:|---:|
| LAMP-Merge | 36 | 0.1958 | 0.0561 | 0.8467 | 0.9292 | 0.2082 |
| M1 only | 36 | 0.1958 | 0.0561 | 0.8467 | 0.9292 | 0.2082 |
| Classifier-head aggregation | 36 | 0.9704 | 0.7832 | -0.0095 | 0.4721 | 0.2082 |
| Global-feature mean | 36 | 0.0000 | -0.0000 | 1.0000 | 1.0000 | 0.2082 |
| Support-only synthetic head | 36 | 0.9879 | 0.9074 | 1.0000 | 1.0000 | 0.2082 |
| Shuffled-label prototype | 36 | 0.1958 | 0.0561 | 0.8467 | 0.7795 | 0.2082 |
| Uniform client weight | 36 | 0.2229 | 0.0832 | 0.8467 | 0.9482 | 0.4343 |
| Binary support only | 36 | 0.2229 | 0.0832 | 0.8467 | 0.9482 | 0.4343 |
| Global client-size weight | 36 | 0.2204 | 0.0801 | 0.8467 | 0.9466 | 0.4080 |
| Uniform prevalence prior | 36 | 0.1958 | 0.0561 | 0.8467 | 0.9292 | 0.2082 |
| Smoothed prevalence prior | 36 | 0.1958 | 0.0561 | 0.8467 | 0.9292 | 0.2082 |

### organsmnist_224

| mode | cases | pairwise dist | nearest-class dist | consistency | proto-client align | evidence entropy |
|---|---:|---:|---:|---:|---:|---:|
| LAMP-Merge | 36 | 0.1939 | 0.0380 | 0.8459 | 0.9245 | 0.1819 |
| M1 only | 36 | 0.1939 | 0.0380 | 0.8459 | 0.9245 | 0.1819 |
| Classifier-head aggregation | 36 | 0.9845 | 0.7722 | -0.0117 | 0.4631 | 0.1819 |
| Global-feature mean | 36 | 0.0000 | -0.0000 | 1.0000 | 1.0000 | 0.1819 |
| Support-only synthetic head | 36 | 0.9879 | 0.9074 | 1.0000 | 1.0000 | 0.1819 |
| Shuffled-label prototype | 36 | 0.1939 | 0.0380 | 0.8459 | 0.7699 | 0.1819 |
| Uniform client weight | 36 | 0.2229 | 0.0671 | 0.8459 | 0.9462 | 0.4242 |
| Binary support only | 36 | 0.2229 | 0.0671 | 0.8459 | 0.9462 | 0.4242 |
| Global client-size weight | 36 | 0.2251 | 0.0691 | 0.8459 | 0.9416 | 0.3899 |
| Uniform prevalence prior | 36 | 0.1939 | 0.0380 | 0.8459 | 0.9245 | 0.1819 |
| Smoothed prevalence prior | 36 | 0.1939 | 0.0380 | 0.8459 | 0.9245 | 0.1819 |

## Interpretation

Prototype separation measures whether class directions occupy distinct positions in the shared reference feature space. Prototype consistency measures whether clients that contain the same diagnosis class agree on its reference-space direction. Evidence entropy measures whether the server assigns class-specific evidence to several clients or concentrates it on one reliable client. These quantities are diagnostic analyses of the uploaded aggregate statistics; they do not require the server to read raw client images.

Prevalence-only variants share the same prototype geometry as LAMP-Merge because they modify the score bias rather than the class prototype construction.
These geometry rows are not accuracy ablation results. They are computed directly from the uploaded prototype and support statistics under each ablation definition; full-scope accuracy completion must be checked in the main ablation table.
