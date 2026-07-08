# LAMP-Merge Prototype Geometry Analysis

This report analyzes uploaded class-level prototype statistics on the formal full grid. The numerical rows aggregate five medical datasets, four backbones, three client counts, and three Dirichlet skew levels unless explicitly filtered.

Expected cases per mode: `180`.

## Dataset-Level Geometry

### bloodmnist_224

| mode | cases | pairwise dist | nearest-class dist | consistency | proto-client align | evidence entropy |
|---|---:|---:|---:|---:|---:|---:|
| Full prototype | 36 | 0.0702 | 0.0323 | 0.9999 | 1.0000 | 0.4160 |
| Classifier-head aggregation | 36 | 0.9645 | 0.8065 | 0.0008 | 0.6308 | 0.4160 |
| Global-feature mean | 36 | -0.0000 | -0.0000 | 1.0000 | 1.0000 | 0.4160 |
| Support-only synthetic head | 36 | 0.9874 | 0.9191 | 1.0000 | 1.0000 | 0.4160 |
| Shuffled-label prototype | 36 | 0.0702 | 0.0323 | 0.9999 | 0.9596 | 0.4160 |
| Uniform client weight | 36 | 0.0702 | 0.0323 | 0.9999 | 1.0000 | 0.4167 |
| Global client-size weight | 36 | 0.0702 | 0.0323 | 0.9999 | 1.0000 | 0.3420 |

### chaoshengmnist_224

| mode | cases | pairwise dist | nearest-class dist | consistency | proto-client align | evidence entropy |
|---|---:|---:|---:|---:|---:|---:|
| Full prototype | 36 | 0.0435 | 0.0218 | 1.0000 | 1.0000 | 0.3750 |
| Classifier-head aggregation | 36 | 0.9732 | 0.8570 | 0.0034 | 0.6272 | 0.3750 |
| Global-feature mean | 36 | -0.0000 | -0.0000 | 1.0000 | 1.0000 | 0.3750 |
| Support-only synthetic head | 36 | 0.9874 | 0.9191 | 1.0000 | 1.0000 | 0.3750 |
| Shuffled-label prototype | 36 | 0.0435 | 0.0218 | 1.0000 | 0.9638 | 0.3750 |
| Uniform client weight | 36 | 0.0435 | 0.0218 | 1.0000 | 1.0000 | 0.3750 |
| Global client-size weight | 36 | 0.0435 | 0.0218 | 1.0000 | 1.0000 | 0.3069 |

### dermamnist_224

| mode | cases | pairwise dist | nearest-class dist | consistency | proto-client align | evidence entropy |
|---|---:|---:|---:|---:|---:|---:|
| Full prototype | 36 | 0.0523 | 0.0281 | 1.0000 | 1.0000 | 0.4127 |
| Classifier-head aggregation | 36 | 0.9840 | 0.8239 | 0.0066 | 0.6320 | 0.4127 |
| Global-feature mean | 36 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.4127 |
| Support-only synthetic head | 36 | 0.9847 | 0.9175 | 1.0000 | 1.0000 | 0.4127 |
| Shuffled-label prototype | 36 | 0.0523 | 0.0281 | 1.0000 | 0.9539 | 0.4127 |
| Uniform client weight | 36 | 0.0523 | 0.0281 | 1.0000 | 1.0000 | 0.4127 |
| Global client-size weight | 36 | 0.0524 | 0.0281 | 1.0000 | 1.0000 | 0.2528 |

### organcmnist_224

| mode | cases | pairwise dist | nearest-class dist | consistency | proto-client align | evidence entropy |
|---|---:|---:|---:|---:|---:|---:|
| Full prototype | 36 | 0.1957 | 0.0562 | 0.9997 | 0.9999 | 0.4338 |
| Classifier-head aggregation | 36 | 0.9699 | 0.7756 | -0.0095 | 0.6177 | 0.4338 |
| Global-feature mean | 36 | -0.0000 | -0.0000 | 1.0000 | 1.0000 | 0.4338 |
| Support-only synthetic head | 36 | 0.9879 | 0.9074 | 1.0000 | 1.0000 | 0.4338 |
| Shuffled-label prototype | 36 | 0.1957 | 0.0562 | 0.9997 | 0.8421 | 0.4338 |
| Uniform client weight | 36 | 0.1958 | 0.0562 | 0.9997 | 0.9999 | 0.4343 |
| Global client-size weight | 36 | 0.1957 | 0.0561 | 0.9997 | 0.9999 | 0.4080 |

### organsmnist_224

| mode | cases | pairwise dist | nearest-class dist | consistency | proto-client align | evidence entropy |
|---|---:|---:|---:|---:|---:|---:|
| Full prototype | 36 | 0.1940 | 0.0379 | 0.9995 | 0.9998 | 0.4231 |
| Classifier-head aggregation | 36 | 0.9751 | 0.7602 | -0.0117 | 0.6080 | 0.4231 |
| Global-feature mean | 36 | 0.0000 | -0.0000 | 1.0000 | 1.0000 | 0.4231 |
| Support-only synthetic head | 36 | 0.9879 | 0.9074 | 1.0000 | 1.0000 | 0.4231 |
| Shuffled-label prototype | 36 | 0.1940 | 0.0379 | 0.9995 | 0.8357 | 0.4231 |
| Uniform client weight | 36 | 0.1940 | 0.0379 | 0.9995 | 0.9998 | 0.4242 |
| Global client-size weight | 36 | 0.1940 | 0.0379 | 0.9995 | 0.9998 | 0.3899 |

## Interpretation

Prototype separation measures whether class directions occupy distinct positions in the shared reference feature space. Prototype consistency measures whether clients that contain the same diagnosis class agree on its reference-space direction. Evidence entropy measures whether the server assigns class-specific evidence to several clients or concentrates it on one reliable client. These quantities are diagnostic analyses of the uploaded aggregate statistics; they do not require the server to read raw client images.

The t-SNE figures in the same figure directory are mechanism visualizations. They use the public test split only to display how reference-space sample clusters relate to uploaded class prototypes; they are not used for model selection, training, or merging.
