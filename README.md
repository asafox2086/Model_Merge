# LAMP-Merge

**LAMP-Merge** (**L**ong-tail-**A**ware **M**edical **P**rototype **Merge**) is a one-shot post-hoc model merging method for multi-center medical image classification.

The repository contains only the LAMP-Merge code path used by the proposed method:

1. client-side export of class-level diagnostic statistics;
2. one-shot server-side construction of the merged checkpoint;
3. evaluation of the merged checkpoint.

It does not ship implementations of comparison baselines. The public code is intentionally scoped to the proposed pipeline.

## Problem Setting

LAMP-Merge targets asynchronous collaboration among medical institutions after local training has finished. Each client owns a locally trained checkpoint and a private local dataset. The server receives the checkpoint and aggregate class-level statistics, but it never receives raw medical images, per-sample features, per-sample logits, or per-sample predictions.

This is not a federated-learning training loop. There is no repeated server broadcast, no client-side fine-tuning after receiving a global model, and no multi-round optimization. The merge is a single post-hoc server operation.

## Method Overview

Let there be `K` clients and `C` diagnostic classes. Client `i` has a private local dataset `D_i` and a local checkpoint. LAMP-Merge uses a shared reference feature extractor `phi_0`, built deterministically from the public experiment configuration: architecture, input channels, number of classes, seed, and pretrained flag.

For each class `c`, client `i` computes the class support count and the reference-space class prototype:

```math
n_{i,c}=|D_{i,c}|,
\qquad
\mu_{i,c}=\frac{1}{n_{i,c}}\sum_{(x,y)\in D_i}\mathbf{1}[y=c]\,\phi_0(T(x)).
```

The server converts class support into reliability evidence:

```math
e_{i,c}=(n_{i,c}+1)^\gamma\mathbf{1}[n_{i,c}>0],
\qquad
\alpha_{i,c}=\frac{e_{i,c}}{\sum_{j=1}^{K}e_{j,c}}.
```

The global diagnostic prototype is:

```math
p_c=\sum_{i=1}^{K}\alpha_{i,c}\mu_{i,c}.
```

LAMP-Merge then synthesizes a cosine prototype classifier on the shared reference backbone:

```math
w_c=s\frac{p_c}{\|p_c\|_2}.
```

For long-tail medical prevalence, clients also upload class prevalence counts `m_{i,c}`. The server estimates:

```math
\pi_c=\frac{\sum_i m_{i,c}}{\sum_{k=1}^{C}\sum_i m_{i,k}},
\qquad
r=C\max_c\pi_c.
```

If `r` exceeds the configured dominance threshold, the classifier bias receives a centered log-prior calibration:

```math
b_c=\lambda\left(\log\pi_c-\frac{1}{C}\sum_{k=1}^{C}\log\pi_k\right).
```

The final score is:

```math
\mathrm{score}_c(x)=w_c^\top\phi_0(T(x))+b_c.
```

The implementation is in [`methods/lamp_merge.py`](methods/lamp_merge.py). A more detailed method note is provided in [`docs/algorithm.md`](docs/algorithm.md).

## What Each Client Uploads

Each client uploads:

- its trained checkpoint;
- `class_feature_mean`: one reference-space feature mean per diagnostic class;
- `class_feature_counts`: support counts used to weight class prototypes;
- `class_prevalence_counts`: local label-count histogram used for long-tail prevalence calibration.

The exported statistics are aggregate class-level tensors. They do not contain raw images or per-sample feature vectors.

## Repository Layout

```text
methods/lamp_merge.py                    # Formal LAMP-Merge implementation
merge.py                                 # One-shot server-side merge entry
evaluate.py                              # Evaluation entry for merged checkpoints
scripts/export_lamp_merge_prototypes.py  # Client-side aggregate statistic export
scripts/run_lamp_merge_eval.py           # Optional batch runner over model_hub/manifest.csv
dataset/                                 # Dataset loaders
model/                                   # Backbones and CLIP wrappers
evaluators/                              # Evaluation routines
utils/                                   # Runtime, checkpoint, and path utilities
configs/lamp_merge/example_small.json    # Minimal example config
docs/algorithm.md                        # Formal method description
```

## Installation

Create an environment with PyTorch, torchvision, timm, and transformers. For example:

```bash
conda create -n lamp-merge python=3.10 -y
conda activate lamp-merge
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build if GPU evaluation is needed.

## Expected Data Layout

The code expects a `model_hub` directory containing trained client checkpoints and metadata:

```text
model_hub/
  manifest.csv
  small/
    bloodmnist_224/
      resnet/
        clients_3/
          beta_0p01/
            seed_42/
              meta.json
              client_0.pt
              client_1.pt
              client_2.pt
```

The corresponding medical image arrays are expected under `Med_data` as `.npz` files:

```text
Med_data/
  bloodmnist_224.npz
  dermamnist_224.npz
  ...
```

The `meta.json` file should describe the task, number of classes, client checkpoints, client sample counts, and class support information. In the formal workflow, each client runs the export step on its own local training split and sends only the resulting aggregate statistics to the server.

## Workflow

### 1. Export client-side aggregate statistics

Run this step on the client side, or in a trusted reproduction environment that has access to the client-local training split:

```bash
python scripts/export_lamp_merge_prototypes.py \
  --model-hub-root model_hub \
  --data-root Med_data \
  --output-root outputs/lamp_merge_client_local_proto_stats \
  --task-type small \
  --datasets bloodmnist_224 \
  --small-models resnet \
  --num-clients 3 \
  --betas 0.01 \
  --seed 42 \
  --split train \
  --device cuda:0
```

This produces:

```text
outputs/lamp_merge_client_local_proto_stats/small/<dataset>/<model>/clients_<K>/beta_<b>/seed_<s>/prototype_stats.pt
```

### 2. Merge once on the server

Edit [`configs/lamp_merge/example_small.json`](configs/lamp_merge/example_small.json) to match the target experiment, then run:

```bash
python merge.py --config configs/lamp_merge/example_small.json
```

The merged checkpoint and merge trace are written under the configured `output_root`.

### 3. Evaluate the merged checkpoint

```bash
python evaluate.py \
  --config configs/lamp_merge/example_small.json \
  --merged-dir outputs/lamp_merge_example/merged/small/bloodmnist_224/resnet/clients_3/beta_0p01/seed_42/lamp_merge
```

### 4. Optional batch run

After exporting all client statistics, run over selected entries in `model_hub/manifest.csv`:

```bash
python scripts/run_lamp_merge_eval.py \
  --model-hub-root model_hub \
  --data-root Med_data \
  --output-root outputs/lamp_merge_eval \
  --lamp-merge-prototype-root outputs/lamp_merge_client_local_proto_stats \
  --task-type small \
  --datasets bloodmnist_224 dermamnist_224 \
  --small-models resnet convnext vit_t swin_tiny \
  --device cuda:0
```

## Default Hyperparameters

| Symbol | Config key | Default | Role |
| --- | --- | ---: | --- |
| `gamma` | `lamp_merge_proto_count_power` | `0.45` | Sublinear class-support evidence power |
| `s` | `lamp_merge_reference_head_scale` | `20.0` | Cosine prototype head scale |
| `tau` | `lamp_merge_reference_prior_threshold` | `2.5` | Dominant-class imbalance threshold |
| `lambda` | `lamp_merge_reference_prior_max_tau` | `5.0` | Centered log-prior calibration strength |

## Privacy Boundary

The server consumes only checkpoints and class-level aggregates. The released merge code validates that prevalence statistics are marked as client-local or client-uploaded. The server-side merge step does not load private training images, public probe images, validation images, per-sample features, or per-sample predictions.

## Citation

If you use this implementation, please cite the corresponding LAMP-Merge paper once available.
