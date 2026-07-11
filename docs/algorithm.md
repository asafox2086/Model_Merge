# LAMP-Merge Algorithm

LAMP-Merge is designed for one-shot asynchronous post-hoc merging of medical image classifiers. The method assumes that local training has already been completed at each medical client. The server performs a single merge step and does not participate in multi-round optimization.

## Setting

There are `K` clients and `C` diagnostic classes. Client `i` has a private local dataset `D_i` and a trained checkpoint. The goal is to construct a merged model without giving the server access to raw images or per-sample representations.

LAMP-Merge relies on a shared reference feature extractor `phi_0`. It is deterministically instantiated from public configuration variables: model architecture, input channels, class count, random seed, and pretrained flag. No client image is used to train this reference backbone at the server.

## Module 1: Diagnostic Prototype Reconstruction

For class `c`, client `i` computes the local class support:

```math
n_{i,c}=|D_{i,c}|.
```

For every class with nonzero support, the client computes a reference-space feature mean:

```math
\mu_{i,c}
=
\frac{1}{n_{i,c}}
\sum_{(x,y)\in D_i}\mathbf{1}[y=c]\phi_0(T(x)).
```

The client uploads `mu_{i,c}` and `n_{i,c}` as aggregate class-level statistics. The server converts support counts into sublinear evidence:

```math
e_{i,c}=(n_{i,c}+1)^\gamma\mathbf{1}[n_{i,c}>0].
```

The per-class client aggregation weight is:

```math
\alpha_{i,c}=\frac{e_{i,c}}{\sum_{j=1}^{K}e_{j,c}}.
```

The global diagnostic prototype is:

```math
p_c=\sum_{i=1}^{K}\alpha_{i,c}\mu_{i,c}.
```

The server starts from the shared reference model and replaces only the classifier head. The class-`c` classifier weight is:

```math
w_c=s\frac{p_c}{\|p_c\|_2}.
```

This module changes the merge object from full-parameter averaging to class-conditional diagnostic evidence reconstruction. It gives every diagnostic class an explicit decision direction in the shared reference feature space.

## Module 2: Long-tail Prevalence Calibration

Medical datasets often contain a dominant diagnosis. In this regime, removing all majority-class bias may reduce overall diagnostic accuracy. LAMP-Merge therefore adds a bounded calibration term when the uploaded prevalence counts indicate strong long-tail imbalance.

Client `i` uploads class prevalence counts `m_{i,c}`. The server estimates the global prevalence prior:

```math
\pi_c=
\frac{\sum_{i=1}^{K}m_{i,c}}
{\sum_{k=1}^{C}\sum_{i=1}^{K}m_{i,k}}.
```

The dominance ratio is:

```math
r=C\max_c\pi_c.
```

If `r` is not larger than the threshold `tau`, the calibration is disabled. If `r > tau`, the server adds a centered log-prior bias:

```math
b_c=
\lambda
\left(
\log\pi_c-\frac{1}{C}\sum_{k=1}^{C}\log\pi_k
\right).
```

The final class score is:

```math
\mathrm{score}_c(x)=w_c^\top\phi_0(T(x))+b_c.
```

Module 2 is not a standalone fusion method. It is defined on top of the diagnostic prototype classifier produced by Module 1. The prototype head provides class-specific decision directions; the prevalence bias adjusts those directions under severe medical long-tail imbalance.

## Communication and Privacy

The communication pattern is:

```text
local training -> client aggregate statistic export -> one-shot server merge
```

The server receives:

- client checkpoints;
- class support counts `n_{i,c}`;
- reference-space class means `mu_{i,c}`;
- class prevalence counts `m_{i,c}`.

The server does not receive raw images, per-sample features, per-sample logits, or per-sample predictions. Therefore, the method is a one-shot post-hoc medical model merging algorithm rather than a federated-learning protocol.
