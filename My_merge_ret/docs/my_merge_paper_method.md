# my_merge: Public-Probe Guided Medical Model Fusion

## Setting

`my_merge` is an asynchronous post-hoc fusion method for medical image models.
Each client trains locally and uploads its checkpoint once. The server merges the
checkpoints in one pass.

This is not federated learning:

- no multi-round communication,
- no candidate model sent back to clients,
- no client feedback loop,
- no server access to client private raw images.

The server is allowed to use a public medical probe set, for example public
MedMNIST-style images or another approved de-identified medical reference set.
The probe set is not client private data.

## Medical Observation

The old high-performing candidate-pool version worked because medical image
fusion needs image evidence to choose between incompatible weight-space
solutions. The privacy issue was not the candidate pool itself; it was using the
target private validation images as the evidence source.

Medical imaging has a practical replacement that NLP usually does not: public
or institution-approved reference images from the same acquisition domain. These
images preserve scanner contrast, organ foreground, lesion boundary, texture,
speckle, and anatomical occupancy statistics. The method therefore moves all
evidence collection to a public medical probe set.

## Module 1: Public Morphological Evidence

For public probe images \(x\), the server computes a radiomics-style
morphological salience vector:

\[
\phi(x) =
[
area,\ boundary,\ contrast,\ texture,\ salience,\ reliability
].
\]

The salience is built from normalized intensity, Sobel boundary magnitude, local
contrast, and local texture variance. Each public sample receives a medical
importance weight:

\[
w(x) =
Normalize\left(
\frac{salience(x)}{E[salience]}
\cdot
(0.65 + 0.35\, reliability(x))
\right).
\]

Each client checkpoint is evaluated on the public probe. If the public probe has
compatible labels, the server estimates ordinary accuracy, morphology-weighted
accuracy, and class-wise public evidence. If labels are unavailable or
incompatible, it falls back to confidence, margin, and agreement with the public
ensemble prediction.

This yields three weight sets:

\[
\alpha^{overall}, \quad \alpha^{morph}, \quad \alpha^{class}_c.
\]

They are not dataset or model-name branches; they are continuous functions of
public medical image evidence.

## Module 2: Fixed Candidate Family and Public Selection

The method builds a fixed family of post-hoc candidate merges:

- `avg`: ordinary checkpoint average.
- `medical_weighted_fusion`: layer/class routed merge using public morphology
  weights.
- `sign_consistent_delta`: TIES-style task-vector trim, sign election, and
  disjoint merge.
- `avg_sign_blend_0p25`: conservative blend between average and sign-consistent
  delta.
- `breadcrumbs`: sparse middle-magnitude task-vector candidate.
- `from`: norm-scaled task-vector candidate.
- `robustmerge`: robust matrix-level candidate.
- `iso_c`: isotropic 2D weight geometry candidate.
- `public_fisher`: Fisher-style curvature candidate computed only on public
  probe images with small micro-batches.

For candidate \(m\), public selection uses:

\[
Score(m) =
0.85\, Acc_{public}(m)
+ 0.15\, Acc_{morph}(m)
- 0.005\, Loss_{public}(m).
\]

Before scoring, BatchNorm statistics are recalibrated on the public probe. This
does not leak client data because the recalibration images are public.

The selected checkpoint is:

\[
\theta^* = \arg\max_m Score(m).
\]

## Why This Is Medical-Specific

The method depends on public medical image phenotype evidence: organ occupancy,
lesion/foreground boundary, scanner or ultrasound texture, local contrast, and
anatomical salience. These are image acquisition and morphology properties, not
token-sequence properties. The method is intentionally scoped to medical image
tasks and rejects non-medical datasets in code.

The public-probe assumption is also natural in medical imaging: public reference
sets such as MedMNIST can provide domain-compatible de-identified images without
exposing private hospital images.

## Privacy Boundary

The server receives:

- client checkpoints,
- task metadata already present in the model hub,
- public medical probe images selected by configuration.

The server does not receive client private raw images, private validation
examples, per-sample private embeddings, logits, activations, or candidate
feedback. Fusion remains one-shot and asynchronous.

## Current Result

Current evaluated setting:

- key client-average groups:
  - `dermamnist_224 / c5_avg`
  - `organcmnist_224 / c3_avg`
  - `organsmnist_224 / c5_avg`
  - `chaoshengmnist_224 / c3_avg`
- models: `resnet`, `convnext`, `vit_t`, `swin_tiny`
- result roots:
  - `outputs/codex_dualprobe_full_20260701_001514_g0`
  - `outputs/codex_dualprobe_full_20260701_001514_g1`

Against `result/all_results.md`, using the beta-averaged `Client Average`
criterion:

```text
my_merge >= best existing method: 16 / 16
strictly better:                  16 / 16
previous restored candidate pool: 10 / 16
```

Main generated reports:

- `My_merge_ret/reports/all_results_my_merge.md`
- `My_merge_ret/reports/codex_dualprobe_full_20260701_001514_my_merge.md`
- `My_merge_ret/reports/codex_dualprobe_full_20260701_001514_combined.md`
- `My_merge_ret/汇总表.md`

## Code

- [methods/my_merge.py](/data/liyapeng_grp/program/MedMNISTMerge/methods/my_merge.py)
- [scripts/run_all_avg_eval.py](/data/liyapeng_grp/program/MedMNISTMerge/scripts/run_all_avg_eval.py)
