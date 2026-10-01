# Collapse Experiment Log

This document records the collapse-control experiments requested in the discussion.
The current repository does not contain NLP text datasets or NLP client checkpoints, so the non-medical control is implemented with natural-image datasets, starting from CIFAR-10.

## Question

Is the single-class collapse caused by non-IID alone, or by a stronger combination of medical-domain factors such as partial label support, class imbalance, local client dominance, weak visual boundaries, and incompatible logit margins?

## Completed Experiments

| ID | Domain | Dataset | Setting | Observation | Key Result |
| --- | --- | --- | --- | --- | --- |
| E7 | Medical image | DermaMNIST | Equal total client size, original local ratios | Collapse reduced but not fixed; dominant class changed | AVG source collapse `0.8120`, effective classes not fully healthy |
| E8 | Medical image | DermaMNIST | Balanced local classes inside each client | Severe collapse disappears | AVG source collapse `0.4304`, effective classes `7` |
| E9 | Natural image | CIFAR-10 | Balanced partial-label clients, 2 classes/client | No single-class collapse | AVG collapse `0.2425`, effective classes `10` |
| E10 | Natural image | CIFAR-10 | Strongly imbalanced partial-label clients | No single-class collapse, but strong drift to the dominant client class | AVG collapse `0.4307`, effective classes `10`, top class `8` |
| E11 | Medical vs natural | DermaMNIST vs CIFAR-10 | Mechanism comparison with output margins, feature geometry, heads, parameters | Medical collapse is decided by all-positive dominant margins | Derma class `5` has `pct_positive=1.0` against all competitors |
| E12 | Natural image | CIFAR-10 | 3-client imbalance control, matching Derma client count | Stronger drift, still not one-class medical collapse | AVG collapse `0.7810`, effective classes `8`, top class `8` |
| E13 | Medical vs natural | DermaMNIST vs CIFAR-10 | CKA, task-vector cosine, head-feature swap, linear probe | Main difference is high-layer update compatibility and head-feature readout, not complete feature destruction | Derma `layer4` task-vector cosine `0.0310`; CIFAR `0.9610`; Derma linear probe BAcc recovers `0.1429 -> 0.5731` |
| E14 | Strict medical vs natural | DermaMNIST vs CIFAR-7 strict | Matched classes, per-class counts, client splits, model, image size, epochs, optimizer, and AVG | Natural images also drift under the exact Derma imbalance, but medical AVG is still more completely collapsed | Derma AVG collapse `1.0000`, effective classes `1`; strict CIFAR AVG collapse `0.9067`, effective classes `2` |

## E10 Design

The E10 natural-image imbalance probe uses the same CIFAR-10 partial-label class grouping as E9:

```text
client_0: [0, 1]
client_1: [2, 3]
client_2: [4, 5]
client_3: [6, 7]
client_4: [8, 9]
```

The sampling is intentionally imbalanced:

```text
client_0: class 0 = 300, class 1 = 300
client_1: class 2 = 300, class 3 = 300
client_2: class 4 = 300, class 5 = 300
client_3: class 6 = 300, class 7 = 300
client_4: class 8 = 4000, class 9 = 80
```

This creates a medical-like pressure case: one large client is dominated by one class.
The exact command was:

```text
/data2/liyapeng_grp/.conda/envs/MM/bin/python medmerge_empirical_study/scripts/natural_image_collapse_probe.py \
  --dataset cifar10_32 \
  --device cuda:0 \
  --epochs 8 \
  --batch-size 256 \
  --group-class-counts '300,300;300,300;300,300;300,300;4000,80' \
  --out-dir medmerge_empirical_study/results/experiment10_natural_image_imbalance_probe \
  --num-workers 4 \
  --amp
```

## E10 Result

| model | accuracy | balanced_accuracy | collapse_ratio | effective_pred_classes | top_pred_class | pred_counts |
| --- | --- | --- | --- | --- | --- | --- |
| client_4 | 0.1116 | 0.1116 | 0.9774 | 2 | 8 | [0, 0, 0, 0, 0, 0, 0, 0, 9774, 226] |
| avg | 0.0832 | 0.0832 | 0.4307 | 10 | 8 | [290, 987, 636, 2073, 142, 382, 872, 194, 4307, 117] |

The dominant imbalanced client itself almost collapses to class 8.
After AVG, class 8 also becomes the most frequent prediction, but the merged model still predicts all 10 classes.
So the natural-image imbalance case shows strong drift, not the medical-style single-class collapse.

## Current Interpretation

The current evidence says non-IID alone is not sufficient to explain collapse.
CIFAR-10 is also partial-label non-IID, but balanced natural-image clients do not collapse to one class after AVG.
DermaMNIST stops collapsing when local class counts are balanced, which means the medical collapse is tightly coupled to imbalance and local margin dominance.

E10 strengthens this conclusion.
Under strong class/client imbalance, natural-image AVG is pulled toward the dominant class, but it does not become a one-class classifier.
The failure mode is therefore not just "non-IID" and not just "imbalance"; the medical setting appears more fragile because imbalance is combined with weak class boundaries, domain-specific visual shortcuts, and less compatible client feature geometry.

## E11/E12 Mechanism Result

E12 controls a possible confound in E10: the original medical case has 3 clients, while E10 used 5 CIFAR clients. The 3-client CIFAR control uses:

```text
client_0: class 0/1/2 = 300/300/300
client_1: class 3/4/5 = 300/300/300
client_2: class 8/9 = 4000/80
```

The AVG result is:

| domain | collapse_ratio | effective_pred_classes | top_class | pred_counts |
| --- | --- | --- | --- | --- |
| DermaMNIST 3-client | `1.0000` | `1` | `5` | `[0, 0, 0, 0, 0, 2005, 0]` |
| CIFAR-10 3-client | `0.7810` | `8` | `8` | `[254, 421, 329, 593, 210, 235, 0, 0, 7810, 148]` |

The decisive difference is the low-tail of the dominant-class margin:

| domain | margin | p05_delta | min_delta | pct_positive |
| --- | --- | --- | --- | --- |
| DermaMNIST | `5-2` | `0.268922` | `0.133353` | `1.000000` |
| DermaMNIST | `5-4` | `0.463657` | `0.071431` | `1.000000` |
| DermaMNIST | `5-6` | `0.688538` | `0.283340` | `1.000000` |
| CIFAR-10 | `8-1` | `-0.049698` | `-1.550948` | `0.937600` |
| CIFAR-10 | `8-3` | `-0.119339` | `-3.122090` | `0.917300` |
| CIFAR-10 | `8-9` | `0.169226` | `-1.132782` | `0.974100` |

The logit decomposition shows this is not a classifier-bias artifact. For Derma `5-2`, `5-4`, and `5-6`, the `bias_delta` is only `0.003848`, `-0.034195`, and `0.002116`; the all-positive margin comes from the final feature projection term `(w_5 - w_c)^T h`.

Feature geometry also separates the two cases. Effective rank at `layer4/classifier_input` is `31.61` for Derma and `237.55` for CIFAR-10. The natural-image AVG keeps much more sample-dependent variation, so even under strong dominant-client drift it still has negative-margin tail samples that can cross decision boundaries.

Detailed report:

```text
medmerge_empirical_study/results/experiment12_natural_image_3client_imbalance_probe/medical_specific_collapse_mechanism.md
```

## E13 Alignment Diagnostics

E13 tests four possible mechanisms:

1. CKA feature similarity.
2. Task-vector cosine from the common reference/init model.
3. Head-feature swaps.
4. Linear probe on frozen AVG features.

The important correction is that CKA does not prove medical activation features are less aligned. In this case, Derma client-client CKA at `layer4` is `0.316591`, while CIFAR is `0.239942`. So CKA is not the main evidence.

The main evidence is task-vector direction compatibility:

| group | Derma cosine | CIFAR cosine |
| --- | --- | --- |
| classifier | `0.107055` | `0.219918` |
| BN running stats | `0.607558` | `0.985049` |
| layer3 | `0.030653` | `0.599215` |
| layer4 | `0.030976` | `0.960982` |

This says Derma clients move high-level parameters in almost orthogonal directions, while CIFAR clients preserve a strong common direction in high layers and BN statistics.

The linear probe result also corrects the interpretation:

| domain | original AVG BAcc | frozen AVG feature + retrained linear head BAcc | reading |
| --- | --- | --- | --- |
| Derma | `0.142857` | `0.573081` | AVG feature still contains class information; original averaged head reads it incorrectly |
| CIFAR | `0.110500` | `0.431400` | AVG feature also contains recoverable class information |

Therefore the accurate mechanism statement is:

```text
AVG does not completely destroy medical features. It breaks the compatibility among the feature extractor, BN statistics, and classifier head. In Derma, the averaged head reads the remaining feature information as class 5 for every sample.
```

Detailed report:

```text
medmerge_empirical_study/results/experiment13_alignment_diagnostics/alignment_diagnostics_interpretation_cn.md
```

## E14 Strict Domain Control

E14 addresses the user's control-variable concern. The natural-image dataset `cifar7_derma_strict` was constructed to match the DermaMNIST setting as tightly as possible:

```text
train counts: [228, 359, 769, 80, 779, 4693, 99]
val counts:   [33, 52, 110, 12, 111, 671, 14]
test counts:  [66, 103, 220, 23, 223, 1341, 29]

client_0: classes 3/4/0 = 80/779/228
client_1: classes 2/1   = 769/359
client_2: classes 6/5   = 99/4693
```

Both domains use ResNet, image size 224, pretrained initialization, 50 epochs, batch size 64, learning rate 0.001, weight decay 0.0001, and weight AVG. The intended variable is the image domain: medical skin images vs natural CIFAR images. Caveat: CIFAR class 5 uses replacement sampling because the requested Derma count exceeds the available CIFAR unique samples.

AVG output:

| domain | accuracy | balanced_accuracy | collapse_ratio | effective_pred_classes | top_class | pred_counts |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| DermaMNIST strict source | 0.668828 | 0.142857 | 1.000000 | 1 | 5 | [0, 0, 0, 0, 0, 2005, 0] |
| CIFAR strict natural | 0.699252 | 0.191548 | 0.906733 | 2 | 5 | [0, 0, 0, 0, 187, 1818, 0] |

The strict-control result changes the earlier interpretation:

```text
Strong claim rejected: natural images never collapse under the same non-IID/imbalance pressure.
Supported claim: under exactly matched partial-label and long-tail pressure, medical AVG collapses more completely.
```

The direct logit reason is the dominant-class margin. In Derma, class 5 beats every other class on every test sample:

| compare | p05_delta | min_delta | pct_positive |
| --- | ---: | ---: | ---: |
| 5-0 | 0.635670 | 0.359340 | 1.000000 |
| 5-1 | 0.411551 | 0.123354 | 1.000000 |
| 5-2 | 0.268922 | 0.133353 | 1.000000 |
| 5-3 | 0.381957 | 0.080625 | 1.000000 |
| 5-4 | 0.463657 | 0.071431 | 1.000000 |
| 5-6 | 0.688538 | 0.283340 | 1.000000 |

In strict CIFAR, the decisive contrast is class `5-4`:

| domain | compare | p05_delta | min_delta | pct_positive |
| --- | --- | ---: | ---: | ---: |
| strict CIFAR | 5-4 | -0.061839 | -0.344914 | 0.906733 |

This negative low-tail explains why strict CIFAR still predicts 187 examples as class 4, while Derma predicts every sample as class 5.

The mechanism diagnosis is:

```text
Non-IID and class imbalance create the dominant-class pressure.
Medical clients additionally have much lower high-layer task-vector compatibility,
so AVG breaks the feature/BN/head matching more severely.
The averaged Derma head then reads all remaining feature evidence as class 5.
```

Strict-control task-vector cosine:

| group | Derma | strict CIFAR |
| --- | ---: | ---: |
| BN running stats | 0.607558 | 0.932364 |
| layer3 | 0.030653 | 0.958356 |
| layer4 | 0.030976 | 0.960622 |

Linear probe confirms that medical features are not completely destroyed:

| domain | original AVG BAcc | frozen AVG feature + linear head BAcc |
| --- | ---: | ---: |
| Derma | 0.142857 | 0.551806 |
| strict CIFAR | 0.191548 | 0.774773 |

Detailed report:

```text
medmerge_empirical_study/results/experiment14_strict_domain_control/strict_domain_control_report_cn.md
```
