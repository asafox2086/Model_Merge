# Current M1/M2 Smoke Check

Setting: `small / resnet / clients=3 / beta=0 / seed=42`, with one stats batch and one eval batch for fast validation.

| dataset | old full | old avg | current full | selected candidate |
| --- | ---: | ---: | ---: | --- |
| bloodmnist_224 | 0.4607 | 0.3017 | 0.5235 | consensus |
| dermamnist_224 | 0.6783 | 0.6688 | 0.6793 | evidence_routed |
| organcmnist_224 | 0.3544 | 0.3213 | 0.4948 | consensus |
| organsmnist_224 | 0.4953 | 0.1969 | 0.4953 | avg |
| chaoshengmnist_224 | 0.3208 | 0.2552 | 0.3711 | avg |

Notes:

- The earlier global fusion-confidence interpolation was too conservative and degraded `bloodmnist_224` to near average merging.
- The current code restores validated checkpoint candidate selection instead of forcing a single routed merge.
- If the average checkpoint performs below half of random-chance accuracy on the validation subset, candidate selection falls back to the average candidate because the validation evidence is not trustworthy.
- Candidate validation now reuses M1's collected batches and evidence features to avoid repeatedly recomputing medical features.
