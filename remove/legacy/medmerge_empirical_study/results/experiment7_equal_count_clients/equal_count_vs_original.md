# Equal Total Count vs Original Unequal Clients

## Sampling Control

| client_id | classes | available_num_samples | sampled_num_samples | sampled_class_counts |
| --- | --- | --- | --- | --- |
| 0 | [3, 4, 0] | 1087 | 1087 | {"3": 80, "4": 779, "0": 228} |
| 1 | [2, 1] | 1128 | 1087 | {"2": 741, "1": 346} |
| 2 | [6, 5] | 4792 | 1087 | {"6": 22, "5": 1065} |

## AVG Merge Comparison

| case | split | acc | bacc | collapse | eff | top | pred_counts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original_unequal | source_test | 0.668828 | 0.142857 | 1.000000 | 1 | 5 | [0, 0, 0, 0, 0, 2005, 0] |
| original_unequal | public_test | 0.147445 | 0.142857 | 1.000000 | 1 | 5 | [0, 0, 0, 0, 0, 685, 0] |
| equal_total | source_test | 0.160100 | 0.286083 | 0.811970 | 6 | 4 | [16, 240, 118, 2, 1628, 0, 1] |
| equal_total | public_test | 0.189781 | 0.185372 | 0.575182 | 4 | 4 | [10, 159, 122, 0, 394, 0, 0] |

## Source-Test Client Comparison

| case | client | acc | bacc | collapse | top | pred_counts |
| --- | --- | --- | --- | --- | --- | --- |
| original_unequal | client_0 | 0.021446 | 0.159242 | 0.895262 | 3 | [148, 0, 0, 1795, 62, 0, 0] |
| original_unequal | client_1 | 0.066334 | 0.160125 | 0.634414 | 1 | [0, 1272, 733, 0, 0, 0, 0] |
| original_unequal | client_2 | 0.577057 | 0.210008 | 0.788529 | 5 | [0, 0, 0, 0, 0, 1581, 424] |
| equal_total | client_0 | 0.143641 | 0.366058 | 0.768080 | 4 | [240, 0, 0, 225, 1540, 0, 0] |
| equal_total | client_1 | 0.151122 | 0.264614 | 0.876309 | 2 | [0, 248, 1757, 0, 0, 0, 0] |
| equal_total | client_2 | 0.675810 | 0.235921 | 0.979052 | 5 | [0, 0, 0, 0, 0, 1963, 42] |

## Mechanism Note

- Equalizing total client count changes the winning default class: original AVG is class 5 for all samples; equal-total AVG is mostly class 4.
- The problem is not solved: source-test BAcc is still low and prediction mass is still dominated by one local class direction.
- This supports the distinction: unequal total sample count explains why the original collapse chose class 5, while incomplete client label support explains why merging remains unreliable.
