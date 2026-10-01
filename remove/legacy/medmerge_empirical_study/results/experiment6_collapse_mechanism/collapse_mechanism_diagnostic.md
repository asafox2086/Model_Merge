# Collapse Mechanism Diagnostic

- Case: `model_hub/small/dermamnist_224/resnet/clients_3/beta_0/seed_42`
- Merge: equal-weight AVG, weights=[0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
- Device: `cuda:0`
- Question: why does the merged model predict one class for almost every image?

## Model-Level Evidence

| model | dataset | accuracy | balanced_accuracy | macro_f1 | collapse_ratio | effective_pred_classes | top_pred_class | top_pred_fraction | mean_top1_top2_margin | true_counts | pred_counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| client_0 | source_test | 0.021446 | 0.159242 | 0.026506 | 0.895262 | 3 | 3 | 0.895262 | 1.875756 | [66, 103, 220, 23, 223, 1341, 29] | [148, 0, 0, 1795, 62, 0, 0] |
| client_0 | public_test | 0.102190 | 0.140845 | 0.027248 | 0.967883 | 3 | 3 | 0.967883 | 1.948316 | [128, 128, 103, 71, 81, 101, 73] | [16, 0, 0, 663, 6, 0, 0] |
| client_1 | source_test | 0.066334 | 0.160125 | 0.030673 | 0.634414 | 2 | 1 | 0.634414 | 2.473911 | [66, 103, 220, 23, 223, 1341, 29] | [0, 1272, 733, 0, 0, 0, 0] |
| client_1 | public_test | 0.192701 | 0.154635 | 0.074449 | 0.718248 | 2 | 1 | 0.718248 | 1.268089 | [128, 128, 103, 71, 81, 101, 73] | [0, 492, 193, 0, 0, 0, 0] |
| client_2 | source_test | 0.577057 | 0.210008 | 0.122725 | 0.788529 | 2 | 5 | 0.788529 | 5.286525 | [66, 103, 220, 23, 223, 1341, 29] | [0, 0, 0, 0, 0, 1581, 424] |
| client_2 | public_test | 0.157664 | 0.169034 | 0.068371 | 0.664234 | 2 | 5 | 0.664234 | 2.571240 | [128, 128, 103, 71, 81, 101, 73] | [0, 0, 0, 0, 0, 455, 230] |
| avg | source_test | 0.668828 | 0.142857 | 0.114508 | 1.000000 | 1 | 5 | 1.000000 | 0.539030 | [66, 103, 220, 23, 223, 1341, 29] | [0, 0, 0, 0, 0, 2005, 0] |
| avg | public_test | 0.147445 | 0.142857 | 0.036714 | 1.000000 | 1 | 5 | 1.000000 | 0.384159 | [128, 128, 103, 71, 81, 101, 73] | [0, 0, 0, 0, 0, 685, 0] |

## AVG Final-Layer Mean Logits on Source Test

| class | mean_wf | std_wf | bias | mean_logit | std_logit | min_logit | max_logit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | -0.993736 | 0.299952 | -0.031668 | -1.025405 | 0.299953 | -1.822449 | -0.340199 |
| 2 | -1.560491 | 0.468140 | -0.035517 | -1.596008 | 0.468140 | -2.775134 | -0.518247 |
| 4 | -1.755148 | 0.382530 | 0.002527 | -1.752621 | 0.382530 | -2.903157 | -0.662283 |
| 1 | -2.013324 | 0.630196 | -0.009575 | -2.022899 | 0.630196 | -3.428988 | -0.528662 |
| 0 | -2.358657 | 0.708230 | -0.011714 | -2.370371 | 0.708230 | -4.082302 | -0.785601 |
| 3 | -2.534899 | 0.998114 | 0.005188 | -2.529711 | 0.998114 | -5.071078 | -0.462911 |
| 6 | -2.861320 | 1.009981 | -0.033785 | -2.895105 | 1.009981 | -5.553578 | -0.687250 |

## Class-5 Margin Decomposition

| dataset | compare | mean_dynamic_delta | std_dynamic_delta | bias_delta | mean_total_delta | min_total_delta | p05_total_delta | pct_total_delta_positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| source_test | 5-0 | 1.364921 | 0.425337 | -0.019955 | 1.344966 | 0.359340 | 0.635670 | 1.000000 |
| source_test | 5-1 | 1.019588 | 0.355431 | -0.022094 | 0.997494 | 0.123354 | 0.411551 | 1.000000 |
| source_test | 5-2 | 0.566755 | 0.203555 | 0.003848 | 0.570603 | 0.133353 | 0.268922 | 1.000000 |
| source_test | 5-3 | 1.541164 | 0.712038 | -0.036857 | 1.504307 | 0.080625 | 0.381957 | 1.000000 |
| source_test | 5-4 | 0.761412 | 0.171161 | -0.034195 | 0.727216 | 0.071431 | 0.463657 | 1.000000 |
| source_test | 5-6 | 1.867584 | 0.723917 | 0.002116 | 1.869700 | 0.283340 | 0.688538 | 1.000000 |
| public_test | 5-0 | 1.065072 | 0.402075 | -0.019955 | 1.045118 | 0.376887 | 0.491843 | 1.000000 |
| public_test | 5-1 | 0.718203 | 0.298015 | -0.022094 | 0.696109 | 0.153107 | 0.273395 | 1.000000 |
| public_test | 5-2 | 0.394615 | 0.163692 | 0.003848 | 0.398464 | 0.072268 | 0.186358 | 1.000000 |
| public_test | 5-3 | 0.903828 | 0.555518 | -0.036857 | 0.866972 | 0.028133 | 0.158060 | 1.000000 |
| public_test | 5-4 | 0.676379 | 0.171956 | -0.034195 | 0.642184 | 0.272741 | 0.383446 | 1.000000 |
| public_test | 5-6 | 1.212459 | 0.562446 | 0.002116 | 1.214576 | 0.314348 | 0.450816 | 1.000000 |

## Short Reading

- AVG source test predicts class 5 for 1.000000 of samples; public test predicts class 5 for 1.000000 of samples.
- Balanced accuracy close to 1/7 means the high source accuracy comes from matching the source test prior, not class-wise discrimination.
- In the pairwise table, `pct_total_delta_positive=1.0` means class 5 logit is larger than that class for every sample; `min_total_delta>0` means even the closest sample cannot cross the decision boundary.
- The collapse is therefore a logit-geometry result: after averaging non-IID clients, the final representation/head has a stable class-5 offset, while image-dependent variation is not enough to change the argmax.
