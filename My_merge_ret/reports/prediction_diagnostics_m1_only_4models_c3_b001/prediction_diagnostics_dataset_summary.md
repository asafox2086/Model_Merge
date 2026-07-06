# Prediction Collapse Diagnostics by Dataset

Each table averages over the evaluated model/client/beta cases within one dataset.

`collapse_ratio` is the fraction of test samples assigned to the most frequent predicted class.

## bloodmnist_224

| method | cases | acc | balanced acc | macro F1 | collapse ratio | effective classes | pred-true TV |
|---|---:|---:|---:|---:|---:|---:|---:|
| my_merge | 4 | 0.8190 | 0.8071 | 0.8023 | 0.1943 | 7.4872 | 0.0394 |

## chaoshengmnist_224

| method | cases | acc | balanced acc | macro F1 | collapse ratio | effective classes | pred-true TV |
|---|---:|---:|---:|---:|---:|---:|---:|
| my_merge | 4 | 0.4620 | 0.4532 | 0.4373 | 0.2037 | 7.4430 | 0.1575 |

## dermamnist_224

| method | cases | acc | balanced acc | macro F1 | collapse ratio | effective classes | pred-true TV |
|---|---:|---:|---:|---:|---:|---:|---:|
| my_merge | 4 | 0.4627 | 0.4492 | 0.2935 | 0.3697 | 5.6490 | 0.3309 |

## organcmnist_224

| method | cases | acc | balanced acc | macro F1 | collapse ratio | effective classes | pred-true TV |
|---|---:|---:|---:|---:|---:|---:|---:|
| my_merge | 4 | 0.6249 | 0.6174 | 0.6029 | 0.1771 | 10.1096 | 0.1280 |

## organsmnist_224

| method | cases | acc | balanced acc | macro F1 | collapse ratio | effective classes | pred-true TV |
|---|---:|---:|---:|---:|---:|---:|---:|
| my_merge | 4 | 0.5736 | 0.5462 | 0.5399 | 0.2021 | 9.5028 | 0.0993 |

