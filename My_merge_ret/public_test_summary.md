# public_test Summary

This table evaluates the already-defined merge method on independent public labeled medical images.
The public images are used only for final external evaluation. They are not used for merge-time selection, training, or candidate tuning.

## Scope

| Item | Value |
|---|---|
| merge method | my_merge |
| source merge data | `/data2/liyapeng_grp/program/MedMNISTMerge/Med_data` |
| client aggregate stats | `/data2/liyapeng_grp/program/MedMNISTMerge/outputs/my_merge_reference_proto_stats_recall_full_table_20260705` |
| public eval data | `/data2/liyapeng_grp/program/MedMNISTMerge/PublicMedFingerprint_data` |
| public split | val |
| device | cuda:0 |
| completed OK cases | 90 |
| total recorded rows | 90 |

## Public Split

| Dataset | Split | Samples | Shape | Classes | Label Counts |
|---|---|---|---|---|---|
| bloodmnist_224 | val | 1024 | 224x224x3 | 8 | 0:128, 1:128, 2:128, 3:128, 4:128, 5:128, 6:128, 7:128 |
| dermamnist_224 | val | 685 | 224x224x3 | 7 | 0:128, 1:128, 2:103, 3:71, 4:81, 5:101, 6:73 |

## Method Summary

| Method | OK Cases | Mean Acc | Min Acc | Max Acc |
|---|---|---|---|---|
| my_merge | 90 | 0.2585 | 0.0938 | 0.4150 |

## Source-to-Public Shift

| Compared method | Matched cases | Mean public-source acc | Min shift | Max shift |
|---|---|---|---|---|
| my_merge | 90 | -0.3551 | -0.6615 | +0.0332 |

## Small

### resnet

#### Raw

<table>
  <thead>
    <tr><th rowspan="2">method</th><th colspan="9">bloodmnist_224</th><th colspan="9">dermamnist_224</th></tr>
    <tr><th>c3_b0</th><th>c3_b0.01</th><th>c3_b0.1</th><th>c5_b0</th><th>c5_b0.01</th><th>c5_b0.1</th><th>c7_b0</th><th>c7_b0.01</th><th>c7_b0.1</th><th>c3_b0</th><th>c3_b0.01</th><th>c3_b0.1</th><th>c5_b0</th><th>c5_b0.01</th><th>c5_b0.1</th><th>c7_b0</th><th>c7_b0.01</th><th>c7_b0.1</th></tr>
  </thead>
  <tbody>
    <tr><td>my_merge</td><td>0.1406</td><td>0.1416</td><td>0.1406</td><td>0.1406</td><td>0.1406</td><td>0.1406</td><td>0.1406</td><td>0.1426</td><td>0.1406</td><td>0.2482</td><td>0.2453</td><td>0.2467</td><td>0.2467</td><td>0.2146</td><td>0.2482</td><td>0.2467</td><td>0.2263</td><td>0.2482</td></tr>
  </tbody>
</table>

#### Client Average

<table>
  <thead>
    <tr><th rowspan="2">method</th><th colspan="3">bloodmnist_224</th><th colspan="3">dermamnist_224</th></tr>
    <tr><th>c3_avg</th><th>c5_avg</th><th>c7_avg</th><th>c3_avg</th><th>c5_avg</th><th>c7_avg</th></tr>
  </thead>
  <tbody>
    <tr><td>my_merge</td><td>0.1410</td><td>0.1406</td><td>0.1413</td><td>0.2467</td><td>0.2365</td><td>0.2404</td></tr>
  </tbody>
</table>

### convnext

#### Raw

<table>
  <thead>
    <tr><th rowspan="2">method</th><th colspan="9">bloodmnist_224</th><th colspan="9">dermamnist_224</th></tr>
    <tr><th>c3_b0</th><th>c3_b0.01</th><th>c3_b0.1</th><th>c5_b0</th><th>c5_b0.01</th><th>c5_b0.1</th><th>c7_b0</th><th>c7_b0.01</th><th>c7_b0.1</th><th>c3_b0</th><th>c3_b0.01</th><th>c3_b0.1</th><th>c5_b0</th><th>c5_b0.01</th><th>c5_b0.1</th><th>c7_b0</th><th>c7_b0.01</th><th>c7_b0.1</th></tr>
  </thead>
  <tbody>
    <tr><td>my_merge</td><td>0.3076</td><td>0.3145</td><td>0.3086</td><td>0.3125</td><td>0.3115</td><td>0.3096</td><td>0.3125</td><td>0.3086</td><td>0.3105</td><td>0.2569</td><td>0.2569</td><td>0.2584</td><td>0.2584</td><td>0.2569</td><td>0.2584</td><td>0.2540</td><td>0.2701</td><td>0.2555</td></tr>
  </tbody>
</table>

#### Client Average

<table>
  <thead>
    <tr><th rowspan="2">method</th><th colspan="3">bloodmnist_224</th><th colspan="3">dermamnist_224</th></tr>
    <tr><th>c3_avg</th><th>c5_avg</th><th>c7_avg</th><th>c3_avg</th><th>c5_avg</th><th>c7_avg</th></tr>
  </thead>
  <tbody>
    <tr><td>my_merge</td><td>0.3102</td><td>0.3112</td><td>0.3105</td><td>0.2574</td><td>0.2579</td><td>0.2599</td></tr>
  </tbody>
</table>

### vit_t

#### Raw

<table>
  <thead>
    <tr><th rowspan="2">method</th><th colspan="9">bloodmnist_224</th><th colspan="9">dermamnist_224</th></tr>
    <tr><th>c3_b0</th><th>c3_b0.01</th><th>c3_b0.1</th><th>c5_b0</th><th>c5_b0.01</th><th>c5_b0.1</th><th>c7_b0</th><th>c7_b0.01</th><th>c7_b0.1</th><th>c3_b0</th><th>c3_b0.01</th><th>c3_b0.1</th><th>c5_b0</th><th>c5_b0.01</th><th>c5_b0.1</th><th>c7_b0</th><th>c7_b0.01</th><th>c7_b0.1</th></tr>
  </thead>
  <tbody>
    <tr><td>my_merge</td><td>0.4102</td><td>0.4111</td><td>0.4111</td><td>0.4121</td><td>0.4150</td><td>0.4141</td><td>0.4141</td><td>0.4141</td><td>0.4121</td><td>0.2847</td><td>0.2832</td><td>0.2847</td><td>0.2818</td><td>0.2730</td><td>0.2818</td><td>0.2818</td><td>0.2891</td><td>0.2818</td></tr>
  </tbody>
</table>

#### Client Average

<table>
  <thead>
    <tr><th rowspan="2">method</th><th colspan="3">bloodmnist_224</th><th colspan="3">dermamnist_224</th></tr>
    <tr><th>c3_avg</th><th>c5_avg</th><th>c7_avg</th><th>c3_avg</th><th>c5_avg</th><th>c7_avg</th></tr>
  </thead>
  <tbody>
    <tr><td>my_merge</td><td>0.4108</td><td>0.4137</td><td>0.4134</td><td>0.2842</td><td>0.2788</td><td>0.2842</td></tr>
  </tbody>
</table>

### swin_tiny

#### Raw

<table>
  <thead>
    <tr><th rowspan="2">method</th><th colspan="9">bloodmnist_224</th><th colspan="9">dermamnist_224</th></tr>
    <tr><th>c3_b0</th><th>c3_b0.01</th><th>c3_b0.1</th><th>c5_b0</th><th>c5_b0.01</th><th>c5_b0.1</th><th>c7_b0</th><th>c7_b0.01</th><th>c7_b0.1</th><th>c3_b0</th><th>c3_b0.01</th><th>c3_b0.1</th><th>c5_b0</th><th>c5_b0.01</th><th>c5_b0.1</th><th>c7_b0</th><th>c7_b0.01</th><th>c7_b0.1</th></tr>
  </thead>
  <tbody>
    <tr><td>my_merge</td><td>0.4004</td><td>0.4053</td><td>0.3955</td><td>0.4043</td><td>0.4014</td><td>0.4033</td><td>0.4053</td><td>0.3965</td><td>0.4023</td><td>0.2394</td><td>0.2423</td><td>0.2423</td><td>0.2423</td><td>0.2321</td><td>0.2409</td><td>0.2438</td><td>0.2540</td><td>0.2423</td></tr>
  </tbody>
</table>

#### Client Average

<table>
  <thead>
    <tr><th rowspan="2">method</th><th colspan="3">bloodmnist_224</th><th colspan="3">dermamnist_224</th></tr>
    <tr><th>c3_avg</th><th>c5_avg</th><th>c7_avg</th><th>c3_avg</th><th>c5_avg</th><th>c7_avg</th></tr>
  </thead>
  <tbody>
    <tr><td>my_merge</td><td>0.4004</td><td>0.4030</td><td>0.4014</td><td>0.2414</td><td>0.2384</td><td>0.2467</td></tr>
  </tbody>
</table>

## VLM

### openai/clip-vit-base-patch32

#### Raw

<table>
  <thead>
    <tr><th rowspan="2">method</th><th colspan="9">bloodmnist_224</th><th colspan="9">dermamnist_224</th></tr>
    <tr><th>c3_b0</th><th>c3_b0.01</th><th>c3_b0.1</th><th>c5_b0</th><th>c5_b0.01</th><th>c5_b0.1</th><th>c7_b0</th><th>c7_b0.01</th><th>c7_b0.1</th><th>c3_b0</th><th>c3_b0.01</th><th>c3_b0.1</th><th>c5_b0</th><th>c5_b0.01</th><th>c5_b0.1</th><th>c7_b0</th><th>c7_b0.01</th><th>c7_b0.1</th></tr>
  </thead>
  <tbody>
    <tr><td>my_merge</td><td>0.1582</td><td>0.1553</td><td>0.1543</td><td>0.1250</td><td>0.1514</td><td>0.1250</td><td>0.1270</td><td>0.0938</td><td>0.1250</td><td>0.1796</td><td>0.1635</td><td>0.1562</td><td>0.1474</td><td>0.1562</td><td>0.1942</td><td>0.1474</td><td>0.1474</td><td>0.1474</td></tr>
  </tbody>
</table>

#### Client Average

<table>
  <thead>
    <tr><th rowspan="2">method</th><th colspan="3">bloodmnist_224</th><th colspan="3">dermamnist_224</th></tr>
    <tr><th>c3_avg</th><th>c5_avg</th><th>c7_avg</th><th>c3_avg</th><th>c5_avg</th><th>c7_avg</th></tr>
  </thead>
  <tbody>
    <tr><td>my_merge</td><td>0.1559</td><td>0.1338</td><td>0.1152</td><td>0.1664</td><td>0.1659</td><td>0.1474</td></tr>
  </tbody>
</table>
