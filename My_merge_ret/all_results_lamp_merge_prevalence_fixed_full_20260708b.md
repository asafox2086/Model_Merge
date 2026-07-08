# Experiment Master Tables

- Layout: aligned with `result/all_results.md`.
- Source output root: `outputs/lamp_merge_release_formula_prevalence_fixed_full_20260708b/resnet, outputs/lamp_merge_release_formula_prevalence_fixed_full_20260708b/convnext, outputs/lamp_merge_release_formula_prevalence_fixed_full_20260708b/vit_t, outputs/lamp_merge_release_formula_prevalence_fixed_full_20260708b/swin_tiny`.
- Extra comparison rows: none.
- Values are filled from real `eval_summary.csv` results for `LAMP-Merge`; missing combinations are shown as `-`.

## Small

### resnet

#### Raw

<table>
  <thead>
    <tr>
      <th rowspan="2">method</th>
      <th colspan="9">bloodmnist_224</th>
      <th colspan="9">dermamnist_224</th>
      <th colspan="9">organcmnist_224</th>
      <th colspan="9">organsmnist_224</th>
      <th colspan="9">chaoshengmnist_224</th>
    </tr>
    <tr>
      <th>c3_b0</th>
      <th>c3_b0.01</th>
      <th>c3_b0.1</th>
      <th>c5_b0</th>
      <th>c5_b0.01</th>
      <th>c5_b0.1</th>
      <th>c7_b0</th>
      <th>c7_b0.01</th>
      <th>c7_b0.1</th>
      <th>c3_b0</th>
      <th>c3_b0.01</th>
      <th>c3_b0.1</th>
      <th>c5_b0</th>
      <th>c5_b0.01</th>
      <th>c5_b0.1</th>
      <th>c7_b0</th>
      <th>c7_b0.01</th>
      <th>c7_b0.1</th>
      <th>c3_b0</th>
      <th>c3_b0.01</th>
      <th>c3_b0.1</th>
      <th>c5_b0</th>
      <th>c5_b0.01</th>
      <th>c5_b0.1</th>
      <th>c7_b0</th>
      <th>c7_b0.01</th>
      <th>c7_b0.1</th>
      <th>c3_b0</th>
      <th>c3_b0.01</th>
      <th>c3_b0.1</th>
      <th>c5_b0</th>
      <th>c5_b0.01</th>
      <th>c5_b0.1</th>
      <th>c7_b0</th>
      <th>c7_b0.01</th>
      <th>c7_b0.1</th>
      <th>c3_b0</th>
      <th>c3_b0.01</th>
      <th>c3_b0.1</th>
      <th>c5_b0</th>
      <th>c5_b0.01</th>
      <th>c5_b0.1</th>
      <th>c7_b0</th>
      <th>c7_b0.01</th>
      <th>c7_b0.1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>LAMP-Merge</td>
      <td>0.7995</td>
      <td>0.7989</td>
      <td>0.8021</td>
      <td>0.8015</td>
      <td>0.7974</td>
      <td>0.8006</td>
      <td>0.8018</td>
      <td>0.8012</td>
      <td>0.7989</td>
      <td>0.6683</td>
      <td>0.6683</td>
      <td>0.6723</td>
      <td>0.6683</td>
      <td>0.6688</td>
      <td>0.6718</td>
      <td>0.6683</td>
      <td>0.6673</td>
      <td>0.6673</td>
      <td>0.5763</td>
      <td>0.5498</td>
      <td>0.5560</td>
      <td>0.5767</td>
      <td>0.6048</td>
      <td>0.6095</td>
      <td>0.5763</td>
      <td>0.6091</td>
      <td>0.6063</td>
      <td>0.4931</td>
      <td>0.4619</td>
      <td>0.4671</td>
      <td>0.4955</td>
      <td>0.5001</td>
      <td>0.4922</td>
      <td>0.4941</td>
      <td>0.4538</td>
      <td>0.4771</td>
      <td>0.4771</td>
      <td>0.4771</td>
      <td>0.4771</td>
      <td>0.4771</td>
      <td>0.4771</td>
      <td>0.4771</td>
      <td>0.4771</td>
      <td>0.4771</td>
      <td>0.4771</td>
    </tr>
  </tbody>
</table>

#### Client Average

<table>
  <thead>
    <tr>
      <th rowspan="2">method</th>
      <th colspan="3">bloodmnist_224</th>
      <th colspan="3">dermamnist_224</th>
      <th colspan="3">organcmnist_224</th>
      <th colspan="3">organsmnist_224</th>
      <th colspan="3">chaoshengmnist_224</th>
    </tr>
    <tr>
      <th>c3_avg</th>
      <th>c5_avg</th>
      <th>c7_avg</th>
      <th>c3_avg</th>
      <th>c5_avg</th>
      <th>c7_avg</th>
      <th>c3_avg</th>
      <th>c5_avg</th>
      <th>c7_avg</th>
      <th>c3_avg</th>
      <th>c5_avg</th>
      <th>c7_avg</th>
      <th>c3_avg</th>
      <th>c5_avg</th>
      <th>c7_avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>LAMP-Merge</td>
      <td>0.8002</td>
      <td>0.7999</td>
      <td>0.8006</td>
      <td>0.6697</td>
      <td>0.6697</td>
      <td>0.6677</td>
      <td>0.5607</td>
      <td>0.5970</td>
      <td>0.5972</td>
      <td>0.4740</td>
      <td>0.4959</td>
      <td>0.4750</td>
      <td>0.4771</td>
      <td>0.4771</td>
      <td>0.4771</td>
    </tr>
  </tbody>
</table>

### convnext

#### Raw

<table>
  <thead>
    <tr>
      <th rowspan="2">method</th>
      <th colspan="9">bloodmnist_224</th>
      <th colspan="9">dermamnist_224</th>
      <th colspan="9">organcmnist_224</th>
      <th colspan="9">organsmnist_224</th>
      <th colspan="9">chaoshengmnist_224</th>
    </tr>
    <tr>
      <th>c3_b0</th>
      <th>c3_b0.01</th>
      <th>c3_b0.1</th>
      <th>c5_b0</th>
      <th>c5_b0.01</th>
      <th>c5_b0.1</th>
      <th>c7_b0</th>
      <th>c7_b0.01</th>
      <th>c7_b0.1</th>
      <th>c3_b0</th>
      <th>c3_b0.01</th>
      <th>c3_b0.1</th>
      <th>c5_b0</th>
      <th>c5_b0.01</th>
      <th>c5_b0.1</th>
      <th>c7_b0</th>
      <th>c7_b0.01</th>
      <th>c7_b0.1</th>
      <th>c3_b0</th>
      <th>c3_b0.01</th>
      <th>c3_b0.1</th>
      <th>c5_b0</th>
      <th>c5_b0.01</th>
      <th>c5_b0.1</th>
      <th>c7_b0</th>
      <th>c7_b0.01</th>
      <th>c7_b0.1</th>
      <th>c3_b0</th>
      <th>c3_b0.01</th>
      <th>c3_b0.1</th>
      <th>c5_b0</th>
      <th>c5_b0.01</th>
      <th>c5_b0.1</th>
      <th>c7_b0</th>
      <th>c7_b0.01</th>
      <th>c7_b0.1</th>
      <th>c3_b0</th>
      <th>c3_b0.01</th>
      <th>c3_b0.1</th>
      <th>c5_b0</th>
      <th>c5_b0.01</th>
      <th>c5_b0.1</th>
      <th>c7_b0</th>
      <th>c7_b0.01</th>
      <th>c7_b0.1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>LAMP-Merge</td>
      <td>0.8506</td>
      <td>0.8471</td>
      <td>0.8465</td>
      <td>0.8495</td>
      <td>0.8486</td>
      <td>0.8471</td>
      <td>0.8498</td>
      <td>0.8474</td>
      <td>0.8498</td>
      <td>0.6155</td>
      <td>0.6105</td>
      <td>0.6254</td>
      <td>0.6204</td>
      <td>0.6105</td>
      <td>0.6180</td>
      <td>0.6175</td>
      <td>0.6050</td>
      <td>0.6140</td>
      <td>0.6551</td>
      <td>0.6551</td>
      <td>0.6509</td>
      <td>0.6574</td>
      <td>0.6496</td>
      <td>0.6459</td>
      <td>0.6574</td>
      <td>0.6512</td>
      <td>0.6487</td>
      <td>0.6101</td>
      <td>0.6038</td>
      <td>0.6045</td>
      <td>0.6105</td>
      <td>0.6028</td>
      <td>0.6027</td>
      <td>0.6103</td>
      <td>0.6107</td>
      <td>0.6053</td>
      <td>0.4268</td>
      <td>0.4268</td>
      <td>0.4268</td>
      <td>0.4268</td>
      <td>0.4268</td>
      <td>0.4268</td>
      <td>0.4268</td>
      <td>0.4268</td>
      <td>0.4268</td>
    </tr>
  </tbody>
</table>

#### Client Average

<table>
  <thead>
    <tr>
      <th rowspan="2">method</th>
      <th colspan="3">bloodmnist_224</th>
      <th colspan="3">dermamnist_224</th>
      <th colspan="3">organcmnist_224</th>
      <th colspan="3">organsmnist_224</th>
      <th colspan="3">chaoshengmnist_224</th>
    </tr>
    <tr>
      <th>c3_avg</th>
      <th>c5_avg</th>
      <th>c7_avg</th>
      <th>c3_avg</th>
      <th>c5_avg</th>
      <th>c7_avg</th>
      <th>c3_avg</th>
      <th>c5_avg</th>
      <th>c7_avg</th>
      <th>c3_avg</th>
      <th>c5_avg</th>
      <th>c7_avg</th>
      <th>c3_avg</th>
      <th>c5_avg</th>
      <th>c7_avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>LAMP-Merge</td>
      <td>0.8481</td>
      <td>0.8484</td>
      <td>0.8490</td>
      <td>0.6171</td>
      <td>0.6163</td>
      <td>0.6121</td>
      <td>0.6537</td>
      <td>0.6510</td>
      <td>0.6524</td>
      <td>0.6061</td>
      <td>0.6053</td>
      <td>0.6088</td>
      <td>0.4268</td>
      <td>0.4268</td>
      <td>0.4268</td>
    </tr>
  </tbody>
</table>

### vit_t

#### Raw

<table>
  <thead>
    <tr>
      <th rowspan="2">method</th>
      <th colspan="9">bloodmnist_224</th>
      <th colspan="9">dermamnist_224</th>
      <th colspan="9">organcmnist_224</th>
      <th colspan="9">organsmnist_224</th>
      <th colspan="9">chaoshengmnist_224</th>
    </tr>
    <tr>
      <th>c3_b0</th>
      <th>c3_b0.01</th>
      <th>c3_b0.1</th>
      <th>c5_b0</th>
      <th>c5_b0.01</th>
      <th>c5_b0.1</th>
      <th>c7_b0</th>
      <th>c7_b0.01</th>
      <th>c7_b0.1</th>
      <th>c3_b0</th>
      <th>c3_b0.01</th>
      <th>c3_b0.1</th>
      <th>c5_b0</th>
      <th>c5_b0.01</th>
      <th>c5_b0.1</th>
      <th>c7_b0</th>
      <th>c7_b0.01</th>
      <th>c7_b0.1</th>
      <th>c3_b0</th>
      <th>c3_b0.01</th>
      <th>c3_b0.1</th>
      <th>c5_b0</th>
      <th>c5_b0.01</th>
      <th>c5_b0.1</th>
      <th>c7_b0</th>
      <th>c7_b0.01</th>
      <th>c7_b0.1</th>
      <th>c3_b0</th>
      <th>c3_b0.01</th>
      <th>c3_b0.1</th>
      <th>c5_b0</th>
      <th>c5_b0.01</th>
      <th>c5_b0.1</th>
      <th>c7_b0</th>
      <th>c7_b0.01</th>
      <th>c7_b0.1</th>
      <th>c3_b0</th>
      <th>c3_b0.01</th>
      <th>c3_b0.1</th>
      <th>c5_b0</th>
      <th>c5_b0.01</th>
      <th>c5_b0.1</th>
      <th>c7_b0</th>
      <th>c7_b0.01</th>
      <th>c7_b0.1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>LAMP-Merge</td>
      <td>0.8527</td>
      <td>0.8524</td>
      <td>0.8515</td>
      <td>0.8524</td>
      <td>0.8521</td>
      <td>0.8518</td>
      <td>0.8524</td>
      <td>0.8509</td>
      <td>0.8524</td>
      <td>0.6214</td>
      <td>0.6130</td>
      <td>0.6214</td>
      <td>0.6155</td>
      <td>0.6140</td>
      <td>0.6135</td>
      <td>0.6145</td>
      <td>0.6020</td>
      <td>0.6060</td>
      <td>0.5885</td>
      <td>0.5896</td>
      <td>0.5826</td>
      <td>0.5881</td>
      <td>0.5817</td>
      <td>0.5758</td>
      <td>0.5880</td>
      <td>0.5797</td>
      <td>0.5764</td>
      <td>0.5506</td>
      <td>0.5464</td>
      <td>0.5455</td>
      <td>0.5507</td>
      <td>0.5400</td>
      <td>0.5448</td>
      <td>0.5505</td>
      <td>0.5522</td>
      <td>0.5436</td>
      <td>0.4618</td>
      <td>0.4618</td>
      <td>0.4618</td>
      <td>0.4618</td>
      <td>0.4618</td>
      <td>0.4618</td>
      <td>0.4618</td>
      <td>0.4618</td>
      <td>0.4618</td>
    </tr>
  </tbody>
</table>

#### Client Average

<table>
  <thead>
    <tr>
      <th rowspan="2">method</th>
      <th colspan="3">bloodmnist_224</th>
      <th colspan="3">dermamnist_224</th>
      <th colspan="3">organcmnist_224</th>
      <th colspan="3">organsmnist_224</th>
      <th colspan="3">chaoshengmnist_224</th>
    </tr>
    <tr>
      <th>c3_avg</th>
      <th>c5_avg</th>
      <th>c7_avg</th>
      <th>c3_avg</th>
      <th>c5_avg</th>
      <th>c7_avg</th>
      <th>c3_avg</th>
      <th>c5_avg</th>
      <th>c7_avg</th>
      <th>c3_avg</th>
      <th>c5_avg</th>
      <th>c7_avg</th>
      <th>c3_avg</th>
      <th>c5_avg</th>
      <th>c7_avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>LAMP-Merge</td>
      <td>0.8522</td>
      <td>0.8521</td>
      <td>0.8519</td>
      <td>0.6186</td>
      <td>0.6143</td>
      <td>0.6075</td>
      <td>0.5869</td>
      <td>0.5819</td>
      <td>0.5814</td>
      <td>0.5475</td>
      <td>0.5452</td>
      <td>0.5487</td>
      <td>0.4618</td>
      <td>0.4618</td>
      <td>0.4618</td>
    </tr>
  </tbody>
</table>

### swin_tiny

#### Raw

<table>
  <thead>
    <tr>
      <th rowspan="2">method</th>
      <th colspan="9">bloodmnist_224</th>
      <th colspan="9">dermamnist_224</th>
      <th colspan="9">organcmnist_224</th>
      <th colspan="9">organsmnist_224</th>
      <th colspan="9">chaoshengmnist_224</th>
    </tr>
    <tr>
      <th>c3_b0</th>
      <th>c3_b0.01</th>
      <th>c3_b0.1</th>
      <th>c5_b0</th>
      <th>c5_b0.01</th>
      <th>c5_b0.1</th>
      <th>c7_b0</th>
      <th>c7_b0.01</th>
      <th>c7_b0.1</th>
      <th>c3_b0</th>
      <th>c3_b0.01</th>
      <th>c3_b0.1</th>
      <th>c5_b0</th>
      <th>c5_b0.01</th>
      <th>c5_b0.1</th>
      <th>c7_b0</th>
      <th>c7_b0.01</th>
      <th>c7_b0.1</th>
      <th>c3_b0</th>
      <th>c3_b0.01</th>
      <th>c3_b0.1</th>
      <th>c5_b0</th>
      <th>c5_b0.01</th>
      <th>c5_b0.1</th>
      <th>c7_b0</th>
      <th>c7_b0.01</th>
      <th>c7_b0.1</th>
      <th>c3_b0</th>
      <th>c3_b0.01</th>
      <th>c3_b0.1</th>
      <th>c5_b0</th>
      <th>c5_b0.01</th>
      <th>c5_b0.1</th>
      <th>c7_b0</th>
      <th>c7_b0.01</th>
      <th>c7_b0.1</th>
      <th>c3_b0</th>
      <th>c3_b0.01</th>
      <th>c3_b0.1</th>
      <th>c5_b0</th>
      <th>c5_b0.01</th>
      <th>c5_b0.1</th>
      <th>c7_b0</th>
      <th>c7_b0.01</th>
      <th>c7_b0.1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>LAMP-Merge</td>
      <td>0.7752</td>
      <td>0.7723</td>
      <td>0.7735</td>
      <td>0.7758</td>
      <td>0.7761</td>
      <td>0.7738</td>
      <td>0.7743</td>
      <td>0.7749</td>
      <td>0.7755</td>
      <td>0.6479</td>
      <td>0.6464</td>
      <td>0.6534</td>
      <td>0.6504</td>
      <td>0.6439</td>
      <td>0.6454</td>
      <td>0.6484</td>
      <td>0.6449</td>
      <td>0.6339</td>
      <td>0.6835</td>
      <td>0.6863</td>
      <td>0.6737</td>
      <td>0.6837</td>
      <td>0.6630</td>
      <td>0.6652</td>
      <td>0.6854</td>
      <td>0.6642</td>
      <td>0.6611</td>
      <td>0.6286</td>
      <td>0.6045</td>
      <td>0.5994</td>
      <td>0.6284</td>
      <td>0.6122</td>
      <td>0.6112</td>
      <td>0.6280</td>
      <td>0.6050</td>
      <td>0.6136</td>
      <td>0.4816</td>
      <td>0.4816</td>
      <td>0.4816</td>
      <td>0.4816</td>
      <td>0.4816</td>
      <td>0.4816</td>
      <td>0.4816</td>
      <td>0.4816</td>
      <td>0.4816</td>
    </tr>
  </tbody>
</table>

#### Client Average

<table>
  <thead>
    <tr>
      <th rowspan="2">method</th>
      <th colspan="3">bloodmnist_224</th>
      <th colspan="3">dermamnist_224</th>
      <th colspan="3">organcmnist_224</th>
      <th colspan="3">organsmnist_224</th>
      <th colspan="3">chaoshengmnist_224</th>
    </tr>
    <tr>
      <th>c3_avg</th>
      <th>c5_avg</th>
      <th>c7_avg</th>
      <th>c3_avg</th>
      <th>c5_avg</th>
      <th>c7_avg</th>
      <th>c3_avg</th>
      <th>c5_avg</th>
      <th>c7_avg</th>
      <th>c3_avg</th>
      <th>c5_avg</th>
      <th>c7_avg</th>
      <th>c3_avg</th>
      <th>c5_avg</th>
      <th>c7_avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>LAMP-Merge</td>
      <td>0.7737</td>
      <td>0.7752</td>
      <td>0.7749</td>
      <td>0.6492</td>
      <td>0.6466</td>
      <td>0.6424</td>
      <td>0.6812</td>
      <td>0.6706</td>
      <td>0.6702</td>
      <td>0.6109</td>
      <td>0.6173</td>
      <td>0.6155</td>
      <td>0.4816</td>
      <td>0.4816</td>
      <td>0.4816</td>
    </tr>
  </tbody>
</table>

## VLM

### openai/clip-vit-base-patch32

#### Raw

<table>
  <thead>
    <tr>
      <th rowspan="2">method</th>
      <th colspan="9">bloodmnist_224</th>
      <th colspan="9">dermamnist_224</th>
      <th colspan="9">organcmnist_224</th>
      <th colspan="9">organsmnist_224</th>
      <th colspan="9">chaoshengmnist_224</th>
    </tr>
    <tr>
      <th>c3_b0</th>
      <th>c3_b0.01</th>
      <th>c3_b0.1</th>
      <th>c5_b0</th>
      <th>c5_b0.01</th>
      <th>c5_b0.1</th>
      <th>c7_b0</th>
      <th>c7_b0.01</th>
      <th>c7_b0.1</th>
      <th>c3_b0</th>
      <th>c3_b0.01</th>
      <th>c3_b0.1</th>
      <th>c5_b0</th>
      <th>c5_b0.01</th>
      <th>c5_b0.1</th>
      <th>c7_b0</th>
      <th>c7_b0.01</th>
      <th>c7_b0.1</th>
      <th>c3_b0</th>
      <th>c3_b0.01</th>
      <th>c3_b0.1</th>
      <th>c5_b0</th>
      <th>c5_b0.01</th>
      <th>c5_b0.1</th>
      <th>c7_b0</th>
      <th>c7_b0.01</th>
      <th>c7_b0.1</th>
      <th>c3_b0</th>
      <th>c3_b0.01</th>
      <th>c3_b0.1</th>
      <th>c5_b0</th>
      <th>c5_b0.01</th>
      <th>c5_b0.1</th>
      <th>c7_b0</th>
      <th>c7_b0.01</th>
      <th>c7_b0.1</th>
      <th>c3_b0</th>
      <th>c3_b0.01</th>
      <th>c3_b0.1</th>
      <th>c5_b0</th>
      <th>c5_b0.01</th>
      <th>c5_b0.1</th>
      <th>c7_b0</th>
      <th>c7_b0.01</th>
      <th>c7_b0.1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>LAMP-Merge</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

#### Client Average

<table>
  <thead>
    <tr>
      <th rowspan="2">method</th>
      <th colspan="3">bloodmnist_224</th>
      <th colspan="3">dermamnist_224</th>
      <th colspan="3">organcmnist_224</th>
      <th colspan="3">organsmnist_224</th>
      <th colspan="3">chaoshengmnist_224</th>
    </tr>
    <tr>
      <th>c3_avg</th>
      <th>c5_avg</th>
      <th>c7_avg</th>
      <th>c3_avg</th>
      <th>c5_avg</th>
      <th>c7_avg</th>
      <th>c3_avg</th>
      <th>c5_avg</th>
      <th>c7_avg</th>
      <th>c3_avg</th>
      <th>c5_avg</th>
      <th>c7_avg</th>
      <th>c3_avg</th>
      <th>c5_avg</th>
      <th>c7_avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>LAMP-Merge</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
</table>
