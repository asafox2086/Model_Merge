# Experiment Master Tables

- Layout: aligned with `result/all_results.md`.
- Source output root: `outputs/my_merge_reference_proto_recall_full_table_20260705`.
- Values are filled from real `eval_summary.csv` results for `my_merge`; missing combinations are shown as `-`.

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
      <td>my_merge</td>
      <td>0.7995</td>
      <td>0.7989</td>
      <td>0.8021</td>
      <td>0.8015</td>
      <td>0.7974</td>
      <td>0.8006</td>
      <td>0.8018</td>
      <td>0.8012</td>
      <td>0.7992</td>
      <td>0.4339</td>
      <td>0.4324</td>
      <td>0.4314</td>
      <td>0.4284</td>
      <td>0.5302</td>
      <td>0.4329</td>
      <td>0.4329</td>
      <td>0.6653</td>
      <td>0.4354</td>
      <td>0.6076</td>
      <td>0.6072</td>
      <td>0.6074</td>
      <td>0.6064</td>
      <td>0.6050</td>
      <td>0.6094</td>
      <td>0.6070</td>
      <td>0.6095</td>
      <td>0.6061</td>
      <td>0.5528</td>
      <td>0.5532</td>
      <td>0.5533</td>
      <td>0.5544</td>
      <td>0.5527</td>
      <td>0.5501</td>
      <td>0.5536</td>
      <td>0.5525</td>
      <td>0.5490</td>
      <td>0.4762</td>
      <td>0.4762</td>
      <td>0.4762</td>
      <td>0.4762</td>
      <td>0.4762</td>
      <td>0.4762</td>
      <td>0.4762</td>
      <td>0.4762</td>
      <td>0.4762</td>
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
      <td>my_merge</td>
      <td>0.8002</td>
      <td>0.7999</td>
      <td>0.8007</td>
      <td>0.4326</td>
      <td>0.4638</td>
      <td>0.5112</td>
      <td>0.6074</td>
      <td>0.6069</td>
      <td>0.6076</td>
      <td>0.5531</td>
      <td>0.5524</td>
      <td>0.5517</td>
      <td>0.4762</td>
      <td>0.4762</td>
      <td>0.4762</td>
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
      <td>my_merge</td>
      <td>0.8506</td>
      <td>0.8471</td>
      <td>0.8468</td>
      <td>0.8495</td>
      <td>0.8483</td>
      <td>0.8471</td>
      <td>0.8498</td>
      <td>0.8474</td>
      <td>0.8506</td>
      <td>0.4354</td>
      <td>0.4349</td>
      <td>0.4334</td>
      <td>0.4319</td>
      <td>0.4594</td>
      <td>0.4324</td>
      <td>0.4294</td>
      <td>0.5282</td>
      <td>0.4319</td>
      <td>0.6490</td>
      <td>0.6453</td>
      <td>0.6503</td>
      <td>0.6503</td>
      <td>0.6491</td>
      <td>0.6447</td>
      <td>0.6501</td>
      <td>0.6510</td>
      <td>0.6475</td>
      <td>0.5993</td>
      <td>0.5968</td>
      <td>0.5969</td>
      <td>0.5994</td>
      <td>0.5991</td>
      <td>0.5970</td>
      <td>0.6002</td>
      <td>0.5996</td>
      <td>0.5971</td>
      <td>0.4259</td>
      <td>0.4259</td>
      <td>0.4259</td>
      <td>0.4259</td>
      <td>0.4259</td>
      <td>0.4259</td>
      <td>0.4259</td>
      <td>0.4259</td>
      <td>0.4259</td>
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
      <td>my_merge</td>
      <td>0.8482</td>
      <td>0.8483</td>
      <td>0.8493</td>
      <td>0.4346</td>
      <td>0.4412</td>
      <td>0.4632</td>
      <td>0.6482</td>
      <td>0.6480</td>
      <td>0.6495</td>
      <td>0.5977</td>
      <td>0.5985</td>
      <td>0.5990</td>
      <td>0.4259</td>
      <td>0.4259</td>
      <td>0.4259</td>
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
      <td>my_merge</td>
      <td>0.8527</td>
      <td>0.8524</td>
      <td>0.8518</td>
      <td>0.8524</td>
      <td>0.8524</td>
      <td>0.8518</td>
      <td>0.8524</td>
      <td>0.8503</td>
      <td>0.8521</td>
      <td>0.5132</td>
      <td>0.5127</td>
      <td>0.5132</td>
      <td>0.5132</td>
      <td>0.5227</td>
      <td>0.5112</td>
      <td>0.5112</td>
      <td>0.5541</td>
      <td>0.5122</td>
      <td>0.5813</td>
      <td>0.5813</td>
      <td>0.5787</td>
      <td>0.5801</td>
      <td>0.5809</td>
      <td>0.5759</td>
      <td>0.5812</td>
      <td>0.5791</td>
      <td>0.5766</td>
      <td>0.5417</td>
      <td>0.5395</td>
      <td>0.5406</td>
      <td>0.5406</td>
      <td>0.5333</td>
      <td>0.5378</td>
      <td>0.5407</td>
      <td>0.5400</td>
      <td>0.5357</td>
      <td>0.4645</td>
      <td>0.4645</td>
      <td>0.4645</td>
      <td>0.4645</td>
      <td>0.4645</td>
      <td>0.4645</td>
      <td>0.4645</td>
      <td>0.4645</td>
      <td>0.4645</td>
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
      <td>my_merge</td>
      <td>0.8523</td>
      <td>0.8522</td>
      <td>0.8516</td>
      <td>0.5131</td>
      <td>0.5157</td>
      <td>0.5259</td>
      <td>0.5805</td>
      <td>0.5790</td>
      <td>0.5790</td>
      <td>0.5406</td>
      <td>0.5372</td>
      <td>0.5388</td>
      <td>0.4645</td>
      <td>0.4645</td>
      <td>0.4645</td>
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
      <td>my_merge</td>
      <td>0.7752</td>
      <td>0.7723</td>
      <td>0.7735</td>
      <td>0.7758</td>
      <td>0.7761</td>
      <td>0.7740</td>
      <td>0.7743</td>
      <td>0.7749</td>
      <td>0.7752</td>
      <td>0.4658</td>
      <td>0.4708</td>
      <td>0.4743</td>
      <td>0.4748</td>
      <td>0.4998</td>
      <td>0.4648</td>
      <td>0.4713</td>
      <td>0.5751</td>
      <td>0.4693</td>
      <td>0.6629</td>
      <td>0.6657</td>
      <td>0.6667</td>
      <td>0.6627</td>
      <td>0.6631</td>
      <td>0.6638</td>
      <td>0.6638</td>
      <td>0.6642</td>
      <td>0.6609</td>
      <td>0.6094</td>
      <td>0.6046</td>
      <td>0.6094</td>
      <td>0.6079</td>
      <td>0.6063</td>
      <td>0.6087</td>
      <td>0.6064</td>
      <td>0.6081</td>
      <td>0.6058</td>
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
      <td>my_merge</td>
      <td>0.7737</td>
      <td>0.7753</td>
      <td>0.7748</td>
      <td>0.4703</td>
      <td>0.4798</td>
      <td>0.5052</td>
      <td>0.6651</td>
      <td>0.6632</td>
      <td>0.6630</td>
      <td>0.6078</td>
      <td>0.6076</td>
      <td>0.6068</td>
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
      <td>my_merge</td>
      <td>0.3063</td>
      <td>0.2955</td>
      <td>0.5706</td>
      <td>0.0918</td>
      <td>0.2274</td>
      <td>0.4844</td>
      <td>0.1798</td>
      <td>0.3227</td>
      <td>0.2423</td>
      <td>0.3397</td>
      <td>0.6753</td>
      <td>0.6788</td>
      <td>0.6648</td>
      <td>0.6688</td>
      <td>0.7202</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.3270</td>
      <td>0.5062</td>
      <td>0.5047</td>
      <td>0.3104</td>
      <td>0.1440</td>
      <td>0.2181</td>
      <td>0.1140</td>
      <td>0.1121</td>
      <td>0.2142</td>
      <td>0.1919</td>
      <td>0.2207</td>
      <td>0.3885</td>
      <td>0.0958</td>
      <td>0.3245</td>
      <td>0.2685</td>
      <td>0.2354</td>
      <td>0.2041</td>
      <td>0.3277</td>
      <td>0.1680</td>
      <td>0.1977</td>
      <td>0.2183</td>
      <td>0.1285</td>
      <td>0.1923</td>
      <td>0.1617</td>
      <td>0.1348</td>
      <td>0.1824</td>
      <td>0.1635</td>
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
      <td>my_merge</td>
      <td>0.3908</td>
      <td>0.2679</td>
      <td>0.2483</td>
      <td>0.5646</td>
      <td>0.6846</td>
      <td>0.6688</td>
      <td>0.4460</td>
      <td>0.2242</td>
      <td>0.1468</td>
      <td>0.2670</td>
      <td>0.2296</td>
      <td>0.2558</td>
      <td>0.1947</td>
      <td>0.1608</td>
      <td>0.1602</td>
    </tr>
  </tbody>
</table>
