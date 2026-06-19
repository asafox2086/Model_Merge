# Experiment Master Tables

- Layout: aligned with `result/all_results.md`.
- Source output root: `outputs/codex_simple_v16e_swin_vit_probe_20260619`.
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
      <td>0.3825</td>
      <td>0.3222</td>
      <td>0.3543</td>
      <td>0.3055</td>
      <td>0.3788</td>
      <td>0.3550</td>
      <td>0.2434</td>
      <td>0.2641</td>
      <td>0.3230</td>
      <td>0.2488</td>
      <td>0.3095</td>
      <td>0.3573</td>
      <td>0.3580</td>
      <td>0.1778</td>
      <td>0.3837</td>
      <td>0.2406</td>
      <td>0.2130</td>
      <td>0.3064</td>
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
      <td>my_merge</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.3530</td>
      <td>0.3464</td>
      <td>0.2769</td>
      <td>0.3052</td>
      <td>0.3065</td>
      <td>0.2534</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
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
      <td>0.2240</td>
      <td>0.1711</td>
      <td>0.0901</td>
      <td>0.2233</td>
      <td>0.0919</td>
      <td>0.0895</td>
      <td>0.2233</td>
      <td>0.2233</td>
      <td>0.0512</td>
      <td>0.1251</td>
      <td>0.0497</td>
      <td>0.0785</td>
      <td>0.0785</td>
      <td>0.0798</td>
      <td>0.1521</td>
      <td>0.2354</td>
      <td>0.2354</td>
      <td>0.1097</td>
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
      <td>my_merge</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.1617</td>
      <td>0.1349</td>
      <td>0.1660</td>
      <td>0.0844</td>
      <td>0.1035</td>
      <td>0.1935</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
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
      <td>0.0730</td>
      <td>0.2723</td>
      <td>0.2090</td>
      <td>0.0896</td>
      <td>0.0952</td>
      <td>0.1852</td>
      <td>0.2791</td>
      <td>0.3173</td>
      <td>0.2100</td>
      <td>0.0642</td>
      <td>0.0569</td>
      <td>0.1000</td>
      <td>0.1026</td>
      <td>0.1459</td>
      <td>0.1099</td>
      <td>0.1532</td>
      <td>0.3469</td>
      <td>0.1594</td>
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
      <td>my_merge</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.1848</td>
      <td>0.1233</td>
      <td>0.2688</td>
      <td>0.0737</td>
      <td>0.1195</td>
      <td>0.2198</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
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
      <td>0.0885</td>
      <td>0.2083</td>
      <td>0.2233</td>
      <td>0.2233</td>
      <td>0.1694</td>
      <td>0.0895</td>
      <td>0.3089</td>
      <td>0.1439</td>
      <td>0.1008</td>
      <td>0.0578</td>
      <td>0.1826</td>
      <td>0.0787</td>
      <td>0.1421</td>
      <td>0.0497</td>
      <td>0.1112</td>
      <td>0.1835</td>
      <td>0.1521</td>
      <td>0.0497</td>
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
      <td>my_merge</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.1734</td>
      <td>0.1607</td>
      <td>0.1845</td>
      <td>0.1064</td>
      <td>0.1010</td>
      <td>0.1285</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
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
      <td>my_merge</td>
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
