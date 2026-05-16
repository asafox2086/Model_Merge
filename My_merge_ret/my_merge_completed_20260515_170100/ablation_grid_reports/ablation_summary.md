# Experiment Master Tables

- Layout: aligned with `result/all_results.md`.
- Source output root: `/data/liyapeng_grp/program/MedMNISTMerge/outputs/my_merge_ablation_three_module_full_20260515_170100`.
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
      <td>0.4098</td>
      <td>0.3327</td>
      <td>0.9044</td>
      <td>0.2534</td>
      <td>0.3148</td>
      <td>0.6279</td>
      <td>0.2023</td>
      <td>0.3087</td>
      <td>0.5621</td>
      <td>0.6589</td>
      <td>0.7007</td>
      <td>0.7601</td>
      <td>0.6688</td>
      <td>0.6678</td>
      <td>0.6678</td>
      <td>0.6678</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.3363</td>
      <td>0.3547</td>
      <td>0.6649</td>
      <td>0.3895</td>
      <td>0.1745</td>
      <td>0.3163</td>
      <td>0.2384</td>
      <td>0.2152</td>
      <td>0.1676</td>
      <td>0.4206</td>
      <td>0.3019</td>
      <td>0.5499</td>
      <td>0.3230</td>
      <td>0.3188</td>
      <td>0.3239</td>
      <td>0.1963</td>
      <td>0.2140</td>
      <td>0.3075</td>
      <td>0.3459</td>
      <td>0.1447</td>
      <td>0.2462</td>
      <td>0.2111</td>
      <td>0.1087</td>
      <td>0.1653</td>
      <td>0.1680</td>
      <td>0.3666</td>
      <td>0.2354</td>
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
      <td>0.5490</td>
      <td>0.3987</td>
      <td>0.3577</td>
      <td>0.7066</td>
      <td>0.6682</td>
      <td>0.6685</td>
      <td>0.4520</td>
      <td>0.2935</td>
      <td>0.2071</td>
      <td>0.4242</td>
      <td>0.3219</td>
      <td>0.2393</td>
      <td>0.2456</td>
      <td>0.1617</td>
      <td>0.2567</td>
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
      <td>0.1947</td>
      <td>0.1374</td>
      <td>0.1947</td>
      <td>0.1947</td>
      <td>0.1947</td>
      <td>0.1947</td>
      <td>0.1947</td>
      <td>0.1947</td>
      <td>0.1947</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.0678</td>
      <td>0.1473</td>
      <td>0.3277</td>
      <td>0.1171</td>
      <td>0.0678</td>
      <td>0.0678</td>
      <td>0.0678</td>
      <td>0.1465</td>
      <td>0.0913</td>
      <td>0.2354</td>
      <td>0.2354</td>
      <td>0.2354</td>
      <td>0.1521</td>
      <td>0.0578</td>
      <td>0.2354</td>
      <td>0.2650</td>
      <td>0.2354</td>
      <td>0.1521</td>
      <td>0.1087</td>
      <td>0.1087</td>
      <td>0.1087</td>
      <td>0.1087</td>
      <td>0.1087</td>
      <td>0.1087</td>
      <td>0.1087</td>
      <td>0.1087</td>
      <td>0.1087</td>
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
      <td>0.1756</td>
      <td>0.1947</td>
      <td>0.1947</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.1809</td>
      <td>0.0842</td>
      <td>0.1019</td>
      <td>0.2354</td>
      <td>0.1484</td>
      <td>0.2175</td>
      <td>0.1087</td>
      <td>0.1087</td>
      <td>0.1087</td>
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
      <td>0.3195</td>
      <td>0.1926</td>
      <td>0.2625</td>
      <td>0.1947</td>
      <td>0.3113</td>
      <td>0.2961</td>
      <td>0.2640</td>
      <td>0.1947</td>
      <td>0.2552</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.1957</td>
      <td>0.1720</td>
      <td>0.1933</td>
      <td>0.1548</td>
      <td>0.1205</td>
      <td>0.1807</td>
      <td>0.2170</td>
      <td>0.1974</td>
      <td>0.0930</td>
      <td>0.1585</td>
      <td>0.3055</td>
      <td>0.2731</td>
      <td>0.2470</td>
      <td>0.1632</td>
      <td>0.2684</td>
      <td>0.3175</td>
      <td>0.3563</td>
      <td>0.2102</td>
      <td>0.1087</td>
      <td>0.1087</td>
      <td>0.1087</td>
      <td>0.1087</td>
      <td>0.1087</td>
      <td>0.1087</td>
      <td>0.1087</td>
      <td>0.1087</td>
      <td>0.1087</td>
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
      <td>0.2582</td>
      <td>0.2674</td>
      <td>0.2379</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.1870</td>
      <td>0.1520</td>
      <td>0.1691</td>
      <td>0.2457</td>
      <td>0.2262</td>
      <td>0.2947</td>
      <td>0.1087</td>
      <td>0.1087</td>
      <td>0.1087</td>
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
      <td>0.1947</td>
      <td>0.1403</td>
      <td>0.1947</td>
      <td>0.1976</td>
      <td>0.1947</td>
      <td>0.1947</td>
      <td>0.1947</td>
      <td>0.1947</td>
      <td>0.1947</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.0678</td>
      <td>0.3093</td>
      <td>0.1171</td>
      <td>0.1508</td>
      <td>0.0678</td>
      <td>0.1008</td>
      <td>0.0678</td>
      <td>0.1520</td>
      <td>0.0913</td>
      <td>0.1521</td>
      <td>0.2028</td>
      <td>0.3484</td>
      <td>0.2164</td>
      <td>0.2415</td>
      <td>0.1987</td>
      <td>0.1229</td>
      <td>0.1893</td>
      <td>0.3013</td>
      <td>0.1087</td>
      <td>0.1087</td>
      <td>0.1087</td>
      <td>0.1087</td>
      <td>0.1087</td>
      <td>0.1087</td>
      <td>0.1087</td>
      <td>0.1087</td>
      <td>0.1087</td>
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
      <td>0.1766</td>
      <td>0.1957</td>
      <td>0.1947</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.1647</td>
      <td>0.1065</td>
      <td>0.1037</td>
      <td>0.2344</td>
      <td>0.2189</td>
      <td>0.2045</td>
      <td>0.1087</td>
      <td>0.1087</td>
      <td>0.1087</td>
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
      <td>0.3332</td>
      <td>0.3312</td>
      <td>0.5878</td>
      <td>0.1947</td>
      <td>0.4087</td>
      <td>0.6469</td>
      <td>0.1947</td>
      <td>0.3905</td>
      <td>0.6106</td>
      <td>0.6758</td>
      <td>0.6873</td>
      <td>0.6998</td>
      <td>0.6693</td>
      <td>0.6958</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6678</td>
      <td>0.2672</td>
      <td>0.5527</td>
      <td>0.6189</td>
      <td>0.3737</td>
      <td>0.1274</td>
      <td>0.5417</td>
      <td>0.1827</td>
      <td>0.1659</td>
      <td>0.3688</td>
      <td>0.2149</td>
      <td>0.3273</td>
      <td>0.5620</td>
      <td>0.3537</td>
      <td>0.3274</td>
      <td>0.4431</td>
      <td>0.1966</td>
      <td>0.2530</td>
      <td>0.5177</td>
      <td>0.1851</td>
      <td>0.1087</td>
      <td>0.1734</td>
      <td>0.1087</td>
      <td>0.1186</td>
      <td>0.1087</td>
      <td>0.1195</td>
      <td>0.1878</td>
      <td>0.1689</td>
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
      <td>0.4174</td>
      <td>0.4167</td>
      <td>0.3986</td>
      <td>0.6876</td>
      <td>0.6780</td>
      <td>0.6685</td>
      <td>0.4796</td>
      <td>0.3476</td>
      <td>0.2391</td>
      <td>0.3681</td>
      <td>0.3747</td>
      <td>0.3224</td>
      <td>0.1557</td>
      <td>0.1120</td>
      <td>0.1587</td>
    </tr>
  </tbody>
</table>
