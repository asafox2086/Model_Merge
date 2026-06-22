# Experiment Master Tables

- Layout: aligned with `result/all_results.md`.
- Source output root: `outputs/custom_methods_codex_cel_pcm_reliability_full_20260621`.
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
      <td>0.3189</td>
      <td>0.3607</td>
      <td>0.4294</td>
      <td>0.3207</td>
      <td>0.3528</td>
      <td>0.4396</td>
      <td>0.3154</td>
      <td>0.4388</td>
      <td>0.2947</td>
      <td>0.6449</td>
      <td>0.6818</td>
      <td>0.6863</td>
      <td>0.6424</td>
      <td>0.6698</td>
      <td>0.6613</td>
      <td>0.6688</td>
      <td>0.6693</td>
      <td>0.6688</td>
      <td>0.3961</td>
      <td>0.4223</td>
      <td>0.3580</td>
      <td>0.2824</td>
      <td>0.2695</td>
      <td>0.2958</td>
      <td>0.2359</td>
      <td>0.2135</td>
      <td>0.1776</td>
      <td>0.1706</td>
      <td>0.3413</td>
      <td>0.3289</td>
      <td>0.2269</td>
      <td>0.1404</td>
      <td>0.2167</td>
      <td>0.2633</td>
      <td>0.2449</td>
      <td>0.4066</td>
      <td>0.2561</td>
      <td>0.3854</td>
      <td>0.2210</td>
      <td>0.1653</td>
      <td>0.2084</td>
      <td>0.1375</td>
      <td>0.2066</td>
      <td>0.1752</td>
      <td>0.1617</td>
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
      <td>0.3697</td>
      <td>0.3710</td>
      <td>0.3496</td>
      <td>0.6710</td>
      <td>0.6579</td>
      <td>0.6690</td>
      <td>0.3921</td>
      <td>0.2825</td>
      <td>0.2090</td>
      <td>0.2803</td>
      <td>0.1947</td>
      <td>0.3049</td>
      <td>0.2875</td>
      <td>0.1704</td>
      <td>0.1812</td>
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
      <td>0.1374</td>
      <td>0.0830</td>
      <td>0.0710</td>
      <td>0.1824</td>
      <td>0.1374</td>
      <td>0.1947</td>
      <td>0.0830</td>
      <td>0.1112</td>
      <td>0.1097</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.1112</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.1102</td>
      <td>0.1095</td>
      <td>0.0913</td>
      <td>0.2233</td>
      <td>0.0525</td>
      <td>0.0895</td>
      <td>0.0913</td>
      <td>0.1008</td>
      <td>0.0913</td>
      <td>0.2354</td>
      <td>0.0785</td>
      <td>0.0785</td>
      <td>0.0785</td>
      <td>0.0798</td>
      <td>0.1521</td>
      <td>0.0798</td>
      <td>0.2354</td>
      <td>0.1097</td>
      <td>0.1051</td>
      <td>0.1734</td>
      <td>0.1734</td>
      <td>0.1734</td>
      <td>0.1051</td>
      <td>0.1734</td>
      <td>0.1087</td>
      <td>0.1051</td>
      <td>0.1617</td>
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
      <td>0.1565</td>
      <td>0.1122</td>
      <td>0.1384</td>
      <td>0.2966</td>
      <td>0.4830</td>
      <td>0.6688</td>
      <td>0.1037</td>
      <td>0.1218</td>
      <td>0.0944</td>
      <td>0.1308</td>
      <td>0.1035</td>
      <td>0.1416</td>
      <td>0.1506</td>
      <td>0.1506</td>
      <td>0.1252</td>
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
      <td>0.1692</td>
      <td>0.0836</td>
      <td>0.1505</td>
      <td>0.0710</td>
      <td>0.0710</td>
      <td>0.1824</td>
      <td>0.1692</td>
      <td>0.0956</td>
      <td>0.0830</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.0539</td>
      <td>0.6504</td>
      <td>0.5506</td>
      <td>0.1097</td>
      <td>0.6608</td>
      <td>0.3367</td>
      <td>0.1112</td>
      <td>0.1610</td>
      <td>0.1506</td>
      <td>0.1854</td>
      <td>0.0525</td>
      <td>0.0923</td>
      <td>0.1683</td>
      <td>0.1255</td>
      <td>0.0613</td>
      <td>0.0988</td>
      <td>0.0983</td>
      <td>0.0587</td>
      <td>0.0963</td>
      <td>0.0530</td>
      <td>0.1426</td>
      <td>0.0599</td>
      <td>0.1274</td>
      <td>0.2354</td>
      <td>0.1502</td>
      <td>0.1339</td>
      <td>0.1743</td>
      <td>0.1267</td>
      <td>0.1258</td>
      <td>0.0988</td>
      <td>0.1734</td>
      <td>0.1087</td>
      <td>0.2228</td>
      <td>0.2336</td>
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
      <td>0.1345</td>
      <td>0.1082</td>
      <td>0.1160</td>
      <td>0.4638</td>
      <td>0.4369</td>
      <td>0.3696</td>
      <td>0.1657</td>
      <td>0.1043</td>
      <td>0.0952</td>
      <td>0.0844</td>
      <td>0.0852</td>
      <td>0.1710</td>
      <td>0.1450</td>
      <td>0.1327</td>
      <td>0.1884</td>
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
      <td>0.1374</td>
      <td>0.1374</td>
      <td>0.0912</td>
      <td>0.1932</td>
      <td>0.0713</td>
      <td>0.1824</td>
      <td>0.1374</td>
      <td>0.1374</td>
      <td>0.0329</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.1112</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.0885</td>
      <td>0.1807</td>
      <td>0.0512</td>
      <td>0.2289</td>
      <td>0.0678</td>
      <td>0.0913</td>
      <td>0.0895</td>
      <td>0.0555</td>
      <td>0.0913</td>
      <td>0.0578</td>
      <td>0.0682</td>
      <td>0.0785</td>
      <td>0.0600</td>
      <td>0.0497</td>
      <td>0.0717</td>
      <td>0.1521</td>
      <td>0.1526</td>
      <td>0.1097</td>
      <td>0.1087</td>
      <td>0.1734</td>
      <td>0.1761</td>
      <td>0.1096</td>
      <td>0.1051</td>
      <td>0.1734</td>
      <td>0.1734</td>
      <td>0.1563</td>
      <td>0.1096</td>
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
      <td>0.1565</td>
      <td>0.1186</td>
      <td>0.1524</td>
      <td>0.4569</td>
      <td>0.4830</td>
      <td>0.6688</td>
      <td>0.1068</td>
      <td>0.1293</td>
      <td>0.0787</td>
      <td>0.0682</td>
      <td>0.0605</td>
      <td>0.1381</td>
      <td>0.1527</td>
      <td>0.1294</td>
      <td>0.1465</td>
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
      <td>0.2675</td>
      <td>0.2698</td>
      <td>0.3043</td>
      <td>0.0909</td>
      <td>0.0830</td>
      <td>0.0830</td>
      <td>0.1374</td>
      <td>0.1947</td>
      <td>0.1672</td>
      <td>0.6683</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.1426</td>
      <td>0.1282</td>
      <td>0.3257</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.1043</td>
      <td>0.1154</td>
      <td>0.1103</td>
      <td>0.0683</td>
      <td>0.0668</td>
      <td>0.2276</td>
      <td>0.2020</td>
      <td>0.2431</td>
      <td>0.2233</td>
      <td>0.0810</td>
      <td>0.2398</td>
      <td>0.3731</td>
      <td>0.0497</td>
      <td>0.0919</td>
      <td>0.2594</td>
      <td>0.2354</td>
      <td>0.0793</td>
      <td>0.1092</td>
      <td>0.1734</td>
      <td>0.1617</td>
      <td>0.1950</td>
      <td>0.1222</td>
      <td>0.1617</td>
      <td>0.1258</td>
      <td>0.1195</td>
      <td>0.1617</td>
      <td>0.1267</td>
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
      <td>0.2805</td>
      <td>0.0856</td>
      <td>0.1664</td>
      <td>0.6687</td>
      <td>0.1988</td>
      <td>0.6688</td>
      <td>0.1100</td>
      <td>0.1209</td>
      <td>0.2228</td>
      <td>0.2313</td>
      <td>0.1337</td>
      <td>0.1413</td>
      <td>0.1767</td>
      <td>0.1366</td>
      <td>0.1360</td>
    </tr>
  </tbody>
</table>
