# Experiment Master Tables

- Layout: aligned with `result/all_results.md`.
- Source output root: `outputs/codex_dualprobe_fulltable_20260701_110908_g0, outputs/codex_dualprobe_fulltable_20260701_110908_g1`.
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
      <td>0.5823</td>
      <td>0.5668</td>
      <td>0.8556</td>
      <td>0.4174</td>
      <td>0.3414</td>
      <td>0.7363</td>
      <td>0.3619</td>
      <td>0.4575</td>
      <td>0.5846</td>
      <td>0.6798</td>
      <td>0.7082</td>
      <td>0.7521</td>
      <td>0.6723</td>
      <td>0.7242</td>
      <td>0.7436</td>
      <td>0.6688</td>
      <td>0.6718</td>
      <td>0.6863</td>
      <td>0.5758</td>
      <td>0.6381</td>
      <td>0.7012</td>
      <td>0.3724</td>
      <td>0.3263</td>
      <td>0.5506</td>
      <td>0.3337</td>
      <td>0.1736</td>
      <td>0.5396</td>
      <td>0.5901</td>
      <td>0.4214</td>
      <td>0.6080</td>
      <td>0.3701</td>
      <td>0.2978</td>
      <td>0.3964</td>
      <td>0.3398</td>
      <td>0.3188</td>
      <td>0.4492</td>
      <td>0.4996</td>
      <td>0.6173</td>
      <td>0.6173</td>
      <td>0.3037</td>
      <td>0.5310</td>
      <td>0.4358</td>
      <td>0.2668</td>
      <td>0.3648</td>
      <td>0.3369</td>
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
      <td>0.6682</td>
      <td>0.4984</td>
      <td>0.4680</td>
      <td>0.7134</td>
      <td>0.7134</td>
      <td>0.6756</td>
      <td>0.6384</td>
      <td>0.4165</td>
      <td>0.3490</td>
      <td>0.5399</td>
      <td>0.3548</td>
      <td>0.3692</td>
      <td>0.5780</td>
      <td>0.4235</td>
      <td>0.3229</td>
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
      <td>0.1947</td>
      <td>0.1947</td>
      <td>0.1947</td>
      <td>0.1947</td>
      <td>0.1824</td>
      <td>0.1947</td>
      <td>0.1947</td>
      <td>0.2520</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.2237</td>
      <td>0.1493</td>
      <td>0.1698</td>
      <td>0.2233</td>
      <td>0.2233</td>
      <td>0.2233</td>
      <td>0.2233</td>
      <td>0.2233</td>
      <td>0.0935</td>
      <td>0.2387</td>
      <td>0.2354</td>
      <td>0.2354</td>
      <td>0.1521</td>
      <td>0.2356</td>
      <td>0.2354</td>
      <td>0.2354</td>
      <td>0.2354</td>
      <td>0.1664</td>
      <td>0.1734</td>
      <td>0.1734</td>
      <td>0.1734</td>
      <td>0.1734</td>
      <td>0.1734</td>
      <td>0.1734</td>
      <td>0.1734</td>
      <td>0.1770</td>
      <td>0.1734</td>
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
      <td>0.1947</td>
      <td>0.1906</td>
      <td>0.2138</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.1809</td>
      <td>0.2233</td>
      <td>0.1801</td>
      <td>0.2365</td>
      <td>0.2077</td>
      <td>0.2124</td>
      <td>0.1734</td>
      <td>0.1734</td>
      <td>0.1746</td>
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
      <td>0.2265</td>
      <td>0.2695</td>
      <td>0.2090</td>
      <td>0.1947</td>
      <td>0.2303</td>
      <td>0.2204</td>
      <td>0.2605</td>
      <td>0.3017</td>
      <td>0.1692</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6693</td>
      <td>0.6688</td>
      <td>0.6693</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.1758</td>
      <td>0.2803</td>
      <td>0.2048</td>
      <td>0.1705</td>
      <td>0.1852</td>
      <td>0.1683</td>
      <td>0.3010</td>
      <td>0.2261</td>
      <td>0.1981</td>
      <td>0.1545</td>
      <td>0.3287</td>
      <td>0.2179</td>
      <td>0.3129</td>
      <td>0.1426</td>
      <td>0.2414</td>
      <td>0.2747</td>
      <td>0.3470</td>
      <td>0.1630</td>
      <td>0.2372</td>
      <td>0.3046</td>
      <td>0.3136</td>
      <td>0.2507</td>
      <td>0.1366</td>
      <td>0.2956</td>
      <td>0.2237</td>
      <td>0.2884</td>
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
      <td>0.2350</td>
      <td>0.2151</td>
      <td>0.2438</td>
      <td>0.6690</td>
      <td>0.6690</td>
      <td>0.6688</td>
      <td>0.2203</td>
      <td>0.1747</td>
      <td>0.2418</td>
      <td>0.2337</td>
      <td>0.2323</td>
      <td>0.2616</td>
      <td>0.2851</td>
      <td>0.2276</td>
      <td>0.2486</td>
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
      <td>0.2245</td>
      <td>0.1947</td>
      <td>0.2564</td>
      <td>0.2134</td>
      <td>0.2192</td>
      <td>0.3139</td>
      <td>0.1947</td>
      <td>0.2084</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.2481</td>
      <td>0.2002</td>
      <td>0.1251</td>
      <td>0.2006</td>
      <td>0.2809</td>
      <td>0.2233</td>
      <td>0.2405</td>
      <td>0.2891</td>
      <td>0.2252</td>
      <td>0.2354</td>
      <td>0.3081</td>
      <td>0.2964</td>
      <td>0.3229</td>
      <td>0.1461</td>
      <td>0.3208</td>
      <td>0.3524</td>
      <td>0.2481</td>
      <td>0.3427</td>
      <td>0.2579</td>
      <td>0.3082</td>
      <td>0.2381</td>
      <td>0.2480</td>
      <td>0.3073</td>
      <td>0.2282</td>
      <td>0.3064</td>
      <td>0.1950</td>
      <td>0.2273</td>
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
      <td>0.2046</td>
      <td>0.2297</td>
      <td>0.2390</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.1911</td>
      <td>0.2349</td>
      <td>0.2516</td>
      <td>0.2800</td>
      <td>0.2633</td>
      <td>0.3144</td>
      <td>0.2680</td>
      <td>0.2612</td>
      <td>0.2429</td>
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
      <td>0.3423</td>
      <td>0.2698</td>
      <td>0.4969</td>
      <td>0.3493</td>
      <td>0.3821</td>
      <td>0.3318</td>
      <td>0.2423</td>
      <td>0.3888</td>
      <td>0.1888</td>
      <td>0.6688</td>
      <td>0.6883</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6963</td>
      <td>0.7072</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.3595</td>
      <td>0.5167</td>
      <td>0.5735</td>
      <td>0.3704</td>
      <td>0.2935</td>
      <td>0.3055</td>
      <td>0.2159</td>
      <td>0.2291</td>
      <td>0.3122</td>
      <td>0.2283</td>
      <td>0.3475</td>
      <td>0.3737</td>
      <td>0.2626</td>
      <td>0.2823</td>
      <td>0.2717</td>
      <td>0.2354</td>
      <td>0.1358</td>
      <td>0.3092</td>
      <td>0.1824</td>
      <td>0.1923</td>
      <td>0.2183</td>
      <td>0.2031</td>
      <td>0.2031</td>
      <td>0.2309</td>
      <td>0.2102</td>
      <td>0.1914</td>
      <td>0.1734</td>
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
      <td>0.3544</td>
      <td>0.2733</td>
      <td>0.6753</td>
      <td>0.6908</td>
      <td>0.6688</td>
      <td>0.4832</td>
      <td>0.3231</td>
      <td>0.2524</td>
      <td>0.3165</td>
      <td>0.2722</td>
      <td>0.2268</td>
      <td>0.1977</td>
      <td>0.2123</td>
      <td>0.1917</td>
    </tr>
  </tbody>
</table>
