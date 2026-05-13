# Experiment Master Tables

- Layout: aligned with `result/all_results.md`.
- Source output root: `/data/liyapeng_grp/program/MedMNISTMerge/outputs/custom_methods_codex_my_merge_small_a4, /data/liyapeng_grp/program/MedMNISTMerge/outputs/custom_methods_codex_my_merge_small_b5, /data/liyapeng_grp/program/MedMNISTMerge/outputs/custom_methods_codex_my_merge_vlm, /data/liyapeng_grp/program/MedMNISTMerge/outputs/my_merge_smoke_med_v4, /data/liyapeng_grp/program/MedMNISTMerge/outputs/my_merge_med_v4_bad_c3, /data/liyapeng_grp/program/MedMNISTMerge/outputs/my_merge_med_v4p4_specialist_probe, /data/liyapeng_grp/program/MedMNISTMerge/outputs/my_merge_med_v4p5_proto_vit_probe, /data/liyapeng_grp/program/MedMNISTMerge/outputs/my_merge_med_v4p6_proto_swin_probe, /data/liyapeng_grp/program/MedMNISTMerge/outputs/my_merge_med_v4p7_proto_vit_organc_probe, /data/liyapeng_grp/program/MedMNISTMerge/outputs/my_merge_med_v4p8_vit_clip_c3, /data/liyapeng_grp/program/MedMNISTMerge/outputs/my_merge_med_v5_focus_probe, /data/liyapeng_grp/program/MedMNISTMerge/outputs/my_merge_med_v5_domain_smoke_derma, /data/liyapeng_grp/program/MedMNISTMerge/outputs/my_merge_med_v5_domain_smoke_organc, /data/liyapeng_grp/program/MedMNISTMerge/outputs/my_merge_med_v5_domain_smoke_chao, /data/liyapeng_grp/program/MedMNISTMerge/outputs/custom_methods_my_merge_full_refresh_fast_20260510_164550`.
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
      <td>0.4469</td>
      <td>0.4516</td>
      <td>0.8536</td>
      <td>0.4230</td>
      <td>0.3023</td>
      <td>0.5861</td>
      <td>0.1988</td>
      <td>0.4566</td>
      <td>0.5767</td>
      <td>0.6783</td>
      <td>0.7027</td>
      <td>0.7561</td>
      <td>0.6758</td>
      <td>0.7431</td>
      <td>0.6504</td>
      <td>0.6678</td>
      <td>0.6718</td>
      <td>0.6828</td>
      <td>0.5461</td>
      <td>0.5759</td>
      <td>0.6657</td>
      <td>0.3807</td>
      <td>0.2115</td>
      <td>0.4618</td>
      <td>0.3144</td>
      <td>0.2731</td>
      <td>0.4464</td>
      <td>0.5058</td>
      <td>0.4553</td>
      <td>0.5054</td>
      <td>0.3537</td>
      <td>0.3097</td>
      <td>0.3263</td>
      <td>0.3069</td>
      <td>0.2839</td>
      <td>0.3561</td>
      <td>0.1914</td>
      <td>0.1734</td>
      <td>0.4726</td>
      <td>0.2363</td>
      <td>0.1959</td>
      <td>0.4043</td>
      <td>0.2435</td>
      <td>0.3333</td>
      <td>0.3235</td>
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
      <td>0.5840</td>
      <td>0.4371</td>
      <td>0.4107</td>
      <td>0.7124</td>
      <td>0.6898</td>
      <td>0.6741</td>
      <td>0.5959</td>
      <td>0.3513</td>
      <td>0.3447</td>
      <td>0.4888</td>
      <td>0.3299</td>
      <td>0.3156</td>
      <td>0.2791</td>
      <td>0.2788</td>
      <td>0.3001</td>
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
      <td>0.1824</td>
      <td>0.1947</td>
      <td>0.1824</td>
      <td>0.1824</td>
      <td>0.1947</td>
      <td>0.1824</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.2233</td>
      <td>0.1607</td>
      <td>0.3236</td>
      <td>0.2233</td>
      <td>0.1716</td>
      <td>0.2233</td>
      <td>0.2233</td>
      <td>0.2233</td>
      <td>0.1171</td>
      <td>0.2354</td>
      <td>0.2354</td>
      <td>0.2354</td>
      <td>0.1521</td>
      <td>0.0483</td>
      <td>0.2354</td>
      <td>0.2802</td>
      <td>0.2354</td>
      <td>0.1434</td>
      <td>0.1051</td>
      <td>0.1087</td>
      <td>0.1087</td>
      <td>0.1087</td>
      <td>0.1087</td>
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
      <td>0.1756</td>
      <td>0.1865</td>
      <td>0.1865</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.2359</td>
      <td>0.2061</td>
      <td>0.1879</td>
      <td>0.2354</td>
      <td>0.1453</td>
      <td>0.2197</td>
      <td>0.1075</td>
      <td>0.1303</td>
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
      <td>0.2783</td>
      <td>0.1833</td>
      <td>0.4455</td>
      <td>0.1824</td>
      <td>0.3315</td>
      <td>0.1915</td>
      <td>0.1824</td>
      <td>0.2385</td>
      <td>0.1733</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6608</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.3559</td>
      <td>0.2669</td>
      <td>0.3772</td>
      <td>0.2853</td>
      <td>0.1325</td>
      <td>0.2097</td>
      <td>0.3202</td>
      <td>0.1604</td>
      <td>0.1646</td>
      <td>0.3360</td>
      <td>0.3817</td>
      <td>0.3275</td>
      <td>0.3575</td>
      <td>0.0636</td>
      <td>0.2919</td>
      <td>0.3750</td>
      <td>0.2354</td>
      <td>0.1758</td>
      <td>0.1168</td>
      <td>0.2075</td>
      <td>0.3001</td>
      <td>0.1950</td>
      <td>0.1087</td>
      <td>0.2309</td>
      <td>0.2120</td>
      <td>0.2606</td>
      <td>0.1653</td>
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
      <td>0.3023</td>
      <td>0.2351</td>
      <td>0.1981</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6662</td>
      <td>0.3333</td>
      <td>0.2092</td>
      <td>0.2151</td>
      <td>0.3484</td>
      <td>0.2377</td>
      <td>0.2621</td>
      <td>0.2081</td>
      <td>0.1782</td>
      <td>0.2126</td>
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
      <td>0.1842</td>
      <td>0.1947</td>
      <td>0.2134</td>
      <td>0.2061</td>
      <td>0.2438</td>
      <td>0.2184</td>
      <td>0.1947</td>
      <td>0.1824</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.0668</td>
      <td>0.3659</td>
      <td>0.2233</td>
      <td>0.2639</td>
      <td>0.2081</td>
      <td>0.2233</td>
      <td>0.2233</td>
      <td>0.2421</td>
      <td>0.2182</td>
      <td>0.1521</td>
      <td>0.2890</td>
      <td>0.2354</td>
      <td>0.1848</td>
      <td>0.1548</td>
      <td>0.3096</td>
      <td>0.2354</td>
      <td>0.2704</td>
      <td>0.3212</td>
      <td>0.1087</td>
      <td>0.2534</td>
      <td>0.2084</td>
      <td>0.2417</td>
      <td>0.1087</td>
      <td>0.1617</td>
      <td>0.2875</td>
      <td>0.1078</td>
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
      <td>0.1912</td>
      <td>0.2211</td>
      <td>0.1985</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.2187</td>
      <td>0.2318</td>
      <td>0.2279</td>
      <td>0.2255</td>
      <td>0.2164</td>
      <td>0.2757</td>
      <td>0.1902</td>
      <td>0.1707</td>
      <td>0.1683</td>
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
      <td>0.3321</td>
      <td>0.3853</td>
      <td>0.5870</td>
      <td>0.3505</td>
      <td>0.4087</td>
      <td>0.6469</td>
      <td>0.2248</td>
      <td>0.4160</td>
      <td>0.6457</td>
      <td>0.6758</td>
      <td>0.6873</td>
      <td>0.7057</td>
      <td>0.6693</td>
      <td>0.6958</td>
      <td>0.7172</td>
      <td>0.6688</td>
      <td>0.6688</td>
      <td>0.6678</td>
      <td>0.2672</td>
      <td>0.5527</td>
      <td>0.6138</td>
      <td>0.3737</td>
      <td>0.4159</td>
      <td>0.5417</td>
      <td>0.3083</td>
      <td>0.2785</td>
      <td>0.3544</td>
      <td>0.2159</td>
      <td>0.3273</td>
      <td>0.5620</td>
      <td>0.3537</td>
      <td>0.3274</td>
      <td>0.3507</td>
      <td>0.3136</td>
      <td>0.2244</td>
      <td>0.4559</td>
      <td>0.1473</td>
      <td>0.1087</td>
      <td>0.2084</td>
      <td>0.1087</td>
      <td>0.1186</td>
      <td>0.1087</td>
      <td>0.1195</td>
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
      <td>0.4348</td>
      <td>0.4687</td>
      <td>0.4288</td>
      <td>0.6896</td>
      <td>0.6941</td>
      <td>0.6685</td>
      <td>0.4779</td>
      <td>0.4438</td>
      <td>0.3137</td>
      <td>0.3684</td>
      <td>0.3439</td>
      <td>0.3313</td>
      <td>0.1548</td>
      <td>0.1120</td>
      <td>0.1123</td>
    </tr>
  </tbody>
</table>
