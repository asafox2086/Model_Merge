# Experiment Master Tables

- Layout: aligned with `result/all_results.md`.
- Source output root: `outputs/lamp_merge_full_client_local_20260708_193654/resnet, outputs/lamp_merge_full_client_local_20260708_193654/convnext, outputs/lamp_merge_full_client_local_20260708_193654/vit_t, outputs/lamp_merge_full_client_local_20260708_193654/swin_tiny`.
- Formal scope: five medical image datasets and four vision backbones (ResNet, ConvNeXt, ViT-Tiny, and Swin-Tiny).
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
      <td>0.8004</td>
      <td>0.7995</td>
      <td>0.8001</td>
      <td>0.8004</td>
      <td>0.8001</td>
      <td>0.7957</td>
      <td>0.8004</td>
      <td>0.7986</td>
      <td>0.7963</td>
      <td>0.6733</td>
      <td>0.6733</td>
      <td>0.6728</td>
      <td>0.6733</td>
      <td>0.6733</td>
      <td>0.6743</td>
      <td>0.6733</td>
      <td>0.6733</td>
      <td>0.6733</td>
      <td>0.5943</td>
      <td>0.5949</td>
      <td>0.5898</td>
      <td>0.5943</td>
      <td>0.5927</td>
      <td>0.5854</td>
      <td>0.5943</td>
      <td>0.5940</td>
      <td>0.5831</td>
      <td>0.5182</td>
      <td>0.5178</td>
      <td>0.5229</td>
      <td>0.5182</td>
      <td>0.5184</td>
      <td>0.5207</td>
      <td>0.5182</td>
      <td>0.5184</td>
      <td>0.5031</td>
      <td>0.4762</td>
      <td>0.4753</td>
      <td>0.4897</td>
      <td>0.4762</td>
      <td>0.4699</td>
      <td>0.4465</td>
      <td>0.4762</td>
      <td>0.4735</td>
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
      <td>LAMP-Merge</td>
      <td>0.8000</td>
      <td>0.7987</td>
      <td>0.7984</td>
      <td>0.6732</td>
      <td>0.6736</td>
      <td>0.6733</td>
      <td>0.5930</td>
      <td>0.5908</td>
      <td>0.5905</td>
      <td>0.5197</td>
      <td>0.5191</td>
      <td>0.5132</td>
      <td>0.4804</td>
      <td>0.4642</td>
      <td>0.4714</td>
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
      <td>0.8483</td>
      <td>0.8492</td>
      <td>0.8465</td>
      <td>0.8483</td>
      <td>0.8483</td>
      <td>0.8462</td>
      <td>0.8483</td>
      <td>0.8515</td>
      <td>0.8465</td>
      <td>0.6045</td>
      <td>0.6035</td>
      <td>0.6020</td>
      <td>0.6045</td>
      <td>0.6050</td>
      <td>0.5696</td>
      <td>0.6045</td>
      <td>0.5975</td>
      <td>0.5940</td>
      <td>0.6566</td>
      <td>0.6542</td>
      <td>0.6467</td>
      <td>0.6566</td>
      <td>0.6570</td>
      <td>0.6366</td>
      <td>0.6566</td>
      <td>0.6521</td>
      <td>0.6463</td>
      <td>0.6067</td>
      <td>0.6043</td>
      <td>0.6076</td>
      <td>0.6067</td>
      <td>0.6064</td>
      <td>0.6048</td>
      <td>0.6067</td>
      <td>0.6079</td>
      <td>0.6067</td>
      <td>0.4259</td>
      <td>0.4268</td>
      <td>0.4313</td>
      <td>0.4259</td>
      <td>0.4304</td>
      <td>0.4286</td>
      <td>0.4259</td>
      <td>0.4214</td>
      <td>0.4043</td>
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
      <td>0.8480</td>
      <td>0.8476</td>
      <td>0.8488</td>
      <td>0.6033</td>
      <td>0.5930</td>
      <td>0.5987</td>
      <td>0.6525</td>
      <td>0.6501</td>
      <td>0.6517</td>
      <td>0.6062</td>
      <td>0.6060</td>
      <td>0.6071</td>
      <td>0.4280</td>
      <td>0.4283</td>
      <td>0.4172</td>
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
      <td>0.8515</td>
      <td>0.8518</td>
      <td>0.8518</td>
      <td>0.8515</td>
      <td>0.8512</td>
      <td>0.8506</td>
      <td>0.8515</td>
      <td>0.8527</td>
      <td>0.8457</td>
      <td>0.6035</td>
      <td>0.6045</td>
      <td>0.5970</td>
      <td>0.6035</td>
      <td>0.6035</td>
      <td>0.5980</td>
      <td>0.6035</td>
      <td>0.5970</td>
      <td>0.6204</td>
      <td>0.5858</td>
      <td>0.5852</td>
      <td>0.5892</td>
      <td>0.5858</td>
      <td>0.5794</td>
      <td>0.5786</td>
      <td>0.5858</td>
      <td>0.5768</td>
      <td>0.5811</td>
      <td>0.5434</td>
      <td>0.5425</td>
      <td>0.5448</td>
      <td>0.5434</td>
      <td>0.5427</td>
      <td>0.5514</td>
      <td>0.5434</td>
      <td>0.5470</td>
      <td>0.5511</td>
      <td>0.4645</td>
      <td>0.4636</td>
      <td>0.4555</td>
      <td>0.4645</td>
      <td>0.4555</td>
      <td>0.4420</td>
      <td>0.4645</td>
      <td>0.4663</td>
      <td>0.4214</td>
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
      <td>0.8517</td>
      <td>0.8511</td>
      <td>0.8499</td>
      <td>0.6017</td>
      <td>0.6017</td>
      <td>0.6070</td>
      <td>0.5867</td>
      <td>0.5813</td>
      <td>0.5812</td>
      <td>0.5436</td>
      <td>0.5458</td>
      <td>0.5472</td>
      <td>0.4612</td>
      <td>0.4540</td>
      <td>0.4507</td>
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
      <td>0.7740</td>
      <td>0.7746</td>
      <td>0.7705</td>
      <td>0.7740</td>
      <td>0.7743</td>
      <td>0.7662</td>
      <td>0.7740</td>
      <td>0.7746</td>
      <td>0.7618</td>
      <td>0.6424</td>
      <td>0.6429</td>
      <td>0.6339</td>
      <td>0.6424</td>
      <td>0.6384</td>
      <td>0.6409</td>
      <td>0.6424</td>
      <td>0.6419</td>
      <td>0.6344</td>
      <td>0.6856</td>
      <td>0.6872</td>
      <td>0.6823</td>
      <td>0.6856</td>
      <td>0.6865</td>
      <td>0.6726</td>
      <td>0.6856</td>
      <td>0.6800</td>
      <td>0.6753</td>
      <td>0.6283</td>
      <td>0.6259</td>
      <td>0.6240</td>
      <td>0.6283</td>
      <td>0.6288</td>
      <td>0.6257</td>
      <td>0.6283</td>
      <td>0.6290</td>
      <td>0.6284</td>
      <td>0.4816</td>
      <td>0.4825</td>
      <td>0.4537</td>
      <td>0.4816</td>
      <td>0.4834</td>
      <td>0.4825</td>
      <td>0.4816</td>
      <td>0.4834</td>
      <td>0.4744</td>
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
      <td>0.7731</td>
      <td>0.7715</td>
      <td>0.7701</td>
      <td>0.6397</td>
      <td>0.6406</td>
      <td>0.6396</td>
      <td>0.6850</td>
      <td>0.6816</td>
      <td>0.6803</td>
      <td>0.6261</td>
      <td>0.6276</td>
      <td>0.6286</td>
      <td>0.4726</td>
      <td>0.4825</td>
      <td>0.4798</td>
    </tr>
  </tbody>
</table>
