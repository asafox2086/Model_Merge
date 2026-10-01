# my_merge 中间现象表（2026-06-12）

用途：这张表只用于观察现象和设计方法，不作为最终结果表。历史好版本包含服务端验证集候选选择，只能当“显微镜”，不能作为最终算法。

| dataset | setting | best baseline | hist selected / delta | v15 selected / delta | v16 selected / delta | v17 selected / delta | v15 conflict/sign/norm | observation |
|---|---:|---:|---|---|---|---|---|---|
| blood | c3_b0 | 0.3826 | `delta_0p50` / 0.2055 | `delta_0p25` / 0.0357 | `delta_0p00` / -0.0809 | `delta_0p50` / 0.0211 | 0.4200/0.3527/0.4676 | v16过度保守/路由更差 |
| blood | c3_b0.01 | 0.3376 | `delta_0p00` / 0.1164 | `delta_0p25` / -0.0555 | `delta_0p00` / -0.0307 | `delta_0p00` / -0.0307 | 0.4072/0.3849/0.3305 | 历史有效形态=delta_0p00 |
| blood | c3_b0.1 | 0.3964 | `delta_0p00` / 0.4577 | `delta_0p25` / -0.0336 | `delta_0p00` / -0.0015 | `delta_0p00` / -0.0015 | 0.4151/0.3481/0.4486 | 历史有效形态=delta_0p00 |
| blood | c5_b0 | 0.2894 | `specialist_client` / 0.0587 | `delta_0p25` / -0.0330 | `delta_0p25` / -0.0330 | `delta_0p50` / -0.0357 | 0.4204/0.5245/0.1028 | 历史有效形态=specialist_client |
| blood | c5_b0.01 | 0.2917 | `specialist_client` / -0.0321 | `delta_0p50` / -0.0885 | `delta_0p25` / -0.0961 | `delta_0p75` / -0.0523 | 0.4540/0.5077/0.3011 | v15明显负优化; 历史有效形态=specialist_client |
| blood | c5_b0.1 | 0.4373 | `delta_1p00` / 0.2765 | `delta_0p50` / 0.0143 | `delta_0p50` / 0.0143 | `specialist_client` / 0.0833 | 0.4570/0.4982/0.3232 | 历史有效形态=delta_1p00 |
| blood | c7_b0 | 0.3560 | `specialist_client` / -0.1166 | `delta_0p50` / -0.0701 | `delta_0p50` / -0.0701 | `delta_0p75` / -0.1675 | 0.4680/0.5783/0.2209 | v17代理评分仍失效; 历史有效形态=specialist_client |
| blood | c7_b0.01 | 0.4250 | `delta_0p50` / 0.0029 | `delta_0p75` / -0.0731 | `delta_1p00` / -0.1359 | `specialist_client` / -0.1172 | 0.5422/0.5217/0.6971 | v16过度保守/路由更差; v17代理评分仍失效 |
| blood | c7_b0.1 | 0.4306 | `delta_0p75` / 0.1341 | `delta_0p50` / 0.0342 | `delta_0p75` / -0.0649 | `delta_0p25` / -0.0599 | 0.4856/0.5702/0.2972 | v16过度保守/路由更差 |
| chaosheng | c3_b0 | 0.2659 | `delta_0p25` / 0.0926 | `delta_0p25` / 0.0027 | `delta_0p00` / -0.0107 | `specialist_client` / -0.0700 | 0.4189/0.3466/0.4757 | 需继续看候选空间指标 |
| chaosheng | c3_b0.01 | 0.4115 | `delta_0p75` / 0.1887 | `medical_weighted_fusion` / 0.0126 | `delta_0p00` / -0.1105 | `specialist_client` / -0.1240 | 0.4014/0.3873/0.2756 | v16过度保守/路由更差; v17代理评分仍失效 |
| chaosheng | c3_b0.1 | 0.4681 | `delta_1p00` / 0.0872 | `delta_0p25` / -0.2650 | `delta_0p00` / -0.2938 | `medical_weighted_fusion` / -0.1213 | 0.4090/0.3850/0.3227 | v15明显负优化; v16过度保守/路由更差; v17代理评分仍失效; 历史有效形态=delta_1p00 |
| chaosheng | c5_b0 | 0.2264 | `specialist_client` / 0.0153 | `delta_0p75` / -0.1141 | `specialist_client` / 0.0351 | `specialist_client` / 0.0351 | 0.4760/0.4896/0.4479 | v15明显负优化; 历史有效形态=specialist_client |
| chaosheng | c5_b0.01 | 0.3468 | `delta_1p00` / 0.1743 | `delta_0p75` / 0.0710 | `delta_0p25` / -0.1123 | `specialist_client` / 0.0970 | 0.4796/0.4814/0.4772 | v16过度保守/路由更差; 历史有效形态=delta_1p00 |
| chaosheng | c5_b0.1 | 0.2552 | `medical_weighted_fusion` / 0.1626 | `delta_0p50` / -0.1258 | `delta_0p25` / -0.1132 | `delta_0p50` / -0.1258 | 0.4335/0.5134/0.1751 | v15明显负优化; v17代理评分仍失效 |
| chaosheng | c7_b0 | 0.2444 | `specialist_client` / -0.0009 | `delta_1p00` / -0.1393 | `specialist_client` / -0.0027 | `specialist_client` / -0.0027 | 0.5204/0.5666/0.4955 | v15明显负优化; 历史有效形态=specialist_client |
| chaosheng | c7_b0.01 | 0.2893 | `specialist_client` / 0.0503 | `delta_1p00` / -0.0404 | `specialist_client` / 0.0521 | `specialist_client` / 0.0521 | 0.5471/0.5287/0.7115 | 历史有效形态=specialist_client |
| chaosheng | c7_b0.1 | 0.2570 | `specialist_client` / 0.0673 | `delta_0p75` / -0.0944 | `specialist_client` / 0.0665 | `delta_1p00` / -0.0953 | 0.4782/0.5635/0.2942 | v15明显负优化; v17代理评分仍失效; 历史有效形态=specialist_client |
| organc | c3_b0 | 0.3824 | `medical_weighted_fusion` / 0.1360 | `delta_0p25` / -0.1350 | `delta_0p00` / -0.0611 | `delta_0p50` / -0.1855 | 0.4071/0.3882/0.3019 | v15明显负优化; v17代理评分仍失效 |
| organc | c3_b0.01 | 0.2657 | `specialist_client` / 0.3054 | `medical_weighted_fusion` / -0.1048 | `specialist_client` / 0.0203 | `specialist_client` / 0.0203 | 0.3679/0.4056/0.0660 | v15明显负优化; 历史有效形态=specialist_client |
| organc | c3_b0.1 | 0.4637 | `delta_0p00` / 0.2022 | `medical_weighted_fusion` / -0.2512 | `delta_0p00` / -0.2506 | `delta_0p50` / -0.0235 | 0.3903/0.3887/0.2242 | v15明显负优化; 历史有效形态=delta_0p00 |
| organc | c5_b0 | 0.2764 | `specialist_client` / 0.0918 | `delta_0p50` / -0.1432 | `specialist_client` / 0.0901 | `specialist_client` / 0.0901 | 0.4372/0.5428/0.1200 | v15明显负优化; 历史有效形态=specialist_client |
| organc | c5_b0.01 | 0.2165 | `delta_1p00` / 0.1040 | `delta_0p50` / -0.0624 | `delta_0p50` / -0.0624 | `delta_0p75` / -0.0724 | 0.4637/0.5193/0.3012 | 历史有效形态=delta_1p00 |
| organc | c5_b0.1 | 0.2991 | `delta_0p25` / 0.1763 | `delta_0p50` / 0.0132 | `delta_0p50` / 0.0132 | `delta_0p75` / -0.0641 | 0.4646/0.5052/0.3330 | 需继续看候选空间指标 |
| organc | c7_b0 | 0.2354 | `delta_1p00` / 0.0202 | `delta_0p75` / -0.0287 | `delta_0p75` / -0.0287 | `delta_1p00` / -0.0700 | 0.5142/0.5716/0.4449 | 历史有效形态=delta_1p00 |
| organc | c7_b0.01 | 0.2153 | `delta_0p00` / -0.0366 | `delta_0p75` / -0.1173 | `delta_0p75` / -0.1173 | `delta_0p50` / -0.1126 | 0.5113/0.5638/0.4362 | v15明显负优化; v17代理评分仍失效; 历史有效形态=delta_0p00 |
| organc | c7_b0.1 | 0.3841 | `delta_1p00` / 0.1556 | `delta_0p50` / -0.0597 | `delta_0p75` / 0.0004 | `delta_0p75` / 0.0004 | 0.4804/0.5833/0.2359 | 历史有效形态=delta_1p00 |

## 当前归纳

- `v15`：闭式冲突规则太单调，几乎总把模型推向 delta，27 格均值 `delta=-0.0686`。
- `v16`：三段式专家保真规则过度保守，27 格均值 `delta=-0.0513`。
- `v17`：无验证集代理评分比 v16 好，但 22 格仍为 `delta=-0.0370`，不能作为主方法。
- 可靠现象：候选形态本身有价值，但最终选择机制尚未找到；下一步应先补充每个候选的模型空间距离、方向一致性、冲突消解比例，再设计规则。
