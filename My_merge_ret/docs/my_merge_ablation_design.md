# my_merge Two-Module Ablation Design

## Goal

消融实验只验证两个主模块是否有效，避免把方法拆成过多小技巧，也避免每个数据集各写一套特殊处理。

## Main Rows

| ablation | meaning |
| --- | --- |
| `full` | 完整 `my_merge` |
| `no_client_information` | 去掉 M1：Diagnostic Evidence Client Information Estimation |
| `no_fusion_selection` | 去掉 M2：Medical Evidence Guided Fusion and Selection |
| `avg_only` | 同时去掉 M1 和 M2，回到平均融合 |

## M1

M1 估计医学诊断信息。它使用所有医学影像任务共用的图像证据：前景结构、边界、局部对比度、纹理异质性、形状紧致度、诊断显著性、类别稀有度、难例表现和预测 margin。

关闭 M1 时，代码同时关闭：

- `image_space`
- `diagnostic_evidence`
- `diagnostic_client_information`
- `class_rarity`
- `focal_weight`
- `domain_focus`

因此 `no_client_information` 不再只是去掉某个弱先验，而是去掉整个客户端诊断信息估计模块。

## M2

M2 使用 M1 的诊断信息做融合和选择。关闭 M2 时，代码同时关闭：

- `balanced_selection`
- `layerwise_merge`
- `sparse_residual`
- `morph_anchor_candidate`
- `specialist_candidate`
- `reference_delta_candidate`
- `prototype_head_candidate`
- `consensus_candidate`
- `candidate_selection`
- `bn_recalibration`
- `head_temperature`

因此 `no_fusion_selection` 是干净的 M2 消融，结果应当回退到平均融合路径。

## Compatibility

旧实验名仍可读取，但不作为默认主消融：

- `no_medical_prior`
- `no_domain_preprocess`
- `no_modality_features`
- `no_medical_preprocess`

这些旧入口现在都映射为关闭 M1，用来兼容历史脚本，而不是继续保留独立的医学先验模块。

## Running

推荐入口：

```bash
bash scripts/run_validated_my_merge_full.sh
```

该脚本会先用 `avg ties` 复现 `result/all_results.md` 中的确定结果；复现检查通过后，再跑 `full no_client_information no_fusion_selection avg_only`，最后自动生成总表和消融摘要。
