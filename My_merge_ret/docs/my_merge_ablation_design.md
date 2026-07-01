# my_merge Ablation Design

当前主方法是 **MedSpec: Medical Specialty Model Merging**。

| 标签 | 含义 |
| --- | --- |
| `full` | 完整 MedSpec |
| `no_client_information` | 不使用 manifest 中的医学专科类别覆盖，退回普通加权平均 |
| `no_fusion_selection` | 关闭专科 head row routing，只保留病例数加权 backbone/head 平均 |

## Module 1 消融

Medical Specialty Map 使用 `{S_i, n_i}`：

```text
omega_i = n_i / sum_j n_j
omega_{c,i} = omega_i 1[c in S_i] / sum_j omega_j 1[c in S_j]
```

关闭后不再区分类别专科覆盖。

## Module 2 消融

Specialist Head Merging 的核心是：

```text
theta_backbone = sum_i omega_i theta_i,backbone
theta_head[c] = sum_i omega_{c,i} theta_i,head[c]
```

关闭后所有 tensor 都使用 `omega_i` 平均。这个消融验证“未训练该疾病/器官类别的客户端 head row 是否会污染医学分类头”。
