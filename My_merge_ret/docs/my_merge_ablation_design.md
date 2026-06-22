# my_merge BN-Only Ablation Design

## Goal

消融只回答两个问题：Clinical Evidence Ledger 是否有用，Protocol-Calibrated Merge 是否有用。方法不包含医学形态特征、线性头二阶矩、候选池、客户端回传验证或按数据集写死的规则。

## Main Rows

| ablation | meaning |
| --- | --- |
| `full` | 完整 `my_merge`：CEL + PCM |
| `no_bml` | 关闭 Clinical Evidence Ledger，使用基础客户端权重 |
| `no_bcm` | 关闭 Protocol-Calibrated Merge，使用 BN 权重普通平均 |
| `avg_only` | sanity baseline，直接使用 client average |

## Module 1: CEL

CEL 使用客户端 checkpoint 中的 BN running moments 构造融合权重：

```text
a_i = Normalize(n_i)
s_i = Normalize(sqrt(n_i))
g_e = Smooth(max_i a_i)
p_i = Normalize((1 - g_e)(0.5 base_i + 0.5 s_i) + g_e a_i)

d_i = mean_l [ ||mu_i^l - mu_bar^l||^2 / v_bar^l
              + 0.5 ||log v_i^l - log v_bar^l||^2 ]

omega_i = Normalize( p_i * (0.35 + 0.65 exp(-d_i / tau)) )
```

关闭 CEL 时，`my_merge` 不再利用客户端 BN 分布偏移，只保留基础平均/样本量先验。

## Module 2: PCM

PCM 根据 CEL 权重做单次闭式融合：

- BN mean 使用算术加权平均。
- BN variance 使用 log-space 几何平均。
- 非分类头可训练参数使用 CEL 权重平均。
- 分类头使用客户端类别覆盖元信息做行级融合。

```text
theta = sum_i omega_i theta_i

omega_{i,c} = Normalize(omega_i * (epsilon + 1[c in C_i]))
rho = Gate(class coverage forms a specialty partition)
lambda_c = rho * (0.30 + 0.25 * max(0, 3 - |S_c|))
theta_c = (1 - lambda_c) sum_i omega_i theta_{i,c}
        + lambda_c sum_i omega_{i,c} theta_{i,c}
```

关闭 PCM 时，`my_merge` 不再执行分类头行级校准和类别先验，结果应退化到 BN 权重下的普通平均。

## Observation Log

当前保留/删除规则来自实际观察：

- 保留 CEL：BN running moments 能表达医院、设备、染色、探头和窗宽窗位带来的 site/protocol shift，而且 checkpoint 已经包含这些聚合统计。
- 保留 PCM：医学 silo 的类别覆盖常反映专科/器官/协议分工，分类头逐类路由能避免专科类别被无关客户端平均掉。
- 修正 singleton-only gate：只看 singleton 比例会修复部分器官数据，但会伤害存在 broad client 的划分；因此现在额外检测 broad client，避免把普通 non-IID 当成专科。
- 增加 compact-specialty gate：`organc` 观察到中等 singleton 比例且没有 broad client 时仍然是专科划分，因此需要保留轻量路由。
- 增加 sparse-overlap specialty gate：`derma c7 b=0.01` 这类 case 不是清晰 singleton partition，而是平均类别支持数低、singleton 不高、没有 broad client、类别先验偏斜明显，符合多中心皮肤病专科之间少量转诊/重叠覆盖的结构；因此在 PCM 内允许路由。
- 保留 fragmented support reliability：`organs c7 b=0.01` 显示高 singleton 碎片化下，低 CEL 支持质量的类别路由会过度信任少量客户端；现在只有 singleton 碎片化和类别先验偏斜同时出现时才降温，避免误伤 `organc c7` 和 `blood c5`。
- 删除 protocol conflict filter：尝试用公共初始化上的 task-vector 符号一致性抑制跨站点冲突方向，但 36-case 探针中 blood c7、organc c3/c7 明显变差，且没有修复 chaosheng，所以不放入正式方法。
- 删除无条件 support-mass head gate：0.13 阈值修复 `organs c7 b=0.01` 但伤害 `organc c7 b=0.01`，0.11 阈值又失去修复效果；正式方法只保留上面的 fragmented support reliability。
- 删除候选池：效果可能好，但需要客户端给候选模型打分，隐私和 single-shot 设定不成立。
- 禁用线性二阶矩：少数 case 有收益，但需要额外上传 `G_i`、运行慢、故事变成三模块，当前只保留兼容开关。

## Compatibility

历史脚本里的旧名字可以继续作为兼容标签处理，但正式论文表只报告：

- `full`
- `no_bml`
- `no_bcm`
- `avg_only`

旧的形态特征、诊断统计和候选池相关消融不再是正式方法的一部分。
