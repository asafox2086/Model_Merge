# my_merge Medical Ablation Design

## 目标

`my_merge` 现在被改成可消融的医学影像模型融合方法。消融实验的目的不是只报告“去掉某个技巧掉了几分”，而是验证一个核心命题：

> 医学影像融合不应该只按全局精度或参数距离合并模型，而应该利用不同医学模态的物理成像机制、病灶边界、空间解剖先验和类别稀缺性，选择更可信的 client 与参数层。

这个命题天然不能迁移到自然语言处理，因为它依赖图像中的核质比、皮肤镜颜色/毛发伪影、CT 窗宽窗位、解剖固定位置、超声 speckle 与声影等物理-形态证据。

## 医学问题与着手点

这里的“找问题”不是泛泛地说某个模型弱，而是先判断每个医学数据集的判别证据来自哪里，再把这些证据变成模型融合时可使用的统计量。自然图像融合常用的参数平均、logit 平均或全局特征相似度并不会显式关心病灶边界、解剖位置、物理灰度窗和少数类风险，这正是医学专攻方法的切入点。

| 数据集 | 医学特定问题 | 关键特征 | 融合着手点 | 对应消融 |
| --- | --- | --- | --- | --- |
| `BloodMNIST` | 血细胞子类外观极像，普通纹理不够，关键在核/胞质比例和细胞核形态 | 细胞面积、核质比、核边界强度、核圆度、染色质对比度 | 用近似细胞核/胞质分割特征判断哪个 client 学到了细胞级几何，而不是只看整体 acc | `no_domain_preprocess`, `no_domain_focus`, `no_layerwise` |
| `DermaMNIST` | 皮肤镜受毛发、气泡、光照和色温影响，且恶性类少 | 病灶面积、边界不规则、不对称性、颜色杂色、毛发/黑帽伪影负载 | 先做颜色恒常和毛发伪影估计，再用少数类和困难样本权重保护恶性/疑难样本 | `no_domain_preprocess`, `no_rarity`, `no_focal` |
| `OrganCMNIST` | 冠状面 CT 软组织对比低，但左右解剖位置相对固定 | 软组织窗对比、组织边界、左右侧带、器官区域中心偏置 | 使用 CT 软组织窗近似和空间位置先验，让融合保留器官位置敏感层 | `no_domain_preprocess`, `no_domain_focus`, `no_layerwise` |
| `OrganSMNIST` | 矢状面形态和冠状面不同，脊柱/腹壁方向性更强 | 纵向延展、侧带结构、局部组织纹理、中心线偏置 | 把矢状面当独立视角，用形状和方向先验选择 client 与中层参数 | `no_domain_preprocess`, `no_domain_focus`, `no_residual` |
| `ChaoshengMNIST` | 超声 speckle 噪声重，边界模糊，声影和边缘形态重要 | 残余 speckle、边缘一致性、后方声影、方向各向异性 | 先用各向异性扩散近似保边去噪，再让边界/声影样本主导候选选择 | `no_domain_preprocess`, `no_domain_focus`, `no_candidate_bank` |
| `VLM / CLIP` | CLIP 输入是 normalized tensor，直接算颜色/灰度会破坏医学物理意义 | 反归一化后的颜色、灰度、边界和模态特征 | 先回到图像空间再提医学特征，并对 transformer/VLM 增加 reference-delta 或 prototype 候选 | `clip_denorm`, `no_reference_delta`, `no_prototype` |

这些着手点共同指向一个设计原则：融合方法要把“哪个 client 对医学关键样本更可靠”估计出来，再把这个估计传播到参数层、分类头和候选模型选择中。

## 算法总览

给定 $K$ 个 client 模型参数 $\theta_1,\dots,\theta_K$，基础权重为 $\pi_k$。`my_merge` 不直接输出一次加权平均，而是先在验证集 $D_v=\{(x_i,y_i)\}_{i=1}^{N}$ 上提取医学特征 $\phi(x_i)$，再构造医学加权样本、client 可信度、类别级专家权重和多个候选融合模型。

算法主线是：

1. 医学特征提取：根据数据集类型计算 $\phi(x)$，例如核质比、边界不规则、软组织窗对比、声影强度。
2. 样本重要性：把医学显著性、类别稀缺和模态重点合成样本权重 $w_i$。
3. Client 评分：用普通准确率、医学加权准确率、困难样本准确率和 margin/focal 指标得到 client 权重。
4. 参数融合：早期层偏向医学形态可靠的 client，中后期层混合总体表现和医学表现，分类头按类别选专家。
5. 候选选择：生成 `avg / morphology / morph_anchor / specialist_client / consensus / reference_delta / prototype_head`，再用医学验证分数选最终模型。

## 核心公式

医学特征向量记作：

$$
\phi(x_i) = [a_i, b_i, c_i, d_i, e_i, s_i]
$$

其中 $s_i$ 是诊断显著性。不同数据集的 $s_i$ 含义不同：血液中偏向核质比和核边界，皮肤镜中偏向病灶边界/颜色/伪影，CT 中偏向软组织窗和空间先验，超声中偏向 speckle/边界/声影。

样本医学重要性：

$$
m_i = \mathrm{clip}\left(\frac{s_i}{\frac{1}{N}\sum_j s_j+\epsilon}, 0.25, 3.5\right)
$$

类别稀缺权重：

$$
r_c = \mathrm{clip}\left(1+\lambda\left(\frac{\sqrt{N/n_c}}{\frac{1}{C}\sum_{t=1}^{C}\sqrt{N/n_t}}-1\right), 0.55, 3.0\right)
$$

其中 $n_c$ 是验证集中类别 $c$ 的样本数。`DermaMNIST` 使用更大的 $\lambda$，因为皮肤病变任务长尾更严重。

最终样本权重：

$$
w_i = \frac{m_i \cdot r_{y_i} \cdot d_i}{\frac{1}{N}\sum_j m_j r_{y_j} d_j+\epsilon}
$$

其中 $d_i$ 是模态重点权重，例如皮肤镜的颜色/伪影负载、CT 的空间先验、超声的边缘加声影强度。消融 `no_domain_focus` 会令 $d_i=1$，`no_rarity` 会令 $r_c=1$。

对第 $k$ 个 client，预测正确指示为 $\mathbb{1}_{i,k}$，真实类 logit margin 为：

$$
q_{i,k} = \sigma\left(z_{k,y_i}(x_i)-\max_{c\ne y_i}z_{k,c}(x_i)\right)
$$

医学加权准确率和困难样本 focal 准确率为：

$$
M_k = \frac{\sum_i w_i \mathbb{1}_{i,k}}{\sum_i w_i+\epsilon}
$$

$$
F_k = \frac{\sum_i w_i\left((1-q_{i,k})^\gamma+\tau\right)\mathbb{1}_{i,k}}{\sum_i w_i\left((1-q_{i,k})^\gamma+\tau\right)+\epsilon}
$$

这里 $F_k$ 会强化“模型虽然总体还行，但在医学困难样本上是否可靠”。消融 `no_focal` 会把 focal 权重退化成普通医学样本权重。

Client 的总体融合权重和医学融合权重分别来自：

$$
S_k^{all}=\pi_k(0.16+A_k+0.26Q_k+0.16H_k+0.18F_k)
$$

$$
S_k^{morph}=\pi_k(0.10+0.82M_k+0.28H_k+0.16Q_k+0.24F_k)
$$

其中 $A_k$ 是普通准确率，$Q_k$ 是 margin 均值，$H_k$ 是高医学重要性样本上的准确率。归一化后得到 $\alpha_k^{all}$ 和 $\alpha_k^{morph}$。

分类头按类别融合。对类别 $c$ 和 client $k$：

$$
G_{c,k}=\pi_k \cdot b_{c,k}\cdot r_c(0.10+0.74A_{c,k}+0.42M_{c,k}+0.24Q_{c,k})
$$

其中 $b_{c,k}$ 是 client 是否见过该类的专家 bonus。分类头第 $c$ 行使用 top-k 类别专家：

$$
\theta_c^{head}=0.78\sum_{k\in \mathrm{TopK}(G_c)} \mathrm{softmax}(G_{c,k}/T)\theta_{c,k}^{head}+0.22\sum_k \alpha_k^{morph}\theta_{c,k}^{head}
$$

非分类头参数按层融合。早期层、中间层、后期层分别使用不同 anchor 比例：

$$
\theta_l=\sum_k \alpha_{l,k}\theta_{l,k}+\rho_l \cdot \mathrm{TopSparse}\left(0.7\Delta_{p,l}+0.3\Delta_{s,l}\right)
$$

其中 $\alpha_{l,k}$ 根据层类型在医学权重、总体权重和 anchor one-hot 之间混合，$\Delta_{p,l}$ 和 $\Delta_{s,l}$ 是主医学专家和次总体专家相对均值的残差。消融 `no_layerwise` 会让所有层使用同一组 consensus 权重，`no_residual` 会令 $\rho_l=0$。

候选模型选择使用验证集医学分数：

$$
\mathrm{Score}=a\cdot Acc+b\cdot BalAcc+c\cdot MorphAcc+d\cdot HardAcc
$$

不同数据集的 $a,b,c,d$ 不同。皮肤镜更重视 `balanced_acc`，超声更重视困难样本和边界样本，器官 CT 更重视医学形态权重。消融 `no_balanced_selection` 会弱化 `balanced_acc`。

## 消融模块与算法含义

`avg_only` 是控制组，输出 $\sum_k \pi_k\theta_k$，用于证明完整方法不是普通平均的换皮。

`no_domain_preprocess` 关闭医学模态特征，退回通用边缘/对比度统计。如果这个消融掉分，说明医学物理先验确实在指导融合。

`no_rarity` 关闭 $r_c$。它主要检验皮肤镜和其他长尾医学类别中，少数类保护是否必要。

`no_focal` 关闭 $F_k$ 中的困难样本权重。它检验模型融合是否需要关注低 margin 的疑难病例。

`no_domain_focus` 令 $d_i=1$。它检验病灶边界、声影、空间先验、伪影负载这些模态重点样本是否应该影响 client 选择。

`no_layerwise` 关闭早/中/晚层差异化。它检验医学证据是否只应该影响分类头，还是也应该影响边界、纹理、形状层。

`no_residual` 关闭稀疏残差回注。它检验是否需要保留医学专家 client 中少量高置信结构，而不是把所有参数都平滑掉。

`no_candidate_bank` 关闭医学 anchor、专家 client、reference-delta、prototype head 和 consensus 候选。它检验医学影像是否需要“先生成多种合理融合，再用医学验证指标选择”，而不是固定一种融合公式。

`no_calibration` 关闭 BN 重校准和 head temperature。它检验后融合数值稳定性是否影响结果，尤其是 CNN 和 logit 爆炸风险。

## 可消融组件

现在可以通过 `--my-merge-ablation` 或 `--my-merge-disable` 关闭组件。主要预设如下：

| preset | 含义 | 主要回答的问题 |
| --- | --- | --- |
| `full` | 完整医学融合方法 | 医学专攻算法的最终表现 |
| `avg_only` | 退化成普通平均控制组 | 完整方法相对平均融合的增益 |
| `no_domain_preprocess` | 关闭各模态医学特征与物理预处理 | 医学图像先验是否带来真实贡献 |
| `no_rarity` | 关闭类别稀缺权重 | 长尾医学类别是否需要单独保护 |
| `no_focal` | 关闭困难样本 margin/focal 权重 | 难病例是否驱动了更好的融合选择 |
| `no_domain_focus` | 关闭模态专属重点样本加权 | 病灶边界、声影、空间先验是否有用 |
| `no_layerwise` | 关闭早/中/晚层差异化融合 | 医学特征是否应该影响不同参数层 |
| `no_residual` | 关闭稀疏残差回注 | 保留专家 client 的高置信结构是否有用 |
| `no_candidate_bank` | 关闭 anchor/specialist/reference/prototype/consensus 候选库 | 自动候选选择是否优于单一路径 |
| `no_calibration` | 关闭 BN 重校准和 head temperature | 后融合数值稳定性是否是必要条件 |

也可以关单个组件，例如：

```bash
--my-merge-disable derma_hair
--my-merge-disable ct_window
--my-merge-disable ultrasound_diffusion,ultrasound_shadow
```

支持的别名包括 `derma_color_constancy`、`hair_removal`、`ct_spatial_prior`、`clip_denorm` 等。

## 如何运行

快速验证单个 case：

```bash
.gpuenv/bin/python scripts/run_all_avg_eval.py \
  --model-hub-root model_hub \
  --data-root Med_data \
  --output-root outputs/my_merge_ablation_smoke \
  --device cuda:0 \
  --task-type small \
  --datasets bloodmnist_224 \
  --small-models resnet \
  --limit 1 \
  --method my_merge \
  --merge-weight-mode equal \
  --my-merge-ablation full \
  --my-merge-stats-max-batches 1 \
  --my-merge-eval-max-batches 1 \
  --my-merge-bn-batches 0
```

运行一组标准消融：

```bash
MODEL_HUB_ROOT=model_hub \
DATA_ROOT=Med_data \
DEVICE=cuda:0 \
ABLATIONS="full no_domain_preprocess no_rarity no_focal no_domain_focus no_layerwise no_residual no_candidate_bank no_calibration avg_only" \
bash scripts/run_my_merge_ablation_grid.sh
```

如果只想先跑表现不好的数据集，例如超声和器官 CT：

```bash
MODEL_HUB_ROOT=model_hub \
DATA_ROOT=Med_data \
DATASETS="chaoshengmnist_224 organcmnist_224 organsmnist_224" \
TASK_TYPES="small" \
SMALL_MODELS="resnet swin_tiny vit_t" \
ABLATIONS="full no_domain_preprocess no_domain_focus no_layerwise no_candidate_bank avg_only" \
bash scripts/run_my_merge_ablation_grid.sh
```

消融脚本会在输出目录生成：

```text
reports/ablation_grid_config.txt
reports/<ablation>_<task_type>.log
reports/ablation_summary.md
```

也可以手动汇总已有消融结果：

```bash
.gpuenv/bin/python scripts/summarize_my_merge_ablations.py \
  --grid-root outputs/my_merge_ablation_grid_xxx \
  --baseline result/all_results.md \
  --dest outputs/my_merge_ablation_grid_xxx/reports/ablation_summary.md
```

## 如何解读

最重要的不是只看 `full` 是否最高，而是看“关掉匹配医学组件后是否定向掉分”：

1. 如果 `no_domain_preprocess` 在 Derma/Organ/Chaosheng 上明显掉分，说明颜色恒常、CT 窗位、超声扩散这些医学物理模块确实贡献了泛化。
2. 如果 `no_rarity` 或 `no_focal` 在 Derma 上掉分，说明方法不是靠良性大类刷总体准确率，而是在保护医学长尾类。
3. 如果 `no_layerwise` 或 `no_residual` 掉分，说明医学证据不只影响最终分类头，也应该指导早期边界/纹理层和中后期语义层。
4. 如果 `no_candidate_bank` 掉分，说明单一融合公式不够，医学影像需要在 anchor、专家 client、原型头和 reference-delta 间按验证集证据选择。
5. 如果某个组件不掉分甚至涨分，应该把它记录为失败假设，继续按数据集定位，而不是把它硬写成贡献。

## 已验证的 smoke 结果

在本机本仓库路径上已验证两个最小 case 可以完整跑通：

| ablation | task | acc | 说明 |
| --- | --- | --- | --- |
| `avg_only` | `small/bloodmnist_224/resnet/c3_b0` | `0.3017` | 普通平均控制组，流程正常 |
| `full` | `small/bloodmnist_224/resnet/c3_b0` | `0.4157` | 医学候选融合正常，选择 `consensus` |

这两个 smoke 不是最终结论，只证明新消融接口、merge/eval 链路和 `merge_result.json` 中的 `ablation_config` 都能正常工作。
