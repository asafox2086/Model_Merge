# Pscore-MLP：TMI 模型融合分类适配

来源：Wimmer et al., *Multi-Task Fusion for Improving Mammography Screening Data Classification*, IEEE Transactions on Medical Imaging, 41(4):937–950, 2022. DOI: [10.1109/TMI.2021.3129068](https://doi.org/10.1109/TMI.2021.3129068)。全文：[arXiv:2112.01320](https://arxiv.org/pdf/2112.01320)。

这是非联邦的预测融合基线，不是参数平均，也不声称把多个分类头压缩成一个线性头。原文 Sec. II-C.1 拼接任务模型概率，Sec. III 的 Prediction score fusion 比较 SVM、随机森林及三种 MLP。本实现选用其中的 MLP 分支；不是 Pfeat，也不是整个原始乳腺摄影系统的复现。

## 与主表一致的设置

- 五个数据集、四个 backbone、K=3/5/7、beta=0/0.01/0.1、seed=42，共 180 个配置。
- 使用与已有 head-only 基线相同的客户端 checkpoint 分类头。每个客户端头作用于同家族、同类别数的冻结参考 backbone 特征；不使用本地 checkpoint 的 backbone。
- 原样复用 NPZ 的 train/val/test，输入变换与 `utils/runtime.py` 一致。参考 backbone 为 eval 模式，CUDA autocast FP16，与既有评估一致；缓存分类头输入，避免重复运行同一 backbone。
- 使用现有 `classification_metrics` 和 `macro_ovr_auc`；逐配置计算 ACC/Macro-F1，再按照既有 beta/K 平均规则汇总。

参考特征与专家 logits 在 CUDA autocast FP16 下计算，专家 softmax 概率转 FP32；小型融合 MLP 在 CPU FP32 下训练和评估。这是当前适配的执行精度，保存于实现与运行来源记录，不表示对原论文的硬件设置作完全复现。

## 必须披露的适配和额外信息

原文融合乳腺密度、病灶等不同任务模型。这里将其替换成 K 个医院在同一分类任务上的专家头；将二分类输出改成 C 分类。每个模型贡献 C 个 softmax 概率，按客户端编号拼接为 K*C 维输入。所有类均保留，不使用 LAMP 的类别原型或支持计数。

原文三种隐藏层结构 `[D,2]`、`[D,D,2]`、`[D,D/2,2]` 改为 `[D,C]`、`[D,D,C]`、`[D,floor(D/2),C]`，其中 D=K*C，C 是输出层。融合器额外消费 **完整训练划分上的专家概率及标签**。同一冻结参考空间只计算一次特征，不改变既有客户端训练。实现集中模拟概率收集与融合器训练；真实部署需要获得这些逐样本概率/标签，不能声称遵守 LAMP 的“仅上传类别聚合统计”接口。

融合器使用 Adam、ReLU、交叉熵；epochs、batch size、lr、weight decay 取当前上游元信息（正式配置为 50、64、0.001、0.0001）。这些是分类适配的统一训练设置，不冒称原文全部超参数。三种架构均从 seed 42 初始化。训练只使用 train；以 val ACC 最大、val 交叉熵最小为先后准则选架构和 epoch，替代原文二分类 validation AUC 选择。测试标签不用于训练、选择或早停；选好后只评估最终模型。完整架构/epoch 历史保存供审查。

因此可保持同一数据、专家、backbone、划分和评价轴，但信息预算与无训练合并方法不同。正式表格应命名 **Pscore-MLP (adapted)**，并说明监督式概率融合，不应仅标为无训练 model merging。

## 运行

```bash
python scripts/run_pscore_baseline.py --models swin_tiny --datasets chaoshengmnist_224 --clients 3 --betas 0 --epochs 2 --output-root outputs/pscore_smoke --device cuda:0
python scripts/run_pscore_baseline.py --models resnet --output-root outputs/pscore_full_20261003 --device cuda:0
python scripts/run_pscore_baseline.py --models convnext --output-root outputs/pscore_full_20261003 --device cuda:1
python scripts/run_pscore_baseline.py --models vit_t --output-root outputs/pscore_full_20261003 --device cuda:2
python scripts/run_pscore_baseline.py --models swin_tiny --output-root outputs/pscore_full_20261003 --device cuda:3
```

正式运行不要传 `--epochs`。每模型写独立 CSV，可并行执行。重复命令可恢复已完成配置；源代码、特征来源或超参数变化会拒绝复用完成结果。每个配置保存 `protocol.json`、`training_history.json`、`fusion.pt`、`test_predictions.npz` 和 `result.json`。融合器重载后与保存前预测严格比较。缓存特征逐批核验能够精确重建参考模型 logits。

本次实际运行进一步按数据集拆分：Blood 使用上述输出根目录；其余数据集使用 `outputs/pscore_full_20261003/shards/<dataset>/` 并传 `--datasets <dataset>`。各分片使用同一代码及训练配置，互不覆盖。汇总递归读取逐配置 `result.json`，不依赖某个分片的中间 CSV：

```bash
python scripts/summarize_pscore_baseline.py --input-root outputs/pscore_full_20261003 --output-dir paper/data/pscore
python scripts/audit_tmi_revision.py
```

汇总器要求 180 组无缺失、无重复；从保存的概率重算混淆矩阵和 ACC/Macro-F1，复核每组完整的 3×50 验证记录及选择结果，并比对旧主表的测试类别计数。覆盖不完整时会拒绝生成正式报告。

新增结果先写 `outputs/`，不覆盖任何历史论文表。性能数值只能在完整运行和结果核验后填入论文。
