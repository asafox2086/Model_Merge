# LAMP-Merge：论文复现与分析

本仓库对应上传的 IEEE/TMI 稿件 `paper/LAMP_Merge_TMI.tex`，研究一次性医学模型融合。上游训练资产由 `model_hub/` 提供；本仓库负责客户端统计导出、融合、评估与分析。

正式方法由 **DPR（Diagnostic Prototype Reconstruction）** 和 **LPC（Long-tail Prevalence Calibration）** 组成。客户端在共享参考 encoder 中上传类别原型、支持数和本地类别计数；服务器保留参考 backbone，重建分类头并按门控加入类别先验偏置。

## 目录

```text
paper/                 IEEE/TMI 论文源码、figures/ 和保留的 data/*.csv
experiments/           按论文组织的配置、统一运行入口与结果汇总
methods/               正式方法、分类头基线、消融实现
merge.py evaluate.py    单任务融合与评估
dataset/ model/        数据读取与模型重建（含 VLM 兼容代码）
evaluators/ utils/     评估、参考缓存、统计量与运行工具
scripts/               输入资产准备、批量融合评估、绘图工具
exp_analyze/           预测诊断、消融、几何分析与历史实验编排
configs/               使用仓库相对路径的单任务示例
docs/                  复现流程、论文实验地图、维护说明
My_merge_ret/          已有汇总表、报告与分析图，保持原路径
remove/                历史版本、旧文档、重复副本及迁移清单
model_hub/ Med_data/    本地 checkpoint 与数据，不进入 Git
reference_cache/       本地参考模型，不进入 Git
outputs/ logs/         本地运行产物，不进入 Git
```

`scripts/` 与 `exp_analyze/` 中的一些旧入口是相对软链接，同一脚本只有一份可维护的实现。正式算法在 `methods/lamp_merge.py`，消融在 `methods/lamp_merge_analysis.py`。

## 从这里开始

使用 Python 3.10；`requirements.txt` 记录整理时 MM 环境的核心依赖版本。GPU 环境先安装 PyTorch 2.6.0 / torchvision 0.21.0 的 CUDA 11.8 构建，再安装其余依赖。本机可运行 `conda activate MM`，或使用 `/data2/liyapeng_grp/.conda/envs/MM/bin/python`。

```bash
python scripts/check_repository.py
python experiments/run.py --list
python experiments/run.py main --check
```

最后一条只打印计划并检查全部 180 个配置的输入路径，不执行评估。全新 clone 需要另外准备数据、客户端 checkpoint、参考缓存和客户端原型统计；这些大文件不随代码推送。

先验证一个模型配置：

```bash
python experiments/run.py main --tag smoke \
  --datasets bloodmnist_224 --models resnet --clients 3 --betas 0 \
  --methods head_avg lamp_merge --device cuda:0 --execute
```

运行完整主实验：

```bash
python experiments/run.py main --tag paper_main --device cuda:0 --execute
python experiments/summarize.py outputs/paper/paper_main/main/reports/metrics.csv
```

统一入口默认只显示命令；加 `--execute` 才执行。主实验包含 12 个 **head-only** 基线与正式 LAMP-Merge，同时收集 ACC、macro-F1、AUC 和预测坍缩指标。输出位于 `outputs/paper/<tag>/<suite>/`，包含计划、日志、逐配置指标、汇总和图，不覆盖论文 CSV。复用同一 tag 可恢复同一计划；更换配置请使用新 tag。

## 论文口径

- 数据：Blood、Derma、Organ-C、Organ-S、Ultrasound；模型：ResNet、ConvNeXt、ViT-Tiny、Swin-Tiny。
- `K={3,5,7}`，`beta={0,0.01,0.1}`，`seed=42`；每方法 180 个配置。
- 固定数据集、模型和 K，对三个 beta 取平均，得到 60 个 client-average 单元。
- 超参数：`gamma=0.55, s=18.75, tau=2.5, lambda=4.25`。
- `head_*` 是最终稿共享参考 backbone 的基线；裸 `avg/ties/...` 是全参数对照，两种口径不可混用。
- `My_merge_ret/汇总表.md` 是历史 ACC 汇总，不能直接当作最终稿 head-only ACC/F1 表。最终数值快照在 `paper/data/`。

详细操作见 [复现指南](docs/reproduction.md)、[论文实验地图](docs/experiments.md)、[整理记录](docs/repository.md)。归档可按 [迁移清单](remove/manifest.json) 恢复。

## 新增 TMI 非联邦融合基线

`Pscore-MLP (adapted)` 来自 Wimmer 等的 TMI 2022 多任务融合论文，复用相同客户端分类头和冻结参考 backbone，拼接专家概率后训练融合器。它使用带标签的训练数据及验证集选择，属于监督式预测融合，不是无训练参数合并。适配定义、信息预算和运行命令见 [Pscore 基线](docs/pscore_baseline.md)。

```bash
python scripts/run_pscore_baseline.py --models resnet --output-root outputs/pscore_full_20261003 --device cuda:0
python scripts/summarize_pscore_baseline.py --input-root outputs/pscore_full_20261003 --output-dir paper/data/pscore
```

新基线使用独立入口，不通过 `merge.py` 或现有 `--methods head_*` 接口冒充线性参数合并。完整数据和验证通过后的结果位于 `paper/data/pscore/`。
