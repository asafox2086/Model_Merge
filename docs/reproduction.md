# 复现流程

## 1. 环境与输入

从仓库根目录运行，激活 Python 3.10 的 MM 环境。`requirements.txt` 是整理当天读取到的核心包版本，不代表已在全新机器验证安装。原实验环境为 PyTorch 2.6.0 / CUDA 11.8。

| 资产 | 路径与内容 |
| --- | --- |
| 数据 | `Med_data/<dataset>.npz`，含 train/val/test splits |
| 客户端模型 | `model_hub/manifest.csv`；`small/<dataset>/<model>/clients_K/beta_B/seed_42/{meta.json,client_*.pt}` |
| 参考模型 | `reference_cache/small/*.pt`，与统计导出时的 encoder 一致 |
| 客户端统计 | `outputs/lamp_merge_client_local_proto_stats/small/.../prototype_stats.pt` |
| 原始分区 | 真实客户端索引，或已上传的完整聚合统计 |

这些资产不进入 Git。`python experiments/run.py main --check` 检查所选配置的全部输入路径。执行时还会验证原型 payload 的格式与本地类别计数来源。不要替换参考 encoder 或伪造客户端分区来补齐资产。

如尚未缓存参考模型：

```bash
python scripts/cache_reference_models.py --model-hub-root model_hub --task-type small
```

该步骤可能需要已有预训练缓存或网络。论文统一入口默认离线运行，并要求参考缓存事先存在，避免触发随机初始化回退。

客户端导出（在有权访问客户端本地数据和真实分区的环境执行）：

```bash
python scripts/export_lamp_merge_prototypes.py \
  --model-hub-root model_hub --data-root Med_data \
  --partition-root /path/to/true_client_partitions \
  --output-root outputs/lamp_merge_client_local_proto_stats --device cuda:0
```

`scripts/prepare_lamp_merge_client_partitions.py` 根据元数据重建的索引不等价于真实原始分区；它保留用于探索，不是默认复现流程。优先保留已有正式统计，不要为了运行成功覆盖它们。

## 2. 主实验与分析

```bash
python experiments/run.py main --tag main_v1 --execute
python experiments/run.py modules --tag modules_v1 --execute
python experiments/run.py baseline_2x2 --tag baseline2x2_v1 --execute
python experiments/run.py dpr --tag dpr_v1 --execute
python experiments/run.py lpc --tag lpc_v1 --execute
python experiments/run.py collapse --tag collapse_v1 --execute
```

默认每个 suite 在一张 GPU 上顺序执行。可在多个终端用 `--models` 拆分，并指定不同 `--device` 和 `--tag`，避免并发写同一 CSV。保留 merged checkpoint 使用 `--keep-merged`。

主实验及消融由预测诊断器评估，包含 macro-F1。单独的 `evaluate.py` / `scripts/run_all_avg_eval.py` 只提供 ACC/loss，不足以重建最终 ACC/F1 主表。

```bash
python experiments/summarize.py outputs/paper/main_v1/main/reports/metrics.csv
```

输出 `paper_client_average.csv`（对 beta 平均）、`paper_dataset_average.csv`（再对 K 平均）和 `paper_overall.csv`。单位为 0–1；论文 CSV 使用百分比。缺少配置会报错。子集实验需显式传相同的 `--datasets/--models/--clients/--betas`，不能把子集均值标为全量结果。

## 3. 超参数

```bash
python experiments/run.py hparam_dpr --tag hparam_v1
python experiments/run.py hparam_dpr --tag hparam_v1 --point 0 --device cuda:0 --execute
python experiments/run.py hparam_lpc --tag hparam_v1 --point 0 --device cuda:1 --execute
```

省略 `--point` 执行 CSV 全部 50 个点，每点 180 个配置。`--point` 从 0 开始，对应 `paper/data/超参数分析_*.csv` 数据行顺序。不同点可并行，同一点不能并发写入。扫描保持完整 LAMP-Merge，仅替换该点两项参数；结果在 `point_NNN/reports/`。

## 4. 图与论文

`paper/figures/` 使用 IEEE/TMI 上传包配图；`paper/data/` 保留原有实验 CSV，不表示已对新稿全部图表重新核对。新图先写独立目录：

```bash
python scripts/plot_latest_paper_figures.py --csv-dir paper/data --paper-dir outputs/paper_preview
```

论文编译：

```bash
cd paper
mkdir -p build
latexmk -pdf -interaction=nonstopmode -outdir=build LAMP_Merge_TMI.tex
```

## 5. 复现边界

- 当前稿件以 GitHub 上传的 IEEE/TMI 包为准；旧独立附录已删除。原有超参数 CSV 和实验配置继续保留，每份 CSV 含 50 个实测点。
- 历史 `exp_analyze/validate_full_outputs.py` 面向旧固定报告集合，含 23 点等历史假设；它不验证新的 `outputs/paper/`。
- 冻结 CSV 未提供每个数值到原始运行目录的完整统一映射。新入口记录计划和指标，不承诺尚未重新运行的结果逐位相同。
- 目录整理不触发全量 GPU 实验；代码检查、输入路径检查与完整数值复现是不同层级。
