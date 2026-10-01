# 最终论文与实验地图

按论文小节定位，避免修订造成图表编号变化。`python experiments/run.py --list` 列出任务。

| 论文内容 | Suite / 入口 | 冻结 CSV 或用途 |
| --- | --- | --- |
| Comparison With the State of the Arts | `main` | `paper/data/主实验_*.csv`；12 个 head-only 基线与 LAMP-Merge |
| DPR/LPC 模块有效性 | `modules` | `paper/data/消融.csv` |
| TIES/DARE DPR×LPC | `baseline_2x2` | `paper/data/基线2x2消融.csv`；DPR 两支跨基线共享 |
| Effectiveness of DPR | `dpr` | `paper/data/诊断原型重建内部消融.csv` |
| Effectiveness of LPC | `lpc` | `paper/data/长尾患病率校准内部消融.csv` |
| Prediction-collapse Diagnostics | `collapse` | `paper/data/预测坍缩诊断.csv`；新运行还包含各客户端 |
| Distribution Analysis | `collapse` 的逐类别指标与图 | 超声与 Derma 类别分布 CSV |
| Hyperparameter Analysis | `hparam_dpr` / `hparam_lpc` | 两份 `paper/data/超参数分析_*.csv` |
| 附录 Algorithmic Specification | `methods/lamp_merge.py` | `paper/appendix.tex` |
| 附录 Organ-S / 数据集统计 | `main` 中 Organ-S；数据统计快照 | 主实验与数据集统计 CSV |
| 扩展：原型几何 / t-SNE | `exp_analyze/analyze_lamp_merge_prototype_geometry.py` | 参数见 `--help` |
| 扩展：公共域评估 | `exp_analyze/run_lamp_merge_public_test.py` | 公共数据仅作评估，不作融合输入 |

## 实现定位

- DPR：`methods/lamp_merge.py::_global_prototypes`，每类支持证据分别归一化。
- 头重建：`_synthesize_reference_prototype_model`，归一化原型乘 s，参考 backbone 保持冻结。
- LPC：`_prevalence_calibration_strength` 与 `_centered_log_prior_bias`，按 `C*max(prior)>tau` 激活。
- 基线：`methods/head_only.py`，只处理 classifier tensor，再覆盖到参考模型。
- 消融：`methods/lamp_merge_analysis.py`；探索分支不要加入正式实现。
- 指标：`exp_analyze/collect_prediction_diagnostics.py`，保存混淆矩阵、类别分布、ACC、macro-F1、AUC。

`experiments/paper.json` 集中固定数据集、backbone、K、beta、seed、超参数和消融模式。新增机制添加独立分析模式与 suite，并使用新 tag。

## 保留的入口

`run_compare_multi_gpu.sh` 默认 small/head-only；全参数对照显式指定 `FORMAL_METHODS`。`run_custom_methods_multi_gpu.sh` 默认 `lamp_merge`，已传客户端原型目录。新实验优先使用 `experiments/run.py`，同时支持预览、输入检查与完成校验。

`exp_analyze/run_*full*.sh` 等旧调度器用于已有实验续跑，可能绑定历史目录、任务数或 GPU 数量，不作为全新复现默认入口。
