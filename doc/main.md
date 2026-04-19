# MedMNSITMerge 仓库目标

更新时间：2026-03-15 12:02

## 这个仓库要做什么

这个仓库是 `MedModelMerging` 的下游融合仓库，目标不是训练，而是把上游已经训练好的 client 模型标准化接进来，完成以下闭环：

1. 从 `model_hub/` 读取一组可融合的 client checkpoints。
2. 在统一接口下实现多种 model merging 方法。
3. 对融合后的模型在对应数据集的 `test` split 上做推理评估。
4. 将结果保存为结构化产物，便于后续复现实验、写表格和继续扩展新方法。

## 当前输入输出约定

输入：
- `model_hub/`
- 其中 small 路径为 `small/<dataset>/<model>/clients_<n>/beta_<beta>/seed_<seed>/`
- 其中 vlm 路径为 `vlm/<dataset>/<clip_model>/clients_<n>/beta_<beta>/seed_<seed>/`

输出：
- `outputs/<run_name>/merged/...`
- `outputs/<run_name>/eval/...`
- `outputs/<run_name>/reports/merge_summary.csv`
- `outputs/<run_name>/reports/eval_summary.csv`

## 当前支持的方法

- `avg`
- `ties`
- `dare_linear`
- `dare_ties`
- `regmean`
- `fisher`

## 当前方法实现原则

- `avg` 继续保留 sample-aware 融合方式，适配当前联邦式 client 设定。
- `ties / dare_linear / dare_ties / regmean / fisher` 的核心融合算子按官方实现思路重写。
- `regmean` 和 `fisher` 的统计量统一从 `.npz` 的 `val` split 提取。
- 批量脚本默认在评估完成后删除 `merged.pt`，只保留 `eval.json`、`merge_result.json` 和汇总表。

## 当前仓库后续要做的事

1. 把更多代表性 merging 方法接进统一接口。
2. 把 small 和 vlm 的评估结果统一汇总成论文直接可用的表。
3. 加入更强的批量调度和失败恢复。
4. 继续保持 `doc/` 下的目标说明、每日更新和实验记录。
