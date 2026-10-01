# 目录整理记录

基准为根目录 `v5.tex`（2026-07-25）、`appendix.tex`（2026-07-26）和论文实验 CSV；比 2026-07-23 打包稿更新。现统一放到 `paper/`。

## 归档原则

1. 旧论文包、旧版本、旧 my_merge 文档进入 `remove/legacy/`。
2. 重复脚本进入 `remove/duplicates/`；原入口改成相对软链接，指向唯一实现。
3. 数据、checkpoint、参考缓存、原型统计与有效实验报告保持原路径。
4. 用途不能确定的报告保留在 `My_merge_ret/`，不会因名称含日期就归档。
5. `remove/manifest.json` 记录原路径、目标路径、原追踪状态、大小与适用的 SHA-256。

`remove/` 默认忽略；原先受 Git 管理的文件作为重命名继续纳入版本控制。原本未跟踪的大文件仅在本机保留。`remove/local/` 保存约 4 GB 数据压缩包。

公共域评估仍需旧 empirical study 的指标函数，已提取到 `exp_analyze/public_prediction_metrics.py`，活动代码不依赖 `remove/`。旧经验研究整体归档供追溯；归档脚本保留原路径，不是新的运行入口。

## 复现修复

- 移除错误的 `methods/*` 忽略规则，补齐所有被导入的基线源码。
- 单任务 JSON 改为相对路径，批量评估默认输出到当前仓库。
- 新入口固定 seed；批量融合空选择或出现失败时返回非零状态。
- 比较脚本默认 head-only；自定义脚本默认 `lamp_merge`。
- 绘图默认指向 `paper/data/` 与 `paper/figures/`。
- 论文正文、CSV 和正式 DPR/LPC 算法保持原内容。

## 维护

运行 `python scripts/check_repository.py` 检查结构。方法变更仍需合成输入或选定配置 smoke；语法检查不等于数值验证。

`model_hub/`、`reference_cache/`、`outputs/`、`logs/`、`My_merge_ret/` 保持名字，以保留历史路径。新报告默认不进 Git；新的正式快照放入 `paper/data/` 并记录来源。不要未经核对覆盖现有快照。
