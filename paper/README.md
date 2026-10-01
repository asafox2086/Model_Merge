# 最终论文快照

- `v5.tex`：2026-07-25 工作稿；`appendix.tex`：2026-07-26 附录。
- `figures/`：稿件图与相关源文件；`data/`：18 份论文实验 CSV。
- `snapshot.json`：整理时正文、附录与 CSV 的 SHA-256。
- `checklist.tex`、`aaai2026.*`：排版辅助文件。

保留现有内容与数值。附录说明 LPC 为 3×10，但保存的 CSV 有 5×10 个点；统一入口按 CSV 逐点复现，本次未替作者修改差异。

在此目录运行 `latexmk -pdf -outdir=build v5.tex` 编译。新图先输出到 `outputs/paper_preview/`，实验结果不能未经核对覆盖快照。
