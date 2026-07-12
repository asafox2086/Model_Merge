# LAMP-Merge v1 Revision Workflow

本文件记录 v1 论文修缮的执行顺序。每完成一项，先检查该项是否满足要求，并确认此前已完成项没有被破坏，再进入下一项。

1. [x] 获取并定位用户新写的 v1 版本，确认编辑入口、图片目录和编译入口。
2. [x] 将全文所有“事后模型融合”统一改为“模型融合”，并检查是否误伤引用名或术语。
3. [x] 压缩 Related Work 到约原长度的一半，采用“已有方法是什么；however；为什么不适用于本文场景”的结构，突出问题并增加必要引用。
4. [x] Experiment 部分分点介绍 datasets、baselines、settings 和其他实验细节。
5. [x] 美化主结果表：非数据表头行用灰色，LAMP-Merge 行用浅黄色，baseline 名称旁加入论文引用；在 K=3、K=5、K=7 后增加三者平均列 Avg，并为 Avg 列上色。
6. [x] 将表格中的 Variable setting 移到表外，用 itemized 文本逐项解释。
7. [x] 重画或美化超参数图，提升字体大小，统一配色、标记形状和论文风格。
8. [x] 将 “Dataset-level client-average accuracy in the module ablation” 表改为图。
9. [x] 调整 t-SNE 图：增大字体；使用 joint 对比图；单行展示；不再放单独的 LAMP-only 图。
10. [x] 调整版面位置：intro 图放第一页右上角，method 图放第二页横向展示。
11. [x] Dataset statistic 不再使用统计表，改为若干示例图像。
12. [x] 给主要表格统一上色，并保持图、表颜色和形状一致；最终编译 PDF 检查，确认排版后提交并 push。
