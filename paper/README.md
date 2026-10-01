# IEEE/TMI 论文

当前唯一主文件：`LAMP_Merge_TMI.tex`。来源为 GitHub 提交 `b82ba8b` 上传的根目录 `LAMP_Merge_TMI.zip`。

- 主稿和配套资源来自上传包；正文保持不变，仅修复三处模板编译问题，见下文。其余 71 个包内文件逐字节一致。
- `aaai2026.bib` 是新稿继续使用的参考文献数据库文件名；参考文献样式是 `IEEEtran`，旧 AAAI 的 `.sty/.bst` 已删除。
- 上传包说明原件保留为 `UPLOAD_README.md`。包内没有启用独立附录，本仓库不再保留旧附录或 checklist。
- `data/` 保留原有 18 份实验 CSV；`snapshot.json` 记录新稿资源和这些 CSV 的校验值。
- 旧 AAAI 主稿、模板、独立附录、图片副本、编译产物及旧论文 ZIP 已直接删除，可通过 Git 历史查阅。

## 编译

在本目录运行：

```bash
mkdir -p build
latexmk -pdf -interaction=nonstopmode -outdir=build LAMP_Merge_TMI.tex
```

使用 pdfLaTeX + BibTeX；需安装 `enumitem`、`cite` 宏包和提供 `IEEEtran.bst` 的 `ieeetran` 包。Overleaf 主文件也选 `LAMP_Merge_TMI.tex`。

## 已验证与兼容修复

2026-10-01 在本机完成编译，生成 `build/LAMP_Merge_TMI.pdf`（9 页）。51 个引用键全部存在，最终编译没有未解析引用；仍有原稿的 4 处 overfull 警告和颜色定义警告，尚未逐页修整版面。

首页目视检查可见模板 `LOGO` 占位字样，以及作者姓名 Bo Du 被拆行。这些上传模板的排版问题本次没有擅自调整，投稿前仍需处理。

为使上传模板可编译，仅作以下修改：

1. 加载 `enumitem` 前加 `\let\labelindent\relax`，解决与 IEEE 类的重复定义。
2. 增加 `\providecommand{\refname}{References}`，补齐模板书目环境需要的名称。
3. 将 `ieeecolor.cls` 中无效的 `\itemsep 0pt plus pt` 修正为 `\itemsep 0pt plus 0pt`。

主稿从 `\begin{document}` 开始的内容与上传稿一致（忽略换行编码差异），18 份实验 CSV 保持原内容。`UPLOAD_README.md` 是上传包自带的历史转换记录，其中旧的未编译说明不代表当前验证状态。

本次接入上传模板并修复编译，不代表已重新审查 TMI 投稿规范或替作者校对内容。新实验与新图先输出到 `outputs/`，不要直接覆盖论文快照。
