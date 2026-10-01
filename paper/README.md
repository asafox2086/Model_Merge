# IEEE/TMI 论文

当前唯一主文件：`LAMP_Merge_TMI.tex`。来源为 GitHub 提交 `b82ba8b` 上传的根目录 `LAMP_Merge_TMI.zip`。

- 主稿和配套资源来自上传包；2026-10-01 已按实验记录完成正文修订，详见 `REVISION_NOTES.md`。上传 ZIP 保留原始版本，当前正文不再与其逐字节一致。
- `aaai2026.bib` 是新稿继续使用的参考文献数据库文件名；参考文献样式是 `IEEEtran`，旧 AAAI 的 `.sty/.bst` 已删除。
- 上传包说明原件保留为 `UPLOAD_README.md`。包内没有启用独立附录，本仓库不再保留旧附录或 checklist。
- `data/` 保留原有 18 份历史实验 CSV，并新增逐配置表及审计记录；`snapshot.json` 记录当前稿资源的校验值。主表依据 `My_merge_ret/reports/head_only_baselines_20260724_v5.csv`，不要用历史 `主实验_*.csv` 覆盖当前主表。
- 旧 AAAI 主稿、模板、独立附录、图片副本、编译产物及旧论文 ZIP 已直接删除，可通过 Git 历史查阅。

## 编译

在本目录运行：

```bash
mkdir -p build
latexmk -pdf -interaction=nonstopmode -outdir=build LAMP_Merge_TMI.tex
```

使用 pdfLaTeX + BibTeX；需安装 `enumitem`、`cite` 宏包和提供 `IEEEtran.bst` 的 `ieeetran` 包。Overleaf 主文件也选 `LAMP_Merge_TMI.tex`。

## 已验证与兼容修复

2026-10-01 修订后在本机完成编译，生成 `build/LAMP_Merge_TMI.pdf`（10 页）。55 个引用键全部存在，最终编译没有未解析引用；仍有原稿首页的 4 处 overfull 警告和颜色定义警告。已目视检查理论说明和新增逐配置表所在页，没有新增溢出警告。

首页目视检查可见模板 `LOGO` 占位字样，以及作者姓名 Bo Du 被拆行。这些上传模板的排版问题本次没有擅自调整，投稿前仍需处理。

为使上传模板可编译，仅作以下修改：

1. 加载 `enumitem` 前加 `\let\labelindent\relax`，解决与 IEEE 类的重复定义。
2. 增加 `\providecommand{\refname}{References}`，补齐模板书目环境需要的名称。
3. 将 `ieeecolor.cls` 中无效的 `\itemsep 0pt plus pt` 修正为 `\itemsep 0pt plus 0pt`。

18 份历史实验 CSV 保持原内容。`UPLOAD_README.md` 是上传包自带的历史转换记录，其中旧的未编译说明不代表当前验证状态。

本次内容修订不代表已完成 TMI 投稿规范审查。新实验与新图先输出到 `outputs/`，不要直接覆盖论文快照。

## 数据复核

在仓库根目录运行（基础核验只需 Python 标准库）：

```bash
python scripts/audit_tmi_revision.py
python scripts/check_repository.py --archive
```

`audit_tmi_revision.py` 核验主表的 768 对指标、2160 个原始混淆矩阵，并检查生成表格和超参数记录是否过期。加 `--write` 可重新生成四个派生文件中的基础三个。完整的 180 组分区审计还需本地 `Med_data/`、分区、原型载荷以及 NumPy/PyTorch：

```bash
python scripts/audit_tmi_revision.py --partitions
```

只有明确更新论文数据时才使用 `--write --partitions`，随后检查差异并更新 `snapshot.json`。原始指标 CSV 已纳入 Git；原图、模型和逐客户端载荷不会因论文核验而上传。
