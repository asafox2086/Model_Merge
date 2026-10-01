# LAMP-Merge — TMI 模板版本

主文件：LAMP_Merge_TMI.tex

## 使用
将本文件夹或 ZIP 上传到 Overleaf，选择 LAMP_Merge_TMI.tex 为主文件，使用 pdfLaTeX。参考文献使用 BibTeX 和标准 IEEEtran.bst（Overleaf / TeX Live 提供）。

## 修改范围
- 使用用户提供的 alternate_tj_latex_template_ap 中的 ieeecolor.cls 和 generic.sty。
- 使用 journal,twoside,web 选项，以及模板默认的字体、双栏、章节编号和图表标题格式。
- 页眉和期刊名设为 IEEE TRANSACTIONS ON MEDICAL IMAGING。
- 作者姓名、顺序、共同贡献关系、单位及邮箱保留，改为 IEEE 作者与脚注格式。
- 从 AAAI 作者年份引用改为 IEEE 数字引用，参考文献样式设为 IEEEtran。
- 原始 LAMP_Merge 文件夹及其 PDF、ZIP 均未修改。

## 内容核对
从原稿的 begin{document} 到 end{document}，除增加 bibliographystyle{IEEEtran} 命令外逐字符一致，包括摘要、正文、公式、图表、数值、标题、引用键和注释。
参考文献数据库与图片直接复制，未修改。
已检查 51 个唯一引用键和 13 处图片引用，均可在工程内找到。
原稿未启用的附录保持未启用；原始独立附录、checklist 和 v5.tex 仍在原文件夹中。
未自行增加关键词、资助信息、收稿日期或作者 IEEE 会员身份。

## 编译状态
已尝试 Codex 内置 LaTeX 编译器，但编译器返回环境错误：Unable to find standard directories for platform。
因此目前交付的是已转换的源码工程，未生成或验证最终 PDF，分页、浮动体位置和溢出仍需在 Overleaf 编译后确认。
本次依据用户提供的模板进行格式迁移，未另行核查期刊最新投稿要求。