#!/usr/bin/env python3
import html
import zipfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEST_DIR = ROOT / "My_merge_ret"
PPTX_PATH = DEST_DIR / "my_merge算法逻辑.pptx"
SCRIPT_PATH = DEST_DIR / "my_merge算法演讲稿.md"

SLIDE_W = 13_333_333
SLIDE_H = 7_500_000


def esc(text):
    return html.escape(str(text), quote=True)


def emu(inches):
    return int(inches * 914400)


def color(hex_value):
    return hex_value.strip("#").upper()


def text_runs(text, size=2400, bold=False, fill="202124"):
    paragraphs = []
    for para in str(text).split("\n"):
        paragraphs.append(
            f"""
            <a:p>
              <a:r>
                <a:rPr lang="zh-CN" sz="{size}"{' b="1"' if bold else ''}>
                  <a:solidFill><a:srgbClr val="{color(fill)}"/></a:solidFill>
                </a:rPr>
                <a:t>{esc(para)}</a:t>
              </a:r>
              <a:endParaRPr lang="zh-CN" sz="{size}"/>
            </a:p>
            """
        )
    return "\n".join(paragraphs)


def textbox(shape_id, x, y, w, h, text, size=2400, bold=False, fill="202124", align="l"):
    return f"""
    <p:sp>
      <p:nvSpPr>
        <p:cNvPr id="{shape_id}" name="TextBox {shape_id}"/>
        <p:cNvSpPr txBox="1"/>
        <p:nvPr/>
      </p:nvSpPr>
      <p:spPr>
        <a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>
        <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        <a:noFill/>
        <a:ln><a:noFill/></a:ln>
      </p:spPr>
      <p:txBody>
        <a:bodyPr wrap="square" rtlCol="0"><a:spAutoFit/></a:bodyPr>
        <a:lstStyle/>
        {text_runs(text, size=size, bold=bold, fill=fill).replace('<a:p>', f'<a:p><a:pPr algn="{align}"/>')}
      </p:txBody>
    </p:sp>
    """


def rect(shape_id, x, y, w, h, fill, line="FFFFFF", radius=False):
    prst = "roundRect" if radius else "rect"
    return f"""
    <p:sp>
      <p:nvSpPr>
        <p:cNvPr id="{shape_id}" name="Shape {shape_id}"/>
        <p:cNvSpPr/>
        <p:nvPr/>
      </p:nvSpPr>
      <p:spPr>
        <a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>
        <a:prstGeom prst="{prst}"><a:avLst/></a:prstGeom>
        <a:solidFill><a:srgbClr val="{color(fill)}"/></a:solidFill>
        <a:ln w="12700"><a:solidFill><a:srgbClr val="{color(line)}"/></a:solidFill></a:ln>
      </p:spPr>
    </p:sp>
    """


def line(shape_id, x1, y1, x2, y2, fill="3C4043", width=25400):
    x = min(x1, x2)
    y = min(y1, y2)
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    return f"""
    <p:cxnSp>
      <p:nvCxnSpPr>
        <p:cNvPr id="{shape_id}" name="Connector {shape_id}"/>
        <p:cNvCxnSpPr/>
        <p:nvPr/>
      </p:nvCxnSpPr>
      <p:spPr>
        <a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>
        <a:prstGeom prst="line"><a:avLst/></a:prstGeom>
        <a:ln w="{width}"><a:solidFill><a:srgbClr val="{color(fill)}"/></a:solidFill></a:ln>
      </p:spPr>
    </p:cxnSp>
    """


def card(shape_id, x, y, w, h, title, body, accent="2563EB"):
    return [
        rect(shape_id, x, y, w, h, "FFFFFF", "D6DEE8", radius=True),
        rect(shape_id + 1, x, y, emu(0.08), h, accent, accent),
        textbox(shape_id + 2, x + emu(0.22), y + emu(0.18), w - emu(0.42), emu(0.34), title, size=1600, bold=True, fill="111827"),
        textbox(shape_id + 3, x + emu(0.22), y + emu(0.58), w - emu(0.42), h - emu(0.72), body, size=1250, fill="374151"),
    ]


def bar(shape_id, x, y, label, value, max_value, fill):
    bw = emu(5.2) * value / max_value
    return [
        textbox(shape_id, x, y, emu(1.55), emu(0.28), label, size=1300, fill="111827"),
        rect(shape_id + 1, x + emu(1.75), y, emu(5.2), emu(0.22), "E5E7EB", "E5E7EB", radius=True),
        rect(shape_id + 2, x + emu(1.75), y, int(bw), emu(0.22), fill, fill, radius=True),
        textbox(shape_id + 3, x + emu(7.05), y - emu(0.03), emu(0.9), emu(0.28), f"{value:.4f}", size=1200, bold=True, fill="111827"),
    ]


def slide_xml(title, shapes, subtitle="my_merge / MedMNISTMerge"):
    title_box = textbox(2, emu(0.55), emu(0.28), emu(12.2), emu(0.55), title, size=3000, bold=True, fill="111827")
    sub_box = textbox(3, emu(0.62), emu(6.95), emu(10.5), emu(0.25), subtitle, size=950, fill="6B7280")
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:bg><p:bgPr><a:solidFill><a:srgbClr val="F8FAFC"/></a:solidFill></p:bgPr></p:bg>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      {title_box}
      {''.join(shapes)}
      {sub_box}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""


def slide_rel_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>"""


def build_slides():
    slides = []

    slides.append(slide_xml(
        "my_merge：医学证据引导模型融合",
        [
            textbox(10, emu(0.85), emu(1.15), emu(10.8), emu(0.6), "一句话：各医院先独立训练模型，my_merge 再用医学证据决定这些 checkpoint 怎么合并。", size=2350, bold=True, fill="111827"),
            *card(20, emu(0.9), emu(2.2), emu(3.6), emu(1.8), "1. 问题", "训练好的医院模型携带的诊断证据不等价。", "2563EB"),
            *card(30, emu(4.95), emu(2.2), emu(3.6), emu(1.8), "2. 方法", "M1 估计证据，M2 用证据指导合并与选择。", "059669"),
            *card(40, emu(8.99), emu(2.2), emu(3.6), emu(1.8), "3. 结果", "full 最好；去掉 M2 会退回普通融合水平。", "DC2626"),
            textbox(50, emu(1.0), emu(4.55), emu(11.0), emu(0.55), "后面只讲三件事：问题是什么，方法怎么合并，为什么有效。", size=1650, fill="4B5563"),
        ],
    ))

    slides.append(slide_xml(
        "问题：医学客户端不是同质的",
        [
            *card(10, emu(0.8), emu(1.35), emu(3.45), emu(1.75), "客户端 A", "边界清楚，形态稳定", "2563EB"),
            *card(20, emu(4.2), emu(1.35), emu(3.45), emu(1.75), "客户端 B", "纹理更明显，局部对比强", "7C3AED"),
            *card(30, emu(7.6), emu(1.35), emu(3.45), emu(1.75), "客户端 C", "少数类多，难例更多", "059669"),
            textbox(40, emu(0.95), emu(3.55), emu(11.4), emu(0.45), "FedAvg 的问题：它把这些证据当成同样重要。", size=1850, bold=True, fill="111827", align="ctr"),
            *card(50, emu(1.1), emu(4.35), emu(10.9), emu(1.15), "真正要学的不是样本数，而是诊断证据", "前景、边界、纹理、形状、类别稀有度、难例 margin。", "DC2626"),
        ],
    ))

    slides.append(slide_xml(
        "为什么不用联邦学习解决",
        [
            *card(10, emu(0.75), emu(1.15), emu(5.55), emu(2.05), "联邦学习解决什么", "多轮通信训练：服务器发全局模型，医院本地训练，再回传更新。重点是训练协议。", "6B7280"),
            *card(20, emu(7.0), emu(1.15), emu(5.55), emu(2.05), "这里解决什么", "模型后融合：医院已经训练好 checkpoint，服务器只负责把现成模型合成一个可用模型。", "2563EB"),
            *card(30, emu(0.75), emu(3.75), emu(5.55), emu(1.55), "联邦学习的错位", "要求反复通信、统一训练入口和同一轮全局模型；真实医疗里常见的是独立训练后的模型交付。", "DC2626"),
            *card(40, emu(7.0), emu(3.75), emu(5.55), emu(1.55), "my_merge 的定位", "不改变医院训练过程，不访问原始数据，只在 checkpoint 层面做医学证据引导合并。", "059669"),
        ],
    ))

    slides.append(slide_xml(
        "流程图：训练后 checkpoint 合并",
        [
            *card(10, emu(0.45), emu(1.55), emu(2.35), emu(1.35), "医院模型", "训练完成的 checkpoints", "9CA3AF"),
            textbox(20, emu(2.92), emu(1.95), emu(0.55), emu(0.4), "→", size=2300, bold=True, fill="6B7280", align="ctr"),
            *card(30, emu(3.35), emu(1.55), emu(2.6), emu(1.35), "M1", "估计客户端诊断信息", "2563EB"),
            textbox(40, emu(6.12), emu(1.95), emu(0.55), emu(0.4), "→", size=2300, bold=True, fill="6B7280", align="ctr"),
            *card(50, emu(6.55), emu(1.55), emu(2.7), emu(1.35), "M2", "证据引导融合选择", "059669"),
            textbox(60, emu(9.42), emu(1.95), emu(0.55), emu(0.4), "→", size=2300, bold=True, fill="6B7280", align="ctr"),
            *card(70, emu(9.85), emu(1.55), emu(2.45), emu(1.35), "最终模型", "输出合并后的全局模型", "DC2626"),
            rect(80, emu(0.8), emu(3.55), emu(11.8), emu(1.55), "F8FAFC", "D6DEE8", radius=True),
            textbox(81, emu(1.05), emu(3.82), emu(11.2), emu(0.95), "M1 输出三组权重：overall / morphology / class。M2 读取这些权重，做层级融合、分类头融合、候选构建和候选选择。", size=1850, bold=True, fill="111827", align="ctr"),
            textbox(82, emu(1.05), emu(4.42), emu(11.2), emu(0.35), "这不是联邦训练：没有多轮通信，只处理已经训练好的模型 checkpoint。", size=1450, fill="4B5563", align="ctr"),
        ],
    ))

    slides.append(slide_xml(
        "算法大概长什么样",
        [
            rect(10, emu(0.75), emu(1.15), emu(11.8), emu(4.7), "FFFFFF", "D6DEE8", radius=True),
            textbox(11, emu(1.05), emu(1.42), emu(11.1), emu(0.35), "Input: checkpoints {θ_i}, validation data D_val", size=1450, bold=True, fill="111827"),
            textbox(12, emu(1.05), emu(2.00), emu(11.1), emu(0.35), "1. Extract medical evidence φ(x): foreground, boundary, contrast, texture, shape", size=1350, fill="374151"),
            textbox(13, emu(1.05), emu(2.48), emu(11.1), emu(0.35), "2. Evaluate each checkpoint on D_val with φ(x), class rarity and hard-case margin", size=1350, fill="374151"),
            textbox(14, emu(1.05), emu(2.96), emu(11.1), emu(0.35), "3. Estimate weights: w_overall, w_morphology, w_class", size=1350, fill="374151"),
            textbox(15, emu(1.05), emu(3.44), emu(11.1), emu(0.35), "4. Build merged candidates: avg, morphology, anchor, specialist, consensus, etc.", size=1350, fill="374151"),
            textbox(16, emu(1.05), emu(3.92), emu(11.1), emu(0.35), "5. Select candidate by validation score; output merged checkpoint θ*", size=1350, fill="374151"),
            rect(20, emu(1.0), emu(4.85), emu(11.3), emu(0.55), "EEF2FF", "C7D2FE", radius=True),
            textbox(21, emu(1.25), emu(5.00), emu(10.8), emu(0.25), "核心：医学证据不是额外说明，而是直接决定参数融合和候选选择。", size=1450, bold=True, fill="1E3A8A", align="ctr"),
        ],
    ))

    slides.append(slide_xml(
        "M1：先估计客户端有什么诊断信息",
        [
            *card(10, emu(0.55), emu(1.25), emu(3.65), emu(2.1), "看什么", "前景面积\n边界强度\n局部对比\n纹理异质性\n形状紧致度", "2563EB"),
            *card(20, emu(4.85), emu(1.25), emu(3.65), emu(2.1), "怎么估", "验证集上看样本、难例、低 margin 和少数类表现。", "7C3AED"),
            *card(30, emu(9.15), emu(1.25), emu(3.65), emu(2.1), "输出什么", "overall_weights\nmorphology_weights\nclass_weights", "059669"),
            rect(40, emu(1.0), emu(4.15), emu(11.3), emu(0.95), "EEF2FF", "C7D2FE", radius=True),
            textbox(41, emu(1.25), emu(4.42), emu(10.8), emu(0.4), "M1 的作用很简单：把“哪个客户端更懂这个病灶”量化出来。", size=1900, bold=True, fill="1E3A8A", align="ctr"),
        ],
    ))

    slides.append(slide_xml(
        "M2：用诊断信息控制融合",
        [
            *card(10, emu(0.55), emu(1.2), emu(3.4), emu(2.05), "早期层", "更看重 morphology_weights", "2563EB"),
            *card(20, emu(4.72), emu(1.2), emu(3.4), emu(2.05), "分类头", "按类别用 class_weights", "7C3AED"),
            *card(30, emu(8.9), emu(1.2), emu(3.4), emu(2.05), "最终选择", "在候选模型里选验证集最好的", "059669"),
            rect(40, emu(0.95), emu(4.15), emu(11.35), emu(1.15), "FEF2F2", "FECACA", radius=True),
            textbox(41, emu(1.2), emu(4.42), emu(10.85), emu(0.6), "M2 不是简单平均，而是“按医学证据挑融合方式”。", size=1900, bold=True, fill="991B1B", align="ctr"),
        ],
    ))

    slides.append(slide_xml(
        "为什么不能直接搬到 NLP",
        [
            *card(10, emu(0.7), emu(1.2), emu(5.5), emu(2.0), "医学影像", "有二维空间证据：前景、边界、纹理、形状、局部对比度。", "2563EB"),
            *card(20, emu(7.1), emu(1.2), emu(5.5), emu(2.0), "NLP", "没有这些像素级证据；要重新定义语言任务里的证据。", "6B7280"),
            *card(30, emu(0.7), emu(3.75), emu(5.5), emu(1.45), "不能直接迁移的部分", "φ(x) 依赖 Sobel 边界、局部纹理、病灶前景和形态紧致度。", "DC2626"),
            *card(40, emu(7.1), emu(3.75), emu(5.5), emu(1.45), "可以迁移的部分", "“先估计来源可靠性，再指导融合”这个抽象框架。", "059669"),
        ],
    ))

    slides.append(slide_xml(
        "消融结果：哪个模块真的有用",
        [
            *bar(10, emu(1.0), emu(1.2), "full", 0.3424, 0.36, "2563EB"),
            *bar(20, emu(1.0), emu(1.9), "no M1", 0.3217, 0.36, "059669"),
            *bar(30, emu(1.0), emu(2.6), "no M2", 0.2227, 0.36, "DC2626"),
            *bar(40, emu(1.0), emu(3.3), "avg only", 0.2227, 0.36, "9CA3AF"),
            rect(50, emu(0.95), emu(4.45), emu(11.3), emu(1.05), "F8FAFC", "D6DEE8", radius=True),
            textbox(51, emu(1.2), emu(4.72), emu(10.8), emu(0.5), "结论：M2 是主贡献，M1 也有用；去掉 M2 基本回到平均融合。", size=1850, bold=True, fill="111827", align="ctr"),
        ],
    ))

    slides.append(slide_xml(
        "最后一句话",
        [
            rect(10, emu(0.95), emu(1.45), emu(11.4), emu(2.2), "EFF6FF", "BFDBFE", radius=True),
            textbox(11, emu(1.25), emu(1.8), emu(10.8), emu(1.1), "my_merge 的核心：把医学影像里的诊断证据，变成模型合并的依据。", size=2300, bold=True, fill="1E3A8A", align="ctr"),
            textbox(12, emu(1.35), emu(3.25), emu(10.6), emu(0.45), "问题 - 方法 - 结果，三句话就够。", size=1750, fill="374151", align="ctr"),
        ],
    ))

    return slides


def content_types(num_slides):
    slide_overrides = "\n".join(
        f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, num_slides + 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
  {slide_overrides}
</Types>"""


def root_rels():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""


def presentation_xml(num_slides):
    ids = "\n".join(f'<p:sldId id="{255+i}" r:id="rId{i}"/>' for i in range(1, num_slides + 1))
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId{num_slides + 1}"/></p:sldMasterIdLst>
  <p:sldIdLst>{ids}</p:sldIdLst>
  <p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>"""


def presentation_rels(num_slides):
    rels = []
    for i in range(1, num_slides + 1):
        rels.append(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>')
    rels.append(f'<Relationship Id="rId{num_slides + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>')
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {''.join(rels)}
</Relationships>"""


def slide_master():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>"""


def slide_master_rels():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>"""


def slide_layout():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>"""


def slide_layout_rels():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>"""


def theme():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="my_merge">
  <a:themeElements>
    <a:clrScheme name="my_merge">
      <a:dk1><a:srgbClr val="111827"/></a:dk1><a:lt1><a:srgbClr val="F8FAFC"/></a:lt1>
      <a:dk2><a:srgbClr val="374151"/></a:dk2><a:lt2><a:srgbClr val="E5E7EB"/></a:lt2>
      <a:accent1><a:srgbClr val="2563EB"/></a:accent1><a:accent2><a:srgbClr val="059669"/></a:accent2>
      <a:accent3><a:srgbClr val="DC2626"/></a:accent3><a:accent4><a:srgbClr val="D97706"/></a:accent4>
      <a:accent5><a:srgbClr val="7C3AED"/></a:accent5><a:accent6><a:srgbClr val="0891B2"/></a:accent6>
      <a:hlink><a:srgbClr val="2563EB"/></a:hlink><a:folHlink><a:srgbClr val="7C3AED"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="my_merge">
      <a:majorFont><a:latin typeface="Arial"/><a:ea typeface="Microsoft YaHei"/></a:majorFont>
      <a:minorFont><a:latin typeface="Arial"/><a:ea typeface="Microsoft YaHei"/></a:minorFont>
    </a:fontScheme>
    <a:fmtScheme name="my_merge"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>
  </a:themeElements>
</a:theme>"""


def app_xml(num_slides):
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
            xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>MedMNISTMerge</Application>
  <PresentationFormat>Widescreen</PresentationFormat>
  <Slides>{num_slides}</Slides>
</Properties>"""


def core_xml():
    now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
                   xmlns:dc="http://purl.org/dc/elements/1.1/"
                   xmlns:dcterms="http://purl.org/dc/terms/"
                   xmlns:dcmitype="http://purl.org/dc/dcmitype/"
                   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>my_merge算法逻辑</dc:title>
  <dc:creator>MedMNISTMerge</dc:creator>
  <cp:lastModifiedBy>MedMNISTMerge</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>"""


def write_pptx():
    slides = build_slides()
    with zipfile.ZipFile(PPTX_PATH, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types(len(slides)))
        z.writestr("_rels/.rels", root_rels())
        z.writestr("docProps/app.xml", app_xml(len(slides)))
        z.writestr("docProps/core.xml", core_xml())
        z.writestr("ppt/presentation.xml", presentation_xml(len(slides)))
        z.writestr("ppt/_rels/presentation.xml.rels", presentation_rels(len(slides)))
        z.writestr("ppt/slideMasters/slideMaster1.xml", slide_master())
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", slide_master_rels())
        z.writestr("ppt/slideLayouts/slideLayout1.xml", slide_layout())
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", slide_layout_rels())
        z.writestr("ppt/theme/theme1.xml", theme())
        for idx, slide in enumerate(slides, start=1):
            z.writestr(f"ppt/slides/slide{idx}.xml", slide)
            z.writestr(f"ppt/slides/_rels/slide{idx}.xml.rels", slide_rel_xml())


def write_script():
    content = """# my_merge 演讲稿

## 1. 开场

今天我讲的是 `my_merge`，一个医学证据引导的模型融合方法。

这里先把问题边界说清楚：我不是在做一个新的联邦学习训练框架。这里的输入不是客户端每一轮上传的梯度，也不是训练过程中的局部更新，而是各医院已经训练完成的 checkpoint。

所以这个任务更准确地叫训练后模型融合，或者 post-hoc model merging。问题是：这些 checkpoint 都来自医学影像任务，但每个医院模型学到的诊断证据并不一样。`my_merge` 要做的，就是先估计每个 checkpoint 携带的医学证据，再用这些证据决定怎么合并参数。

## 2. 问题

医学影像里的模型融合，不能简单理解成“多个模型平均一下”。

不同医院的数据分布不同，病灶形态、边界清晰度、纹理复杂度、少数类比例和难例比例都会不同。一个医院模型可能更擅长边界清楚的样本，另一个模型可能更擅长纹理复杂或者少数类样本。

普通平均的问题是，它默认所有 checkpoint 的信息价值差不多。但医学诊断不是这样。真正有价值的不是某个模型参数本身，而是这个模型在哪些医学证据上可靠。

因此，`my_merge` 的核心问题是：怎样从医学影像证据出发，判断哪个模型更值得信任，并把这种信任写进模型融合过程。

## 3. 为什么不能用联邦学习解决

这里需要明确区分模型融合和联邦学习。

经典联邦学习解决的是训练协议问题。它通常要求服务端维护一个全局模型，每一轮把全局模型发给客户端，客户端本地训练，再把更新传回来，服务端做轮次聚合。

但这里的场景不是这样。这里各医院已经独立训练好了模型，服务器拿到的是 checkpoint。我们不关心如何组织多轮训练，也不要求医院围绕同一个全局模型继续训练。我们关心的是训练完成之后，怎么把这些现成模型合并成一个更好的模型。

所以联邦学习和这里的问题不在同一层。联邦学习关注训练过程，`my_merge` 关注后融合过程。直接用 FedAvg 这类方法，只会把模型当成同质来源平均掉，无法回答“哪个 checkpoint 在边界、纹理、少数类、难例上更可靠”。

因此，这个方法不是联邦学习的替代训练协议，而是面向医学场景的 checkpoint-level model merging。

## 4. 方法流程

整个流程可以分成五步。

第一步，输入各医院训练完成的 checkpoints，以及一个验证集。

第二步，M1 从验证图像里提取医学影像证据，包括前景面积、边界强度、局部对比度、纹理异质性、形状紧致度和诊断显著性。

第三步，M1 用这些证据评估每个 checkpoint。它不只看普通准确率，还看医学加权准确率、hard case、low-margin 样本和类别稀有度。最后输出三类权重：整体权重、形态证据权重和类别级权重。

第四步，M2 使用这些权重做融合。早期层更关注形态证据，后期层更关注整体诊断可靠性，分类头按类别使用不同 checkpoint 的专长权重。

第五步，M2 构建多个候选合并模型，并在验证集上选择最终模型。

一句话概括：M1 负责回答“谁更可靠”，M2 负责回答“这些可靠性怎么写进参数合并”。

## 5. 算法大概是什么样

算法可以写成一个很直接的流程。

输入是多个 checkpoint，记作 theta_i，还有验证集 D_val。

首先定义医学影像证据函数 phi(x)。这个函数从图像中提取前景、边界、纹理、对比度和形态信息。

然后对每个 checkpoint，在验证集上计算它对这些医学证据的响应。比如普通样本上表现如何，医学证据强的样本上表现如何，少数类上表现如何，低 margin 难例上表现如何。

接着根据这些表现生成三组权重：

overall_weights 表示一个 checkpoint 的整体可靠性；
morphology_weights 表示它在形态和纹理证据上的可靠性；
class_weights 表示它对不同类别的专长。

最后进入融合阶段。普通层根据 overall 和 morphology 做加权融合；分类头根据 class_weights 按类别融合；同时构建 avg、morphology、anchor、specialist、consensus 等候选模型。最后用验证集分数选择最终输出。

所以它不是“手动选一个先验”，而是把医学影像证据转化成融合权重。

## 6. 为什么不是数据集特判

现在的方法不再针对 blood、derma、organ、ultrasound 写特例。

这点很重要。旧版本如果给每个数据集写一套特殊规则，就容易被质疑成“具体问题具体调参”。现在的方法把数据集特判收掉，只保留医学影像共通的证据。

这些证据包括空间结构、边界、局部对比、纹理变化、形状紧致度、类别稀有度和难例 margin。它们不是某个数据集独有的，而是医学影像任务中普遍存在的诊断线索。

所以这个方法是医学影像专用，但不是某一个医学数据集专用。

## 7. 为什么不能直接迁移到 NLP

能迁移的是“先估计客户端信息，再指导融合”的框架。

不能直接迁移的是这里真正起作用的证据函数 phi(x)。

医学影像是二维空间信号。前景、边界、局部纹理、形状紧致度、局部对比度，这些量在图像中有明确含义，也和诊断过程直接相关。

NLP 是 token 序列。文本里没有稳定的病灶前景，没有 Sobel 边界，没有局部纹理异质性，也没有形状紧致度。把这套证据直接搬到 NLP，会导致证据函数失效。

当然，NLP 可以借鉴这个思想：先估计不同来源模型的任务证据，再指导融合。但 NLP 必须重新定义语言证据，比如实体覆盖、句法结构、领域术语、长上下文依赖、语义歧义和难例类型。那会是另一个方法，不是当前的 `my_merge`。

## 8. 实验结果

full 的平均准确率是 0.3424。
去掉 M1 后是 0.3217，下降 0.0206。
去掉 M2 后是 0.2227，和 avg only 一样，下降 0.1197。

这组消融说明两个模块都有作用。

去掉 M1，模型仍然保留融合和候选选择结构，所以不会完全崩掉，但会失去医学证据下的 checkpoint 可靠性估计，因此性能下降。

去掉 M2，模型没有办法把医学证据真正写进参数融合和候选选择，结果基本退回普通平均融合水平。因此 M2 是主要贡献，M1 是必要的信息来源。

## 9. 结论

一句话总结：`my_merge` 不是做“医学数据集特判”，而是把医学影像证据变成模型合并的依据。

它和联邦学习的区别是：联邦学习解决训练过程，`my_merge` 解决训练后 checkpoint 融合。

它和 NLP 方法的区别是：`my_merge` 的核心证据函数依赖医学影像的二维空间结构，不能直接搬到文本任务。

最终，这个方法的故事是：医学 checkpoint 的价值不等价；这种不等价可以通过医学影像证据估计；估计出来的证据可以指导模型融合，并且消融结果证明两个模块都有作用。
"""
    SCRIPT_PATH.write_text(content, encoding="utf-8")


def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    write_pptx()
    write_script()
    print(f"wrote {PPTX_PATH}")
    print(f"wrote {SCRIPT_PATH}")


if __name__ == "__main__":
    main()
