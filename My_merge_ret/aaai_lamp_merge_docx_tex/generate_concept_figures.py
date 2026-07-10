from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FONT_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def text(draw, xy, content, size=28, fill=(20, 24, 33), bold=False, anchor=None):
    draw.text(xy, content, font=font(size, bold=bold), fill=fill, anchor=anchor)


def multiline(draw, xy, lines, size=24, fill=(20, 24, 33), bold=False, leading=34):
    x, y = xy
    for i, line in enumerate(lines):
        text(draw, (x, y + i * leading), line, size=size, fill=fill, bold=bold)


def rounded(draw, xy, radius, fill, outline, width=3):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw, start, end, fill=(75, 86, 105), width=6):
    draw.line([start, end], fill=fill, width=width)
    x1, y1 = start
    x2, y2 = end
    angle = math.atan2(y2 - y1, x2 - x1)
    head = 24
    spread = 0.42
    p1 = (x2 - head * math.cos(angle - spread), y2 - head * math.sin(angle - spread))
    p2 = (x2 - head * math.cos(angle + spread), y2 - head * math.sin(angle + spread))
    draw.polygon([end, p1, p2], fill=fill)


def bars(draw, x, y, vals, colors, bar_w=28, height=82, gap=10):
    base = y + height
    draw.line([(x, base), (x + len(vals) * (bar_w + gap) - gap, base)], fill=(120, 130, 145), width=2)
    for i, val in enumerate(vals):
        h = int(height * val)
        x0 = x + i * (bar_w + gap)
        draw.rectangle([x0, base - h, x0 + bar_w, base], fill=colors[i])


BLUE = (44, 103, 178)
TEAL = (29, 143, 139)
ORANGE = (226, 108, 35)
RED = (202, 50, 52)
PURPLE = (112, 80, 190)
GREEN = (67, 150, 74)
INK = (19, 24, 34)
GRAY = (73, 83, 99)
CLASS_COLORS = [BLUE, TEAL, ORANGE, RED, PURPLE]


def draw_problem():
    img = Image.new("RGB", (2400, 1240), "white")
    d = ImageDraw.Draw(img)

    text(d, (80, 70), "为什么类别无关的后置模型融合会在多中心医学图像中坍缩", 48, INK, True)
    text(d, (84, 128), "本地模型仍具有局部诊断能力，但通用融合不知道哪个客户端在哪个诊断类别上有可靠证据。", 30, (70, 78, 94))

    text(d, (420, 215), "独立医院客户端", 36, INK, True, anchor="mm")
    text(d, (1200, 215), "通用后置融合", 36, INK, True, anchor="mm")
    text(d, (1980, 215), "坍缩的全局预测器", 36, INK, True, anchor="mm")

    clients = [
        ("客户端 1", "本地可见类别：甲、乙、丙", [0.39, 0.33, 0.21, 0.05, 0.02], "坍缩强度 0.39"),
        ("客户端 2", "本地可见类别：乙、丁", [0.07, 0.44, 0.06, 0.37, 0.06], "坍缩强度 0.44"),
        ("客户端 3", "本地可见类别：丙、戊", [0.04, 0.10, 0.36, 0.08, 0.42], "坍缩强度 0.42"),
    ]
    ys = [290, 560, 830]
    for (name, visible, vals, rho), y in zip(clients, ys):
        rounded(d, (110, y, 740, y + 170), 22, (247, 252, 255), (177, 198, 225), 3)
        text(d, (150, y + 34), name, 30, INK, True)
        text(d, (150, y + 78), visible, 24, (48, 56, 70))
        text(d, (150, y + 116), "预测分布未退化为单类", 24, GREEN)
        text(d, (150, y + 150), rho, 24, (48, 56, 70), True)
        bars(d, 500, y + 55, vals, CLASS_COLORS, bar_w=30, height=72, gap=8)

    rounded(d, (900, 420, 1530, 710), 30, (255, 243, 224), (206, 132, 39), 4)
    text(d, (1215, 485), "参数级整体融合", 36, (126, 64, 10), True, anchor="mm")
    multiline(
        d,
        (955, 545),
        [
            "只处理完整模型参数",
            "融合变量仅为各客户端参数",
            "没有类别证据变量",
            "未见类别不会被显式屏蔽",
        ],
        26,
        (47, 39, 29),
        leading=40,
    )
    rounded(d, (940, 765, 1490, 870), 18, (255, 249, 232), (214, 160, 65), 3)
    text(d, (1215, 805), "缺失类方向与多数类方向", 26, (110, 69, 6), True, anchor="mm")
    text(d, (1215, 843), "在同一参数空间中被混合", 24, (110, 69, 6), anchor="mm")

    arrow(d, (740, 375), (900, 520))
    arrow(d, (740, 645), (900, 575))
    arrow(d, (740, 915), (900, 630))
    arrow(d, (1530, 565), (1720, 565))

    rounded(d, (1720, 330, 2320, 785), 30, (255, 231, 229), (205, 65, 63), 5)
    text(d, (2020, 405), "融合后模型", 36, (130, 30, 25), True, anchor="mm")
    text(d, (2020, 455), "预测类别分布", 24, (70, 55, 70), anchor="mm")
    bars(d, 1860, 520, [0.03, 0.04, 0.05, 0.92, 0.03], CLASS_COLORS, bar_w=62, height=120, gap=17)
    text(d, (2020, 700), "坍缩强度接近一", 32, (130, 30, 25), True, anchor="mm")
    text(d, (2020, 748), "退化为单类或少数类预测器", 25, (110, 54, 50), anchor="mm")

    rounded(d, (170, 1060, 2230, 1168), 22, (242, 247, 252), (181, 194, 211), 3)
    text(d, (220, 1110), "原因：", 30, (38, 46, 62), True)
    text(d, (310, 1110), "类别可及性依赖客户端，但通用融合是类别无关的。", 29, INK)
    text(d, (310, 1150), "多数类方向更容易保留；稀有类或局部缺失类的诊断方向被吸收。", 29, INK)

    img.save(FIG_DIR / "concept_predictive_collapse.png", dpi=(220, 220))


def draw_method():
    img = Image.new("RGB", (2400, 1320), "white")
    d = ImageDraw.Draw(img)

    text(d, (80, 70), "长尾自适应医学原型融合", 54, INK, True)
    text(d, (84, 132), "客户端本地计算诊断证据；服务端重建原型分类头，并用有界患病率先验进行长尾校准。", 30, (70, 78, 94))

    rounded(d, (70, 235, 820, 1215), 30, (247, 252, 255), (177, 198, 225), 3)
    text(d, (445, 295), "客户端侧", 38, INK, True, anchor="mm")
    text(d, (445, 340), "原始图像仅留在本地", 26, RED, True, anchor="mm")

    rounded(d, (1630, 235, 2330, 1215), 30, (248, 252, 246), (185, 214, 180), 3)
    text(d, (1980, 295), "服务端侧", 38, INK, True, anchor="mm")
    text(d, (1980, 340), "一次性事后构造", 26, GREEN, True, anchor="mm")

    client_vals = [
        [0.55, 0.23, 0.06, 0.12, 0.04],
        [0.04, 0.48, 0.05, 0.38, 0.05],
        [0.03, 0.06, 0.42, 0.07, 0.44],
    ]
    for i, y in enumerate([405, 650, 895], start=1):
        rounded(d, (145, y, 735, y + 168), 18, (255, 255, 255), (187, 201, 224), 3)
        text(d, (185, y + 36), f"客户端 {i}：本地训练", 30, INK, True)
        text(d, (185, y + 80), "私有图像数据，类别覆盖不完整", 24, (65, 72, 88))
        text(d, (185, y + 120), "用共享参考骨干提取类别特征", 24, (65, 72, 88))
        bars(d, 555, y + 45, client_vals[i - 1], CLASS_COLORS, bar_w=24, height=70, gap=8)

    rounded(d, (900, 470, 1540, 835), 30, (238, 247, 255), BLUE, 5)
    text(d, (1220, 540), "本地训练结束后一次上传", 36, (32, 78, 140), True, anchor="mm")
    multiline(
        d,
        (960, 590),
        [
            "模型检查点",
            "类别原型",
            "类别支持数",
            "患病率计数",
        ],
        28,
        (30, 42, 62),
        leading=36,
    )
    rounded(d, (955, 735, 1485, 815), 18, (255, 246, 246), (220, 116, 116), 4)
    text(d, (1220, 766), "不上传：原始图像、逐样本输出、", 24, (145, 25, 25), True, anchor="mm")
    text(d, (1220, 796), "逐样本特征或验证集数据", 24, (145, 25, 25), True, anchor="mm")

    arrow(d, (735, 485), (900, 585))
    arrow(d, (735, 735), (900, 655))
    arrow(d, (735, 980), (900, 725))
    arrow(d, (1540, 655), (1630, 655))

    rounded(d, (1695, 385, 2265, 600), 20, (218, 246, 241), TEAL, 4)
    text(d, (1745, 430), "模块一：诊断原型重建", 30, (9, 91, 88), True)
    text(d, (1745, 478), "按类别支持数评估证据可靠性", 24, INK)
    text(d, (1745, 515), "同一诊断类别的原型加权汇聚", 24, INK)
    text(d, (1745, 548), "归一化后形成全局诊断分类头", 24, INK)

    rounded(d, (1695, 690, 2265, 900), 20, (255, 240, 220), ORANGE, 4)
    text(d, (1745, 735), "模块二：长尾患病率校准", 30, (135, 62, 9), True)
    text(d, (1745, 782), "由客户端计数估计全局患病率", 23, INK)
    text(d, (1745, 818), "主导类别过强时激活校准", 23, INK)
    text(d, (1745, 854), "对分类偏置做有界长尾修正", 23, INK)

    rounded(d, (1695, 1000, 2265, 1190), 20, (231, 248, 229), GREEN, 4)
    text(d, (1745, 1045), "融合后的全局诊断模型", 30, (35, 102, 42), True)
    text(d, (1745, 1095), "类别得分 = 原型相似度 + 长尾偏置", 24, INK)
    text(d, (1745, 1135), "保留类别级诊断证据", 24, (53, 74, 55))
    text(d, (1745, 1170), "对真实医学长尾先验做有界校准", 24, (53, 74, 55))

    arrow(d, (1980, 600), (1980, 690))
    arrow(d, (1980, 900), (1980, 1000))

    rounded(d, (890, 965, 1545, 1155), 22, (246, 246, 255), (160, 156, 205), 3)
    text(d, (1218, 1015), "形式化设计原则", 30, (71, 63, 128), True, anchor="mm")
    text(d, (945, 1072), "融合类别证据，而不是只融合参数", 26, INK)
    text(d, (945, 1110), "用类别支持数屏蔽缺失类别并加权可靠证据", 26, INK)
    text(d, (945, 1148), "用患病率计数保留医学长尾先验信息", 26, INK)

    img.save(FIG_DIR / "concept_lamp_merge_pipeline.png", dpi=(220, 220))


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    draw_problem()
    draw_method()


if __name__ == "__main__":
    main()
