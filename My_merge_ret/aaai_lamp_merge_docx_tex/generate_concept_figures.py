from pathlib import Path
import math
import subprocess


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"


def esc(text):
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class EPS:
    def __init__(self, path, width=1200, height=620):
        self.path = Path(path)
        self.width = width
        self.height = height
        self.buf = [
            "%!PS-Adobe-3.0 EPSF-3.0",
            f"%%BoundingBox: 0 0 {width} {height}",
            "%%LanguageLevel: 2",
            "%%Pages: 1",
            "%%EndComments",
            "/Helvetica findfont 10 scalefont setfont",
            "1 setlinejoin 1 setlinecap",
        ]

    def color(self, rgb):
        r, g, b = rgb
        self.buf.append(f"{r:.3f} {g:.3f} {b:.3f} setrgbcolor")

    def lw(self, width):
        self.buf.append(f"{width:.2f} setlinewidth")

    def rect(self, x, y, w, h, fill, stroke=(0.12, 0.16, 0.22), lw=1.2):
        self.color(fill)
        self.buf.append(f"newpath {x} {y} moveto {w} 0 rlineto 0 {h} rlineto {-w} 0 rlineto closepath fill")
        self.color(stroke)
        self.lw(lw)
        self.buf.append(f"newpath {x} {y} moveto {w} 0 rlineto 0 {h} rlineto {-w} 0 rlineto closepath stroke")

    def round_rect(self, x, y, w, h, r, fill, stroke=(0.12, 0.16, 0.22), lw=1.2):
        # PostScript rounded rectangle.
        self.color(fill)
        self.buf.append(
            f"newpath {x+r} {y} moveto {x+w-r} {y} lineto "
            f"{x+w-r} {y} {x+w} {y} {x+w} {y+r} curveto "
            f"{x+w} {y+h-r} lineto {x+w} {y+h-r} {x+w} {y+h} {x+w-r} {y+h} curveto "
            f"{x+r} {y+h} lineto {x+r} {y+h} {x} {y+h} {x} {y+h-r} curveto "
            f"{x} {y+r} lineto {x} {y+r} {x} {y} {x+r} {y} curveto closepath fill"
        )
        self.color(stroke)
        self.lw(lw)
        self.buf.append(
            f"newpath {x+r} {y} moveto {x+w-r} {y} lineto "
            f"{x+w-r} {y} {x+w} {y} {x+w} {y+r} curveto "
            f"{x+w} {y+h-r} lineto {x+w} {y+h-r} {x+w} {y+h} {x+w-r} {y+h} curveto "
            f"{x+r} {y+h} lineto {x+r} {y+h} {x} {y+h} {x} {y+h-r} curveto "
            f"{x} {y+r} lineto {x} {y+r} {x} {y} {x+r} {y} curveto closepath stroke"
        )

    def text(self, x, y, text, size=16, color=(0.08, 0.10, 0.14), font="Helvetica"):
        self.color(color)
        self.buf.append(f"/{font} findfont {size} scalefont setfont")
        self.buf.append(f"{x} {y} moveto ({esc(text)}) show")

    def text_center(self, x, y, text, size=16, color=(0.08, 0.10, 0.14), font="Helvetica"):
        self.color(color)
        self.buf.append(f"/{font} findfont {size} scalefont setfont")
        self.buf.append(f"({esc(text)}) dup stringwidth pop 2 div neg {x} add {y} moveto show")

    def multiline(self, x, y, lines, size=14, leading=18, color=(0.08, 0.10, 0.14), font="Helvetica"):
        for idx, line in enumerate(lines):
            self.text(x, y - idx * leading, line, size=size, color=color, font=font)

    def line(self, x1, y1, x2, y2, color=(0.14, 0.18, 0.24), lw=1.2, dash=None):
        self.color(color)
        self.lw(lw)
        if dash:
            self.buf.append(f"[{dash}] 0 setdash")
        self.buf.append(f"newpath {x1} {y1} moveto {x2} {y2} lineto stroke")
        if dash:
            self.buf.append("[] 0 setdash")

    def arrow(self, x1, y1, x2, y2, color=(0.14, 0.18, 0.24), lw=2.0):
        self.line(x1, y1, x2, y2, color=color, lw=lw)
        ang = math.atan2(y2 - y1, x2 - x1)
        head = 13
        spread = 0.42
        p1 = (x2 - head * math.cos(ang - spread), y2 - head * math.sin(ang - spread))
        p2 = (x2 - head * math.cos(ang + spread), y2 - head * math.sin(ang + spread))
        self.color(color)
        self.buf.append(
            f"newpath {x2} {y2} moveto {p1[0]:.2f} {p1[1]:.2f} lineto {p2[0]:.2f} {p2[1]:.2f} lineto closepath fill"
        )

    def small_bars(self, x, y, vals, colors, w=16, h=64, gap=6, label=None):
        if label:
            self.text(x, y + h + 12, label, size=11, color=(0.25, 0.29, 0.36))
        self.line(x, y, x + len(vals) * (w + gap) - gap, y, color=(0.55, 0.59, 0.64), lw=0.8)
        for idx, val in enumerate(vals):
            bh = h * val
            self.rect(x + idx * (w + gap), y, w, bh, fill=colors[idx], stroke=colors[idx], lw=0.4)

    def circle(self, x, y, r, fill, stroke=(0.12, 0.16, 0.22), lw=1.0):
        self.color(fill)
        self.buf.append(f"newpath {x} {y} {r} 0 360 arc fill")
        self.color(stroke)
        self.lw(lw)
        self.buf.append(f"newpath {x} {y} {r} 0 360 arc stroke")

    def save(self):
        self.buf += ["showpage", "%%EOF"]
        self.path.write_text("\n".join(self.buf) + "\n")


PALETTE = {
    "blue": (0.18, 0.42, 0.73),
    "teal": (0.10, 0.55, 0.54),
    "green": (0.29, 0.58, 0.31),
    "orange": (0.86, 0.43, 0.14),
    "red": (0.78, 0.22, 0.22),
    "purple": (0.45, 0.32, 0.75),
    "gray": (0.42, 0.46, 0.53),
    "light_blue": (0.88, 0.94, 1.00),
    "light_teal": (0.86, 0.96, 0.94),
    "light_red": (1.00, 0.91, 0.90),
    "light_orange": (1.00, 0.94, 0.86),
    "light_green": (0.91, 0.97, 0.90),
    "ink": (0.08, 0.10, 0.14),
}


def draw_problem_figure():
    eps = EPS(FIG_DIR / "concept_predictive_collapse.eps")
    eps.rect(0, 0, eps.width, eps.height, fill=(1, 1, 1), stroke=(1, 1, 1), lw=0)
    eps.text(40, 575, "Why class-agnostic post-hoc merging collapses in multi-center medical imaging", size=25, font="Helvetica-Bold")
    eps.text(42, 548, "Local classifiers retain partial diagnostic ability, but parameter-only fusion ignores which client has evidence for each class.", size=14, color=(0.25, 0.29, 0.36))

    # Column headers.
    eps.text_center(210, 512, "Independent hospitals", size=17, font="Helvetica-Bold")
    eps.text_center(600, 512, "Generic post-hoc merge", size=17, font="Helvetica-Bold")
    eps.text_center(990, 512, "Collapsed global predictor", size=17, font="Helvetica-Bold")

    class_cols = [PALETTE["blue"], PALETTE["teal"], PALETTE["orange"], PALETTE["red"], PALETTE["purple"]]
    clients = [
        ("Client 1", "local labels: A, B, C", [0.39, 0.33, 0.21, 0.05, 0.02], "rho=0.39"),
        ("Client 2", "local labels: B, D", [0.07, 0.44, 0.06, 0.37, 0.06], "rho=0.44"),
        ("Client 3", "local labels: C, E", [0.04, 0.10, 0.36, 0.08, 0.42], "rho=0.42"),
    ]
    ys = [390, 255, 120]
    for (name, labels, vals, rho), y in zip(clients, ys):
        eps.round_rect(55, y, 310, 98, 12, fill=(0.97, 0.99, 1.00), stroke=(0.70, 0.78, 0.88), lw=1.2)
        eps.text(75, y + 70, name, size=15, font="Helvetica-Bold")
        eps.text(75, y + 49, labels, size=12, color=(0.26, 0.30, 0.36))
        eps.text(75, y + 30, "non-degenerate local predictions", size=12, color=PALETTE["green"])
        eps.text(75, y + 12, rho, size=12, color=(0.22, 0.26, 0.32), font="Helvetica-Bold")
        eps.small_bars(245, y + 18, vals, class_cols, w=14, h=54, gap=4)

    # Merge module.
    eps.round_rect(440, 262, 320, 160, 16, fill=PALETTE["light_orange"], stroke=(0.80, 0.55, 0.20), lw=1.5)
    eps.text_center(600, 390, "parameter-only fusion", size=18, font="Helvetica-Bold", color=(0.46, 0.25, 0.07))
    eps.multiline(
        467,
        357,
        [
            "operates on whole checkpoints",
            "theta_1, ..., theta_K",
            "no class evidence variable",
            "n_i,c = 0 is not masked",
        ],
        size=13,
        leading=21,
        color=(0.27, 0.22, 0.16),
    )
    eps.round_rect(463, 182, 274, 54, 10, fill=(1.0, 0.98, 0.90), stroke=(0.83, 0.65, 0.32), lw=1.0)
    eps.text_center(600, 214, "missing-class and majority directions", size=13, font="Helvetica-Bold", color=(0.44, 0.28, 0.04))
    eps.text_center(600, 196, "are mixed in the same parameter space", size=12, color=(0.44, 0.28, 0.04))

    eps.arrow(365, 438, 440, 368, color=(0.35, 0.39, 0.46), lw=2.0)
    eps.arrow(365, 304, 440, 342, color=(0.35, 0.39, 0.46), lw=2.0)
    eps.arrow(365, 168, 440, 318, color=(0.35, 0.39, 0.46), lw=2.0)
    eps.arrow(760, 342, 860, 342, color=(0.35, 0.39, 0.46), lw=2.2)

    # Collapse result.
    eps.round_rect(860, 235, 285, 230, 16, fill=PALETTE["light_red"], stroke=(0.78, 0.25, 0.24), lw=1.5)
    eps.text_center(1002, 432, "merged predictor", size=18, font="Helvetica-Bold", color=(0.50, 0.12, 0.10))
    eps.small_bars(922, 312, [0.03, 0.04, 0.05, 0.92, 0.03], class_cols, w=32, h=88, gap=10, label="predicted class distribution q(c)")
    eps.text_center(1002, 282, "rho = max_c q(c) -> 1", size=15, font="Helvetica-Bold", color=(0.50, 0.12, 0.10))
    eps.text_center(1002, 258, "single-class or few-class prediction", size=12, color=(0.38, 0.18, 0.17))

    # Cause strip.
    eps.round_rect(85, 36, 1030, 54, 12, fill=(0.95, 0.97, 0.99), stroke=(0.70, 0.75, 0.82), lw=1.1)
    eps.text(110, 67, "Cause:", size=15, font="Helvetica-Bold", color=(0.16, 0.20, 0.28))
    eps.text(168, 67, "class availability is client-dependent, but generic fusion is class-agnostic.", size=14)
    eps.text(168, 47, "Majority and observed-class directions survive; rare or locally missing diagnostic directions are absorbed.", size=14)

    eps.save()


def draw_method_figure():
    eps = EPS(FIG_DIR / "concept_lamp_merge_pipeline.eps", width=1200, height=660)
    eps.rect(0, 0, eps.width, eps.height, fill=(1, 1, 1), stroke=(1, 1, 1), lw=0)
    eps.text(40, 615, "LAMP-Merge: one-shot class-level medical model merging", size=26, font="Helvetica-Bold")
    eps.text(42, 588, "Clients compute diagnostic evidence locally; the server reconstructs a prototype head and calibrates it with bounded prevalence priors.", size=14, color=(0.25, 0.29, 0.36))

    # Privacy boundary background.
    eps.round_rect(35, 70, 360, 480, 18, fill=(0.97, 0.99, 1.00), stroke=(0.70, 0.78, 0.88), lw=1.2)
    eps.text_center(215, 520, "Client side", size=18, font="Helvetica-Bold")
    eps.text_center(215, 500, "raw images stay local", size=12, color=PALETTE["red"], font="Helvetica-Bold")

    eps.round_rect(805, 70, 360, 480, 18, fill=(0.98, 0.99, 0.97), stroke=(0.74, 0.84, 0.70), lw=1.2)
    eps.text_center(985, 520, "Server side", size=18, font="Helvetica-Bold")
    eps.text_center(985, 500, "single post-hoc construction", size=12, color=PALETTE["green"], font="Helvetica-Bold")

    # Clients.
    for idx, y in enumerate([405, 285, 165], start=1):
        eps.round_rect(70, y, 285, 84, 12, fill=(1.00, 1.00, 1.00), stroke=(0.74, 0.80, 0.88), lw=1.0)
        eps.text(90, y + 58, f"Client {idx}: local training", size=14, font="Helvetica-Bold")
        eps.text(90, y + 38, "private data D_i, class subset", size=11, color=(0.30, 0.34, 0.40))
        eps.text(90, y + 20, "fixed reference z = phi_0(T(x))", size=11, color=(0.30, 0.34, 0.40))
        eps.small_bars(270, y + 18, [0.55, 0.23, 0.06, 0.12, 0.04] if idx == 1 else ([0.04, 0.48, 0.05, 0.38, 0.05] if idx == 2 else [0.03, 0.06, 0.42, 0.07, 0.44]), [PALETTE["blue"], PALETTE["teal"], PALETTE["orange"], PALETTE["red"], PALETTE["purple"]], w=10, h=42, gap=3)

    # Upload packet.
    eps.round_rect(440, 255, 310, 190, 16, fill=(0.94, 0.97, 1.00), stroke=PALETTE["blue"], lw=1.4)
    eps.text_center(595, 414, "one upload after local training", size=17, font="Helvetica-Bold", color=(0.12, 0.28, 0.52))
    eps.multiline(
        465,
        383,
        [
            "checkpoint: theta_i",
            "class prototype: mu_i,c",
            "support count: n_i,c",
            "prevalence count: m_i,c",
        ],
        size=13,
        leading=22,
        color=(0.12, 0.18, 0.28),
    )
    eps.round_rect(465, 266, 260, 47, 9, fill=(1.00, 0.96, 0.96), stroke=(0.82, 0.45, 0.45), lw=1.0)
    eps.text_center(595, 294, "not uploaded: raw images, logits,", size=12, color=(0.55, 0.10, 0.10), font="Helvetica-Bold")
    eps.text_center(595, 278, "per-sample features or validation data", size=12, color=(0.55, 0.10, 0.10), font="Helvetica-Bold")

    eps.arrow(355, 445, 440, 395, color=(0.28, 0.34, 0.44), lw=2.0)
    eps.arrow(355, 327, 440, 350, color=(0.28, 0.34, 0.44), lw=2.0)
    eps.arrow(355, 207, 440, 305, color=(0.28, 0.34, 0.44), lw=2.0)
    eps.arrow(750, 350, 805, 350, color=(0.28, 0.34, 0.44), lw=2.0)

    # Server M1/M2/output.
    eps.round_rect(835, 390, 300, 88, 12, fill=PALETTE["light_teal"], stroke=PALETTE["teal"], lw=1.3)
    eps.text(858, 453, "M1: prototype reconstruction", size=15, font="Helvetica-Bold", color=(0.05, 0.34, 0.34))
    eps.text(858, 431, "e_i,c = (n_i,c+1)^gamma 1[n_i,c>0]", size=11)
    eps.text(858, 414, "p_c = sum_i alpha_i,c mu_i,c", size=11)
    eps.text(858, 397, "w_c = s p_c / ||p_c||_2", size=11)

    eps.round_rect(835, 255, 300, 90, 12, fill=PALETTE["light_orange"], stroke=PALETTE["orange"], lw=1.3)
    eps.text(858, 320, "M2: long-tail calibration", size=15, font="Helvetica-Bold", color=(0.55, 0.25, 0.06))
    eps.text(858, 298, "pi_c = sum_i m_i,c / sum_k sum_i m_i,k", size=11)
    eps.text(858, 281, "activate if r = C max_c pi_c > tau", size=11)
    eps.text(858, 264, "b_c = lambda(centered log pi_c)", size=11)

    eps.arrow(985, 390, 985, 345, color=(0.28, 0.34, 0.44), lw=2.0)

    eps.round_rect(835, 112, 300, 93, 12, fill=PALETTE["light_green"], stroke=PALETTE["green"], lw=1.3)
    eps.text(858, 178, "merged diagnostic model", size=15, font="Helvetica-Bold", color=(0.15, 0.38, 0.16))
    eps.text(858, 154, "score_c(x) = w_c^T phi_0(T(x)) + b_c", size=12)
    eps.text(858, 134, "class-wise evidence is preserved;", size=11, color=(0.25, 0.31, 0.25))
    eps.text(858, 118, "long-tail prevalence is boundedly calibrated", size=11, color=(0.25, 0.31, 0.25))

    eps.arrow(985, 255, 985, 205, color=(0.28, 0.34, 0.44), lw=2.0)

    # Bottom summary.
    eps.round_rect(435, 80, 320, 108, 14, fill=(0.96, 0.96, 0.99), stroke=(0.62, 0.61, 0.78), lw=1.1)
    eps.text_center(595, 160, "formal design principle", size=15, font="Helvetica-Bold", color=(0.28, 0.25, 0.49))
    eps.text(458, 134, "merge class evidence, not only parameters", size=12)
    eps.text(458, 114, "mask absent classes through n_i,c", size=12)
    eps.text(458, 94, "retain prevalence through m_i,c", size=12)

    eps.save()


def convert_eps_to_pdf(stem):
    eps_path = FIG_DIR / f"{stem}.eps"
    pdf_path = FIG_DIR / f"{stem}.pdf"
    subprocess.run(
        [
            "gs",
            "-dBATCH",
            "-dNOPAUSE",
            "-dSAFER",
            "-sDEVICE=pdfwrite",
            "-dEPSCrop",
            f"-sOutputFile={pdf_path}",
            str(eps_path),
        ],
        check=True,
    )
    eps_path.unlink()


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    draw_problem_figure()
    draw_method_figure()
    convert_eps_to_pdf("concept_predictive_collapse")
    convert_eps_to_pdf("concept_lamp_merge_pipeline")


if __name__ == "__main__":
    main()
