import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


METHOD_COLORS = {
    "resnet": "#4F81BD",
    "convnext": "#9BBB59",
    "vit_t": "#C0504D",
    "swin_tiny": "#8064A2",
}


def setup_style(kind="line"):
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "ieee", "no-latex"])
    except Exception:
        plt.style.use("default")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.titlesize": 22,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "axes.linewidth": 1.6,
            "grid.linewidth": 0.9,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "lines.linewidth": 2.8,
            "lines.markersize": 9,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )


def darken_color(color, factor=0.7):
    return tuple(max(0, channel * factor) for channel in mcolors.to_rgb(color))


def polish_axes(axis, y_grid=True, x_grid=False):
    if y_grid:
        axis.yaxis.grid(True, linestyle="--", color="grey", alpha=0.45, zorder=0)
    else:
        axis.yaxis.grid(False)
    if x_grid:
        axis.xaxis.grid(True, linestyle="--", color="grey", alpha=0.25, zorder=0)
    else:
        axis.xaxis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.6)


def save_png_pdf(figure, out_base, dpi=300):
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    figure.savefig(out_base + ".png", dpi=dpi, bbox_inches="tight")
    figure.savefig(out_base + ".pdf", bbox_inches="tight")
