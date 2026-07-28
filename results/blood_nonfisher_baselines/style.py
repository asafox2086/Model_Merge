import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


def setup_style():
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "ieee", "no-latex"])
    except Exception:
        plt.style.use("default")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 18,
            "axes.titlesize": 24,
            "axes.labelsize": 22,
            "xtick.labelsize": 20,
            "ytick.labelsize": 18,
            "axes.linewidth": 1.6,
            "grid.linewidth": 0.9,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )


def darken_color(color, factor=0.7):
    return tuple(max(0, channel * factor) for channel in mcolors.to_rgb(color))


def polish_axes(axis):
    axis.yaxis.grid(True, linestyle="--", color="grey", alpha=0.45, zorder=0)
    axis.xaxis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.6)


def save_png_pdf(figure, out_base, dpi=300):
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    figure.savefig(out_base + ".png", dpi=dpi, bbox_inches="tight")
    figure.savefig(out_base + ".pdf", bbox_inches="tight")
