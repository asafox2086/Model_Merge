import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


METHOD_COLORS = {
    "Method A": "#4F81BD",
    "Method B": "#9BBB59",
    "Method C": "#C0504D",
    "Baseline": "#7F7F7F",
}

ABLATION_COLORS = ["#F4CCCC", "#DAE8FC", "#FFF2CC", "#1F77B4"]


def setup_style(kind="paper"):
    try:
        import scienceplots  # noqa: F401

        if kind in {"paper", "bar", "line", "scatter", "dashboard", "heatmap"}:
            plt.style.use(["science", "ieee", "no-latex"])
        else:
            plt.style.use(["science", "no-latex"])
    except Exception:
        plt.style.use("default")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "axes.linewidth": 1.6,
            "grid.linewidth": 0.9,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "lines.linewidth": 2.6,
            "lines.markersize": 8,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )


def darken_color(color, factor=0.7):
    return tuple(max(0, channel * factor) for channel in mcolors.to_rgb(color))


def polish_axes(ax, y_grid=True, x_grid=False):
    if y_grid:
        ax.yaxis.grid(True, linestyle="--", color="grey", alpha=0.45, zorder=0)
    else:
        ax.yaxis.grid(False)
    if x_grid:
        ax.xaxis.grid(True, linestyle="--", color="grey", alpha=0.25, zorder=0)
    else:
        ax.xaxis.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.6)


def save_png(fig, out_base, dpi=300):
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    fig.savefig(out_base + ".png", dpi=dpi, bbox_inches="tight")
