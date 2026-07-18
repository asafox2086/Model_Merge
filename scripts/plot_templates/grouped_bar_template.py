import matplotlib.pyplot as plt
import numpy as np

from style import darken_color, polish_axes, save_png_pdf, setup_style


def plot_grouped_bar(
    categories,
    series,
    title="",
    xlabel="Dataset",
    ylabel="Score",
    out_base="grouped_bar",
):
    if not categories or not series:
        raise ValueError("categories and series must not be empty")
    for item in series:
        values = np.asarray(item["values"])
        if values.shape != (len(categories),) or not np.isfinite(values).all():
            raise ValueError("every series must contain one finite value per category")
        errors = item.get("errors")
        if errors is not None and np.asarray(errors).shape != values.shape:
            raise ValueError("errors must match the series values")

    setup_style("bar")
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=300)
    x = np.arange(len(categories))
    width = 0.78 / max(1, len(series))

    for index, item in enumerate(series):
        offset = (index - (len(series) - 1) / 2) * width
        color = item.get("color", f"C{index}")
        ax.bar(
            x + offset,
            item["values"],
            width=width,
            label=item["label"],
            yerr=item.get("errors"),
            capsize=3 if item.get("errors") is not None else 0,
            color=color,
            edgecolor=darken_color(color, 0.65),
            linewidth=1.4,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=10, fontweight="bold")
    ax.legend(loc="best", frameon=False)
    polish_axes(ax, y_grid=True, x_grid=False)
    fig.tight_layout()
    save_png_pdf(fig, out_base)
    plt.close(fig)


if __name__ == "__main__":
    plot_grouped_bar(
        ["Dataset 1", "Dataset 2", "Dataset 3"],
        [
            {"label": "Method A", "values": [71, 75, 78], "color": "#4F81BD"},
            {"label": "Method B", "values": [74, 77, 81], "color": "#C0504D"},
        ],
        title="Method Comparison",
        out_base="grouped_bar_demo",
    )
