import matplotlib.pyplot as plt
import numpy as np

from style import METHOD_COLORS, darken_color, polish_axes, save_png_pdf, setup_style


LINE_STYLES = [("o", "-"), ("D", "--"), ("^", "-."), ("s", ":")]


def plot_line_chart(
    x_values,
    series,
    title="",
    xlabel="Communication Rounds",
    ylabel="Score",
    out_base="line_chart",
):
    x_values = np.asarray(x_values)
    if x_values.ndim != 1 or not np.isfinite(x_values).all():
        raise ValueError("x_values must be a finite one-dimensional sequence")
    for item in series:
        values = np.asarray(item["values"])
        if values.shape != x_values.shape or not np.isfinite(values).all():
            raise ValueError("every series must contain one finite value per x position")
        lower, upper = item.get("lower"), item.get("upper")
        if (lower is None) != (upper is None):
            raise ValueError("uncertainty requires both lower and upper bounds")
        if lower is not None:
            lower, upper = np.asarray(lower), np.asarray(upper)
            if lower.shape != x_values.shape or upper.shape != x_values.shape:
                raise ValueError("uncertainty bounds must match x_values")

    setup_style("line")
    fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=300)

    for index, item in enumerate(series):
        label = item["label"]
        marker, linestyle = LINE_STYLES[index % len(LINE_STYLES)]
        color = item.get("color", METHOD_COLORS.get(label, f"C{index}"))
        ax.plot(
            x_values,
            item["values"],
            label=label,
            color=color,
            linestyle=item.get("linestyle", linestyle),
            marker=item.get("marker", marker),
            markerfacecolor=color,
            markeredgecolor=darken_color(color, 0.65),
            markeredgewidth=1.8,
            linewidth=item.get("linewidth", 2.6),
            zorder=3,
        )

        lower = item.get("lower")
        upper = item.get("upper")
        if lower is not None and upper is not None:
            ax.fill_between(x_values, lower, upper, color=color, alpha=0.16, linewidth=0)

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
    rounds = [0, 10, 20, 30, 40, 50]
    series = [
        {"label": "Method A", "values": [22, 27, 31, 34, 36, 37]},
        {"label": "Method B", "values": [24, 30, 35, 39, 41, 43]},
        {"label": "Method C", "values": [25, 33, 39, 43, 46, 48]},
    ]
    plot_line_chart(rounds, series, title="Metric Evolution", out_base="line_chart_demo")
