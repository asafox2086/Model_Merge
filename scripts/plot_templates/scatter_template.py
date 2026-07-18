import matplotlib.pyplot as plt
import numpy as np

from style import darken_color, polish_axes, save_png_pdf, setup_style


MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]


def plot_embedding(
    coordinates,
    labels,
    class_names=None,
    title="",
    xlabel="Component 1",
    ylabel="Component 2",
    out_base="embedding_scatter",
):
    coordinates = np.asarray(coordinates)
    labels = np.asarray(labels)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape (n_samples, 2)")
    if len(labels) != len(coordinates):
        raise ValueError("labels and coordinates must have the same length")
    if not np.isfinite(coordinates).all():
        raise ValueError("coordinates must be finite")

    setup_style("scatter")
    fig, ax = plt.subplots(figsize=(7.0, 5.6), dpi=300)
    classes = np.unique(labels)
    if class_names is not None and len(class_names) != len(classes):
        raise ValueError("class_names must contain one name per unique label")
    for index, value in enumerate(classes):
        mask = labels == value
        name = class_names[index] if class_names is not None else str(value)
        color = f"C{index % 10}"
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            label=name,
            marker=MARKERS[index % len(MARKERS)],
            s=42,
            alpha=0.78,
            color=color,
            edgecolor=darken_color(color, 0.65),
            linewidth=0.6,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=10, fontweight="bold")
    ax.legend(loc="best", frameon=False)
    polish_axes(ax, y_grid=False, x_grid=False)
    fig.tight_layout()
    save_png_pdf(fig, out_base)
    plt.close(fig)


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    coordinates = np.vstack([rng.normal((0, 0), 0.7, (40, 2)), rng.normal((3, 2), 0.7, (40, 2))])
    labels = np.repeat([0, 1], 40)
    plot_embedding(coordinates, labels, class_names=["Class A", "Class B"], out_base="scatter_demo")
