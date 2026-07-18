import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from style import save_png_pdf, setup_style


def plot_heatmap(
    matrix,
    x_labels,
    y_labels,
    title="",
    xlabel="Column",
    ylabel="Row",
    cmap="mako",
    center=None,
    out_base="heatmap",
):
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if matrix.shape != (len(y_labels), len(x_labels)):
        raise ValueError("label counts must match matrix rows and columns")
    if not np.isfinite(matrix).all():
        raise ValueError("matrix values must be finite")

    setup_style("heatmap")
    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=300)
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        center=center,
        cbar=True,
        xticklabels=x_labels,
        yticklabels=y_labels,
        linewidths=0.35,
        linecolor="white",
    )
    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    save_png_pdf(fig, out_base)
    plt.close(fig)


if __name__ == "__main__":
    matrix = [[0.82, 0.77, 0.65], [0.79, 0.81, 0.62], [0.58, 0.61, 0.76]]
    plot_heatmap(
        matrix,
        x_labels=["Class A", "Class B", "Class C"],
        y_labels=["Client 1", "Client 2", "Client 3"],
        title="Client-by-Class Similarity",
        out_base="heatmap_demo",
    )
