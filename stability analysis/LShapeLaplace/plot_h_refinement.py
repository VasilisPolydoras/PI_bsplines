"""Plot the newest L-shaped singular-solution h-refinement experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUT_PATH = HERE / "lshape_h_refinement.pdf"
BASE_FONT_SIZE = 18
TITLE_SIZE = 22
LABEL_SIZE = 20
TICK_SIZE = 17
LEGEND_SIZE = 16
PDF_DPI = 650
plt.rcParams.update({"font.size": BASE_FONT_SIZE})


def latest_result_path() -> Path:
    candidates = list(HERE.glob("lshape_h_refinement_results*.npz"))
    if not candidates:
        raise FileNotFoundError(
            f"No L-shape refinement results were found in {HERE}. "
            "Run run_h_refinement.py first."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def main() -> Path:
    data_path = latest_result_path()
    print(f"Loading L-shape refinement results: {data_path}")
    with np.load(data_path, allow_pickle=False) as data:
        n_basis = np.asarray(data["n_basis"], dtype=int)
        mesh_sizes = np.asarray(data["h"], dtype=float)
        global_l2 = np.asarray(
            data["validation_relative_l2_errors"], dtype=float
        )
        corner_l2 = np.asarray(data["corner_relative_l2_errors"], dtype=float)
        h1_error = np.asarray(
            data["validation_relative_h1_seminorm_errors"], dtype=float
        )
        corner_radius = float(data["corner_radius"])

    if np.any(global_l2 <= 0.0) or np.any(corner_l2 <= 0.0) or np.any(
        h1_error <= 0.0
    ):
        raise ValueError("The logarithmic error axis requires positive errors")
    order = np.argsort(mesh_sizes)[::-1]
    n_basis = n_basis[order]
    mesh_sizes = mesh_sizes[order]
    global_l2 = global_l2[order]
    corner_l2 = corner_l2[order]
    h1_error = h1_error[order]

    for n_value, h_value, global_error, corner_error, gradient_error in zip(
        n_basis, mesh_sizes, global_l2, corner_l2, h1_error
    ):
        print(
            f"  n={n_value:2d}, h={h_value:.6e}: global L2={global_error:.6e}, "
            f"corner L2={corner_error:.6e}, H1={gradient_error:.6e}"
        )

    figure, axis = plt.subplots(figsize=(11, 7), constrained_layout=True)
    axis.semilogy(
        mesh_sizes,
        global_l2,
        color="tab:blue",
        marker="^",
        linewidth=2.7,
        markersize=10,
        label=r"Global relative $L^2$ error",
    )
    axis.semilogy(
        mesh_sizes,
        corner_l2,
        color="tab:red",
        marker="^",
        linestyle="--",
        linewidth=2.7,
        markersize=10,
        label=rf"Corner relative $L^2$ error ($r\leq {corner_radius:g}$)",
    )
    axis.semilogy(
        mesh_sizes,
        h1_error,
        color="tab:green",
        marker="^",
        linestyle=":",
        linewidth=2.7,
        markersize=10,
        label=(
            r"Relative gradient error "
            r"$\|\nabla\widehat{u}-\nabla u\|_{L^2}/\|\nabla u\|_{L^2}$"
        ),
    )
    for n_value, h_value, error in zip(n_basis, mesh_sizes, global_l2):
        axis.annotate(
            rf"$n={n_value}$",
            (h_value, error),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=13,
        )

    axis.invert_xaxis()
    axis.set_xlabel(r"Uniform patch mesh size $h$", fontsize=LABEL_SIZE)
    axis.set_ylabel("Relative error", fontsize=LABEL_SIZE)
    axis.set_title(
        "L-shaped singular-solution refinement",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        pad=12,
    )
    axis.tick_params(axis="both", which="both", labelsize=TICK_SIZE)
    axis.grid(True, which="both", linestyle="--", alpha=0.45)
    axis.legend(loc="best", fontsize=LEGEND_SIZE)
    axis.margins(x=0.06, y=0.22)
    figure.savefig(OUTPUT_PATH, format="pdf", dpi=PDF_DPI, bbox_inches="tight")
    print(f"Saved L-shape refinement figure: {OUTPUT_PATH}")
    plt.show()
    return OUTPUT_PATH


if __name__ == "__main__":
    main()
