"""Plot the saved Diffusion B-spline h-refinement results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUT_PATH = HERE / "diffusion_h_refinement.pdf"

# Match Diffusion/postprocessing/plot_results.py.
BASE_FONT_SIZE = 18
TITLE_SIZE = 22
LABEL_SIZE = 20
TICK_SIZE = 17
LEGEND_SIZE = 16
PDF_DPI = 650

plt.rcParams.update({"font.size": BASE_FONT_SIZE})


def latest_result_path() -> Path:
    """Return the most recently written refinement archive."""
    candidates = list(HERE.glob("diffusion_h_refinement_results*.npz"))
    if not candidates:
        raise FileNotFoundError(
            f"No stability results were found in {HERE}. "
            "Run run_h_refinement.py first."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def main() -> None:
    data_path = latest_result_path()
    print(f"Loading refinement results: {data_path}")

    with np.load(data_path, allow_pickle=False) as data:
        n_basis = np.asarray(data["n_basis"], dtype=int)
        mesh_sizes = np.asarray(data["h"], dtype=float)
        errors = np.asarray(
            data["validation_relative_l2_errors"], dtype=float
        )

    if np.any(mesh_sizes <= 0.0) or np.any(errors <= 0.0):
        raise ValueError("A logarithmic plot requires positive h and L2 errors")

    observed_rates = (
        np.log(errors[:-1] / errors[1:])
        / np.log(mesh_sizes[:-1] / mesh_sizes[1:])
    )
    print("Observed convergence rates")
    for left, right, rate in zip(
        n_basis[:-1], n_basis[1:], observed_rates
    ):
        print(f"  n={left} -> n={right}: {rate:.4f}")

    figure, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
    axis.semilogy(
        mesh_sizes,
        errors,
        color="tab:blue",
        marker="^",
        linestyle="-",
        linewidth=2.5,
        markersize=11,
        label="PI-TP-BsA",
    )
    for index, (h_value, error, basis) in enumerate(
        zip(mesh_sizes, errors, n_basis)
    ):
        above_point = index % 2 == 0
        if index == 0:
            label_offset = (15, 12)
            horizontal_alignment = "left"
        elif index == len(n_basis) - 1:
            label_offset = (15, 12)
            horizontal_alignment = "left"
        else:
            label_offset = (0, 14 if above_point else -25)
            horizontal_alignment = "center"
        axis.annotate(
            rf"$n={basis}$",
            (h_value, error),
            xytext=label_offset,
            textcoords="offset points",
            ha=horizontal_alignment,
            va="bottom" if above_point else "top",
            fontsize=TICK_SIZE,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.9,
                "pad": 1.5,
            },
        )

    axis.invert_xaxis()
    axis.margins(x=0.16, y=0.15)
    axis.set_title(
        r"Diffusion B-spline $h$-refinement",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        pad=12,
    )
    axis.set_xlabel(r"Mesh size $h$", fontsize=LABEL_SIZE)
    axis.set_ylabel(
        r"Validation relative $L^2$ error", fontsize=LABEL_SIZE
    )
    axis.tick_params(axis="both", which="both", labelsize=TICK_SIZE)
    axis.xaxis.get_offset_text().set_fontsize(TICK_SIZE)
    axis.yaxis.get_offset_text().set_fontsize(TICK_SIZE)
    axis.grid(True, which="both", linestyle="--", alpha=0.45)
    axis.legend(loc="best", fontsize=LEGEND_SIZE)

    figure.savefig(
        OUTPUT_PATH,
        format="pdf",
        dpi=PDF_DPI,
        bbox_inches="tight",
    )
    print(f"Saved refinement figure: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
