"""Plot the most recently saved Diffusion B-spline p-refinement results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUT_PATH = HERE / "diffusion_p_refinement.pdf"
BASE_FONT_SIZE = 18
TITLE_SIZE = 22
LABEL_SIZE = 20
TICK_SIZE = 17
LEGEND_SIZE = 16
PDF_DPI = 650
plt.rcParams.update({"font.size": BASE_FONT_SIZE})


def latest_result_path() -> Path:
    candidates = list(HERE.glob("diffusion_p_refinement_results*.npz"))
    if not candidates:
        raise FileNotFoundError(
            f"No p-refinement results were found in {HERE}. "
            "Run run_p_refinement.py first."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def main() -> None:
    data_path = latest_result_path()
    print(f"Loading p-refinement results: {data_path}")
    with np.load(data_path, allow_pickle=False) as data:
        degrees = np.asarray(data["degree"], dtype=int)
        n_basis = np.asarray(data["n_basis"], dtype=int)
        errors = np.asarray(data["validation_relative_l2_errors"], dtype=float)
        mesh_sizes = np.asarray(data["h"], dtype=float)
    if np.any(errors <= 0.0):
        raise ValueError("A logarithmic error axis requires positive L2 errors")
    if not np.allclose(mesh_sizes, mesh_sizes[0]):
        raise ValueError("The saved experiment does not have a fixed h")

    print(f"Fixed mesh size: h={mesh_sizes[0]:.8f}")
    for degree, basis, error in zip(degrees, n_basis, errors):
        print(f"  p={degree}, n_basis={basis}: relative L2={error:.8e}")

    figure, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
    axis.semilogy(
        degrees, errors, color="tab:green", marker="^", linestyle="-",
        linewidth=2.5, markersize=11, label="PI-TP-BsA",
    )
    for index, (degree, error, basis) in enumerate(
        zip(degrees, errors, n_basis)
    ):
        if index == 0:
            offset, alignment = (14, 12), "left"
        elif index == len(degrees) - 1:
            offset, alignment = (-14, 12), "right"
        else:
            above = index % 2 == 0
            offset, alignment = (0, 14 if above else -25), "center"
        axis.annotate(
            rf"$n={basis}$", (degree, error), xytext=offset,
            textcoords="offset points", ha=alignment,
            va="bottom" if offset[1] > 0 else "top", fontsize=TICK_SIZE,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5},
        )

    axis.margins(x=0.08, y=0.15)
    axis.set_title(
        r"Diffusion B-spline $p$-refinement", fontsize=TITLE_SIZE,
        fontweight="bold", pad=12,
    )
    axis.set_xlabel(r"B-spline degree $p$", fontsize=LABEL_SIZE)
    axis.set_ylabel(r"Validation relative $L^2$ error", fontsize=LABEL_SIZE)
    axis.set_xticks(degrees)
    axis.tick_params(axis="both", which="both", labelsize=TICK_SIZE)
    axis.xaxis.get_offset_text().set_fontsize(TICK_SIZE)
    axis.yaxis.get_offset_text().set_fontsize(TICK_SIZE)
    axis.grid(True, which="both", linestyle="--", alpha=0.45)
    axis.legend(loc="best", fontsize=LEGEND_SIZE)
    figure.savefig(OUTPUT_PATH, format="pdf", dpi=PDF_DPI, bbox_inches="tight")
    print(f"Saved p-refinement figure: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
