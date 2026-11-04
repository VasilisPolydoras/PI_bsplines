"""Plot the newest Diffusion B-spline collocation-density results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUT_PATH = HERE / "diffusion_rho_refinement.pdf"
BASE_FONT_SIZE = 18
TITLE_SIZE = 22
LABEL_SIZE = 20
TICK_SIZE = 17
LEGEND_SIZE = 16
PDF_DPI = 650
plt.rcParams.update({"font.size": BASE_FONT_SIZE})


def latest_result_path() -> Path:
    candidates = list(HERE.glob("diffusion_rho_refinement_results*.npz"))
    if not candidates:
        raise FileNotFoundError(
            f"No collocation-density results were found in {HERE}. "
            "Run run_rho_refinement.py first."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def main() -> None:
    data_path = latest_result_path()
    print(f"Loading collocation-density results: {data_path}")
    with np.load(data_path, allow_pickle=False) as data:
        rho = np.asarray(data["effective_rho"], dtype=float)
        counts = np.asarray(data["interior_collocation_points"], dtype=int)
        errors = np.asarray(data["validation_relative_l2_errors"], dtype=float)
    if np.any(rho <= 0.0) or np.any(errors <= 0.0):
        raise ValueError("Logarithmic axes require positive rho and L2 errors")

    for density, count, error in zip(rho, counts, errors):
        print(
            f"rho={density:.8f}, N_Omega={count}: relative L2={error:.8e}"
        )

    figure, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
    axis.semilogy(
        rho, errors, color="tab:orange", marker="^", linestyle="-",
        linewidth=2.5, markersize=11, label="PI-TP-BsA",
    )
    # Hand-tuned offsets keep the labels separated, particularly for the
    # nearly equal errors at rho=4, 8, and 10.
    label_layout = (
        ((15, 16), "left", "bottom"),
        ((15, -22), "left", "top"),
        ((15, 18), "left", "bottom"),
        ((15, 44), "left", "bottom"),
        ((15, -44), "left", "top"),
        ((15, 16), "left", "bottom"),
        ((15, -32), "left", "top"),
        ((15, 20), "left", "bottom"),
    )
    for (density, error, count), (offset, alignment, vertical_alignment) in zip(
        zip(rho, errors, counts), label_layout
    ):
        axis.annotate(
            rf"$N_\Omega={count}$", (density, error), xytext=offset,
            textcoords="offset points", ha=alignment,
            va=vertical_alignment, fontsize=TICK_SIZE,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5},
            arrowprops={"arrowstyle": "-", "color": "0.4", "linewidth": 0.8},
        )

    axis.margins(x=0.0, y=0.25)
    axis.set_xlim(0.0, 50.0)
    axis.set_xticks(np.arange(0.0, 51.0, 5.0))
    axis.set_title(
        "Diffusion B-spline collocation-density study",
        fontsize=TITLE_SIZE, fontweight="bold", pad=12,
    )
    axis.set_xlabel(r"Collocation ratio $\rho=N_\Omega/N_C$", fontsize=LABEL_SIZE)
    axis.set_ylabel(r"Validation relative $L^2$ error", fontsize=LABEL_SIZE)
    axis.tick_params(axis="both", which="both", labelsize=TICK_SIZE)
    axis.xaxis.get_offset_text().set_fontsize(TICK_SIZE)
    axis.yaxis.get_offset_text().set_fontsize(TICK_SIZE)
    axis.grid(True, which="both", linestyle="--", alpha=0.45)
    axis.legend(loc="best", fontsize=LEGEND_SIZE)
    figure.savefig(OUTPUT_PATH, format="pdf", dpi=PDF_DPI, bbox_inches="tight")
    print(f"Saved collocation-density figure: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
