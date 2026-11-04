"""Plot the newest Burgers spatial knot-concentration experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUT_PATH = HERE / "burgers_knot_concentration.pdf"
BASE_FONT_SIZE = 18
TITLE_SIZE = 22
LABEL_SIZE = 20
TICK_SIZE = 17
LEGEND_SIZE = 16
PDF_DPI = 650
plt.rcParams.update({"font.size": BASE_FONT_SIZE})


def latest_result_path() -> Path:
    candidates = list(HERE.glob("burgers_knot_concentration_results*.npz"))
    if not candidates:
        raise FileNotFoundError(
            f"No knot-concentration results were found in {HERE}. "
            "Run run_knot_concentration.py first."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def main() -> Path:
    data_path = latest_result_path()
    print(f"Loading knot-concentration results: {data_path}")
    with np.load(data_path, allow_pickle=False) as data:
        concentration = np.asarray(data["spatial_clustering"], dtype=float)
        minimum_spans = np.asarray(data["minimum_knot_span"], dtype=float)
        global_errors = np.asarray(
            data["validation_relative_l2_errors"], dtype=float
        )
        local_errors = np.asarray(data["local_relative_l2_errors"], dtype=float)
        tuned_concentration = float(data["tuned_spatial_clustering"])
        local_half_width = float(data["local_half_width"])

    if np.any(global_errors <= 0.0) or np.any(local_errors <= 0.0):
        raise ValueError("The logarithmic error axis requires positive errors")
    order = np.argsort(concentration)
    concentration = concentration[order]
    minimum_spans = minimum_spans[order]
    global_errors = global_errors[order]
    local_errors = local_errors[order]

    print(f"Tuned concentration: gamma={tuned_concentration:g}")
    for gamma, h_min, global_error, local_error in zip(
        concentration, minimum_spans, global_errors, local_errors
    ):
        print(
            f"  gamma={gamma:4.1f}: h_min={h_min:.6e}, "
            f"global L2={global_error:.6e}, local L2={local_error:.6e}"
        )

    figure, axis = plt.subplots(figsize=(11, 7), constrained_layout=True)
    axis.semilogy(
        concentration,
        global_errors,
        color="tab:blue",
        marker="^",
        linewidth=2.7,
        markersize=10,
        label=r"Global relative $L^2$ error",
    )
    axis.semilogy(
        concentration,
        local_errors,
        color="tab:red",
        marker="^",
        linestyle="--",
        linewidth=2.7,
        markersize=10,
        label=rf"Local relative $L^2$ error ($|x|\leq {local_half_width:g}$)",
    )
    for gamma, h_min, error in zip(
        concentration, minimum_spans, global_errors
    ):
        axis.annotate(
            rf"$h_{{\min}}={h_min:.1e}$",
            (gamma, error),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
            rotation=35,
        )

    axis.set_xlabel(r"Spatial knot-concentration parameter $\gamma$", fontsize=LABEL_SIZE)
    axis.set_ylabel(r"Relative $L^2$ error", fontsize=LABEL_SIZE)
    axis.set_title(
        "Burgers spatial knot-concentration sensitivity",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        pad=12,
    )
    axis.set_xticks(concentration)
    axis.tick_params(axis="both", which="both", labelsize=TICK_SIZE)
    axis.grid(True, which="both", linestyle="--", alpha=0.45)
    axis.legend(loc="best", fontsize=LEGEND_SIZE)
    axis.margins(x=0.05, y=0.25)
    figure.savefig(OUTPUT_PATH, format="pdf", dpi=PDF_DPI, bbox_inches="tight")
    print(f"Saved knot-concentration figure: {OUTPUT_PATH}")
    plt.show()
    return OUTPUT_PATH


if __name__ == "__main__":
    main()
