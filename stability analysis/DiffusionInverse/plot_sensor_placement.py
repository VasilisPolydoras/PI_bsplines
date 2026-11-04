"""Plot sensor layouts and errors for the placement-sensitivity study."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUT_PATH = HERE / "diffusion_inverse_sensor_placement.pdf"
BASE_FONT_SIZE = 18
TITLE_SIZE = 22
LABEL_SIZE = 20
TICK_SIZE = 16
LEGEND_SIZE = 15
PDF_DPI = 650
plt.rcParams.update({"font.size": BASE_FONT_SIZE})


def latest_result_path() -> Path:
    candidates = list(
        HERE.glob("diffusion_inverse_sensor_placement_results*.npz")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No placement results were found in {HERE}. "
            "Run run_sensor_placement.py first."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def main() -> Path:
    data_path = latest_result_path()
    print(f"Loading sensor-placement results: {data_path}")
    with np.load(data_path, allow_pickle=False) as data:
        labels = np.asarray(data["placement_labels"]).astype(str)
        locations = np.asarray(data["sensor_locations"], dtype=float)
        estimated = np.asarray(data["estimated_diffusivities"], dtype=float)
        parameter_errors = np.asarray(
            data["parameter_relative_errors"], dtype=float
        )
        field_errors = np.asarray(
            data["validation_relative_l2_errors"], dtype=float
        )
        true_diffusivity = float(data["true_diffusivity"])
        sensor_count = int(data["sensor_count"])
        noise_std = float(data["noise_std"])

    if np.any(parameter_errors <= 0.0) or np.any(field_errors <= 0.0):
        raise ValueError("Logarithmic axes require positive error values")
    print(
        f"D_true={true_diffusivity:.8f}; N_s={sensor_count}; "
        f"noise_std={noise_std:g}"
    )
    for label, estimate, parameter_error, field_error in zip(
        labels, estimated, parameter_errors, field_errors
    ):
        print(
            f"  {label:10s}: D_hat={estimate:.8f}, "
            f"D error={parameter_error:.6e}, L2={field_error:.6e}"
        )

    figure = plt.figure(figsize=(16, 10), constrained_layout=True)
    grid = figure.add_gridspec(2, 4, height_ratios=(1.0, 1.15))
    for index, label in enumerate(labels):
        axis = figure.add_subplot(grid[0, index])
        axis.scatter(
            locations[index, :, 0],
            locations[index, :, 1],
            s=18,
            marker="^",
            color="tab:blue",
            alpha=0.75,
            edgecolors="none",
        )
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(label, fontsize=BASE_FONT_SIZE, fontweight="bold")
        axis.set_xlabel(r"$x$", fontsize=BASE_FONT_SIZE)
        if index == 0:
            axis.set_ylabel(r"$t$", fontsize=BASE_FONT_SIZE)
        axis.tick_params(axis="both", labelsize=13)
        axis.grid(True, linestyle="--", alpha=0.35)

    left_axis = figure.add_subplot(grid[1, :])
    right_axis = left_axis.twinx()
    positions = np.arange(len(labels))
    parameter_line = left_axis.plot(
        positions,
        parameter_errors,
        color="tab:purple",
        marker="^",
        linewidth=2.7,
        markersize=10,
        label="Relative parameter error",
    )[0]
    field_line = right_axis.plot(
        positions,
        field_errors,
        color="tab:green",
        marker="^",
        linestyle="--",
        linewidth=2.7,
        markersize=10,
        label=r"Validation relative $L^2$ error",
    )[0]
    left_axis.set_yscale("log")
    right_axis.set_yscale("log")
    left_axis.set_xticks(positions, labels)
    left_axis.set_xlabel("Sensor placement", fontsize=LABEL_SIZE)
    left_axis.set_ylabel(
        r"Relative diffusivity error $|\widehat D-D|/D$",
        fontsize=LABEL_SIZE,
        color="tab:purple",
    )
    right_axis.set_ylabel(
        r"Validation relative $L^2$ error",
        fontsize=LABEL_SIZE,
        color="tab:green",
    )
    left_axis.tick_params(axis="x", labelsize=TICK_SIZE)
    left_axis.tick_params(
        axis="y", which="both", labelsize=TICK_SIZE, colors="tab:purple"
    )
    right_axis.tick_params(
        axis="y", which="both", labelsize=TICK_SIZE, colors="tab:green"
    )
    left_axis.grid(True, which="both", linestyle="--", alpha=0.45)
    left_axis.legend(
        handles=[parameter_line, field_line], loc="best", fontsize=LEGEND_SIZE
    )
    left_axis.margins(x=0.08, y=0.18)
    right_axis.margins(y=0.18)
    figure.suptitle(
        "Inverse-diffusion sensitivity to sensor placement",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )
    figure.savefig(OUTPUT_PATH, format="pdf", dpi=PDF_DPI, bbox_inches="tight")
    print(f"Saved sensor-placement figure: {OUTPUT_PATH}")
    plt.show()
    return OUTPUT_PATH


if __name__ == "__main__":
    main()
