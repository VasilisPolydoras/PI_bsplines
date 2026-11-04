"""Plot the newest inverse-diffusion sensor-number experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter


HERE = Path(__file__).resolve().parent
BASE_OUTPUT_PATH = HERE / "diffusion_inverse_sensor_number.pdf"
BASE_FONT_SIZE = 18
TITLE_SIZE = 22
LABEL_SIZE = 20
TICK_SIZE = 17
LEGEND_SIZE = 16
PDF_DPI = 650
plt.rcParams.update({"font.size": BASE_FONT_SIZE})

def latest_result_path() -> Path:
    candidates = list(HERE.glob("diffusion_inverse_sensor_number_results*.npz"))
    if not candidates:
        raise FileNotFoundError(
            f"No sensor-number results were found in {HERE}. "
            "Run run_sensor_number.py first."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def main() -> Path:
    data_path = latest_result_path()
    print(f"Loading sensor-number results: {data_path}")
    with np.load(data_path, allow_pickle=False) as data:
        sensor_counts = np.asarray(data["sensor_counts"], dtype=int)
        parameter_errors = np.asarray(
            data["parameter_relative_errors"], dtype=float
        ).reshape(-1)
        field_errors = np.asarray(
            data["validation_relative_l2_errors"], dtype=float
        ).reshape(-1)
        estimated = np.asarray(
            data["estimated_diffusivities"], dtype=float
        ).reshape(-1)
        true_diffusivity = float(data["true_diffusivity"])
        noise_std = float(data["noise_std"])

    if np.any(parameter_errors <= 0.0) or np.any(field_errors <= 0.0):
        raise ValueError("Logarithmic axes require positive error values")
    print(f"D_true={true_diffusivity:.8f}; noise_std={noise_std:g}")
    for index, count in enumerate(sensor_counts):
        print(
            f"  N_s={count:3d}: D_hat={estimated[index]:.8f}, "
            f"D error={parameter_errors[index]:.6e}, "
            f"L2={field_errors[index]:.6e}"
        )

    figure, left_axis = plt.subplots(
        figsize=(11, 7), constrained_layout=True
    )
    right_axis = left_axis.twinx()
    parameter_line = left_axis.plot(
        sensor_counts,
        parameter_errors,
        color="tab:purple",
        marker="^",
        linestyle="-",
        linewidth=2.7,
        markersize=10,
        label=r"Relative parameter error",
    )[0]
    field_line = right_axis.plot(
        sensor_counts,
        field_errors,
        color="tab:green",
        marker="^",
        linestyle="--",
        linewidth=2.7,
        markersize=9,
        label=r"Validation relative $L^2$ error",
    )[0]

    left_axis.set_xscale("log")
    left_axis.set_yscale("log")
    right_axis.set_yscale("log")
    left_axis.set_xticks(sensor_counts)
    left_axis.get_xaxis().set_major_formatter(ScalarFormatter())
    left_axis.set_xlabel(r"Number of sensors $N_s$", fontsize=LABEL_SIZE)
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
    left_axis.tick_params(
        axis="x", which="both", labelsize=TICK_SIZE
    )
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
    left_axis.margins(x=0.06, y=0.18)
    right_axis.margins(y=0.18)
    left_axis.set_title(
        "Inverse-diffusion sensitivity to sensor number",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        pad=12,
    )
    output_path = BASE_OUTPUT_PATH
    figure.savefig(output_path, format="pdf", dpi=PDF_DPI, bbox_inches="tight")
    print(f"Saved sensor-number figure: {output_path}")
    plt.show()
    return output_path


if __name__ == "__main__":
    main()
