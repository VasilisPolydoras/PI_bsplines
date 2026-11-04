"""Plot the newest inverse-diffusion sensor-noise experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUT_PATH = HERE / "diffusion_inverse_sensor_noise.pdf"
BASE_FONT_SIZE = 18
TITLE_SIZE = 22
LABEL_SIZE = 20
TICK_SIZE = 17
LEGEND_SIZE = 16
PDF_DPI = 650
plt.rcParams.update({"font.size": BASE_FONT_SIZE})


def latest_result_path() -> Path:
    candidates = list(HERE.glob("diffusion_inverse_sensor_noise_results*.npz"))
    if not candidates:
        raise FileNotFoundError(
            f"No sensor-noise results were found in {HERE}. "
            "Run run_sensor_noise.py first."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def main() -> Path:
    data_path = latest_result_path()
    print(f"Loading sensor-noise results: {data_path}")
    with np.load(data_path, allow_pickle=False) as data:
        noise_std = np.asarray(data["noise_std"], dtype=float)
        estimated = np.asarray(data["estimated_diffusivities"], dtype=float)
        parameter_errors = np.asarray(
            data["parameter_relative_errors"], dtype=float
        )
        field_errors = np.asarray(
            data["validation_relative_l2_errors"], dtype=float
        )
        true_diffusivity = float(data["true_diffusivity"])
        sensor_count = int(data["sensor_count"])

    if np.any(parameter_errors <= 0.0) or np.any(field_errors <= 0.0):
        raise ValueError("Logarithmic axes require positive error values")
    order = np.argsort(noise_std)
    noise_std = noise_std[order]
    estimated = estimated[order]
    parameter_errors = parameter_errors[order]
    field_errors = field_errors[order]

    print(f"D_true={true_diffusivity:.8f}; N_s={sensor_count}")
    for noise, estimate, parameter_error, field_error in zip(
        noise_std, estimated, parameter_errors, field_errors
    ):
        print(
            f"  noise_std={noise:.3f}: D_hat={estimate:.8f}, "
            f"D error={parameter_error:.6e}, L2={field_error:.6e}"
        )

    figure, left_axis = plt.subplots(
        figsize=(11, 7), constrained_layout=True
    )
    right_axis = left_axis.twinx()
    parameter_line = left_axis.plot(
        noise_std,
        parameter_errors,
        color="tab:purple",
        marker="^",
        linewidth=2.7,
        markersize=10,
        label="Relative parameter error",
    )[0]
    field_line = right_axis.plot(
        noise_std,
        field_errors,
        color="tab:green",
        marker="^",
        linestyle="--",
        linewidth=2.7,
        markersize=10,
        label=r"Validation relative $L^2$ error",
    )[0]
    positive_noise = noise_std[noise_std > 0.0]
    linear_threshold = (
        float(positive_noise.min()) if positive_noise.size else 1.0e-3
    )
    left_axis.set_xscale(
        "symlog", linthresh=linear_threshold, linscale=1.0, base=10
    )
    left_axis.set_yscale("log")
    right_axis.set_yscale("log")
    left_axis.set_xticks(noise_std)
    left_axis.set_xlabel(
        r"Absolute noise standard deviation $\sigma_{\mathrm{noise}}$",
        fontsize=LABEL_SIZE,
    )
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
    left_axis.tick_params(axis="x", which="both", labelsize=TICK_SIZE)
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
    left_axis.margins(x=0.05, y=0.18)
    right_axis.margins(y=0.18)
    left_axis.set_title(
        "Inverse-diffusion sensitivity to measurement noise",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        pad=12,
    )
    figure.savefig(OUTPUT_PATH, format="pdf", dpi=PDF_DPI, bbox_inches="tight")
    print(f"Saved sensor-noise figure: {OUTPUT_PATH}")
    plt.show()
    return OUTPUT_PATH


if __name__ == "__main__":
    main()
