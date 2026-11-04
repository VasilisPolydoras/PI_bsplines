"""Plot the newest fixed-D profile-loss identifiability experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
BASE_OUTPUT_PATH = HERE / "diffusion_inverse_identifiability_profile.pdf"
BASE_FONT_SIZE = 18
TITLE_SIZE = 22
LABEL_SIZE = 20
TICK_SIZE = 17
LEGEND_SIZE = 16
PDF_DPI = 650
plt.rcParams.update({"font.size": BASE_FONT_SIZE})


def latest_result_path() -> Path:
    candidates = list(
        HERE.glob("diffusion_inverse_identifiability_results*.npz")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No identifiability results were found in {HERE}. "
            "Run run_identifiability.py first."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def next_output_path() -> Path:
    if not BASE_OUTPUT_PATH.exists():
        return BASE_OUTPUT_PATH
    run_number = 2
    while True:
        candidate = HERE / (
            f"diffusion_inverse_identifiability_profile_run_{run_number:02d}.pdf"
        )
        if not candidate.exists():
            return candidate
        run_number += 1


def main() -> Path:
    data_path = latest_result_path()
    print(f"Loading identifiability results: {data_path}")
    with np.load(data_path, allow_pickle=False) as data:
        diffusivities = np.asarray(data["candidate_diffusivities"], dtype=float)
        true_diffusivity = float(data["true_diffusivity"])
        total_losses = np.asarray(data["profile_total_loss"], dtype=float)
        pde_losses = np.asarray(data["profile_pde_loss"], dtype=float)
        data_losses = np.asarray(data["profile_data_loss"], dtype=float)
        noise_std = float(data["noise_std"])

    if np.any(total_losses <= 0.0) or np.any(pde_losses <= 0.0) or np.any(
        data_losses <= 0.0
    ):
        raise ValueError("The logarithmic loss axis requires positive losses")

    order = np.argsort(diffusivities)
    diffusivities = diffusivities[order]
    total_losses = total_losses[order]
    pde_losses = pde_losses[order]
    data_losses = data_losses[order]
    minimum_index = int(np.argmin(total_losses))
    profile_minimizer = diffusivities[minimum_index]

    print(f"D_true={true_diffusivity:.8f}; noise_std={noise_std:.6g}")
    for diffusivity, total, pde, data in zip(
        diffusivities, total_losses, pde_losses, data_losses
    ):
        print(
            f"  D={diffusivity:.6f}: profile={total:.8e}, "
            f"PDE={pde:.8e}, data={data:.8e}"
        )
    print(f"Minimum tested profile loss occurs at D={profile_minimizer:.8f}")

    figure, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
    axis.semilogy(
        diffusivities,
        total_losses,
        color="tab:blue",
        marker="^",
        linewidth=2.7,
        markersize=10,
        label=r"Profile loss $\Phi(D)$",
    )
    axis.semilogy(
        diffusivities,
        pde_losses,
        color="tab:orange",
        linestyle="--",
        linewidth=2.0,
        label=r"PDE component",
    )
    axis.semilogy(
        diffusivities,
        data_losses,
        color="tab:green",
        linestyle=":",
        linewidth=2.5,
        label=r"Data component",
    )
    axis.axvline(
        true_diffusivity,
        color="black",
        linestyle="--",
        linewidth=2.0,
        label=rf"True $D={true_diffusivity:g}$",
    )
    axis.scatter(
        [profile_minimizer],
        [total_losses[minimum_index]],
        color="crimson",
        marker="*",
        s=190,
        zorder=5,
        label=rf"Profile minimum $D={profile_minimizer:g}$",
    )
    axis.annotate(
        rf"$D_{{\min}}={profile_minimizer:g}$",
        (profile_minimizer, total_losses[minimum_index]),
        xytext=(14, 14),
        textcoords="offset points",
        fontsize=TICK_SIZE,
        bbox={
            "facecolor": "white",
            "edgecolor": "0.7",
            "alpha": 0.9,
            "pad": 2.0,
        },
    )
    axis.set_title(
        "Inverse-diffusion parameter identifiability",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        pad=12,
    )
    axis.set_xlabel(r"Fixed candidate diffusivity $D_k$", fontsize=LABEL_SIZE)
    axis.set_ylabel(r"Minimized loss", fontsize=LABEL_SIZE)
    axis.tick_params(axis="both", which="both", labelsize=TICK_SIZE)
    axis.xaxis.get_offset_text().set_fontsize(TICK_SIZE)
    axis.yaxis.get_offset_text().set_fontsize(TICK_SIZE)
    axis.grid(True, which="both", linestyle="--", alpha=0.45)
    axis.legend(loc="best", fontsize=LEGEND_SIZE)
    axis.margins(x=0.04, y=0.16)

    output_path = next_output_path()
    figure.savefig(output_path, format="pdf", dpi=PDF_DPI, bbox_inches="tight")
    print(f"Saved identifiability figure: {output_path}")
    plt.show()
    return output_path


if __name__ == "__main__":
    main()
