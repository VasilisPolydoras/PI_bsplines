"""Recreate publication-ready wave-equation comparisons without retraining."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
BASE_FONT_SIZE = 18
TITLE_SIZE = 22
LABEL_SIZE = 20
TICK_SIZE = 17
LEGEND_SIZE = 16

plt.rcParams.update({"font.size": BASE_FONT_SIZE})

SOLVERS = {
    "bspline-hard": ("bspline_wave_hard", "PI-TP-BsA"),
    "pinn-hard": ("pinn_wave_hard", "PINN hard"),
    "pinn-soft": ("pinn_wave_soft", "PINN soft"),
    "pikan-hard": ("pikan_wave_hard", "PIKAN hard"),
    "pikan-soft": ("pikan_wave_soft", "PIKAN soft"),
}

STATISTICS_FILES = {
    "bspline-hard": "bspline_statistics.npz",
    "pinn-hard": "pinn_hard_statistics.npz",
    "pinn-soft": "pinn_soft_statistics.npz",
    "pikan-hard": "pikan_hard_statistics.npz",
    "pikan-soft": "pikan_soft_statistics.npz",
}


def print_statistics(key: str) -> None:
    """Print a saved repeated-run summary for one solver configuration."""
    statistics_path = HERE / STATISTICS_FILES[key]
    if not statistics_path.exists():
        print(f"No saved statistics for {key}: {statistics_path}")
        return

    with np.load(statistics_path, allow_pickle=False) as data:
        runs = int(data["l2_errors"].size)
        l2_mean = float(data["l2_mean"])
        l2_std = float(data["l2_std"])
        l2_ci = np.asarray(data["l2_ci95"], dtype=float)
        time_mean = float(data["time_mean_seconds"])
        time_std = float(data["time_std_seconds"])
        time_ci = np.asarray(data["time_ci95_seconds"], dtype=float)

    _, title = SOLVERS[key]
    print(f"\nSaved statistics: {title} ({runs} runs)")
    print(f"  Relative L2 mean +/- std: {l2_mean:.6e} +/- {l2_std:.6e}")
    print(f"  Relative L2 95% CI: [{l2_ci[0]:.6e}, {l2_ci[1]:.6e}]")
    print(f"  Training time mean +/- std: {time_mean:.3f} +/- {time_std:.3f} s")
    print(f"  Training-time 95% CI: [{time_ci[0]:.3f}, {time_ci[1]:.3f}] s")


def load_solution(data_path: Path) -> dict[str, np.ndarray | float]:
    with np.load(data_path, allow_pickle=False) as data:
        return {
            "x": data["x"],
            "time": data["time"],
            "prediction": data["prediction"],
            "absolute_error": data["absolute_error"],
            "l2_error": float(data["validation_relative_l2_error"]),
            "wall_time": float(data["training_wall_time_seconds"]),
        }


def plot_comparison(keys: tuple[str, ...], output_path: Path) -> None:
    records: dict[str, dict[str, np.ndarray | float]] = {}
    for key in keys:
        stem, _ = SOLVERS[key]
        data_path = HERE / f"{stem}_solution_data.npz"
        if data_path.exists():
            records[key] = load_solution(data_path)
        else:
            print(f"Missing solution data for {key}: {data_path}")
    if not records:
        raise FileNotFoundError("No saved wave-equation solution data were found")

    figure, axes = plt.subplots(
        len(keys), 2, figsize=(17, 5.2 * len(keys)), squeeze=False,
        sharex=True, sharey=True, constrained_layout=True,
    )
    for row, key in enumerate(keys):
        _, title = SOLVERS[key]
        prediction_axis, error_axis = axes[row]
        if key not in records:
            for axis in (prediction_axis, error_axis):
                axis.text(
                    0.5, 0.5, "Saved data unavailable", ha="center", va="center",
                    transform=axis.transAxes,
                )
                axis.set_axis_off()
            continue

        record = records[key]
        x = np.asarray(record["x"])
        times = np.asarray(record["time"])
        extent = [float(x[0]), float(x[-1]), float(times[0]), float(times[-1])]
        solution_image = prediction_axis.imshow(
            np.asarray(record["prediction"]), extent=extent, origin="lower",
            aspect="auto", cmap="viridis",
        )
        error_image = error_axis.imshow(
            np.asarray(record["absolute_error"]), extent=extent, origin="lower",
            aspect="auto", cmap="magma", vmin=0.0,
        )
        for axis in (prediction_axis, error_axis):
            axis.set_xlabel("$x$", fontsize=LABEL_SIZE)
            axis.set_ylabel("$t$", fontsize=LABEL_SIZE)
            axis.tick_params(axis="both", labelsize=TICK_SIZE)
        prediction_axis.annotate(
            title, xy=(-0.23, 0.5), xycoords="axes fraction", ha="center",
            va="center", rotation=90, fontsize=TITLE_SIZE, fontweight="bold",
        )
        solution_bar = figure.colorbar(
            solution_image, ax=prediction_axis, pad=0.02, fraction=0.05
        )
        solution_bar.ax.tick_params(labelsize=TICK_SIZE)
        solution_bar.ax.yaxis.get_offset_text().set_fontsize(TICK_SIZE)
        error_bar = figure.colorbar(
            error_image, ax=error_axis, pad=0.02, fraction=0.05
        )
        error_bar.ax.tick_params(labelsize=TICK_SIZE)
        error_bar.ax.yaxis.get_offset_text().set_fontsize(TICK_SIZE)
        error_axis.text(
            0.97, 0.95, rf"relative $L^2={float(record['l2_error']):.2e}$",
            ha="right", va="top", transform=error_axis.transAxes,
            fontsize=TICK_SIZE,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    axes[0, 0].set_title("Predicted solution", fontsize=TITLE_SIZE, pad=12)
    axes[0, 1].set_title("Absolute error", fontsize=TITLE_SIZE, pad=12)
    figure.savefig(output_path, format="pdf", dpi=650, bbox_inches="tight")


def load_training(data_path: Path) -> dict[str, np.ndarray | None]:
    with np.load(data_path, allow_pickle=False) as data:
        return {
            "epoch": data["epoch"],
            "loss": data["loss"],
            "initial_epoch": data["initial_epoch"] if "initial_epoch" in data else None,
            "initial_loss": data["initial_loss"] if "initial_loss" in data else None,
        }


def plot_training_comparison(keys: tuple[str, ...], output_path: Path) -> None:
    figure, axes = plt.subplots(
        len(keys), 1, figsize=(13, 4.2 * len(keys)), squeeze=False,
        constrained_layout=True,
    )
    for row, key in enumerate(keys):
        stem, title = SOLVERS[key]
        axis = axes[row, 0]
        data_path = HERE / f"{stem}_training_data.npz"
        if not data_path.exists():
            axis.text(
                0.5, 0.5, "Saved training data unavailable", ha="center",
                va="center", transform=axis.transAxes,
            )
            axis.set_title(title)
            axis.set_axis_off()
            print(f"Missing training data for {key}: {data_path}")
            continue

        record = load_training(data_path)
        if record["initial_epoch"] is not None:
            axis.semilogy(
                record["initial_epoch"], record["initial_loss"],
                label="Initial-condition pretraining", linewidth=2.0,
            )
        loss_label = (
            "Total soft-constraint loss" if key.endswith("soft")
            else "PDE-residual training loss"
        )
        axis.semilogy(record["epoch"], record["loss"], label=loss_label, linewidth=2.0)
        axis.set_title(title, fontsize=TITLE_SIZE, pad=8)
        axis.set_xlabel("Epoch", fontsize=LABEL_SIZE)
        axis.set_ylabel("Loss", fontsize=LABEL_SIZE)
        axis.tick_params(axis="both", labelsize=TICK_SIZE)
        axis.xaxis.get_offset_text().set_fontsize(TICK_SIZE)
        axis.yaxis.get_offset_text().set_fontsize(TICK_SIZE)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(loc="best", fontsize=LEGEND_SIZE)

    figure.savefig(output_path, format="pdf", dpi=650, bbox_inches="tight")


def main(name: str, show: bool) -> None:
    selected = tuple(SOLVERS) if name == "all" else (name,)
    if name == "all":
        comparison_groups = {
            "wave_method_comparison.pdf": (
                "bspline-hard",
                "pinn-soft",
                "pinn-hard",
                "pikan-soft",
                "pikan-hard",
            ),
        }
    else:
        comparison_groups = {f"{SOLVERS[name][0]}_comparison.pdf": (name,)}
    for comparison_name, comparison_keys in comparison_groups.items():
        comparison_path = HERE / comparison_name
        plot_comparison(comparison_keys, comparison_path)
        print(f"Saved comparison figure: {comparison_path}")
    training_name = (
        "wave_loss_evolution.pdf"
        if name == "all" else f"{SOLVERS[name][0]}_loss_evolution.pdf"
    )
    training_path = HERE / training_name
    plot_training_comparison(selected, training_path)
    print(f"Saved loss-evolution figure: {training_path}")

    for key in selected:
        stem, _ = SOLVERS[key]
        solution_path = HERE / f"{stem}_solution_data.npz"
        if solution_path.exists():
            record = load_solution(solution_path)
            error = float(record["l2_error"])
            wall_time = float(record["wall_time"])
            print(
                f"{key}: validation relative L2={error:.6e}, "
                f"time={wall_time:.3f} s"
            )
        print_statistics(key)
    if show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replot saved wave-equation results")
    parser.add_argument("--solver", choices=("all", *SOLVERS), default="all")
    parser.add_argument("--show", action="store_true")
    arguments = parser.parse_args()
    main(arguments.solver, arguments.show)
