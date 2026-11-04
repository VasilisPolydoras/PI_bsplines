"""Create publication-ready comparisons for the irregular L-shaped problem."""

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
    "bspline": ("bspline_xpinn_irregular", "PI-TP-BsA"),
    "xpinn": ("xpinn_irregular", "XPINN hard"),
}

STATISTICS_FILES = {
    "bspline": "bspline_statistics.npz",
    "xpinn": "xpinn_statistics.npz",
}


def print_statistics(key: str) -> None:
    """Print the repeated-run summary when its archive is available."""
    statistics_path = HERE / STATISTICS_FILES[key]
    if not statistics_path.exists():
        print(f"No saved statistics for {key}: {statistics_path}")
        return

    with np.load(statistics_path, allow_pickle=False) as data:
        required = {
            "l2_errors",
            "l2_mean",
            "l2_std",
            "l2_ci95",
            "time_mean_seconds",
            "time_std_seconds",
            "time_ci95_seconds",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            print(
                f"Statistics archive for {key} is missing: "
                f"{', '.join(missing)}"
            )
            return
        runs = int(data["l2_errors"].size)
        l2_mean = float(data["l2_mean"])
        l2_std = float(data["l2_std"])
        l2_ci = np.asarray(data["l2_ci95"], dtype=float)
        time_mean = float(data["time_mean_seconds"])
        time_std = float(data["time_std_seconds"])
        time_ci = np.asarray(data["time_ci95_seconds"], dtype=float)

    _, method_label = SOLVERS[key]
    print(f"\nSaved statistics: {method_label} ({runs} runs)")
    print(f"  Relative L2 mean +/- std: {l2_mean:.6e} +/- {l2_std:.6e}")
    print(f"  Relative L2 95% CI: [{l2_ci[0]:.6e}, {l2_ci[1]:.6e}]")
    print(f"  Training time mean +/- std: {time_mean:.3f} +/- {time_std:.3f} s")
    print(f"  Training-time 95% CI: [{time_ci[0]:.3f}, {time_ci[1]:.3f}] s")


def contour_levels(values: np.ndarray, *, start_at_zero: bool = False):
    """Return robust contour levels for the three-patch data."""
    lower = 0.0 if start_at_zero else float(np.nanmin(values))
    upper = float(np.nanmax(values))
    if np.isclose(lower, upper):
        upper = lower + 1.0e-12
    return np.linspace(lower, upper, 51)


def load_solution(path: Path) -> dict[str, np.ndarray | float]:
    """Load one solver's saved solution arrays without retraining."""
    with np.load(path, allow_pickle=False) as data:
        return {
            "x": data["x"],
            "y": data["y"],
            "prediction": data["prediction"],
            "absolute_error": data["absolute_error"],
            "l2_error": float(data["validation_relative_l2_error"]),
        }


def draw_patch_field(axis, x, y, values, levels, color_map):
    """Draw the same scalar field over all three rectangular patches."""
    contour = None
    for patch_index in range(x.shape[0]):
        contour = axis.contourf(
            x[patch_index],
            y[patch_index],
            values[patch_index],
            levels=levels,
            cmap=color_map,
            extend="max" if levels[0] == 0.0 else "both",
        )
    return contour


def annotate_domain_decomposition(axis) -> None:
    """Mark both internal interfaces and identify the three patches."""
    line_style = {
        "color": "0.45",
        "linestyle": "--",
        "linewidth": 2.0,
        "zorder": 10,
    }
    axis.plot([0.0, 0.0], [0.0, 1.0], **line_style)
    axis.plot([-1.0, 0.0], [0.0, 0.0], **line_style)
    for label, x_position, y_position in (
        ("Domain 1", 0.50, 0.50),
        ("Domain 2", -0.50, 0.50),
        ("Domain 3", -0.50, -0.50),
    ):
        axis.text(
            x_position,
            y_position,
            label,
            ha="center",
            va="center",
            fontsize=TICK_SIZE,
            color="0.25",
            zorder=11,
            bbox={
                "facecolor": "white",
                "alpha": 0.70,
                "edgecolor": "none",
                "pad": 2.0,
            },
        )


def plot_comparison(keys: tuple[str, ...], output_path: Path) -> None:
    """Save prediction/error rows for the selected methods as one PDF."""
    records = {}
    for key in keys:
        stem, _ = SOLVERS[key]
        data_path = HERE / f"{stem}_solution_data.npz"
        if data_path.exists():
            records[key] = load_solution(data_path)
        else:
            print(f"Missing solution data for {key}: {data_path}")
    if not records:
        raise FileNotFoundError("No saved irregular solution data were found")

    figure, axes = plt.subplots(
        len(keys), 2, figsize=(17, 5.2 * len(keys)), squeeze=False,
        sharex=True, sharey=True, constrained_layout=True,
    )
    for row, key in enumerate(keys):
        _, method_label = SOLVERS[key]
        prediction_axis, error_axis = axes[row]
        if key not in records:
            for axis in (prediction_axis, error_axis):
                axis.text(
                    0.5, 0.5, "Saved data unavailable", ha="center",
                    va="center", transform=axis.transAxes,
                )
                axis.set_axis_off()
            continue

        record = records[key]
        x = np.asarray(record["x"])
        y = np.asarray(record["y"])
        prediction = np.asarray(record["prediction"])
        absolute_error = np.asarray(record["absolute_error"])
        solution_contour = draw_patch_field(
            prediction_axis, x, y, prediction,
            contour_levels(prediction), "viridis",
        )
        error_contour = draw_patch_field(
            error_axis, x, y, absolute_error,
            contour_levels(absolute_error, start_at_zero=True), "magma",
        )
        for axis in (prediction_axis, error_axis):
            axis.set_aspect("equal")
            axis.set_xlabel("$x$", fontsize=LABEL_SIZE)
            axis.set_ylabel("$y$", fontsize=LABEL_SIZE)
            axis.tick_params(axis="both", labelsize=TICK_SIZE)
            annotate_domain_decomposition(axis)
        prediction_axis.annotate(
            method_label, xy=(-0.23, 0.5), xycoords="axes fraction",
            ha="center", va="center", rotation=90,
            fontsize=TITLE_SIZE, fontweight="bold",
        )
        for contour, axis in (
            (solution_contour, prediction_axis),
            (error_contour, error_axis),
        ):
            color_bar = figure.colorbar(
                contour, ax=axis, pad=0.02, fraction=0.05
            )
            color_bar.ax.tick_params(labelsize=TICK_SIZE)
            color_bar.ax.yaxis.get_offset_text().set_fontsize(TICK_SIZE)
        error_axis.text(
            0.97, 0.95,
            rf"relative $L^2={float(record['l2_error']):.2e}$",
            ha="right", va="top", transform=error_axis.transAxes,
            fontsize=TICK_SIZE,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    axes[0, 0].set_title("Predicted solution", fontsize=TITLE_SIZE, pad=12)
    axes[0, 1].set_title("Absolute error", fontsize=TITLE_SIZE, pad=12)
    figure.savefig(output_path, format="pdf", dpi=650, bbox_inches="tight")


def load_training(path: Path) -> dict[str, np.ndarray | float | None]:
    """Load interior and optional boundary-pretraining histories."""
    with np.load(path, allow_pickle=False) as data:
        return {
            "epoch": data["epoch"],
            "loss": data["loss"],
            "boundary_epoch": (
                data["boundary_epoch"] if "boundary_epoch" in data else None
            ),
            "boundary_loss": (
                data["boundary_loss"] if "boundary_loss" in data else None
            ),
            "training_time_seconds": (
                float(data["training_time_seconds"])
                if "training_time_seconds" in data
                else None
            ),
        }


def plot_training_comparison(keys: tuple[str, ...], output_path: Path) -> None:
    """Save both methods' loss evolution using the Diffusion styling."""
    figure, axes = plt.subplots(
        len(keys), 1, figsize=(13, 4.2 * len(keys)), squeeze=False,
        constrained_layout=True,
    )
    for row, key in enumerate(keys):
        stem, method_label = SOLVERS[key]
        axis = axes[row, 0]
        data_path = HERE / f"{stem}_training_data.npz"
        if not data_path.exists():
            axis.text(
                0.5, 0.5, "Saved training data unavailable", ha="center",
                va="center", transform=axis.transAxes,
            )
            axis.set_axis_off()
            print(f"Missing training data for {key}: {data_path}")
            continue

        record = load_training(data_path)
        if (
            record["boundary_epoch"] is not None
            and np.asarray(record["boundary_epoch"]).size
        ):
            axis.semilogy(
                record["boundary_epoch"], record["boundary_loss"],
                label="Dirichlet pretraining", linewidth=2.0,
            )
        axis.semilogy(
            record["epoch"], record["loss"],
            label="PDE/interface training loss", linewidth=2.0,
        )
        axis.set_title(method_label, fontsize=TITLE_SIZE, pad=8)
        axis.set_xlabel("Epoch", fontsize=LABEL_SIZE)
        axis.set_ylabel("Loss", fontsize=LABEL_SIZE)
        axis.tick_params(axis="both", labelsize=TICK_SIZE)
        axis.xaxis.get_offset_text().set_fontsize(TICK_SIZE)
        axis.yaxis.get_offset_text().set_fontsize(TICK_SIZE)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(loc="best", fontsize=LEGEND_SIZE)

    figure.savefig(output_path, format="pdf", dpi=650, bbox_inches="tight")


def main(solver: str = "all", show: bool = False) -> None:
    selected = tuple(SOLVERS) if solver == "all" else (solver,)
    comparison_name = (
        "lshape_method_comparison.pdf"
        if solver == "all"
        else f"{SOLVERS[solver][0]}_comparison.pdf"
    )
    comparison_path = HERE / comparison_name
    plot_comparison(selected, comparison_path)
    print(f"Saved comparison figure: {comparison_path}")

    loss_name = (
        "lshape_loss_evolution.pdf"
        if solver == "all"
        else f"{SOLVERS[solver][0]}_loss_evolution.pdf"
    )
    loss_path = HERE / loss_name
    plot_training_comparison(selected, loss_path)
    print(f"Saved loss-evolution figure: {loss_path}")

    for key in selected:
        stem, method_label = SOLVERS[key]
        solution_path = HERE / f"{stem}_solution_data.npz"
        training_path = HERE / f"{stem}_training_data.npz"
        single_run_time = None
        if training_path.exists():
            single_run_time = load_training(training_path)[
                "training_time_seconds"
            ]
        if solution_path.exists():
            record = load_solution(solution_path)
            message = (
                f"{method_label}: validation relative L2="
                f"{float(record['l2_error']):.6e}"
            )
            if single_run_time is not None:
                message += f", training time={float(single_run_time):.3f} s"
            print(message)
        elif single_run_time is not None:
            print(
                f"{method_label}: training time="
                f"{float(single_run_time):.3f} s"
            )
        print_statistics(key)
    if show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replot saved irregular L-shaped results"
    )
    parser.add_argument("--solver", choices=("all", *SOLVERS), default="all")
    parser.add_argument("--show", action="store_true")
    arguments = parser.parse_args()
    main(arguments.solver, arguments.show)
