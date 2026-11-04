"""Recreate publication-ready Burgers comparisons without retraining."""

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
    "bspline": ("bspline_burgers", "PI-TP-BsA"),
    "pinn-soft": ("pinn_burgers_soft", "PINN soft"),
    "pinn-hard": ("pinn_burgers_hard", "PINN hard"),
    "pikan-soft": ("pikan_burgers_soft", "PIKAN soft"),
    "pikan-hard": ("pikan_burgers_hard", "PIKAN hard"),
}


def load_solution(path: Path) -> dict[str, np.ndarray | float]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "x": data["x"],
            "time": data["time"],
            "prediction": data["prediction"],
            "absolute_error": data["absolute_error"],
            "l2_error": float(data["validation_relative_l2_error"]),
            "wall_time": (
                float(data["training_wall_time_seconds"])
                if "training_wall_time_seconds" in data.files
                else float("nan")
            ),
        }


def plot_comparison(keys: tuple[str, ...], output_path: Path) -> None:
    records = {}
    for key in keys:
        stem, _ = SOLVERS[key]
        path = HERE / f"{stem}_solution_data.npz"
        if path.exists():
            records[key] = load_solution(path)
        else:
            print(f"Missing solution data for {key}: {path}")

    if not records:
        raise FileNotFoundError("No saved Burgers solution data were found")

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
                    0.5, 0.5, "Saved data unavailable", ha="center",
                    va="center", transform=axis.transAxes, fontsize=LABEL_SIZE,
                )
                axis.set_axis_off()
            prediction_axis.annotate(
                title, xy=(-0.23, 0.5), xycoords="axes fraction",
                ha="center", va="center", rotation=90,
                fontsize=TITLE_SIZE, fontweight="bold",
            )
            continue

        record = records[key]
        x, times = np.asarray(record["x"]), np.asarray(record["time"])
        extent = [float(x[0]), float(x[-1]), float(times[0]), float(times[-1])]
        solution_image = prediction_axis.imshow(
            np.asarray(record["prediction"]), extent=extent, origin="lower",
            aspect="auto", cmap="viridis",
        )
        error_image = error_axis.imshow(
            np.asarray(record["absolute_error"]), extent=extent,
            origin="lower", aspect="auto", cmap="magma", vmin=0.0,
        )
        for axis in (prediction_axis, error_axis):
            axis.set_xlabel("$x$", fontsize=LABEL_SIZE)
            axis.set_ylabel("$t$", fontsize=LABEL_SIZE)
            axis.tick_params(axis="both", labelsize=TICK_SIZE)
        prediction_axis.annotate(
            title, xy=(-0.23, 0.5), xycoords="axes fraction",
            ha="center", va="center", rotation=90,
            fontsize=TITLE_SIZE, fontweight="bold",
        )
        for axis, image in (
            (prediction_axis, solution_image), (error_axis, error_image)
        ):
            colorbar = figure.colorbar(
                image, ax=axis, pad=0.02, fraction=0.05
            )
            colorbar.ax.tick_params(labelsize=TICK_SIZE)
            colorbar.ax.yaxis.get_offset_text().set_fontsize(TICK_SIZE)
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


def load_training(path: Path) -> dict[str, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "epoch": data["epoch"],
            "loss": data["loss"],
            "initial_epoch": data["initial_epoch"] if "initial_epoch" in data.files else None,
            "initial_loss": data["initial_loss"] if "initial_loss" in data.files else None,
        }


def plot_training(keys: tuple[str, ...], output_path: Path) -> None:
    figure, axes = plt.subplots(
        len(keys), 1, figsize=(13, 4.2 * len(keys)), squeeze=False,
        constrained_layout=True,
    )
    for row, key in enumerate(keys):
        stem, title = SOLVERS[key]
        axis = axes[row, 0]
        path = HERE / f"{stem}_training_data.npz"
        if not path.exists():
            axis.text(
                0.5, 0.5, "Saved training data unavailable", ha="center",
                va="center", transform=axis.transAxes,
            )
            axis.set_title(title, fontsize=TITLE_SIZE)
            axis.set_axis_off()
            continue
        record = load_training(path)
        if record["initial_epoch"] is not None:
            axis.semilogy(
                record["initial_epoch"], record["initial_loss"],
                label="Initial-condition pretraining", linewidth=2,
            )
        label = (
            "Total soft-constraint loss" if key.endswith("soft")
            else "PDE-residual training loss"
        )
        axis.semilogy(record["epoch"], record["loss"], label=label, linewidth=2)
        axis.set_title(title, fontsize=TITLE_SIZE)
        axis.set_xlabel("Epoch", fontsize=LABEL_SIZE)
        axis.set_ylabel("Loss", fontsize=LABEL_SIZE)
        axis.tick_params(labelsize=TICK_SIZE)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=LEGEND_SIZE)
    figure.savefig(output_path, format="pdf", dpi=650, bbox_inches="tight")


def main(solver: str, show: bool) -> None:
    selected = tuple(SOLVERS) if solver == "all" else (solver,)
    comparison_name = (
        "burgers_method_comparison.pdf" if solver == "all"
        else f"{SOLVERS[solver][0]}_comparison.pdf"
    )
    comparison_path = HERE / comparison_name
    plot_comparison(selected, comparison_path)
    print(f"Saved comparison figure: {comparison_path}")

    training_name = (
        "burgers_loss_evolution.pdf" if solver == "all"
        else f"{SOLVERS[solver][0]}_loss_evolution.pdf"
    )
    training_path = HERE / training_name
    plot_training(selected, training_path)
    print(f"Saved loss-evolution figure: {training_path}")

    for key in selected:
        stem, _ = SOLVERS[key]
        path = HERE / f"{stem}_solution_data.npz"
        if not path.exists():
            continue
        record = load_solution(path)
        timing = (
            f"{float(record['wall_time']):.3f} s"
            if np.isfinite(float(record["wall_time"])) else "unavailable"
        )
        print(
            f"{key}: validation relative L2="
            f"{float(record['l2_error']):.6e}, time={timing}"
        )
    if show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=("all", *SOLVERS), default="all")
    parser.add_argument("--show", action="store_true")
    arguments = parser.parse_args()
    main(arguments.solver, arguments.show)
