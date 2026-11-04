"""Recreate publication-ready Klein--Gordon comparisons without retraining."""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE = 22, 20, 17, 16
MAX_SCATTER_POINTS = 60000
plt.rcParams.update({"font.size": 18})

SOLVERS = {
    "bspline-hard": ("bspline_klein_gordon_hard", "PI-TP-BsA"),
    "pinn-soft": ("pinn_klein_gordon_soft", "PINN soft"),
    "pinn-hard": ("pinn_klein_gordon_hard", "PINN hard"),
    "pikan-soft": ("pikan_klein_gordon_soft", "PIKAN soft"),
    "pikan-hard": ("pikan_klein_gordon_hard", "PIKAN hard"),
}


def load_solution(path: Path) -> dict[str, np.ndarray | float]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "x": data["x"], "y": data["y"], "box_time": data["box_time"],
            "prediction": data["volume_prediction"], "error": data["volume_error"],
            "l2": float(data["validation_relative_l2_error"]),
            "wall_time": (float(data["training_wall_time_seconds"])
                          if "training_wall_time_seconds" in data.files else float("nan")),
        }


def plot_comparison(keys: tuple[str, ...], output: Path) -> None:
    records = {}
    for key in keys:
        path = HERE / f"{SOLVERS[key][0]}_solution_data.npz"
        if path.exists():
            records[key] = load_solution(path)
        else:
            print(f"Missing solution data for {key}: {path}")
    if not records:
        raise FileNotFoundError("No saved Klein--Gordon solution data were found")
    figure = plt.figure(figsize=(18, 7.0 * len(keys)), constrained_layout=True)
    for row, key in enumerate(keys):
        title = SOLVERS[key][1]
        if key not in records:
            for column in range(2):
                axis = figure.add_subplot(len(keys), 2, 2 * row + column + 1)
                axis.text(0.5, 0.5, "Saved data unavailable", ha="center", va="center")
                axis.set_axis_off()
            continue
        record = records[key]
        time, y, x = np.meshgrid(record["box_time"], record["y"], record["x"], indexing="ij")
        stride = max(1, int(np.ceil(x.size / MAX_SCATTER_POINTS)))
        indices = slice(None, None, stride)
        for column, (values, heading, cmap) in enumerate((
            (record["prediction"], "Predicted solution", "viridis"),
            (record["error"], "Absolute error", "magma"),
        )):
            axis = figure.add_subplot(len(keys), 2, 2 * row + column + 1, projection="3d")
            cloud = axis.scatter(
                time.ravel()[indices], x.ravel()[indices], y.ravel()[indices],
                c=np.asarray(values).ravel()[indices], s=1.2, cmap=cmap,
                linewidths=0, rasterized=True,
            )
            axis.set_xlabel("$t$", fontsize=LABEL_SIZE)
            axis.set_ylabel("$x$", fontsize=LABEL_SIZE)
            axis.set_zlabel("$y$", fontsize=LABEL_SIZE)
            axis.tick_params(labelsize=TICK_SIZE)
            axis.set_title(heading if row == 0 else "", fontsize=TITLE_SIZE, pad=12)
            bar = figure.colorbar(cloud, ax=axis, shrink=0.62, pad=0.08)
            bar.ax.tick_params(labelsize=TICK_SIZE)
            if column == 0:
                axis.text2D(-0.12, 0.5, title, transform=axis.transAxes,
                            rotation=90, va="center", ha="center",
                            fontsize=TITLE_SIZE, fontweight="bold")
            else:
                axis.text2D(0.96, 0.94, rf"relative $L^2={record['l2']:.2e}$",
                            transform=axis.transAxes, ha="right", va="top",
                            fontsize=TICK_SIZE,
                            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    figure.savefig(output, format="pdf", dpi=650, bbox_inches="tight")


def plot_losses(keys: tuple[str, ...], output: Path) -> None:
    figure, axes = plt.subplots(len(keys), 1, figsize=(13, 4.2 * len(keys)),
                                squeeze=False, constrained_layout=True)
    for row, key in enumerate(keys):
        path = HERE / f"{SOLVERS[key][0]}_training_data.npz"
        axis = axes[row, 0]
        if not path.exists():
            axis.text(0.5, 0.5, "Saved training data unavailable", ha="center", va="center")
            axis.set_axis_off()
            continue
        with np.load(path, allow_pickle=False) as data:
            if key == "bspline-hard":
                for prefix, label in (
                    ("boundary", "Boundary preprocessing"),
                    ("initial_value", "Initial-value preprocessing"),
                    ("initial_velocity", "Initial-velocity preprocessing"),
                ):
                    epoch_key, loss_key = f"{prefix}_epoch", f"{prefix}_loss"
                    if epoch_key in data.files and loss_key in data.files:
                        axis.semilogy(
                            data[epoch_key], data[loss_key],
                            linewidth=2, label=label,
                        )
            label = "Total soft-constraint loss" if key.endswith("soft") else "PDE-residual loss"
            axis.semilogy(data["epoch"], data["loss"], linewidth=2, label=label)
        axis.set_title(SOLVERS[key][1], fontsize=TITLE_SIZE)
        axis.set_xlabel("Epoch", fontsize=LABEL_SIZE)
        axis.set_ylabel("Loss", fontsize=LABEL_SIZE)
        axis.tick_params(labelsize=TICK_SIZE)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=LEGEND_SIZE)
    figure.savefig(output, format="pdf", dpi=650, bbox_inches="tight")


def main(name: str, show: bool) -> None:
    selected = tuple(SOLVERS) if name == "all" else (name,)
    groups = ({
        "klein_gordon_bspline_comparison.pdf": ("bspline-hard",),
        "klein_gordon_pinn_comparison.pdf": ("pinn-soft", "pinn-hard"),
        "klein_gordon_pikan_comparison.pdf": ("pikan-soft", "pikan-hard"),
    } if name == "all" else {f"{SOLVERS[name][0]}_comparison.pdf": (name,)})
    for filename, keys in groups.items():
        try:
            plot_comparison(keys, HERE / filename)
            print(f"Saved comparison figure: {HERE / filename}")
        except FileNotFoundError as error:
            print(f"Skipped {filename}: {error}")
    loss_name = "klein_gordon_loss_evolution.pdf" if name == "all" else f"{SOLVERS[name][0]}_loss_evolution.pdf"
    plot_losses(selected, HERE / loss_name)
    print(f"Saved loss-evolution figure: {HERE / loss_name}")
    for key in selected:
        path = HERE / f"{SOLVERS[key][0]}_solution_data.npz"
        if path.exists():
            record = load_solution(path)
            print(f"{key}: validation relative L2={record['l2']:.6e}, time={record['wall_time']:.3f} s")
    if show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replot saved Klein--Gordon results")
    parser.add_argument("--solver", choices=("all", *SOLVERS), default="all")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    main(args.solver, args.show)
