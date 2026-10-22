"""Recreate Burgers solver figures from saved data without retraining."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


POSTPROCESSING_DIR = Path(__file__).resolve().parent
SOLVERS = {
    "bspline": {
        "stem": "bspline_burgers",
        "prediction_title": "B-spline prediction",
        "history_label": "Interior PDE training",
        "loss_label": "Loss",
        "training_title": "Viscous Burgers B-spline Loss",
    },
    "pinn-soft": {
        "stem": "pinn_burgers_soft",
        "prediction_title": "Soft-constraint PINN prediction",
        "history_label": "Training loss",
        "loss_label": "PINN loss",
        "training_title": "Burgers PINN (soft) training",
    },
    "pinn-hard": {
        "stem": "pinn_burgers_hard",
        "prediction_title": "Hard-constraint PINN prediction",
        "history_label": "Training loss",
        "loss_label": "PINN loss",
        "training_title": "Burgers PINN (hard) training",
    },
    "pikan-soft": {
        "stem": "pikan_burgers_soft",
        "prediction_title": "Soft-constraint PIKAN prediction",
        "history_label": "Training loss",
        "loss_label": "PIKAN loss",
        "training_title": "Burgers PIKAN (soft) training",
    },
    "pikan-hard": {
        "stem": "pikan_burgers_hard",
        "prediction_title": "Hard-constraint PIKAN prediction",
        "history_label": "Training loss",
        "loss_label": "PIKAN loss",
        "training_title": "Burgers PIKAN (hard) training",
    },
}


def plot_solution(solver: str = "bspline", *, show: bool = False) -> Path:
    """Recreate the prediction/reference/error figure from its archive."""
    settings = SOLVERS[solver]
    data_path = POSTPROCESSING_DIR / f"{settings['stem']}_solution_data.npz"
    output_path = POSTPROCESSING_DIR / f"{settings['stem']}_solution.png"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing {data_path}. Run the corresponding solver first."
        )
    with np.load(data_path, allow_pickle=False) as data:
        x = data["x"]
        times = data["time"]
        prediction = data["prediction"]
        reference = data["reference"]
        absolute_error = data["absolute_error"]
        relative_error = float(data["validation_relative_l2_error"])

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    extent = [float(x[0]), float(x[-1]), float(times[0]), float(times[-1])]
    for axis, values, title in zip(
        axes,
        (prediction, reference, absolute_error),
        (settings["prediction_title"], "Numerical reference", "Absolute error"),
    ):
        image = axis.imshow(
            values, extent=extent, origin="lower", aspect="auto"
        )
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("t")
        figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)
    print(f"Recreated: {output_path}")
    print(f"Validation relative L2 error: {relative_error:.6e}")
    return output_path


def plot_training_history(
    solver: str = "bspline", *, show: bool = False
) -> Path:
    """Recreate the interior PDE-loss figure from its archive."""
    settings = SOLVERS[solver]
    data_path = POSTPROCESSING_DIR / f"{settings['stem']}_training_data.npz"
    output_path = POSTPROCESSING_DIR / f"{settings['stem']}_training.png"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing {data_path}. Run the corresponding solver first."
        )
    with np.load(data_path, allow_pickle=False) as data:
        epochs = data["epoch"]
        loss = data["loss"]

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(epochs, loss, label=settings["history_label"])
    axis.set_yscale("log")
    axis.set_xlabel("Epochs")
    axis.set_ylabel(settings["loss_label"])
    axis.set_title(settings["training_title"])
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)
    print(f"Recreated: {output_path}")
    return output_path


def main(solver: str = "all", show: bool = False) -> None:
    selected = SOLVERS if solver == "all" else (solver,)
    for name in selected:
        try:
            plot_solution(name, show=show)
            plot_training_history(name, show=show)
        except FileNotFoundError as error:
            if solver != "all":
                raise
            print(f"Skipped {name}: {error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recreate Burgers plots without rerunning training"
    )
    parser.add_argument(
        "--solver",
        choices=(
            "all", "bspline", "pinn-soft", "pinn-hard",
            "pikan-soft", "pikan-hard",
        ),
        default="all",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display figures in addition to saving them",
    )
    arguments = parser.parse_args()
    main(arguments.solver, arguments.show)
