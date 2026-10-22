"""Regenerate L-shaped XPINN figures from saved NumPy data.

Run any solver once to create its ``*_data.npz`` files. Afterwards this
script can recreate the figures without loading a model or repeating training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LSHAPE_DIR = Path(__file__).resolve().parent.parent
SOLVER_SETTINGS = {
    "xpinn": {
        "solution_data": "xpinn_regular_solution_data.npz",
        "solution_plot": "xpinn_regular_solution.png",
        "training_data": "xpinn_regular_training_data.npz",
        "training_plot": "xpinn_regular_training.png",
        "solution_title": "XPINN solution",
        "exact_title": r"Exact solution: $e^x\sin(y)$",
        "loss_label": "XPINN loss",
        "history_title": "XPINN training history",
    },
    "bspline": {
        "solution_data": "bspline_xpinn_regular_solution_data.npz",
        "solution_plot": "bspline_xpinn_regular_solution.png",
        "training_data": "bspline_xpinn_regular_training_data.npz",
        "training_plot": "bspline_xpinn_regular_training.png",
        "solution_title": "B-spline XPINN solution",
        "exact_title": r"Exact solution: $e^x\sin(y)$",
        "loss_label": "B-spline XPINN loss",
        "history_title": "B-spline XPINN training history",
    },
    "xpinn_irregular": {
        "solution_data": "xpinn_irregular_solution_data.npz",
        "solution_plot": "xpinn_irregular_solution.png",
        "training_data": "xpinn_irregular_training_data.npz",
        "training_plot": "xpinn_irregular_training.png",
        "solution_title": "XPINN singular solution",
        "exact_title": r"Exact singular solution: $r^{2/3}\sin(2\theta/3)$",
        "loss_label": "XPINN loss",
        "history_title": "XPINN singular-solution training history",
    },
    "bspline_irregular": {
        "solution_data": "bspline_xpinn_irregular_solution_data.npz",
        "solution_plot": "bspline_xpinn_irregular_solution.png",
        "training_data": "bspline_xpinn_irregular_training_data.npz",
        "training_plot": "bspline_xpinn_irregular_training.png",
        "solution_title": "B-spline XPINN singular solution",
        "exact_title": r"Exact singular solution: $r^{2/3}\sin(2\theta/3)$",
        "loss_label": "B-spline XPINN loss",
        "history_title": "B-spline XPINN singular-solution training history",
    },
    "xpinn_refined_irregular": {
        "solution_data": "xpinn_refined_irregular_solution_data.npz",
        "solution_plot": "xpinn_refined_irregular_solution.png",
        "training_data": "xpinn_refined_irregular_training_data.npz",
        "training_plot": "xpinn_refined_irregular_training.png",
        "solution_title": "Refined XPINN singular solution",
        "exact_title": r"Exact singular solution: $r^{2/3}\sin(2\theta/3)$",
        "loss_label": "XPINN loss",
        "history_title": "Refined XPINN training history",
    },
    "bspline_refined_irregular": {
        "solution_data": "bspline_xpinn_refined_irregular_solution_data.npz",
        "solution_plot": "bspline_xpinn_refined_irregular_solution.png",
        "training_data": "bspline_xpinn_refined_irregular_training_data.npz",
        "training_plot": "bspline_xpinn_refined_irregular_training.png",
        "solution_title": "Refined B-spline XPINN singular solution",
        "exact_title": r"Exact singular solution: $r^{2/3}\sin(2\theta/3)$",
        "loss_label": "B-spline XPINN loss",
        "history_title": "Refined B-spline XPINN training history",
    },
}


def contour_levels(values: np.ndarray) -> np.ndarray:
    """Return robust common contour levels for all three patches."""
    lower = float(values.min())
    upper = float(values.max())
    if np.isclose(lower, upper):
        lower -= 1.0e-12
        upper += 1.0e-12
    return np.linspace(lower, upper, 51)


def postprocessing_directory(solver: str) -> Path:
    """Return the family-specific directory containing a solver's data."""
    if "refined_irregular" in solver:
        family = "RefinedIrregular"
    elif "irregular" in solver:
        family = "Irregular"
    else:
        family = "Regular"
    return LSHAPE_DIR / family / "postprocessing"


def plot_solution(solver: str, *, show: bool = False) -> Path:
    """Recreate one solution/exact/error figure from its saved arrays."""
    settings = SOLVER_SETTINGS[solver]
    output_directory = postprocessing_directory(solver)
    data_path = output_directory / settings["solution_data"]
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing {data_path}. Run the {solver} solver once first."
        )

    with np.load(data_path, allow_pickle=False) as data:
        x = data["x"]
        y = data["y"]
        prediction = data["prediction"]
        exact = data["exact"]
        absolute_error = data["absolute_error"]
        validation_relative_l2_error = (
            float(data["validation_relative_l2_error"])
            if "validation_relative_l2_error" in data.files
            else None
        )

    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for axis, values, title, color_map in (
        (axes[0], prediction, settings["solution_title"], "viridis"),
        (axes[1], exact, settings["exact_title"], "viridis"),
        (axes[2], absolute_error, "Absolute error", "magma"),
    ):
        levels = contour_levels(values)
        contour = None
        for patch_index in range(x.shape[0]):
            contour = axis.contourf(
                x[patch_index],
                y[patch_index],
                values[patch_index],
                levels=levels,
                cmap=color_map,
            )
        figure.colorbar(contour, ax=axis)
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.set_xlabel("x")
        axis.set_ylabel("y")

    output_path = output_directory / settings["solution_plot"]
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)
    print(f"Recreated: {output_path}")
    if validation_relative_l2_error is None:
        print(
            f"Validation relative L2 error ({solver}): not stored in this "
            "legacy solution archive"
        )
    else:
        print(
            f"Validation relative L2 error ({solver}): "
            f"{validation_relative_l2_error:.6e}"
        )
    return output_path


def plot_training_history(solver: str, *, show: bool = False) -> Path:
    """Recreate one loss-history figure from its saved arrays."""
    settings = SOLVER_SETTINGS[solver]
    output_directory = postprocessing_directory(solver)
    data_path = output_directory / settings["training_data"]
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing {data_path}. Run the {solver} solver once first."
        )

    with np.load(data_path, allow_pickle=False) as data:
        epochs = data["epoch"]
        loss = data["loss"]

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.semilogy(epochs, loss)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(settings["loss_label"])
    axis.set_title(settings["history_title"])
    figure.tight_layout()
    output_path = output_directory / settings["training_plot"]
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)
    print(f"Recreated: {output_path}")
    return output_path


def main(solver: str = "all", show: bool = False) -> None:
    selected = SOLVER_SETTINGS if solver == "all" else (solver,)
    for solver_name in selected:
        try:
            plot_solution(solver_name, show=show)
            plot_training_history(solver_name, show=show)
        except FileNotFoundError as error:
            if solver != "all":
                raise
            print(f"Skipped {solver_name}: {error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recreate solver figures without rerunning training"
    )
    parser.add_argument(
        "--solver",
        choices=(
            "all",
            "xpinn",
            "bspline",
            "xpinn_irregular",
            "bspline_irregular",
            "xpinn_refined_irregular",
            "bspline_refined_irregular",
        ),
        default="all",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures in addition to saving them",
    )
    arguments = parser.parse_args()
    main(arguments.solver, arguments.show)
