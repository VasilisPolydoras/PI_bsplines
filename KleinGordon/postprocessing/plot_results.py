"""Recreate Klein--Gordon figures from NPZ archives without retraining."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from KleinGordon._common import plot_saved_solution, plot_saved_training

SOLVERS = {
    "bspline-soft": ("bspline_klein_gordon_soft", "Klein-Gordon B-spline (soft)"),
    "bspline-hard": ("bspline_klein_gordon_hard", "Klein-Gordon B-spline (hard)"),
    "pinn-soft": ("pinn_klein_gordon_soft", "Klein-Gordon PINN (soft)"),
    "pinn-hard": ("pinn_klein_gordon_hard", "Klein-Gordon PINN (hard)"),
    "pikan-soft": ("pikan_klein_gordon_soft", "Klein-Gordon PIKAN (soft)"),
    "pikan-hard": ("pikan_klein_gordon_hard", "Klein-Gordon PIKAN (hard)"),
}


def reproduce(name: str) -> None:
    stem, title = SOLVERS[name]
    solution_data = SCRIPT_DIR / f"{stem}_solution_data.npz"
    training_data = SCRIPT_DIR / f"{stem}_training_data.npz"
    if not solution_data.exists() or not training_data.exists():
        raise FileNotFoundError(f"Run the {name} solver first; saved NPZ data are missing")
    plot_saved_solution(solution_data, SCRIPT_DIR / f"{stem}_solution.png", title)
    plot_saved_training(training_data, SCRIPT_DIR / f"{stem}_training.png", title)
    with np.load(solution_data, allow_pickle=False) as data:
        error = float(data["validation_relative_l2_error"])
    print(f"{name} validation relative L2 error: {error:.6e}")


def main(solver: str, show: bool = False) -> None:
    names = SOLVERS if solver == "all" else (solver,)
    for name in names:
        try:
            reproduce(name)
        except FileNotFoundError as error:
            if solver != "all":
                raise
            print(f"Skipped {name}: {error}")
    if show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recreate Klein-Gordon plots")
    parser.add_argument("--solver", choices=("all", *SOLVERS), default="all")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the recreated figures interactively after saving them",
    )
    args = parser.parse_args()
    main(args.solver, args.show)
