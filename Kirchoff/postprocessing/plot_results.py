"""Recreate Kirchhoff solution and training plots without retraining."""

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

from Kirchoff._common import plot_solution, plot_training

SOLVERS = {
    "bspline-hard": ("bspline_kirchhoff_hard", "B-spline (hard)"),
    "pinn-soft": ("pinn_kirchhoff_soft", "PINN (soft)"),
    "pinn-hard": ("pinn_kirchhoff_hard", "PINN (hard)"),
    "pikan-soft": ("pikan_kirchhoff_soft", "PIKAN (soft)"),
    "pikan-hard": ("pikan_kirchhoff_hard", "PIKAN (hard)"),
}


def reproduce(name: str) -> None:
    stem, title = SOLVERS[name]
    solution = SCRIPT_DIR / f"{stem}_solution_data.npz"
    training = SCRIPT_DIR / f"{stem}_training_data.npz"
    if not solution.exists() or not training.exists():
        raise FileNotFoundError(f"Run {name} first; its NPZ archives are missing")
    plot_solution(solution, SCRIPT_DIR / f"{stem}_solution.png", title)
    plot_training(training, SCRIPT_DIR / f"{stem}_training.png", title)
    with np.load(solution, allow_pickle=False) as data:
        print(
            f"{name} validation relative L2 error: "
            f"{float(data['validation_relative_l2_error']):.6e}"
        )


def main(solver: str, show: bool) -> None:
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=("all", *SOLVERS), default="all")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    main(args.solver, args.show)

