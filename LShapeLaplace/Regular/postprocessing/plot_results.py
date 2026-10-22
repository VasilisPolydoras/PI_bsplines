"""Recreate regular L-shape figures from saved data without retraining."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LShapeLaplace.postprocessing.plot_results import (
    plot_solution,
    plot_training_history,
)

SOLVERS = {"bspline": "bspline", "xpinn": "xpinn"}


def main(solver: str = "all", show: bool = False) -> None:
    selected = SOLVERS if solver == "all" else (solver,)
    for short_name in selected:
        solver_name = SOLVERS[short_name]
        try:
            plot_solution(solver_name, show=show)
            plot_training_history(solver_name, show=show)
        except FileNotFoundError as error:
            if solver != "all":
                raise
            print(f"Skipped {short_name}: {error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recreate regular L-shape plots from saved NumPy data"
    )
    parser.add_argument(
        "--solver", choices=("all", "bspline", "xpinn"), default="all"
    )
    parser.add_argument("--show", action="store_true")
    arguments = parser.parse_args()
    main(arguments.solver, arguments.show)
