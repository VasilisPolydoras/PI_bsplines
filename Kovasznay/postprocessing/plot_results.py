"""Recreate Kovasznay PINN and PI-KAN plots without retraining."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DIRECTORY = Path(__file__).resolve().parent
SOLVERS = {
    "pinn-soft": ("pinn_kovasznay_soft", "Soft-constraint PINN", "PINN"),
    "pinn-hard": ("pinn_kovasznay_hard", "Hard-constraint PINN", "PINN"),
    "pikan-soft": ("pikan_kovasznay_soft", "Soft-constraint PIKAN", "PIKAN"),
    "pikan-hard": ("pikan_kovasznay_hard", "Hard-constraint PIKAN", "PIKAN"),
}


def plot_solution(solver: str, show: bool = False) -> Path:
    stem, label, _ = SOLVERS[solver]
    data_path = DIRECTORY / f"{stem}_solution_data.npz"
    output_path = DIRECTORY / f"{stem}_solution.png"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}. Run the solver first.")
    with np.load(data_path, allow_pickle=False) as data:
        x, y = data["x"], data["y"]
        prediction, reference = data["prediction"], data["reference"]
        error = data["absolute_error"]
        errors = data["validation_relative_l2_errors"]
    figure, axes = plt.subplots(3, 3, figsize=(14, 12))
    for row, name in enumerate(("u", "v", "p")):
        titles = (f"{label} {name}", f"Exact {name}", f"Error {name}")
        for column, (values, title) in enumerate(zip(
            (prediction[:, :, row], reference[:, :, row], error[:, :, row]), titles
        )):
            image = axes[row, column].imshow(
                values, extent=[x[0], x[-1], y[0], y[-1]],
                origin="lower", aspect="auto",
            )
            axes[row, column].set_title(title)
            axes[row, column].set_xlabel("x")
            axes[row, column].set_ylabel("y")
            figure.colorbar(image, ax=axes[row, column])
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)
    print(f"Recreated: {output_path}")
    for name, value in zip(("u", "v", "p"), errors):
        print(f"Validation relative L2 error {name}: {value:.6e}")
    return output_path


def plot_training(solver: str, show: bool = False) -> Path:
    stem, label, loss_label = SOLVERS[solver]
    data_path = DIRECTORY / f"{stem}_training_data.npz"
    output_path = DIRECTORY / f"{stem}_training.png"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}. Run the solver first.")
    with np.load(data_path, allow_pickle=False) as data:
        epoch, loss = data["epoch"], data["loss"]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.semilogy(epoch, loss)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(f"{loss_label} loss")
    axis.set_title(f"Kovasznay {label} training")
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
            plot_solution(name, show)
            plot_training(name, show)
        except FileNotFoundError as error:
            if solver != "all":
                raise
            print(f"Skipped {name}: {error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recreate Kovasznay plots without retraining"
    )
    parser.add_argument(
        "--solver", choices=("all", *SOLVERS), default="all"
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    main(args.solver, args.show)
