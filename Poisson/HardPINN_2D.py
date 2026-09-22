import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Poisson.PINN_2D import PINN, poisson_residual_loss, select_device
from training import train


HYPERPARAMETER_PATH = (
    Path(__file__).resolve().parent
    / "results"
    / "best_hard_pinn_hyperparameters.json"
)
DEFAULT_DEVICE = "gpu"  # Choose "cpu" or "gpu".


class HardConstraintPINN(PINN):
    """PINN satisfying homogeneous Dirichlet conditions by construction."""

    def forward(self, points):
        x = points[:, 0]
        y = points[:, 1]
        distance_function = x * (1 - x) * y * (1 - y)
        unconstrained_solution = self.network(points).squeeze(-1)
        return distance_function * unconstrained_solution


def load_best_hard_pinn_hyperparameters(path=HYPERPARAMETER_PATH):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Hard-PINN hyperparameter file not found: {path}. "
            "Run Poisson/tune_hard_pinn.py first."
        )
    with path.open("r", encoding="utf-8") as file:
        result = json.load(file)
    try:
        n_layers = int(result["best_params"]["n_layers"])
        n_neurons = int(result["best_params"]["n_neurons"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid Hard-PINN hyperparameter file: {path}") from error
    if n_layers < 1 or n_neurons < 1:
        raise ValueError("n_layers and n_neurons must be positive")
    return n_layers, n_neurons


def main(device_name=DEFAULT_DEVICE):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = select_device(device_name)
    print(f"Using device: {device}")

    n_layers, n_neurons = load_best_hard_pinn_hyperparameters()
    print(
        f"Using tuned hard-constraint PINN: "
        f"n_layers={n_layers}, n_neurons={n_neurons}"
    )
    model = HardConstraintPINN(n_layers, n_neurons).to(device)
    collocation_points = torch.rand(5000, 2, device=device)

    def loss_fn():
        return poisson_residual_loss(model, collocation_points)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=50, min_lr=1e-5
    )
    loss_history = train(
        parameters=model.parameters(),
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=20000,
        scheduler=scheduler,
        early_stopping_patience=500,
        min_delta=1e-8,
        print_every=100,
    )

    x = y = np.linspace(0, 1, 100)
    grid_x, grid_y = np.meshgrid(x, y)
    grid = torch.tensor(
        np.column_stack((grid_x.ravel(), grid_y.ravel())),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        prediction = model(grid).cpu().numpy().reshape(grid_x.shape)
    exact = np.sin(np.pi * grid_x) * np.sin(np.pi * grid_y)
    error = np.abs(prediction - exact)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, values, title, color_map in zip(
        axes,
        (prediction, exact, error),
        ("Hard-Constraint PINN", "Exact Solution", "Absolute Error"),
        ("viridis", "viridis", "hot"),
    ):
        image = axis.imshow(
            values, extent=[0, 1, 0, 1], origin="lower", cmap=color_map
        )
        axis.set_title(title, fontsize=14)
        axis.set_xlabel("x", fontsize=14)
        axis.set_ylabel("y", fontsize=14)
        fig.colorbar(image, ax=axis)
    plt.tight_layout()
    plt.show()

    plt.plot(np.asarray(loss_history))
    plt.yscale("log")
    plt.title("Hard-Constraint PINN Loss Curve", fontsize=14)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the hard-constraint 2D Poisson PINN"
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default=DEFAULT_DEVICE,
        help=f"Training device (default: {DEFAULT_DEVICE})",
    )
    arguments = parser.parse_args()
    main(arguments.device)
