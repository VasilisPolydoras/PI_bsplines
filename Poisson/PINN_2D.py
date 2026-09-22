import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training import train


HYPERPARAMETER_PATH = (
    Path(__file__).resolve().parent / "results" / "best_pinn_hyperparameters.json"
)
DEFAULT_DEVICE = "gpu"  # Choose "cpu" or "gpu".


class PINN(nn.Module):
    """Fully connected tanh network for the 2D Poisson problem."""

    def __init__(self, n_layers, n_neurons):
        super().__init__()
        layers = [nn.Linear(2, n_neurons), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers.extend((nn.Linear(n_neurons, n_neurons), nn.Tanh()))
        layers.append(nn.Linear(n_neurons, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, points):
        return self.network(points).squeeze(-1)


def source_function(x, y):
    return -2 * torch.pi**2 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)


def poisson_residual_loss(model, points):
    """Mean squared strong-form Poisson residual at the supplied points."""
    # training.train evaluates the monitoring loss under no_grad. Explicitly
    # re-enable gradients because coordinate derivatives are required here.
    with torch.enable_grad():
        coordinates = points.detach().clone().requires_grad_(True)
        solution = model(coordinates)
        gradient = torch.autograd.grad(
            solution,
            coordinates,
            grad_outputs=torch.ones_like(solution),
            create_graph=True,
        )[0]
        u_xx = torch.autograd.grad(
            gradient[:, 0],
            coordinates,
            grad_outputs=torch.ones_like(gradient[:, 0]),
            create_graph=True,
        )[0][:, 0]
        u_yy = torch.autograd.grad(
            gradient[:, 1],
            coordinates,
            grad_outputs=torch.ones_like(gradient[:, 1]),
            create_graph=True,
        )[0][:, 1]
        residual = u_xx + u_yy - source_function(
            coordinates[:, 0], coordinates[:, 1]
        )
        return residual.square().mean()


def boundary_loss(model, boundary_points):
    """Mean squared error for the homogeneous Dirichlet boundary condition."""
    return model(boundary_points).square().mean()


def generate_boundary_points(points_per_edge, device):
    """Generate points on all four edges of the unit square."""
    coordinate = torch.linspace(0, 1, points_per_edge, device=device).unsqueeze(1)
    zero = torch.zeros_like(coordinate)
    one = torch.ones_like(coordinate)
    return torch.cat(
        (
            torch.cat((coordinate, zero), dim=1),
            torch.cat((coordinate, one), dim=1),
            torch.cat((zero, coordinate), dim=1),
            torch.cat((one, coordinate), dim=1),
        ),
        dim=0,
    )


def load_best_pinn_hyperparameters(path=HYPERPARAMETER_PATH):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"PINN hyperparameter file not found: {path}. "
            "Run Poisson/tune_pinn.py first."
        )

    with path.open("r", encoding="utf-8") as file:
        result = json.load(file)

    try:
        n_layers = int(result["best_params"]["n_layers"])
        n_neurons = int(result["best_params"]["n_neurons"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid PINN hyperparameter file: {path}") from error

    if n_layers < 1 or n_neurons < 1:
        raise ValueError("n_layers and n_neurons must be positive")
    return n_layers, n_neurons


def select_device(device_name):
    """Map the user-facing CPU/GPU choice to a PyTorch device."""
    if device_name == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU was requested, but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cpu")


def main(device_name=DEFAULT_DEVICE):
    torch.manual_seed(42)
    device = select_device(device_name)
    print(f"Using device: {device}")
    n_layers, n_neurons = load_best_pinn_hyperparameters()
    print(f"Using tuned PINN: n_layers={n_layers}, n_neurons={n_neurons}")

    model = PINN(n_layers, n_neurons).to(device)
    collocation_points = torch.rand(5000, 2, device=device)
    boundary_points = generate_boundary_points(250, device)

    def loss_fn():
        return (
            poisson_residual_loss(model, collocation_points)
            + boundary_loss(model, boundary_points)
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=50, min_lr=1e-5
    )
    loss_history = train(
        parameters=model.parameters(),
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=200000,
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
        ("PINN Solution", "Exact Solution", "Absolute Error"),
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
    plt.title("PINN Loss Curve", fontsize=14)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the 2D Poisson PINN")
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default=DEFAULT_DEVICE,
        help=f"Training device (default: {DEFAULT_DEVICE})",
    )
    arguments = parser.parse_args()
    main(arguments.device)
