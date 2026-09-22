import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from kan import KAN

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training import train


HYPERPARAMETER_PATH = (
    Path(__file__).resolve().parent / "results" / "best_kan_hyperparameters.json"
)
DEFAULT_DEVICE = "gpu"  # Choose "cpu" or "gpu".


def create_kan(n_layers, n_neurons, grid, spline_order, device, seed=42):
    """Construct a pykan model without checkpoint or symbolic overhead."""
    width = [2] + [n_neurons] * n_layers + [1]
    return KAN(
        width=width,
        grid=grid,
        k=spline_order,
        seed=seed,
        device=device,
        auto_save=False,
        save_act=False,
        symbolic_enabled=False,
    ).to(device)


def source_function(x, y):
    return -2 * torch.pi**2 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)


def poisson_residual_loss(model, points):
    """Mean squared strong-form Poisson residual at the supplied points."""
    with torch.enable_grad():
        coordinates = points.detach().clone().requires_grad_(True)
        solution = model(coordinates).squeeze(-1)
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
    return model(boundary_points).squeeze(-1).square().mean()


def generate_boundary_points(points_per_edge, device):
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


def load_best_kan_hyperparameters(path=HYPERPARAMETER_PATH):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"KAN hyperparameter file not found: {path}. "
            "Run Poisson/tune_kan.py first."
        )
    with path.open("r", encoding="utf-8") as file:
        result = json.load(file)
    try:
        params = result["best_params"]
        values = (
            int(params["n_layers"]),
            int(params["n_neurons"]),
            int(params["grid"]),
            int(params["spline_order"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid KAN hyperparameter file: {path}") from error
    if any(value < 1 for value in values):
        raise ValueError("All KAN hyperparameters must be positive")
    return values


def select_device(device_name):
    if device_name == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU was requested, but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cpu")


def main(device_name=DEFAULT_DEVICE):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = select_device(device_name)
    print(f"Using device: {device}")

    n_layers, n_neurons, grid, spline_order = load_best_kan_hyperparameters()
    print(
        "Using tuned KAN: "
        f"n_layers={n_layers}, n_neurons={n_neurons}, "
        f"grid={grid}, spline_order={spline_order}"
    )
    model = create_kan(
        n_layers, n_neurons, grid, spline_order, device=device, seed=42
    )
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
        epochs=20000,
        scheduler=scheduler,
        early_stopping_patience=500,
        min_delta=1e-8,
        print_every=100,
    )

    x = y = np.linspace(0, 1, 100)
    grid_x, grid_y = np.meshgrid(x, y)
    evaluation_points = torch.tensor(
        np.column_stack((grid_x.ravel(), grid_y.ravel())),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        prediction = (
            model(evaluation_points).squeeze(-1).cpu().numpy().reshape(grid_x.shape)
        )
    exact = np.sin(np.pi * grid_x) * np.sin(np.pi * grid_y)
    error = np.abs(prediction - exact)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, values, title, color_map in zip(
        axes,
        (prediction, exact, error),
        ("PI-KAN Solution", "Exact Solution", "Absolute Error"),
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
    plt.title("PI-KAN Loss Curve", fontsize=14)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the 2D Poisson PI-KAN")
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default=DEFAULT_DEVICE,
        help=f"Training device (default: {DEFAULT_DEVICE})",
    )
    arguments = parser.parse_args()
    main(arguments.device)
