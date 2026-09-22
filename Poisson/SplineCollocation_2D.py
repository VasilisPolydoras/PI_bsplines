import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Poisson.Bspline_2D import (
    bsplines,
    build_poisson_system,
    load_best_bspline_hyperparameters,
    select_device,
)


DEFAULT_DEVICE = "gpu"  # Choose "cpu" or "gpu".


def interior_greville_points(model):
    """Return Greville abscissae associated with the interior B-splines."""
    degree = model.degree
    knots = model.knot_vector
    indices = range(1, model.n_basis - 1)
    return torch.stack(
        [knots[index + 1:index + degree + 1].mean() for index in indices]
    )


def tensor_product_points(one_dimensional_points):
    """Construct the Cartesian product of a one-dimensional point set."""
    x, y = torch.meshgrid(
        one_dimensional_points,
        one_dimensional_points,
        indexing="ij",
    )
    return torch.stack((x.reshape(-1), y.reshape(-1)), dim=1)


def solve_spline_collocation(n_basis, degree, device):
    """Solve the Poisson equation by direct B-spline collocation."""
    model = bsplines(n_basis, degree, device=device)
    greville_points = interior_greville_points(model)
    collocation_points = tensor_product_points(greville_points)
    laplacian_matrix, source = build_poisson_system(model, collocation_points)

    # There is one equation per interior tensor-product coefficient.
    coefficients = torch.linalg.solve(laplacian_matrix, source)
    with torch.no_grad():
        model.coeffs.zero_()
        model.coeffs[1:-1, 1:-1].copy_(
            coefficients.reshape(n_basis - 2, n_basis - 2)
        )

    residual = laplacian_matrix @ coefficients - source
    return model, collocation_points, residual


def main(device_name=DEFAULT_DEVICE):
    device = select_device(device_name)
    print(f"Using device: {device}")

    n_basis, degree = load_best_bspline_hyperparameters()
    print(f"Using B-splines: n_basis={n_basis}, degree={degree}")
    model, collocation_points, residual = solve_spline_collocation(
        n_basis, degree, device
    )
    print(f"Linear-system size: {collocation_points.shape[0]} x {collocation_points.shape[0]}")
    print(f"Maximum collocation residual: {residual.abs().max().item():.6e}")

    x = y = np.linspace(0, 1, 100)
    grid_x, grid_y = np.meshgrid(x, y)
    evaluation_points = torch.tensor(
        np.column_stack((grid_x.ravel(), grid_y.ravel())),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        prediction = (
            model.forward(evaluation_points).cpu().numpy().reshape(grid_x.shape)
        )
    exact = np.sin(np.pi * grid_x) * np.sin(np.pi * grid_y)
    error = np.abs(prediction - exact)
    relative_l2_error = np.linalg.norm(prediction - exact) / np.linalg.norm(exact)
    print(f"Relative L2 error: {relative_l2_error:.6e}")
    print(f"Maximum absolute error: {error.max():.6e}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, values, title, color_map in zip(
        axes,
        (prediction, exact, error),
        ("Spline Collocation", "Exact Solution", "Absolute Error"),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve the 2D Poisson equation by B-spline collocation"
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default=DEFAULT_DEVICE,
        help=f"Linear-solve device (default: {DEFAULT_DEVICE})",
    )
    arguments = parser.parse_args()
    main(arguments.device)
