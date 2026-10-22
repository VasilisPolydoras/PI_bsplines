"""One-patch geometry-fitted B-spline Poisson solver.

The tensor-product spline lives on [0,1]^2 and the same regular polynomial map
as the PINN sends that square to the curved physical domain. Boundary rows and
columns of spline coefficients are identically zero, imposing homogeneous
Dirichlet data exactly on the mapped boundary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LShapeLaplace.Regular.Bspline_2D import basis_derivative_matrix
from IrregularPoisson.PINN_2D import (
    DEFAULT_DEVICE,
    LEARNING_RATE,
    POSTPROCESSING_DIR,
    SEED,
    exact_solution,
    geometry_map,
    geometry_operator_terms,
    mapped_laplacian_from_derivatives,
    sample_parameter_domain,
    select_device,
    set_seed,
    source_function,
    weighted_mean,
)
from training import train

N_BASIS = 18
DEGREE = 3
N_INTERIOR_POINTS = 5_000
EPOCHS = 20_000
RESULT_PATH = (
    Path(__file__).resolve().parent
    / "results"
    / "best_bspline_hyperparameters.json"
)


def open_uniform_knots(
    n_basis: int, degree: int, device: torch.device
) -> torch.Tensor:
    if degree < 2:
        raise ValueError("The strong Poisson formulation requires degree >= 2")
    if n_basis <= degree:
        raise ValueError("n_basis must be greater than degree")
    interior_count = n_basis - degree - 1
    interior = (
        torch.linspace(0.0, 1.0, interior_count + 2, device=device)[1:-1]
        if interior_count > 0
        else torch.empty(0, device=device)
    )
    return torch.cat(
        (
            torch.zeros(degree + 1, device=device),
            interior,
            torch.ones(degree + 1, device=device),
        )
    )


class bsplines(nn.Module):
    """Tensor-product solution spline on the geometry's parameter square."""

    def __init__(self, n_basis: int, degree: int, device: torch.device) -> None:
        super().__init__()
        if n_basis < 3:
            raise ValueError("n_basis must permit at least one interior row")
        self.n_basis = n_basis
        self.degree = degree
        self.register_buffer("knots", open_uniform_knots(n_basis, degree, device))
        self.interior_coefficients = nn.Parameter(
            torch.zeros(n_basis - 2, n_basis - 2, device=device)
        )

    def coefficient_tensor(self) -> torch.Tensor:
        """Pad trainable interior coefficients with exact zero boundary traces."""
        return torch.nn.functional.pad(self.interior_coefficients, (1, 1, 1, 1))

    def operators(self, parametric_points: torch.Tensor, max_order: int = 2):
        xi, eta = parametric_points[:, 0], parametric_points[:, 1]
        return {
            "x": tuple(
                basis_derivative_matrix(xi, self.knots, self.degree, order)
                for order in range(max_order + 1)
            ),
            "y": tuple(
                basis_derivative_matrix(eta, self.knots, self.degree, order)
                for order in range(max_order + 1)
            ),
        }

    def field(self, operators, dx: int = 0, dy: int = 0) -> torch.Tensor:
        return torch.einsum(
            "ni,ij,nj->n",
            operators["x"][dx],
            self.coefficient_tensor(),
            operators["y"][dy],
        )

    def forward(self, parametric_points: torch.Tensor) -> torch.Tensor:
        return self.field(self.operators(parametric_points, max_order=0))


def precompute_operators(
    model: bsplines, parametric_points: torch.Tensor, max_order: int = 2
):
    return model.operators(parametric_points, max_order=max_order)


def spline_derivatives(model: bsplines, operators):
    gradient = torch.stack(
        (model.field(operators, dx=1), model.field(operators, dy=1)), dim=1
    )
    return (
        gradient,
        model.field(operators, dx=2),
        model.field(operators, dx=1, dy=1),
        model.field(operators, dy=2),
    )


def poisson_residual(
    model: bsplines,
    parametric_points: torch.Tensor,
    operators,
    geometry_terms,
    source: torch.Tensor,
) -> torch.Tensor:
    laplacian = mapped_laplacian_from_derivatives(
        *spline_derivatives(model, operators), geometry_terms
    )
    return -laplacian - source


def poisson_residual_loss(
    model: bsplines,
    parametric_points: torch.Tensor,
    operators,
    geometry_terms,
    source: torch.Tensor,
) -> torch.Tensor:
    residual = poisson_residual(
        model, parametric_points, operators, geometry_terms, source
    )
    return weighted_mean(residual.square(), geometry_terms[0])


def relative_l2_error(
    model: bsplines,
    parametric_points: torch.Tensor,
    operators,
    geometry_terms,
) -> torch.Tensor:
    prediction = model.field(operators)
    exact = exact_solution(parametric_points)
    numerator = weighted_mean((prediction - exact).square(), geometry_terms[0])
    denominator = weighted_mean(exact.square(), geometry_terms[0])
    return torch.sqrt(numerator / denominator)


def load_hyperparameters(path: Path = RESULT_PATH) -> tuple[int, int]:
    if not path.exists():
        raise FileNotFoundError(
            f"No B-spline tuning result found at {path}. Run tune_bsplines.py "
            "first or use --no-use-tuned."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("best_params", payload)
    return int(params["n_basis"]), int(params["degree"])


def save_plots(
    model: bsplines, loss_history: list[float], resolution: int = 201
) -> None:
    device = model.knots.device
    axis = torch.linspace(0.0, 1.0, resolution, device=device)
    xi, eta = torch.meshgrid(axis, axis, indexing="ij")
    parametric = torch.stack((xi.flatten(), eta.flatten()), dim=1)
    physical = geometry_map(parametric)
    with torch.no_grad():
        prediction = model(parametric)
        exact = exact_solution(parametric)
    error = (prediction - exact).abs()
    x = physical[:, 0].reshape(resolution, resolution).cpu().numpy()
    y = physical[:, 1].reshape(resolution, resolution).cpu().numpy()
    fields = [
        value.reshape(resolution, resolution).cpu().numpy()
        for value in (prediction, exact, error)
    ]
    POSTPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    data_path = POSTPROCESSING_DIR / "bspline_solution_data.npz"
    np.savez_compressed(
        data_path, x=x, y=y, prediction=fields[0], exact=fields[1],
        absolute_error=fields[2], loss=np.asarray(loss_history),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for panel, values, title, color_map in zip(
        axes, fields, ("B-spline solution", "Exact solution", "Absolute error"),
        ("viridis", "viridis", "magma"),
    ):
        image = panel.pcolormesh(x, y, values, shading="auto", cmap=color_map)
        panel.set_title(title)
        panel.set_aspect("equal")
        panel.set_xlabel("x")
        panel.set_ylabel("y")
        figure.colorbar(image, ax=panel)
    figure_path = POSTPROCESSING_DIR / "bspline_solution.png"
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    history_figure, history_axis = plt.subplots(figsize=(6, 4))
    history_axis.semilogy(loss_history)
    history_axis.set(xlabel="Epoch", ylabel="Loss", title="B-spline training")
    history_axis.grid(True, which="both", alpha=0.3)
    history_path = POSTPROCESSING_DIR / "bspline_training.png"
    history_figure.savefig(history_path, dpi=300, bbox_inches="tight")
    print(f"Saved solution data to: {data_path}")
    print(f"Saved plots to: {figure_path} and {history_path}")


def main(
    device_name: str = DEFAULT_DEVICE,
    n_basis: int = N_BASIS,
    degree: int = DEGREE,
    epochs: int = EPOCHS,
    use_tuned: bool | None = None,
) -> None:
    set_seed()
    device = select_device(device_name)
    should_load = RESULT_PATH.exists() if use_tuned is None else use_tuned
    if should_load:
        n_basis, degree = load_hyperparameters()
        print(f"Loaded tuned hyperparameters from: {RESULT_PATH}")
    print(f"Using device: {device}")
    print(f"Geometry-fitted B-spline: n_basis={n_basis}, degree={degree}")
    model = bsplines(n_basis, degree, device)
    generator = torch.Generator(device=device).manual_seed(SEED)
    training_points = sample_parameter_domain(N_INTERIOR_POINTS, device, generator)
    training_operators = precompute_operators(model, training_points, 2)
    training_geometry = geometry_operator_terms(training_points)
    training_source = source_function(training_points, training_geometry)
    loss_fn = lambda: poisson_residual_loss(
        model, training_points, training_operators,
        training_geometry, training_source,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=100, min_lr=1.0e-5
    )
    loss_history = train(
        model.parameters(), loss_fn, optimizer, epochs=epochs,
        scheduler=scheduler, early_stopping_patience=1_000,
        min_delta=1.0e-10, print_every=100,
    )
    validation_points = sample_parameter_domain(20_000, device, generator)
    validation_operators = precompute_operators(model, validation_points, 2)
    validation_geometry = geometry_operator_terms(validation_points)
    validation_source = source_function(validation_points, validation_geometry)
    with torch.no_grad():
        error = relative_l2_error(
            model, validation_points, validation_operators, validation_geometry
        )
        residual = poisson_residual_loss(
            model, validation_points, validation_operators,
            validation_geometry, validation_source,
        )
    print(f"Relative physical L2 error: {error.item():.6e}")
    print(f"Physical Poisson residual MSE: {residual.item():.6e}")
    print(
        f"Geometry det(J): [{validation_geometry[0].min().item():.4f}, "
        f"{validation_geometry[0].max().item():.4f}]"
    )
    save_plots(model, loss_history)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Geometry-fitted B-spline Poisson solver")
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--n-basis", type=int, default=N_BASIS)
    parser.add_argument("--degree", type=int, default=DEGREE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    tuned = parser.add_mutually_exclusive_group()
    tuned.add_argument("--use-tuned", dest="use_tuned", action="store_true")
    tuned.add_argument("--no-use-tuned", dest="use_tuned", action="store_false")
    parser.set_defaults(use_tuned=None)
    args = parser.parse_args()
    main(args.device, args.n_basis, args.degree, args.epochs, args.use_tuned)
