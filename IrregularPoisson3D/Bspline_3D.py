"""One-patch geometry-fitted tensor-product B-spline for 3D Poisson."""

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
from IrregularPoisson3D.PINN_3D import (
    DEFAULT_DEVICE, LEARNING_RATE, POSTPROCESSING_DIR, SEED,
    exact_solution, geometry_map, geometry_operator_terms,
    mapped_laplacian_from_derivatives, sample_parameter_domain,
    plot_domain_wireframe, select_device, set_seed, source_function, weighted_mean,
)
from training import train

N_BASIS = 12
DEGREE = 3
N_INTERIOR_POINTS = 3_000
EPOCHS = 20_000
RESULT_PATH = (
    Path(__file__).resolve().parent / "results" / "best_bspline_hyperparameters.json"
)


def open_uniform_knots(n_basis: int, degree: int, device: torch.device):
    if degree < 2:
        raise ValueError("The strong Poisson formulation requires degree >= 2")
    if n_basis <= degree:
        raise ValueError("n_basis must be greater than degree")
    count = n_basis-degree-1
    interior = (torch.linspace(0, 1, count+2, device=device)[1:-1]
                if count > 0 else torch.empty(0, device=device))
    return torch.cat((torch.zeros(degree+1, device=device), interior,
                      torch.ones(degree+1, device=device)))


class bsplines(nn.Module):
    """Trivariate solution spline with all six boundary faces fixed to zero."""

    def __init__(self, n_basis: int, degree: int, device: torch.device):
        super().__init__()
        self.n_basis, self.degree = n_basis, degree
        self.register_buffer("knots", open_uniform_knots(n_basis, degree, device))
        self.interior_coefficients = nn.Parameter(
            torch.zeros(n_basis-2, n_basis-2, n_basis-2, device=device)
        )

    def coefficient_tensor(self):
        return torch.nn.functional.pad(
            self.interior_coefficients, (1, 1, 1, 1, 1, 1)
        )

    def operators(self, points: torch.Tensor, max_order: int = 2):
        return {
            axis: tuple(
                basis_derivative_matrix(
                    points[:, index], self.knots, self.degree, order
                )
                for order in range(max_order+1)
            )
            for index, axis in enumerate(("x", "y", "z"))
        }

    def field(self, operators, dx=0, dy=0, dz=0):
        return torch.einsum(
            "pa,abc,pb,pc->p", operators["x"][dx], self.coefficient_tensor(),
            operators["y"][dy], operators["z"][dz]
        )

    def forward(self, points: torch.Tensor):
        return self.field(self.operators(points, 0))


def precompute_operators(model: bsplines, points: torch.Tensor, max_order=2):
    return model.operators(points, max_order)


def spline_derivatives(model: bsplines, operators):
    ux = model.field(operators, dx=1)
    uy = model.field(operators, dy=1)
    uz = model.field(operators, dz=1)
    uxx = model.field(operators, dx=2)
    uyy = model.field(operators, dy=2)
    uzz = model.field(operators, dz=2)
    uxy = model.field(operators, dx=1, dy=1)
    uxz = model.field(operators, dx=1, dz=1)
    uyz = model.field(operators, dy=1, dz=1)
    gradient = torch.stack((ux, uy, uz), dim=1)
    hessian = torch.stack(
        (
            torch.stack((uxx, uxy, uxz), dim=1),
            torch.stack((uxy, uyy, uyz), dim=1),
            torch.stack((uxz, uyz, uzz), dim=1),
        ), dim=1,
    )
    return gradient, hessian


def poisson_residual(model, points, operators, geometry_terms, source):
    laplacian = mapped_laplacian_from_derivatives(
        *spline_derivatives(model, operators), geometry_terms
    )
    return -laplacian-source


def poisson_residual_loss(model, points, operators, geometry_terms, source):
    residual = poisson_residual(model, points, operators, geometry_terms, source)
    return weighted_mean(residual.square(), geometry_terms[0])


def relative_l2_error(model, points, operators, geometry_terms):
    prediction, exact = model.field(operators), exact_solution(points)
    numerator = weighted_mean((prediction-exact).square(), geometry_terms[0])
    denominator = weighted_mean(exact.square(), geometry_terms[0])
    return torch.sqrt(numerator/denominator)


def load_hyperparameters(path: Path = RESULT_PATH):
    if not path.exists():
        raise FileNotFoundError(f"No 3D B-spline tuning result found at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("best_params", payload)
    return int(params["n_basis"]), int(params["degree"])


def save_plots(model: bsplines, history: list[float], resolution=121):
    device = model.knots.device
    axis = torch.linspace(0, 1, resolution, device=device)
    xi, eta = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack(
        (xi.flatten(), eta.flatten(), torch.full_like(xi.flatten(), .5)), dim=1
    )
    physical = geometry_map(points)
    with torch.no_grad():
        prediction, exact = model(points), exact_solution(points)
    error = (prediction-exact).abs()
    x = physical[:, 0].reshape(resolution, resolution).cpu().numpy()
    y = physical[:, 1].reshape(resolution, resolution).cpu().numpy()
    z = physical[:, 2].reshape(resolution, resolution).cpu().numpy()
    fields = [v.reshape(resolution, resolution).cpu().numpy()
              for v in (prediction, exact, error)]
    POSTPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    data_path = POSTPROCESSING_DIR / "bspline_central_slice_data.npz"
    np.savez_compressed(data_path, x=x, y=y, z=z, prediction=fields[0],
                        exact=fields[1], absolute_error=fields[2],
                        loss=np.asarray(history))
    figure = plt.figure(figsize=(18, 5.5), constrained_layout=True)
    axes = [figure.add_subplot(1, 3, index + 1, projection="3d")
            for index in range(3)]
    for panel, values, title, cmap_name in zip(
        axes, fields,
        ("B-spline central slice", "Exact central slice", "Absolute error"),
        ("viridis", "viridis", "magma"),
    ):
        plot_domain_wireframe(panel, device)
        lower, upper = float(np.nanmin(values)), float(np.nanmax(values))
        if np.isclose(lower, upper):
            upper = lower + 1.0e-12
        normalizer = plt.Normalize(lower, upper)
        color_map = plt.get_cmap(cmap_name)
        surface = panel.plot_surface(
            x, y, z, facecolors=color_map(normalizer(values)),
            linewidth=0.0, antialiased=True, shade=False, alpha=0.95,
        )
        panel.set_title(title)
        panel.set_xlabel("x"); panel.set_ylabel("y"); panel.set_zlabel("z")
        panel.set_box_aspect((1.0, 1.0, 1.0))
        panel.view_init(elev=28, azim=-135)
        color_bar_source = plt.cm.ScalarMappable(norm=normalizer, cmap=color_map)
        color_bar_source.set_array(values)
        figure.colorbar(color_bar_source, ax=panel, shrink=0.65, pad=0.10)
    figure_path = POSTPROCESSING_DIR / "bspline_central_slice.png"
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    history_figure, history_axis = plt.subplots(figsize=(6, 4))
    history_axis.semilogy(history); history_axis.set(
        xlabel="Epoch", ylabel="Loss", title="3D B-spline training"
    )
    history_axis.grid(True, which="both", alpha=.3)
    history_path = POSTPROCESSING_DIR / "bspline_training.png"
    history_figure.savefig(history_path, dpi=300, bbox_inches="tight")
    print(f"Saved slice data to: {data_path}")
    print(f"Saved plots to: {figure_path} and {history_path}")


def main(device_name=DEFAULT_DEVICE, n_basis=N_BASIS, degree=DEGREE,
         epochs=EPOCHS, use_tuned: bool | None = None):
    set_seed(); device = select_device(device_name)
    should_load = RESULT_PATH.exists() if use_tuned is None else use_tuned
    if should_load:
        n_basis, degree = load_hyperparameters()
        print(f"Loaded tuned hyperparameters from: {RESULT_PATH}")
    print(f"Using device: {device}")
    print(f"3D geometry-fitted B-spline: n_basis={n_basis}, degree={degree}")
    model = bsplines(n_basis, degree, device)
    generator = torch.Generator(device=device).manual_seed(SEED)
    points = sample_parameter_domain(N_INTERIOR_POINTS, device, generator)
    operators = precompute_operators(model, points, 2)
    geometry = geometry_operator_terms(points); source = source_function(points, geometry)
    loss_fn = lambda: poisson_residual_loss(
        model, points, operators, geometry, source
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.8, patience=100, min_lr=1e-5
    )
    history = train(model.parameters(), loss_fn, optimizer, epochs=epochs,
                    scheduler=scheduler, early_stopping_patience=1_000,
                    min_delta=1e-10, print_every=100)
    validation = sample_parameter_domain(10_000, device, generator)
    validation_operators = precompute_operators(model, validation, 2)
    validation_geometry = geometry_operator_terms(validation)
    validation_source = source_function(validation, validation_geometry)
    with torch.no_grad():
        error = relative_l2_error(
            model, validation, validation_operators, validation_geometry
        )
        residual = poisson_residual_loss(
            model, validation, validation_operators,
            validation_geometry, validation_source
        )
    print(f"Relative physical L2 error: {error.item():.6e}")
    print(f"Physical residual MSE: {residual.item():.6e}")
    print(f"det(J) range: [{validation_geometry[0].min().item():.4f}, "
          f"{validation_geometry[0].max().item():.4f}]")
    save_plots(model, history); plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D geometry-fitted B-spline Poisson")
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--n-basis", type=int, default=N_BASIS)
    parser.add_argument("--degree", type=int, default=DEGREE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    tuned = parser.add_mutually_exclusive_group()
    tuned.add_argument("--use-tuned", dest="use_tuned", action="store_true")
    tuned.add_argument("--no-use-tuned", dest="use_tuned", action="store_false")
    parser.set_defaults(use_tuned=None); args = parser.parse_args()
    main(args.device, args.n_basis, args.degree, args.epochs, args.use_tuned)
