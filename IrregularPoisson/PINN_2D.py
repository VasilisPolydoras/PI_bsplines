"""Geometry-fitted PINN for Poisson's equation on a curved irregular domain.

A regular polynomial map F sends the parameter square [0,1]^2 to one smooth,
asymmetric physical domain. The network receives physical coordinates, while
the physical Laplacian is evaluated through the exact mapping metric.
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

from training import train

SEED = 42
DEFAULT_DEVICE = "gpu"
N_LAYERS = 3
N_NEURONS = 40
N_INTERIOR_POINTS = 5_000
EPOCHS = 20_000
LEARNING_RATE = 1.0e-3
RESULT_PATH = (
    Path(__file__).resolve().parent / "results" / "best_pinn_hyperparameters.json"
)
POSTPROCESSING_DIR = Path(__file__).resolve().parent / "postprocessing"


def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_name: str) -> torch.device:
    if device_name == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU was requested, but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cpu")


def geometry_map(parametric_points: torch.Tensor) -> torch.Tensor:
    """Map [0,1]^2 bijectively to one asymmetric curved quadrilateral."""
    q = 2.0 * parametric_points[:, 0] - 1.0
    r = 2.0 * parametric_points[:, 1] - 1.0
    x = q + 0.22 * (1.0 - r.square()) * (1.0 + 0.25 * q + 0.18 * r)
    y = r + 0.18 * (1.0 - q.square()) * (-0.25 + 0.15 * q - 0.20 * r)
    return torch.stack((x, y), dim=1)


def boundary_envelope(parametric_points: torch.Tensor) -> torch.Tensor:
    """Unit-square envelope that vanishes on all four parametric edges."""
    xi, eta = parametric_points[:, 0], parametric_points[:, 1]
    return 16.0 * xi * (1.0 - xi) * eta * (1.0 - eta)


def sample_parameter_domain(
    count: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    return torch.rand(count, 2, device=device, generator=generator)


def geometry_operator_terms(parametric_points: torch.Tensor):
    """Precompute det(J), A, and div(A) for the mapped Laplacian.

    Here A=det(J) J^{-1}J^{-T} and
    Delta_x u = det(J)^(-1) div_xi(A grad_xi u).
    """
    with torch.enable_grad():
        coordinates = parametric_points.detach().clone().requires_grad_(True)
        physical = geometry_map(coordinates)
        grad_x = torch.autograd.grad(
            physical[:, 0], coordinates, torch.ones_like(physical[:, 0]),
            create_graph=True,
        )[0]
        grad_y = torch.autograd.grad(
            physical[:, 1], coordinates, torch.ones_like(physical[:, 1]),
            create_graph=True,
        )[0]
        jacobian = torch.stack((grad_x, grad_y), dim=1)
        determinant = torch.linalg.det(jacobian)
        if torch.any(determinant <= 0.0):
            raise ValueError("The geometry map has a non-positive Jacobian")
        inverse = torch.linalg.inv(jacobian)
        metric = determinant[:, None, None] * torch.bmm(
            inverse, inverse.transpose(1, 2)
        )
        grad_a11 = torch.autograd.grad(
            metric[:, 0, 0], coordinates, torch.ones_like(metric[:, 0, 0]),
            create_graph=True, retain_graph=True,
        )[0]
        grad_a21 = torch.autograd.grad(
            metric[:, 1, 0], coordinates, torch.ones_like(metric[:, 1, 0]),
            create_graph=True, retain_graph=True,
        )[0]
        grad_a12 = torch.autograd.grad(
            metric[:, 0, 1], coordinates, torch.ones_like(metric[:, 0, 1]),
            create_graph=True, retain_graph=True,
        )[0]
        grad_a22 = torch.autograd.grad(
            metric[:, 1, 1], coordinates, torch.ones_like(metric[:, 1, 1]),
            create_graph=False,
        )[0]
        divergence = torch.stack(
            (
                grad_a11[:, 0] + grad_a21[:, 1],
                grad_a12[:, 0] + grad_a22[:, 1],
            ),
            dim=1,
        )
    return determinant.detach(), metric.detach(), divergence.detach()


def mapped_laplacian_from_derivatives(
    gradient: torch.Tensor,
    u_xixi: torch.Tensor,
    u_xieta: torch.Tensor,
    u_etaeta: torch.Tensor,
    geometry_terms,
) -> torch.Tensor:
    determinant, metric, divergence = geometry_terms
    second_order = (
        metric[:, 0, 0] * u_xixi
        + (metric[:, 0, 1] + metric[:, 1, 0]) * u_xieta
        + metric[:, 1, 1] * u_etaeta
    )
    return (second_order + (divergence * gradient).sum(dim=1)) / determinant


def exact_solution(parametric_points: torch.Tensor) -> torch.Tensor:
    physical = geometry_map(parametric_points)
    return boundary_envelope(parametric_points) * torch.exp(
        physical[:, 0] + physical[:, 1]
    )


def field_derivatives(field: torch.Tensor, coordinates: torch.Tensor):
    gradient = torch.autograd.grad(
        field, coordinates, torch.ones_like(field), create_graph=True
    )[0]
    derivative_xi = torch.autograd.grad(
        gradient[:, 0], coordinates, torch.ones_like(gradient[:, 0]),
        create_graph=True, retain_graph=True,
    )[0]
    derivative_eta = torch.autograd.grad(
        gradient[:, 1], coordinates, torch.ones_like(gradient[:, 1]),
        create_graph=True,
    )[0]
    return gradient, derivative_xi[:, 0], derivative_xi[:, 1], derivative_eta[:, 1]


def source_function(parametric_points: torch.Tensor, geometry_terms=None) -> torch.Tensor:
    """Manufactured source f=-Delta_x u_exact on the mapped domain."""
    geometry_terms = (
        geometry_operator_terms(parametric_points)
        if geometry_terms is None else geometry_terms
    )
    with torch.enable_grad():
        coordinates = parametric_points.detach().clone().requires_grad_(True)
        derivatives = field_derivatives(exact_solution(coordinates), coordinates)
        laplacian = mapped_laplacian_from_derivatives(*derivatives, geometry_terms)
    return -laplacian.detach()


class PINN(nn.Module):
    """Global physical-coordinate network with a parametric hard boundary."""

    def __init__(self, n_layers: int, n_neurons: int) -> None:
        super().__init__()
        modules: list[nn.Module] = [nn.Linear(2, n_neurons), nn.Tanh()]
        for _ in range(n_layers - 1):
            modules.extend((nn.Linear(n_neurons, n_neurons), nn.Tanh()))
        modules.append(nn.Linear(n_neurons, 1))
        self.network = nn.Sequential(*modules)
        self.n_layers = n_layers
        self.n_neurons = n_neurons

    def forward(self, parametric_points: torch.Tensor) -> torch.Tensor:
        physical = geometry_map(parametric_points)
        free_field = self.network(physical).squeeze(-1)
        return boundary_envelope(parametric_points) * free_field


def poisson_residual(
    model: PINN,
    parametric_points: torch.Tensor,
    geometry_terms=None,
    source: torch.Tensor | None = None,
) -> torch.Tensor:
    geometry_terms = (
        geometry_operator_terms(parametric_points)
        if geometry_terms is None else geometry_terms
    )
    source = source_function(parametric_points, geometry_terms) if source is None else source
    with torch.enable_grad():
        coordinates = parametric_points.detach().clone().requires_grad_(True)
        derivatives = field_derivatives(model(coordinates), coordinates)
        laplacian = mapped_laplacian_from_derivatives(*derivatives, geometry_terms)
    return -laplacian - source


def weighted_mean(values: torch.Tensor, determinant: torch.Tensor) -> torch.Tensor:
    return (determinant * values).sum() / determinant.sum()


def poisson_residual_loss(
    model: PINN,
    parametric_points: torch.Tensor,
    geometry_terms=None,
    source: torch.Tensor | None = None,
) -> torch.Tensor:
    geometry_terms = (
        geometry_operator_terms(parametric_points)
        if geometry_terms is None else geometry_terms
    )
    residual = poisson_residual(model, parametric_points, geometry_terms, source)
    return weighted_mean(residual.square(), geometry_terms[0])


def relative_l2_error(
    model: PINN, parametric_points: torch.Tensor, geometry_terms=None
) -> torch.Tensor:
    geometry_terms = (
        geometry_operator_terms(parametric_points)
        if geometry_terms is None else geometry_terms
    )
    prediction = model(parametric_points)
    exact = exact_solution(parametric_points)
    numerator = weighted_mean((prediction - exact).square(), geometry_terms[0])
    denominator = weighted_mean(exact.square(), geometry_terms[0])
    return torch.sqrt(numerator / denominator)


def load_hyperparameters(path: Path = RESULT_PATH) -> tuple[int, int]:
    if not path.exists():
        raise FileNotFoundError(
            f"No PINN tuning result found at {path}. Run tune_pinn.py first "
            "or use --no-use-tuned."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("best_params", payload)
    return int(params["n_layers"]), int(params["n_neurons"])


def save_plots(model: PINN, loss_history: list[float], resolution: int = 201) -> None:
    device = next(model.parameters()).device
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
    data_path = POSTPROCESSING_DIR / "pinn_solution_data.npz"
    np.savez_compressed(
        data_path, x=x, y=y, prediction=fields[0], exact=fields[1],
        absolute_error=fields[2], loss=np.asarray(loss_history),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for panel, values, title, color_map in zip(
        axes, fields, ("PINN solution", "Exact solution", "Absolute error"),
        ("viridis", "viridis", "magma"),
    ):
        image = panel.pcolormesh(x, y, values, shading="auto", cmap=color_map)
        panel.set_title(title)
        panel.set_aspect("equal")
        panel.set_xlabel("x")
        panel.set_ylabel("y")
        figure.colorbar(image, ax=panel)
    figure_path = POSTPROCESSING_DIR / "pinn_solution.png"
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    history_figure, history_axis = plt.subplots(figsize=(6, 4))
    history_axis.semilogy(loss_history)
    history_axis.set(xlabel="Epoch", ylabel="Loss", title="PINN training")
    history_axis.grid(True, which="both", alpha=0.3)
    history_path = POSTPROCESSING_DIR / "pinn_training.png"
    history_figure.savefig(history_path, dpi=300, bbox_inches="tight")
    print(f"Saved solution data to: {data_path}")
    print(f"Saved plots to: {figure_path} and {history_path}")


def main(
    device_name: str = DEFAULT_DEVICE,
    n_layers: int = N_LAYERS,
    n_neurons: int = N_NEURONS,
    epochs: int = EPOCHS,
    use_tuned: bool | None = None,
) -> None:
    set_seed()
    device = select_device(device_name)
    should_load = RESULT_PATH.exists() if use_tuned is None else use_tuned
    if should_load:
        n_layers, n_neurons = load_hyperparameters()
        print(f"Loaded tuned hyperparameters from: {RESULT_PATH}")
    print(f"Using device: {device}")
    print(f"Geometry-fitted PINN: layers={n_layers}, neurons={n_neurons}")
    model = PINN(n_layers, n_neurons).to(device)
    generator = torch.Generator(device=device).manual_seed(SEED)
    training_points = sample_parameter_domain(N_INTERIOR_POINTS, device, generator)
    training_geometry = geometry_operator_terms(training_points)
    training_source = source_function(training_points, training_geometry)
    loss_fn = lambda: poisson_residual_loss(
        model, training_points, training_geometry, training_source
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=100, min_lr=1.0e-5
    )
    loss_history = train(
        model.parameters(), loss_fn, optimizer, epochs=epochs,
        scheduler=scheduler, early_stopping_patience=1_000,
        min_delta=1.0e-9, print_every=100,
    )
    validation_points = sample_parameter_domain(20_000, device, generator)
    validation_geometry = geometry_operator_terms(validation_points)
    validation_source = source_function(validation_points, validation_geometry)
    with torch.no_grad():
        error = relative_l2_error(model, validation_points, validation_geometry)
    residual = poisson_residual_loss(
        model, validation_points, validation_geometry, validation_source
    ).detach()
    print(f"Relative physical L2 error: {error.item():.6e}")
    print(f"Physical Poisson residual MSE: {residual.item():.6e}")
    print(
        f"Geometry det(J): [{validation_geometry[0].min().item():.4f}, "
        f"{validation_geometry[0].max().item():.4f}]"
    )
    save_plots(model, loss_history)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Geometry-fitted Poisson PINN")
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--layers", type=int, default=N_LAYERS)
    parser.add_argument("--neurons", type=int, default=N_NEURONS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    tuned = parser.add_mutually_exclusive_group()
    tuned.add_argument("--use-tuned", dest="use_tuned", action="store_true")
    tuned.add_argument("--no-use-tuned", dest="use_tuned", action="store_false")
    parser.set_defaults(use_tuned=None)
    args = parser.parse_args()
    main(args.device, args.layers, args.neurons, args.epochs, args.use_tuned)
