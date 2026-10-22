"""Geometry-fitted hard-constrained PINN for 3D Poisson's equation."""

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
N_NEURONS = 50
N_INTERIOR_POINTS = 3_000
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
    """Map [0,1]^3 to a curved asymmetric hexahedral domain."""
    q = 2.0 * parametric_points[:, 0] - 1.0
    r = 2.0 * parametric_points[:, 1] - 1.0
    s = 2.0 * parametric_points[:, 2] - 1.0
    x = q + 0.16 * (1-r.square()) * (1-s.square()) * (
        1.0 + 0.20*q + 0.12*r - 0.10*s
    )
    y = r + 0.13 * (1-q.square()) * (1-s.square()) * (
        -0.30 + 0.15*q - 0.20*r + 0.10*s
    )
    z = s + 0.15 * (1-q.square()) * (1-r.square()) * (
        0.20 - 0.10*q + 0.15*r + 0.20*s
    )
    return torch.stack((x, y, z), dim=1)


def boundary_envelope(parametric_points: torch.Tensor) -> torch.Tensor:
    xi, eta, zeta = (
        parametric_points[:, 0], parametric_points[:, 1], parametric_points[:, 2]
    )
    return (
        64.0 * xi*(1-xi) * eta*(1-eta) * zeta*(1-zeta)
    )


def sample_parameter_domain(
    count: int, device: torch.device, generator: torch.Generator | None = None
) -> torch.Tensor:
    return torch.rand(count, 3, device=device, generator=generator)


def geometry_operator_terms(parametric_points: torch.Tensor):
    """Return det(J), A=det(J)J^-1J^-T, and div(A)."""
    with torch.enable_grad():
        coordinates = parametric_points.detach().clone().requires_grad_(True)
        physical = geometry_map(coordinates)
        rows = [
            torch.autograd.grad(
                physical[:, component], coordinates,
                torch.ones_like(physical[:, component]), create_graph=True,
                retain_graph=True,
            )[0]
            for component in range(3)
        ]
        jacobian = torch.stack(rows, dim=1)
        determinant = torch.linalg.det(jacobian)
        if torch.any(determinant <= 0.0):
            raise ValueError("The 3D geometry map has a non-positive Jacobian")
        inverse = torch.linalg.inv(jacobian)
        metric = determinant[:, None, None] * torch.bmm(
            inverse, inverse.transpose(1, 2)
        )
        divergence_components = []
        for beta in range(3):
            component = torch.zeros_like(determinant)
            for alpha in range(3):
                derivative = torch.autograd.grad(
                    metric[:, alpha, beta], coordinates,
                    torch.ones_like(metric[:, alpha, beta]), create_graph=True,
                    retain_graph=True,
                )[0]
                component = component + derivative[:, alpha]
            divergence_components.append(component)
        divergence = torch.stack(divergence_components, dim=1)
    return determinant.detach(), metric.detach(), divergence.detach()


def mapped_laplacian_from_derivatives(
    gradient: torch.Tensor, hessian: torch.Tensor, geometry_terms
) -> torch.Tensor:
    determinant, metric, divergence = geometry_terms
    second_order = torch.einsum("nij,nij->n", metric, hessian)
    first_order = (divergence * gradient).sum(dim=1)
    return (second_order + first_order) / determinant


def field_derivatives(field: torch.Tensor, coordinates: torch.Tensor):
    gradient = torch.autograd.grad(
        field, coordinates, torch.ones_like(field), create_graph=True
    )[0]
    hessian_rows = [
        torch.autograd.grad(
            gradient[:, component], coordinates,
            torch.ones_like(gradient[:, component]), create_graph=True,
            retain_graph=True,
        )[0]
        for component in range(3)
    ]
    return gradient, torch.stack(hessian_rows, dim=1)


def exact_solution(parametric_points: torch.Tensor) -> torch.Tensor:
    physical = geometry_map(parametric_points)
    return boundary_envelope(parametric_points) * torch.exp(physical.sum(dim=1))


def source_function(parametric_points: torch.Tensor, geometry_terms=None):
    geometry_terms = (
        geometry_operator_terms(parametric_points)
        if geometry_terms is None else geometry_terms
    )
    with torch.enable_grad():
        coordinates = parametric_points.detach().clone().requires_grad_(True)
        gradient, hessian = field_derivatives(exact_solution(coordinates), coordinates)
        laplacian = mapped_laplacian_from_derivatives(
            gradient, hessian, geometry_terms
        )
    return -laplacian.detach()


class PINN(nn.Module):
    def __init__(self, n_layers: int, n_neurons: int) -> None:
        super().__init__()
        modules: list[nn.Module] = [nn.Linear(3, n_neurons), nn.Tanh()]
        for _ in range(n_layers - 1):
            modules.extend((nn.Linear(n_neurons, n_neurons), nn.Tanh()))
        modules.append(nn.Linear(n_neurons, 1))
        self.network = nn.Sequential(*modules)
        self.n_layers, self.n_neurons = n_layers, n_neurons

    def forward(self, parametric_points: torch.Tensor) -> torch.Tensor:
        physical = geometry_map(parametric_points)
        return boundary_envelope(parametric_points) * self.network(physical).squeeze(-1)


def poisson_residual(
    model: PINN, parametric_points: torch.Tensor, geometry_terms, source
) -> torch.Tensor:
    with torch.enable_grad():
        coordinates = parametric_points.detach().clone().requires_grad_(True)
        gradient, hessian = field_derivatives(model(coordinates), coordinates)
        laplacian = mapped_laplacian_from_derivatives(
            gradient, hessian, geometry_terms
        )
    return -laplacian - source


def weighted_mean(values: torch.Tensor, determinant: torch.Tensor) -> torch.Tensor:
    return (determinant * values).sum() / determinant.sum()


def poisson_residual_loss(
    model: PINN, parametric_points: torch.Tensor, geometry_terms, source
) -> torch.Tensor:
    residual = poisson_residual(model, parametric_points, geometry_terms, source)
    return weighted_mean(residual.square(), geometry_terms[0])


def relative_l2_error(model: PINN, parametric_points, geometry_terms):
    prediction, exact = model(parametric_points), exact_solution(parametric_points)
    numerator = weighted_mean((prediction-exact).square(), geometry_terms[0])
    denominator = weighted_mean(exact.square(), geometry_terms[0])
    return torch.sqrt(numerator/denominator)


def load_hyperparameters(path: Path = RESULT_PATH) -> tuple[int, int]:
    if not path.exists():
        raise FileNotFoundError(f"No 3D PINN tuning result found at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("best_params", payload)
    return int(params["n_layers"]), int(params["n_neurons"])


def plot_domain_wireframe(axis_object, device: torch.device, resolution: int = 17):
    """Draw the retained zeta<=0.5 half-body below the exposed center cut."""
    coordinate = torch.linspace(0.0, 1.0, resolution, device=device)
    half_coordinate = torch.linspace(0.0, 0.5, resolution, device=device)
    first, second = torch.meshgrid(
        coordinate, half_coordinate, indexing="ij"
    )
    flat_first, flat_half = first.flatten(), second.flatten()
    zero, one = torch.zeros_like(flat_first), torch.ones_like(flat_first)
    cap_first, cap_second = torch.meshgrid(
        coordinate, coordinate, indexing="ij"
    )
    cap_first, cap_second = cap_first.flatten(), cap_second.flatten()
    faces = (
        torch.stack((zero, flat_first, flat_half), dim=1),
        torch.stack((one, flat_first, flat_half), dim=1),
        torch.stack((flat_first, zero, flat_half), dim=1),
        torch.stack((flat_first, one, flat_half), dim=1),
        torch.stack(
            (cap_first, cap_second, torch.zeros_like(cap_first)), dim=1
        ),
    )
    with torch.no_grad():
        physical_faces = [geometry_map(face).cpu().numpy() for face in faces]
    for physical in physical_faces:
        physical_x = physical[:, 0].reshape(resolution, resolution)
        physical_y = physical[:, 1].reshape(resolution, resolution)
        physical_z = physical[:, 2].reshape(resolution, resolution)
        axis_object.plot_surface(
            physical_x, physical_y, physical_z,
            color="lightsteelblue", linewidth=0.0, shade=True, alpha=0.34,
        )
        axis_object.plot_wireframe(
            physical_x, physical_y, physical_z,
            rstride=2, cstride=2, color="0.35", linewidth=0.45, alpha=0.38,
        )


def save_plots(model: PINN, loss_history: list[float], resolution: int = 121):
    """Save prediction/exact/error on the central zeta=0.5 curved slice."""
    device = next(model.parameters()).device
    axis = torch.linspace(0.0, 1.0, resolution, device=device)
    xi, eta = torch.meshgrid(axis, axis, indexing="ij")
    parametric = torch.stack(
        (xi.flatten(), eta.flatten(), torch.full_like(xi.flatten(), 0.5)), dim=1
    )
    physical = geometry_map(parametric)
    with torch.no_grad():
        prediction, exact = model(parametric), exact_solution(parametric)
    error = (prediction-exact).abs()
    x = physical[:, 0].reshape(resolution, resolution).cpu().numpy()
    y = physical[:, 1].reshape(resolution, resolution).cpu().numpy()
    z = physical[:, 2].reshape(resolution, resolution).cpu().numpy()
    fields = [v.reshape(resolution, resolution).cpu().numpy() for v in (prediction, exact, error)]
    POSTPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    data_path = POSTPROCESSING_DIR / "pinn_central_slice_data.npz"
    np.savez_compressed(data_path, x=x, y=y, z=z, prediction=fields[0],
                        exact=fields[1], absolute_error=fields[2],
                        loss=np.asarray(loss_history))
    figure = plt.figure(figsize=(18, 5.5), constrained_layout=True)
    axes = [figure.add_subplot(1, 3, index + 1, projection="3d")
            for index in range(3)]
    for panel, values, title, cmap_name in zip(
        axes, fields,
        ("PINN central slice", "Exact central slice", "Absolute error"),
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
    figure_path = POSTPROCESSING_DIR / "pinn_central_slice.png"
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    history_figure, history_axis = plt.subplots(figsize=(6, 4))
    history_axis.semilogy(loss_history); history_axis.set(
        xlabel="Epoch", ylabel="Loss", title="3D PINN training"
    )
    history_axis.grid(True, which="both", alpha=0.3)
    history_path = POSTPROCESSING_DIR / "pinn_training.png"
    history_figure.savefig(history_path, dpi=300, bbox_inches="tight")
    print(f"Saved slice data to: {data_path}")
    print(f"Saved plots to: {figure_path} and {history_path}")


def main(
    device_name: str = DEFAULT_DEVICE, n_layers: int = N_LAYERS,
    n_neurons: int = N_NEURONS, epochs: int = EPOCHS,
    use_tuned: bool | None = None,
) -> None:
    set_seed(); device = select_device(device_name)
    should_load = RESULT_PATH.exists() if use_tuned is None else use_tuned
    if should_load:
        n_layers, n_neurons = load_hyperparameters()
        print(f"Loaded tuned hyperparameters from: {RESULT_PATH}")
    print(f"Using device: {device}")
    print(f"3D geometry-fitted PINN: layers={n_layers}, neurons={n_neurons}")
    model = PINN(n_layers, n_neurons).to(device)
    generator = torch.Generator(device=device).manual_seed(SEED)
    points = sample_parameter_domain(N_INTERIOR_POINTS, device, generator)
    geometry = geometry_operator_terms(points); source = source_function(points, geometry)
    loss_fn = lambda: poisson_residual_loss(model, points, geometry, source)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=100, min_lr=1e-5
    )
    history = train(model.parameters(), loss_fn, optimizer, epochs=epochs,
                    scheduler=scheduler, early_stopping_patience=1_000,
                    min_delta=1e-9, print_every=100)
    validation = sample_parameter_domain(10_000, device, generator)
    validation_geometry = geometry_operator_terms(validation)
    validation_source = source_function(validation, validation_geometry)
    with torch.no_grad():
        error = relative_l2_error(model, validation, validation_geometry)
    residual = poisson_residual_loss(
        model, validation, validation_geometry, validation_source
    ).detach()
    print(f"Relative physical L2 error: {error.item():.6e}")
    print(f"Physical residual MSE: {residual.item():.6e}")
    print(f"det(J) range: [{validation_geometry[0].min().item():.4f}, "
          f"{validation_geometry[0].max().item():.4f}]")
    save_plots(model, history); plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D geometry-fitted Poisson PINN")
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--layers", type=int, default=N_LAYERS)
    parser.add_argument("--neurons", type=int, default=N_NEURONS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    tuned = parser.add_mutually_exclusive_group()
    tuned.add_argument("--use-tuned", dest="use_tuned", action="store_true")
    tuned.add_argument("--no-use-tuned", dest="use_tuned", action="store_false")
    parser.set_defaults(use_tuned=None); args = parser.parse_args()
    main(args.device, args.layers, args.neurons, args.epochs, args.use_tuned)
