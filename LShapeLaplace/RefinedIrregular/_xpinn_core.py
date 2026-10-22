"""Internal XPINN networks, residuals, losses, and plotting utilities."""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

PATCH_NAMES = ("upper_right", "upper_left", "lower_left")
PATCH_BOUNDS = {
    "upper_right": ((0.0, 0.0), (1.0, 1.0)),
    "upper_left": ((-1.0, 0.0), (0.0, 1.0)),
    "lower_left": ((-1.0, -1.0), (0.0, 0.0)),
}
INTERFACE_NORMALS = {
    "upper_right_upper_left": ((-1.0, 0.0), (1.0, 0.0)),
    "upper_left_lower_left": ((0.0, -1.0), (0.0, 1.0)),
}


def set_seed(seed: int) -> None:
    """Seed every random-number generator used by this solver."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU requested, but CUDA is not available")
        return torch.device("cuda")
    raise ValueError("device must be 'cpu' or 'gpu'")


def map_patch(name: str, local_points: torch.Tensor) -> torch.Tensor:
    """Map coordinates in [0,1]^2 to one of the three L-shaped patches."""
    xi, eta = local_points[:, 0], local_points[:, 1]
    if name == "upper_right":
        return torch.stack((xi, eta), dim=1)
    if name == "upper_left":
        return torch.stack((xi - 1.0, eta), dim=1)
    if name == "lower_left":
        return torch.stack((xi - 1.0, eta - 1.0), dim=1)
    raise ValueError(f"Unknown patch: {name}")


def exact_solution(points: torch.Tensor) -> torch.Tensor:
    """Smooth harmonic field u(x,y)=exp(x) sin(y)."""
    return torch.exp(points[:, 0]) * torch.sin(points[:, 1])


class SubdomainNetwork(nn.Module):
    """Tanh MLP with coordinates normalized on its own square patch."""

    def __init__(
        self,
        n_layers: int,
        n_neurons: int,
        lower_bound: tuple[float, float],
        upper_bound: tuple[float, float],
    ) -> None:
        super().__init__()
        if n_layers < 1 or n_neurons < 1:
            raise ValueError("n_layers and n_neurons must be positive")
        self.register_buffer("lower_bound", torch.tensor(lower_bound))
        self.register_buffer("upper_bound", torch.tensor(upper_bound))

        modules: list[nn.Module] = []
        input_size = 2
        for _ in range(n_layers):
            linear = nn.Linear(input_size, n_neurons)
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            modules.extend((linear, nn.Tanh()))
            input_size = n_neurons
        output = nn.Linear(input_size, 1)
        nn.init.xavier_normal_(output.weight)
        nn.init.zeros_(output.bias)
        modules.append(output)
        self.network = nn.Sequential(*modules)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        normalized = 2.0 * (points - self.lower_bound) / (
            self.upper_bound - self.lower_bound
        ) - 1.0
        return self.network(normalized).squeeze(-1)


class XPINN(nn.Module):
    """Three independent neural networks on the fitted L-shaped partition."""

    patch_names = PATCH_NAMES
    output_stem = "xpinn_regular"
    solution_title = "XPINN regular solution"
    exact_solution_label = r"Exact solution: $e^x\sin(y)$"

    def __init__(self, n_layers: int, n_neurons: int) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.networks = nn.ModuleDict(
            {
                name: SubdomainNetwork(
                    n_layers,
                    n_neurons,
                    PATCH_BOUNDS[name][0],
                    PATCH_BOUNDS[name][1],
                )
                for name in self.patch_names
            }
        )

    def forward(self, patch_name: str, points: torch.Tensor) -> torch.Tensor:
        return self.networks[patch_name](points)

    @staticmethod
    def exact_solution(points: torch.Tensor) -> torch.Tensor:
        return exact_solution(points)


def sample_interior_points(
    count: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Sample the same number of uniformly distributed points in each patch."""
    return {
        name: map_patch(
            name,
            torch.rand(count, 2, device=device, generator=generator),
        )
        for name in PATCH_NAMES
    }


def generate_boundary_points(
    points_per_edge: int, device: torch.device
) -> dict[str, torch.Tensor]:
    """Return only the exterior edges; internal patch edges are excluded."""
    line = torch.linspace(0.0, 1.0, points_per_edge, device=device)
    zero, one = torch.zeros_like(line), torch.ones_like(line)
    local_edges = {
        "upper_right": (
            torch.stack((line, zero), dim=1),
            torch.stack((one, line), dim=1),
            torch.stack((line, one), dim=1),
        ),
        "upper_left": (
            torch.stack((zero, line), dim=1),
            torch.stack((line, one), dim=1),
        ),
        "lower_left": (
            torch.stack((zero, line), dim=1),
            torch.stack((line, zero), dim=1),
            torch.stack((one, line), dim=1),
        ),
    }
    return {
        name: torch.cat(
            [map_patch(name, edge) for edge in edges], dim=0
        )
        for name, edges in local_edges.items()
    }


def generate_interface_points(
    count: int, device: torch.device
) -> dict[str, tuple[str, torch.Tensor, str, torch.Tensor]]:
    """Return paired physical points on the two internal interfaces."""
    line = torch.linspace(0.0, 1.0, count, device=device)
    zero, one = torch.zeros_like(line), torch.ones_like(line)
    return {
        "upper_right_upper_left": (
            "upper_right",
            map_patch("upper_right", torch.stack((zero, line), dim=1)),
            "upper_left",
            map_patch("upper_left", torch.stack((one, line), dim=1)),
        ),
        "upper_left_lower_left": (
            "upper_left",
            map_patch("upper_left", torch.stack((line, zero), dim=1)),
            "lower_left",
            map_patch("lower_left", torch.stack((line, one), dim=1)),
        ),
    }


def laplace_residual(
    model: XPINN, patch_name: str, points: torch.Tensor
) -> torch.Tensor:
    """Evaluate u_xx + u_yy while retaining the graph for parameter training."""
    with torch.enable_grad():
        coordinates = points.detach().clone().requires_grad_(True)
        solution = model(patch_name, coordinates)
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
        return u_xx + u_yy


def normal_flux(
    model: XPINN,
    patch_name: str,
    points: torch.Tensor,
    normal: tuple[float, float],
    *,
    create_graph: bool,
) -> torch.Tensor:
    """Evaluate q.n for q=-grad(u) on one side of an interface."""
    with torch.enable_grad():
        coordinates = points.detach().clone().requires_grad_(True)
        solution = model(patch_name, coordinates)
        gradient = torch.autograd.grad(
            solution,
            coordinates,
            grad_outputs=torch.ones_like(solution),
            create_graph=create_graph,
        )[0]
        return -(
            gradient[:, 0] * normal[0] + gradient[:, 1] * normal[1]
        )


def xpinn_loss_components(
    model: XPINN,
    interior_points: dict[str, torch.Tensor],
    boundary_points: dict[str, torch.Tensor],
    interface_points: dict[
        str, tuple[str, torch.Tensor, str, torch.Tensor]
    ],
) -> dict[str, torch.Tensor]:
    """Calculate all interior, boundary, and interface loss components."""
    pde_loss = torch.stack(
        [
            laplace_residual(model, name, interior_points[name]).square().mean()
            for name in model.patch_names
        ]
    ).mean()

    boundary_loss = torch.stack(
        [
            (
                model(name, boundary_points[name])
                - model.exact_solution(boundary_points[name])
            )
            .square()
            .mean()
            for name in model.patch_names
        ]
    ).mean()

    solution_terms = []
    residual_terms = []
    flux_terms = []
    for interface_name, data in interface_points.items():
        name_a, points_a, name_b, points_b = data
        solution_a = model(name_a, points_a)
        solution_b = model(name_b, points_b)
        average_solution = 0.5 * (solution_a + solution_b)
        # This is the average-solution interface condition in the XPINN paper.
        solution_terms.append(
            0.5
            * (
                (solution_a - average_solution).square().mean()
                + (solution_b - average_solution).square().mean()
            )
        )
        residual_a = laplace_residual(model, name_a, points_a)
        residual_b = laplace_residual(model, name_b, points_b)
        residual_terms.append((residual_a - residual_b).square().mean())

        normal_a, normal_b = INTERFACE_NORMALS[interface_name]
        flux_a = normal_flux(
            model, name_a, points_a, normal_a, create_graph=True
        )
        flux_b = normal_flux(
            model, name_b, points_b, normal_b, create_graph=True
        )
        flux_terms.append((flux_a + flux_b).square().mean())

    return {
        "pde": pde_loss,
        "boundary": boundary_loss,
        "interface_solution": torch.stack(solution_terms).mean(),
        "interface_residual": torch.stack(residual_terms).mean(),
        "interface_flux": torch.stack(flux_terms).mean(),
    }


def xpinn_loss(
    model: XPINN,
    interior_points: dict[str, torch.Tensor],
    boundary_points: dict[str, torch.Tensor],
    interface_points: dict[
        str, tuple[str, torch.Tensor, str, torch.Tensor]
    ],
    *,
    pde_weight: float,
    boundary_weight: float,
    interface_solution_weight: float,
    interface_residual_weight: float,
    interface_flux_weight: float,
) -> torch.Tensor:
    terms = xpinn_loss_components(
        model, interior_points, boundary_points, interface_points
    )
    return (
        pde_weight * terms["pde"]
        + boundary_weight * terms["boundary"]
        + interface_solution_weight * terms["interface_solution"]
        + interface_residual_weight * terms["interface_residual"]
        + interface_flux_weight * terms["interface_flux"]
    )


def relative_l2_error(
    model: XPINN, points: dict[str, torch.Tensor]
) -> torch.Tensor:
    squared_error = torch.zeros((), device=next(model.parameters()).device)
    squared_exact = torch.zeros_like(squared_error)
    with torch.no_grad():
        for name in model.patch_names:
            prediction = model(name, points[name])
            exact = model.exact_solution(points[name])
            squared_error = squared_error + (prediction - exact).square().mean()
            squared_exact = squared_exact + exact.square().mean()
    return torch.sqrt(squared_error / squared_exact)


def maximum_boundary_error(
    model: XPINN, boundary_points: dict[str, torch.Tensor]
) -> torch.Tensor:
    with torch.no_grad():
        return torch.stack(
            [
                (
                    model(name, boundary_points[name])
                    - model.exact_solution(boundary_points[name])
                )
                .abs()
                .max()
                for name in model.patch_names
            ]
        ).max()


def interface_diagnostics(
    model: XPINN,
    interface_points: dict[
        str, tuple[str, torch.Tensor, str, torch.Tensor]
    ],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return maximum solution, residual, and outward-normal flux jumps."""
    solution_jumps = []
    residual_jumps = []
    flux_jumps = []
    for interface_name, data in interface_points.items():
        name_a, points_a, name_b, points_b = data
        with torch.no_grad():
            solution_jumps.append(
                (model(name_a, points_a) - model(name_b, points_b)).abs().max()
            )
        residual_jumps.append(
            (
                laplace_residual(model, name_a, points_a)
                - laplace_residual(model, name_b, points_b)
            )
            .detach()
            .abs()
            .max()
        )

        normal_a, normal_b = INTERFACE_NORMALS[interface_name]
        flux_parts = (
            normal_flux(
                model, name_a, points_a, normal_a, create_graph=False
            ),
            normal_flux(
                model, name_b, points_b, normal_b, create_graph=False
            ),
        )
        flux_jumps.append((flux_parts[0] + flux_parts[1]).detach().abs().max())

    return (
        torch.stack(solution_jumps).max(),
        torch.stack(residual_jumps).max(),
        torch.stack(flux_jumps).max(),
    )


def plot_solution(
    model: XPINN,
    postprocessing_dir: Path,
    resolution: int = 101,
) -> None:
    """Save the plotted fields and all arrays needed to reproduce the plot."""
    device = next(model.parameters()).device
    axis = torch.linspace(0.0, 1.0, resolution, device=device)
    xi, eta = torch.meshgrid(axis, axis, indexing="ij")
    local = torch.stack((xi.flatten(), eta.flatten()), dim=1)
    physical_parts, prediction_parts, exact_parts = [], [], []
    with torch.no_grad():
        for name in model.patch_names:
            physical = map_patch(name, local)
            prediction = model(name, physical)
            exact = model.exact_solution(physical)
            physical_parts.append(physical.cpu().numpy())
            prediction_parts.append(prediction.cpu().numpy())
            exact_parts.append(exact.cpu().numpy())
    error_parts = [
        np.abs(prediction - exact)
        for prediction, exact in zip(prediction_parts, exact_parts)
    ]

    postprocessing_dir.mkdir(parents=True, exist_ok=True)
    solution_data_path = (
        postprocessing_dir / f"{model.output_stem}_solution_data.npz"
    )
    np.savez_compressed(
        solution_data_path,
        patch_names=np.asarray(model.patch_names),
        x=np.stack(
            [part[:, 0].reshape(resolution, resolution) for part in physical_parts]
        ),
        y=np.stack(
            [part[:, 1].reshape(resolution, resolution) for part in physical_parts]
        ),
        prediction=np.stack(
            [part.reshape(resolution, resolution) for part in prediction_parts]
        ),
        exact=np.stack(
            [part.reshape(resolution, resolution) for part in exact_parts]
        ),
        absolute_error=np.stack(
            [part.reshape(resolution, resolution) for part in error_parts]
        ),
    )

    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for axis_object, values, title, color_map in (
        (axes[0], prediction_parts, model.solution_title, "viridis"),
        (axes[1], exact_parts, model.exact_solution_label, "viridis"),
        (axes[2], error_parts, "Absolute error", "magma"),
    ):
        combined = np.concatenate(values)
        levels = np.linspace(combined.min(), combined.max(), 51)
        if np.isclose(levels[0], levels[-1]):
            levels = np.linspace(levels[0] - 1.0e-12, levels[-1] + 1.0e-12, 51)
        contour = None
        for physical, patch_values in zip(physical_parts, values):
            contour = axis_object.contourf(
                physical[:, 0].reshape(resolution, resolution),
                physical[:, 1].reshape(resolution, resolution),
                patch_values.reshape(resolution, resolution),
                levels=levels,
                cmap=color_map,
            )
        figure.colorbar(contour, ax=axis_object)
        axis_object.set_title(title)
        axis_object.set_aspect("equal")
        axis_object.set_xlabel("x")
        axis_object.set_ylabel("y")
    figure_path = postprocessing_dir / f"{model.output_stem}_solution.png"
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    print(f"Saved solution data to: {solution_data_path}")
    print(f"Saved solution plot to: {figure_path}")


def plot_training_history(
    loss_history: list[float],
    model: XPINN,
    postprocessing_dir: Path,
) -> None:
    """Save the loss history as both numerical data and a figure."""
    postprocessing_dir.mkdir(parents=True, exist_ok=True)
    loss = np.asarray(loss_history, dtype=float)
    epochs = np.arange(1, loss.size + 1)
    data_path = postprocessing_dir / f"{model.output_stem}_training_data.npz"
    np.savez_compressed(data_path, epoch=epochs, loss=loss)

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.semilogy(epochs, loss)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("XPINN loss")
    axis.set_title(f"{model.solution_title} training history")
    figure.tight_layout()
    figure_path = postprocessing_dir / f"{model.output_stem}_training.png"
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    print(f"Saved training data to: {data_path}")
    print(f"Saved training plot to: {figure_path}")

