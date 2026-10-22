"""Twelve-patch XPINN for the singular L-shaped Laplace problem.

Each of the three original unit squares is split into four rectangles at a
distance ``CORNER_SIZE`` from the re-entrant point. Equal point counts per
patch consequently produce a substantially higher physical sampling density
in the three small corner patches.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LShapeLaplace.RefinedIrregular._xpinn_core import (
    SubdomainNetwork,
    laplace_residual,
    normal_flux,
    plot_training_history,
    select_device,
    set_seed,
)
from training import train


SEED = 42
DEFAULT_DEVICE = "gpu"
N_LAYERS = 3
N_NEURONS = 40
N_INTERIOR_POINTS_PER_PATCH = 2_000
N_BOUNDARY_POINTS_PER_EDGE = 250
N_INTERFACE_POINTS = 500
EPOCHS = 100_000
LEARNING_RATE = 1.0e-3
CORNER_SIZE = 0.6
PDE_WEIGHT = 1.0
BOUNDARY_WEIGHT = 1.0
INTERFACE_SOLUTION_WEIGHT = 1.0
INTERFACE_RESIDUAL_WEIGHT = 1.0
INTERFACE_FLUX_WEIGHT = 1.0
RESULT_PATH = (
    Path(__file__).resolve().parent
    / "results"
    / "best_xpinn_refined_irregular_hyperparameters.json"
)
POSTPROCESSING_DIR = Path(__file__).resolve().parent / "postprocessing"


def build_patch_bounds(
    corner_size: float = CORNER_SIZE,
) -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
    """Return the 12 rectangles forming the refined L-shaped partition."""
    if not 0.0 < corner_size < 1.0:
        raise ValueError("corner_size must lie strictly between zero and one")
    r = corner_size
    return {
        "ur_corner": ((0.0, 0.0), (r, r)),
        "ur_right": ((r, 0.0), (1.0, r)),
        "ur_top": ((0.0, r), (r, 1.0)),
        "ur_outer": ((r, r), (1.0, 1.0)),
        "ul_corner": ((-r, 0.0), (0.0, r)),
        "ul_left": ((-1.0, 0.0), (-r, r)),
        "ul_top": ((-r, r), (0.0, 1.0)),
        "ul_outer": ((-1.0, r), (-r, 1.0)),
        "ll_corner": ((-r, -r), (0.0, 0.0)),
        "ll_left": ((-1.0, -r), (-r, 0.0)),
        "ll_bottom": ((-r, -1.0), (0.0, -r)),
        "ll_outer": ((-1.0, -1.0), (-r, -r)),
    }


PATCH_BOUNDS = build_patch_bounds()
PATCH_NAMES = tuple(PATCH_BOUNDS)

# Physical exterior edges of each refined rectangle.
EXTERIOR_EDGES = {
    "ur_corner": ("bottom",),
    "ur_right": ("bottom", "right"),
    "ur_top": ("top",),
    "ur_outer": ("right", "top"),
    "ul_corner": (),
    "ul_left": ("left",),
    "ul_top": ("top",),
    "ul_outer": ("left", "top"),
    "ll_corner": ("right",),
    "ll_left": ("left",),
    "ll_bottom": ("bottom", "right"),
    "ll_outer": ("left", "bottom"),
}


def discover_interfaces(bounds=PATCH_BOUNDS) -> tuple[dict, ...]:
    """Find all positive-length common edges in the rectangular partition."""
    names = tuple(bounds)
    interfaces = []
    tolerance = 1.0e-12
    for index, name_a in enumerate(names):
        (ax0, ay0), (ax1, ay1) = bounds[name_a]
        for name_b in names[index + 1 :]:
            (bx0, by0), (bx1, by1) = bounds[name_b]

            y0, y1 = max(ay0, by0), min(ay1, by1)
            if y1 - y0 > tolerance:
                if abs(ax1 - bx0) < tolerance:
                    interfaces.append(
                        {
                            "name": f"{name_a}__{name_b}",
                            "a": name_a,
                            "b": name_b,
                            "orientation": "vertical",
                            "constant": ax1,
                            "start": y0,
                            "end": y1,
                            "edge_a": "right",
                            "edge_b": "left",
                            "normal_a": (1.0, 0.0),
                            "normal_b": (-1.0, 0.0),
                        }
                    )
                elif abs(bx1 - ax0) < tolerance:
                    interfaces.append(
                        {
                            "name": f"{name_a}__{name_b}",
                            "a": name_a,
                            "b": name_b,
                            "orientation": "vertical",
                            "constant": ax0,
                            "start": y0,
                            "end": y1,
                            "edge_a": "left",
                            "edge_b": "right",
                            "normal_a": (-1.0, 0.0),
                            "normal_b": (1.0, 0.0),
                        }
                    )

            x0, x1 = max(ax0, bx0), min(ax1, bx1)
            if x1 - x0 > tolerance:
                if abs(ay1 - by0) < tolerance:
                    interfaces.append(
                        {
                            "name": f"{name_a}__{name_b}",
                            "a": name_a,
                            "b": name_b,
                            "orientation": "horizontal",
                            "constant": ay1,
                            "start": x0,
                            "end": x1,
                            "edge_a": "top",
                            "edge_b": "bottom",
                            "normal_a": (0.0, 1.0),
                            "normal_b": (0.0, -1.0),
                        }
                    )
                elif abs(by1 - ay0) < tolerance:
                    interfaces.append(
                        {
                            "name": f"{name_a}__{name_b}",
                            "a": name_a,
                            "b": name_b,
                            "orientation": "horizontal",
                            "constant": ay0,
                            "start": x0,
                            "end": x1,
                            "edge_a": "bottom",
                            "edge_b": "top",
                            "normal_a": (0.0, -1.0),
                            "normal_b": (0.0, 1.0),
                        }
                    )
    return tuple(interfaces)


INTERFACES = discover_interfaces()
if len(INTERFACES) != 16:
    raise RuntimeError(f"Expected 16 refined interfaces, found {len(INTERFACES)}")


def exact_solution(points: torch.Tensor) -> torch.Tensor:
    """Return r^(2/3) sin(2 theta/3) with theta in [0, 2*pi)."""
    x, y = points[:, 0], points[:, 1]
    radius = torch.sqrt(x.square() + y.square())
    theta = torch.remainder(torch.atan2(y, x), 2.0 * math.pi)
    return radius.pow(2.0 / 3.0) * torch.sin(2.0 * theta / 3.0)


def map_patch(name: str, local_points: torch.Tensor) -> torch.Tensor:
    """Map local unit-square coordinates to a refined physical rectangle."""
    lower, upper = PATCH_BOUNDS[name]
    lower_tensor = local_points.new_tensor(lower)
    upper_tensor = local_points.new_tensor(upper)
    return lower_tensor + local_points * (upper_tensor - lower_tensor)


class XPINN(nn.Module):
    """Twelve independent subnetworks on the corner-refined partition."""

    patch_names = PATCH_NAMES
    output_stem = "xpinn_refined_irregular"
    solution_title = "Refined XPINN singular solution"
    exact_solution_label = r"Exact singular solution: $r^{2/3}\sin(2\theta/3)$"

    def __init__(self, n_layers: int, n_neurons: int) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.networks = nn.ModuleDict(
            {
                name: SubdomainNetwork(
                    n_layers, n_neurons, PATCH_BOUNDS[name][0], PATCH_BOUNDS[name][1]
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
    """Use equal counts per patch, producing dense corner sampling."""
    return {
        name: map_patch(
            name, torch.rand(count, 2, device=device, generator=generator)
        )
        for name in PATCH_NAMES
    }


def edge_points(
    patch_name: str, edge: str, count: int, device: torch.device
) -> torch.Tensor:
    line = torch.linspace(0.0, 1.0, count, device=device)
    zero, one = torch.zeros_like(line), torch.ones_like(line)
    if edge == "left":
        local = torch.stack((zero, line), dim=1)
    elif edge == "right":
        local = torch.stack((one, line), dim=1)
    elif edge == "bottom":
        local = torch.stack((line, zero), dim=1)
    elif edge == "top":
        local = torch.stack((line, one), dim=1)
    else:
        raise ValueError(f"Unknown edge: {edge}")
    return map_patch(patch_name, local)


def generate_boundary_points(
    points_per_edge: int, device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        name: torch.cat(
            [edge_points(name, edge, points_per_edge, device) for edge in edges],
            dim=0,
        )
        for name, edges in EXTERIOR_EDGES.items()
        if edges
    }


def generate_interface_points(
    count: int, device: torch.device
) -> dict[str, tuple[str, torch.Tensor, tuple, str, torch.Tensor, tuple]]:
    result = {}
    for interface in INTERFACES:
        line = torch.linspace(
            interface["start"], interface["end"], count, device=device
        )
        constant = torch.full_like(line, interface["constant"])
        if interface["orientation"] == "vertical":
            points = torch.stack((constant, line), dim=1)
        else:
            points = torch.stack((line, constant), dim=1)
        result[interface["name"]] = (
            interface["a"],
            points,
            interface["normal_a"],
            interface["b"],
            points.clone(),
            interface["normal_b"],
        )
    return result


def xpinn_loss_components(
    model: XPINN,
    interior_points: dict[str, torch.Tensor],
    boundary_points: dict[str, torch.Tensor],
    interface_points,
) -> dict[str, torch.Tensor]:
    pde = torch.stack(
        [
            laplace_residual(model, name, interior_points[name]).square().mean()
            for name in model.patch_names
        ]
    ).mean()
    boundary = torch.stack(
        [
            (
                model(name, points) - model.exact_solution(points)
            ).square().mean()
            for name, points in boundary_points.items()
        ]
    ).mean()

    solution_terms, residual_terms, flux_terms = [], [], []
    for name_a, points_a, normal_a, name_b, points_b, normal_b in (
        interface_points.values()
    ):
        solution_a = model(name_a, points_a)
        solution_b = model(name_b, points_b)
        average = 0.5 * (solution_a + solution_b)
        solution_terms.append(
            0.5
            * (
                (solution_a - average).square().mean()
                + (solution_b - average).square().mean()
            )
        )
        residual_a = laplace_residual(model, name_a, points_a)
        residual_b = laplace_residual(model, name_b, points_b)
        residual_terms.append((residual_a - residual_b).square().mean())
        flux_a = normal_flux(
            model, name_a, points_a, normal_a, create_graph=True
        )
        flux_b = normal_flux(
            model, name_b, points_b, normal_b, create_graph=True
        )
        flux_terms.append((flux_a + flux_b).square().mean())

    return {
        "pde": pde,
        "boundary": boundary,
        "interface_solution": torch.stack(solution_terms).mean(),
        "interface_residual": torch.stack(residual_terms).mean(),
        "interface_flux": torch.stack(flux_terms).mean(),
    }


def xpinn_loss(
    model: XPINN,
    interior_points,
    boundary_points,
    interface_points,
    *,
    pde_weight: float = PDE_WEIGHT,
    boundary_weight: float = BOUNDARY_WEIGHT,
    interface_solution_weight: float = INTERFACE_SOLUTION_WEIGHT,
    interface_residual_weight: float = INTERFACE_RESIDUAL_WEIGHT,
    interface_flux_weight: float = INTERFACE_FLUX_WEIGHT,
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


def relative_l2_error(model: XPINN, points: dict[str, torch.Tensor]) -> torch.Tensor:
    squared_error = torch.zeros((), device=next(model.parameters()).device)
    squared_exact = torch.zeros_like(squared_error)
    with torch.no_grad():
        for name, values in points.items():
            area = math.prod(
                upper - lower
                for lower, upper in zip(*PATCH_BOUNDS[name])
            )
            prediction = model(name, values)
            exact = model.exact_solution(values)
            squared_error += area * (prediction - exact).square().mean()
            squared_exact += area * exact.square().mean()
    return torch.sqrt(squared_error / squared_exact)


def maximum_boundary_error(model: XPINN, boundary_points) -> torch.Tensor:
    with torch.no_grad():
        return torch.stack(
            [
                (model(name, points) - model.exact_solution(points)).abs().max()
                for name, points in boundary_points.items()
            ]
        ).max()


def interface_diagnostics(model: XPINN, interface_points):
    solution_jumps, residual_jumps, flux_jumps = [], [], []
    for name_a, points_a, normal_a, name_b, points_b, normal_b in (
        interface_points.values()
    ):
        with torch.no_grad():
            solution_jumps.append(
                (model(name_a, points_a) - model(name_b, points_b)).abs().max()
            )
        residual_jumps.append(
            (
                laplace_residual(model, name_a, points_a)
                - laplace_residual(model, name_b, points_b)
            ).detach().abs().max()
        )
        flux_a = normal_flux(
            model, name_a, points_a, normal_a, create_graph=False
        )
        flux_b = normal_flux(
            model, name_b, points_b, normal_b, create_graph=False
        )
        flux_jumps.append((flux_a + flux_b).detach().abs().max())
    return (
        torch.stack(solution_jumps).max(),
        torch.stack(residual_jumps).max(),
        torch.stack(flux_jumps).max(),
    )


def load_hyperparameters(path: Path = RESULT_PATH) -> tuple[int, int]:
    if not path.exists():
        raise FileNotFoundError(
            f"No refined XPINN tuning result found at {path}. Supply that "
            "JSON file or run with --no-use-tuned."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("best_params", payload)
    return int(params["n_layers"]), int(params["n_neurons"])


def plot_solution(model: XPINN, resolution: int = 101) -> None:
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

    POSTPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    data_path = POSTPROCESSING_DIR / f"{model.output_stem}_solution_data.npz"
    np.savez_compressed(
        data_path,
        patch_names=np.asarray(model.patch_names),
        x=np.stack([p[:, 0].reshape(resolution, resolution) for p in physical_parts]),
        y=np.stack([p[:, 1].reshape(resolution, resolution) for p in physical_parts]),
        prediction=np.stack([p.reshape(resolution, resolution) for p in prediction_parts]),
        exact=np.stack([p.reshape(resolution, resolution) for p in exact_parts]),
        absolute_error=np.stack([p.reshape(resolution, resolution) for p in error_parts]),
    )

    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for axis_object, values, title, color_map in (
        (axes[0], prediction_parts, model.solution_title, "viridis"),
        (axes[1], exact_parts, model.exact_solution_label, "viridis"),
        (axes[2], error_parts, "Absolute error", "magma"),
    ):
        combined = np.concatenate(values)
        lower, upper = combined.min(), combined.max()
        if np.isclose(lower, upper):
            lower, upper = lower - 1.0e-12, upper + 1.0e-12
        levels = np.linspace(lower, upper, 51)
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
    figure_path = POSTPROCESSING_DIR / f"{model.output_stem}_solution.png"
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    print(f"Saved solution data to: {data_path}")
    print(f"Saved solution plot to: {figure_path}")


def main(
    device_name: str = DEFAULT_DEVICE,
    n_layers: int = N_LAYERS,
    n_neurons: int = N_NEURONS,
    epochs: int = EPOCHS,
    use_tuned: bool | None = None,
) -> None:
    set_seed(SEED)
    device = select_device(device_name)
    should_load = RESULT_PATH.exists() if use_tuned is None else use_tuned
    if should_load:
        n_layers, n_neurons = load_hyperparameters()
        print(f"Loaded tuned hyperparameters from: {RESULT_PATH}")
    print(f"Using device: {device}")
    print(f"Refined XPINN: 12 patches, rho={CORNER_SIZE}")
    print(f"Subnetwork architecture: layers={n_layers}, neurons={n_neurons}")

    model = XPINN(n_layers, n_neurons).to(device)
    generator = torch.Generator(device=device).manual_seed(SEED)
    interior = sample_interior_points(
        N_INTERIOR_POINTS_PER_PATCH, device, generator
    )
    boundary = generate_boundary_points(N_BOUNDARY_POINTS_PER_EDGE, device)
    interfaces = generate_interface_points(N_INTERFACE_POINTS, device)
    loss_fn = lambda: xpinn_loss(model, interior, boundary, interfaces)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=50, min_lr=1.0e-5
    )
    loss_history = train(
        model.parameters(),
        loss_fn,
        optimizer,
        epochs=epochs,
        scheduler=scheduler,
        early_stopping_patience=500,
        min_delta=1.0e-9,
        print_every=100,
    )

    validation = sample_interior_points(2_000, device, generator)
    validation_interfaces = generate_interface_points(501, device)
    validation_boundary = generate_boundary_points(251, device)
    relative_error = relative_l2_error(model, validation)
    residual_mse = torch.stack(
        [
            laplace_residual(model, name, validation[name]).detach().square().mean()
            for name in model.patch_names
        ]
    ).mean()
    boundary_error = maximum_boundary_error(model, validation_boundary)
    solution_jump, residual_jump, flux_jump = interface_diagnostics(
        model, validation_interfaces
    )
    print(f"Relative physical L2 error: {relative_error.item():.6e}")
    print(f"Interior Laplace residual MSE: {residual_mse.item():.6e}")
    print(f"Maximum boundary error: {boundary_error.item():.6e}")
    print(f"Maximum interface solution jump: {solution_jump.item():.6e}")
    print(f"Maximum interface residual jump: {residual_jump.item():.6e}")
    print(f"Maximum interface normal-flux jump: {flux_jump.item():.6e}")
    plot_solution(model)
    plot_training_history(
        loss_history, model, postprocessing_dir=POSTPROCESSING_DIR
    )
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve singular L-shaped Laplace on a 12-patch XPINN"
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--layers", type=int, default=N_LAYERS)
    parser.add_argument("--neurons", type=int, default=N_NEURONS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    tuned_group = parser.add_mutually_exclusive_group()
    tuned_group.add_argument("--use-tuned", dest="use_tuned", action="store_true")
    tuned_group.add_argument("--no-use-tuned", dest="use_tuned", action="store_false")
    parser.set_defaults(use_tuned=None)
    arguments = parser.parse_args()
    main(arguments.device, arguments.layers, arguments.neurons, arguments.epochs, arguments.use_tuned)
