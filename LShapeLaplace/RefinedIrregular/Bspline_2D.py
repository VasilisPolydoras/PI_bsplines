"""Twelve-patch B-spline XPINN for the singular L-shaped Laplace problem.

The partition matches ``XPINN_2D.py``. Compatible interface
traces are shared exactly, while residual and physical normal-flux continuity
are optimized at all 16 internal interfaces. Affine-map scale factors are
included in every physical derivative. The basis budget of each original
unit-square patch is split anisotropically between its corner and outer
coordinate intervals, concentrating resolution near the re-entrant corner
without multiplying the original coefficient count by four.
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LShapeLaplace.RefinedIrregular._bspline_core import (
    basis_derivative_matrix,
    basis_matrix,
    greville_abscissae,
    open_endpoint_clustered_knots,
    plot_training_history,
    select_device,
    set_seed,
)
from LShapeLaplace.RefinedIrregular.XPINN_2D import (
    CORNER_SIZE,
    EXTERIOR_EDGES,
    INTERFACES,
    PATCH_BOUNDS,
    PATCH_NAMES,
    map_patch,
)
from training import train


SEED = 42
DEFAULT_DEVICE = "gpu"
N_BASIS = 18
DEGREE = 3
N_INTERIOR_POINTS_PER_PATCH = 4_000
N_INTERFACE_POINTS = 500
EPOCHS = 100_000
LEARNING_RATE = 1.0e-3
PDE_WEIGHT = 1.0
INTERFACE_RESIDUAL_WEIGHT = 1.0
INTERFACE_FLUX_WEIGHT = 1.0
CORNER_BASIS_FRACTION = 0.60
RESULT_PATH = (
    Path(__file__).resolve().parent
    / "results"
    / "best_bspline_xpinn_refined_irregular_hyperparameters.json"
)
POSTPROCESSING_DIR = Path(__file__).resolve().parent / "postprocessing"


# ``corner`` denotes a coordinate interval touching the re-entrant point;
# ``outer`` denotes the remaining interval. Matching labels in the tangential
# direction guarantee equal trace sizes on every conforming interface.
PATCH_AXIS_REGIONS = {
    "ur_corner": ("corner", "corner"),
    "ur_right": ("outer", "corner"),
    "ur_top": ("corner", "outer"),
    "ur_outer": ("outer", "outer"),
    "ul_corner": ("corner", "corner"),
    "ul_left": ("outer", "corner"),
    "ul_top": ("corner", "outer"),
    "ul_outer": ("outer", "outer"),
    "ll_corner": ("corner", "corner"),
    "ll_left": ("outer", "corner"),
    "ll_bottom": ("corner", "outer"),
    "ll_outer": ("outer", "outer"),
}

# Physical quadrature measures used when combining patchwise and interface
# mean-square residuals. Without these factors, a 0.2 x 0.2 corner patch would
# receive the same loss weight as a 0.8 x 0.8 outer patch.
PATCH_AREAS = {
    name: (upper[0] - lower[0]) * (upper[1] - lower[1])
    for name, (lower, upper) in PATCH_BOUNDS.items()
}
TOTAL_DOMAIN_AREA = sum(PATCH_AREAS.values())
INTERFACE_LENGTHS = {
    interface["name"]: interface["end"] - interface["start"]
    for interface in INTERFACES
}
TOTAL_INTERFACE_LENGTH = sum(INTERFACE_LENGTHS.values())


def split_basis_count(
    n_basis: int,
    degree: int,
    corner_fraction: float = CORNER_BASIS_FRACTION,
) -> tuple[int, int]:
    """Split a former one-patch basis count into corner and outer counts.

    Since ``n_corner + n_outer == n_basis``, the four refined rectangles of
    each original square contain exactly ``n_basis**2`` raw coefficients.
    """
    if not 0.5 < corner_fraction < 1.0:
        raise ValueError("corner_fraction must lie strictly between 0.5 and 1")
    n_corner = int(round(n_basis * corner_fraction))
    n_outer = n_basis - n_corner
    if min(n_corner, n_outer) <= degree:
        raise ValueError(
            "Both refined basis counts must exceed degree; increase n_basis "
            "or reduce degree/corner_fraction"
        )
    return n_corner, n_outer


def exact_solution(points: torch.Tensor) -> torch.Tensor:
    x, y = points[:, 0], points[:, 1]
    radius = torch.sqrt(x.square() + y.square())
    theta = torch.remainder(torch.atan2(y, x), 2.0 * math.pi)
    return radius.pow(2.0 / 3.0) * torch.sin(2.0 * theta / 3.0)


def edge_local_points(
    edge: str, values: torch.Tensor
) -> torch.Tensor:
    zero, one = torch.zeros_like(values), torch.ones_like(values)
    if edge == "left":
        return torch.stack((zero, values), dim=1)
    if edge == "right":
        return torch.stack((one, values), dim=1)
    if edge == "bottom":
        return torch.stack((values, zero), dim=1)
    if edge == "top":
        return torch.stack((values, one), dim=1)
    raise ValueError(f"Unknown edge: {edge}")


def extract_edge(coefficients: torch.Tensor, edge: str) -> torch.Tensor:
    if edge == "left":
        return coefficients[0, :]
    if edge == "right":
        return coefficients[-1, :]
    if edge == "bottom":
        return coefficients[:, 0]
    if edge == "top":
        return coefficients[:, -1]
    raise ValueError(f"Unknown edge: {edge}")


def replace_edge(
    coefficients: torch.Tensor, edge: str, values: torch.Tensor
) -> torch.Tensor:
    """Return a differentiable tensor with one trace replaced."""
    if edge == "left":
        return torch.cat((values[None, :], coefficients[1:, :]), dim=0)
    if edge == "right":
        return torch.cat((coefficients[:-1, :], values[None, :]), dim=0)
    if edge == "bottom":
        return torch.cat((values[:, None], coefficients[:, 1:]), dim=1)
    if edge == "top":
        return torch.cat((coefficients[:, :-1], values[:, None]), dim=1)
    raise ValueError(f"Unknown edge: {edge}")


# Every entry copies a complete coefficient trace from an already assembled
# neighboring patch. The ordering makes all 16 interfaces exact C0 constraints.
ASSEMBLY_RECIPES = {
    "ur_corner": (),
    "ur_right": (("left", "ur_corner", "right"),),
    "ur_top": (("bottom", "ur_corner", "top"),),
    "ur_outer": (
        ("left", "ur_top", "right"),
        ("bottom", "ur_right", "top"),
    ),
    "ul_corner": (("right", "ur_corner", "left"),),
    "ul_left": (("right", "ul_corner", "left"),),
    "ul_top": (
        ("bottom", "ul_corner", "top"),
        ("right", "ur_top", "left"),
    ),
    "ul_outer": (
        ("right", "ul_top", "left"),
        ("bottom", "ul_left", "top"),
    ),
    "ll_corner": (("top", "ul_corner", "bottom"),),
    "ll_left": (
        ("right", "ll_corner", "left"),
        ("top", "ul_left", "bottom"),
    ),
    "ll_bottom": (("top", "ll_corner", "bottom"),),
    "ll_outer": (
        ("right", "ll_bottom", "left"),
        ("top", "ll_left", "bottom"),
    ),
}


class bsplines(torch.nn.Module):
    """Compatible tensor-product splines on the 12 refined rectangles."""

    patch_names = PATCH_NAMES
    output_stem = "bspline_xpinn_refined_irregular"
    solution_title = "Refined B-spline XPINN singular solution"
    exact_solution_label = r"Exact singular solution: $r^{2/3}\sin(2\theta/3)$"

    def __init__(
        self,
        n_basis: int,
        degree: int,
        device: torch.device,
        corner_fraction: float = CORNER_BASIS_FRACTION,
    ) -> None:
        super().__init__()
        if degree < 2:
            raise ValueError("The strong Laplace formulation requires degree >= 2")
        n_corner, n_outer = split_basis_count(
            n_basis, degree, corner_fraction
        )
        self.n_basis = n_basis
        self.n_basis_corner = n_corner
        self.n_basis_outer = n_outer
        self.corner_fraction = corner_fraction
        self.degree = degree
        self.register_buffer(
            "knots_corner",
            open_endpoint_clustered_knots(n_corner, degree, device),
        )
        self.register_buffer(
            "knots_outer",
            open_endpoint_clustered_knots(n_outer, degree, device),
        )
        self.raw_coefficients = torch.nn.ParameterDict(
            {
                name: torch.nn.Parameter(
                    torch.zeros(*self.patch_basis_shape(name), device=device)
                )
                for name in self.patch_names
            }
        )
        masks, values = self._build_boundary_data(device)
        for name in self.patch_names:
            self.register_buffer(f"{name}_fixed_mask", masks[name])
            self.register_buffer(f"{name}_fixed_values", values[name])

    @staticmethod
    def exact_solution(points: torch.Tensor) -> torch.Tensor:
        return exact_solution(points)

    def patch_scale(self, name: str) -> tuple[float, float]:
        lower, upper = PATCH_BOUNDS[name]
        return upper[0] - lower[0], upper[1] - lower[1]

    def basis_count(self, region: str) -> int:
        return (
            self.n_basis_corner if region == "corner" else self.n_basis_outer
        )

    def knots_for_region(self, region: str) -> torch.Tensor:
        return getattr(self, f"knots_{region}")

    def patch_basis_shape(self, name: str) -> tuple[int, int]:
        region_x, region_y = PATCH_AXIS_REGIONS[name]
        return self.basis_count(region_x), self.basis_count(region_y)

    def patch_knots(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        region_x, region_y = PATCH_AXIS_REGIONS[name]
        return self.knots_for_region(region_x), self.knots_for_region(region_y)

    def _boundary_coefficients(self, name: str, edge: str) -> torch.Tensor:
        knots_x, knots_y = self.patch_knots(name)
        tangential_knots = knots_y if edge in ("left", "right") else knots_x
        greville = greville_abscissae(tangential_knots, self.degree)
        local = edge_local_points(edge, greville)
        physical = map_patch(name, local)
        matrix = basis_matrix(greville, tangential_knots, self.degree)
        return torch.linalg.solve(matrix, self.exact_solution(physical))

    def _build_boundary_data(self, device: torch.device):
        masks = {
            name: torch.zeros(
                *self.patch_basis_shape(name), dtype=torch.bool, device=device
            )
            for name in self.patch_names
        }
        values = {
            name: torch.zeros(*self.patch_basis_shape(name), device=device)
            for name in self.patch_names
        }
        for name, edges in EXTERIOR_EDGES.items():
            for edge in edges:
                coefficients = self._boundary_coefficients(name, edge)
                if edge == "left":
                    masks[name][0, :] = True
                    values[name][0, :] = coefficients
                elif edge == "right":
                    masks[name][-1, :] = True
                    values[name][-1, :] = coefficients
                elif edge == "bottom":
                    masks[name][:, 0] = True
                    values[name][:, 0] = coefficients
                elif edge == "top":
                    masks[name][:, -1] = True
                    values[name][:, -1] = coefficients
        return masks, values

    def coefficient_tensors(self) -> dict[str, torch.Tensor]:
        assembled: dict[str, torch.Tensor] = {}
        for name in self.patch_names:
            candidate = self.raw_coefficients[name]
            for target_edge, source_name, source_edge in ASSEMBLY_RECIPES[name]:
                candidate = replace_edge(
                    candidate, target_edge,
                    extract_edge(assembled[source_name], source_edge),
                )
            candidate = torch.where(
                getattr(self, f"{name}_fixed_mask"),
                getattr(self, f"{name}_fixed_values"),
                candidate,
            )
            assembled[name] = candidate
        return assembled

    def operators(
        self, patch_name: str, local_points: torch.Tensor, max_order: int = 2
    ):
        xi, eta = local_points[:, 0], local_points[:, 1]
        knots_x, knots_y = self.patch_knots(patch_name)
        return {
            "x": tuple(
                basis_derivative_matrix(xi, knots_x, self.degree, order)
                for order in range(max_order + 1)
            ),
            "y": tuple(
                basis_derivative_matrix(eta, knots_y, self.degree, order)
                for order in range(max_order + 1)
            ),
        }

    def evaluate(
        self,
        patch_name: str,
        coefficients: torch.Tensor,
        operators,
        dx: int = 0,
        dy: int = 0,
    ) -> torch.Tensor:
        hx, hy = self.patch_scale(patch_name)
        local_value = torch.einsum(
            "ni,ij,nj->n",
            operators["x"][dx],
            coefficients,
            operators["y"][dy],
        )
        return local_value / (hx**dx * hy**dy)

    def forward(self, patch_name: str, local_points: torch.Tensor) -> torch.Tensor:
        coefficients = self.coefficient_tensors()[patch_name]
        return self.evaluate(
            patch_name,
            coefficients,
            self.operators(patch_name, local_points, 0),
        )[:, None]


def sample_local_points(
    count: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    return torch.rand(count, 2, device=device, generator=generator)


def precompute_operators(
    model: bsplines,
    local_points: torch.Tensor,
    max_order: int = 2,
    patch_name: str | None = None,
):
    if patch_name is not None:
        return model.operators(patch_name, local_points, max_order=max_order)
    return {
        name: model.operators(name, local_points, max_order=max_order)
        for name in model.patch_names
    }


def generate_interface_local_points(
    count: int, device: torch.device
) -> dict[str, tuple[str, torch.Tensor, tuple, str, torch.Tensor, tuple]]:
    line = torch.linspace(0.0, 1.0, count, device=device)
    return {
        interface["name"]: (
            interface["a"],
            edge_local_points(interface["edge_a"], line),
            interface["normal_a"],
            interface["b"],
            edge_local_points(interface["edge_b"], line),
            interface["normal_b"],
        )
        for interface in INTERFACES
    }


def precompute_interface_operators(model: bsplines, interface_points):
    return {
        key: (
            name_a,
            precompute_operators(
                model, points_a, max_order=2, patch_name=name_a
            ),
            normal_a,
            name_b,
            precompute_operators(
                model, points_b, max_order=2, patch_name=name_b
            ),
            normal_b,
        )
        for key, (
            name_a,
            points_a,
            normal_a,
            name_b,
            points_b,
            normal_b,
        ) in interface_points.items()
    }


def patch_laplacian(model, coefficients, name, operators):
    return model.evaluate(name, coefficients, operators, dx=2, dy=0) + model.evaluate(
        name, coefficients, operators, dx=0, dy=2
    )


def laplace_residual_loss(model: bsplines, operators) -> torch.Tensor:
    """Physical-area-weighted mean-square Laplace residual."""
    coefficients = model.coefficient_tensors()
    weighted_terms = torch.stack(
        [
            PATCH_AREAS[name]
            * patch_laplacian(model, coefficients[name], name, operators[name])
            .square().mean()
            for name in model.patch_names
        ]
    )
    return weighted_terms.sum() / TOTAL_DOMAIN_AREA


def interface_residual_loss(model: bsplines, interface_operators) -> torch.Tensor:
    """Physical-length-weighted residual-continuity loss."""
    coefficients = model.coefficient_tensors()
    terms = []
    for interface_name, (
        name_a, operators_a, _, name_b, operators_b, _
    ) in interface_operators.items():
        residual_a = patch_laplacian(
            model, coefficients[name_a], name_a, operators_a
        )
        residual_b = patch_laplacian(
            model, coefficients[name_b], name_b, operators_b
        )
        terms.append(
            INTERFACE_LENGTHS[interface_name]
            * (residual_a - residual_b).square().mean()
        )
    return torch.stack(terms).sum() / TOTAL_INTERFACE_LENGTH


def normal_flux(model, coefficients, name, operators, normal):
    u_x = model.evaluate(name, coefficients, operators, dx=1, dy=0)
    u_y = model.evaluate(name, coefficients, operators, dx=0, dy=1)
    return -(normal[0] * u_x + normal[1] * u_y)


def interface_flux_loss(model: bsplines, interface_operators) -> torch.Tensor:
    """Physical-length-weighted normal-flux-continuity loss."""
    coefficients = model.coefficient_tensors()
    terms = []
    for interface_name, (
        name_a, operators_a, normal_a, name_b, operators_b, normal_b
    ) in interface_operators.items():
        flux_a = normal_flux(
            model, coefficients[name_a], name_a, operators_a, normal_a
        )
        flux_b = normal_flux(
            model, coefficients[name_b], name_b, operators_b, normal_b
        )
        terms.append(
            INTERFACE_LENGTHS[interface_name]
            * (flux_a + flux_b).square().mean()
        )
    return torch.stack(terms).sum() / TOTAL_INTERFACE_LENGTH


def strong_loss_components(model, interior_operators, interface_operators):
    return {
        "pde": laplace_residual_loss(model, interior_operators),
        "interface_residual": interface_residual_loss(model, interface_operators),
        "interface_flux": interface_flux_loss(model, interface_operators),
    }


def strong_residual_loss(
    model,
    interior_operators,
    interface_operators,
    *,
    pde_weight: float = PDE_WEIGHT,
    interface_residual_weight: float = INTERFACE_RESIDUAL_WEIGHT,
    interface_flux_weight: float = INTERFACE_FLUX_WEIGHT,
):
    terms = strong_loss_components(model, interior_operators, interface_operators)
    return (
        pde_weight * terms["pde"]
        + interface_residual_weight * terms["interface_residual"]
        + interface_flux_weight * terms["interface_flux"]
    )


def relative_l2_error(model: bsplines, local_points, operators=None):
    if operators is None:
        operators = precompute_operators(model, local_points, 0)
    coefficients = model.coefficient_tensors()
    squared_error = torch.zeros((), device=local_points.device)
    squared_exact = torch.zeros_like(squared_error)
    for name in model.patch_names:
        hx, hy = model.patch_scale(name)
        physical = map_patch(name, local_points)
        prediction = model.evaluate(name, coefficients[name], operators[name])
        exact = model.exact_solution(physical)
        squared_error += hx * hy * (prediction - exact).square().mean()
        squared_exact += hx * hy * exact.square().mean()
    return torch.sqrt(squared_error / squared_exact)


def boundary_and_interface_errors(model: bsplines, sample_count: int = 501):
    device = model.knots_corner.device
    line = torch.linspace(0.0, 1.0, sample_count, device=device)
    maximum_boundary_error = torch.zeros((), device=device)
    for name, edges in EXTERIOR_EDGES.items():
        for edge in edges:
            local = edge_local_points(edge, line)
            prediction = model(name, local).squeeze(1)
            exact = model.exact_solution(map_patch(name, local))
            maximum_boundary_error = torch.maximum(
                maximum_boundary_error, (prediction - exact).abs().max()
            )

    maximum_jump = torch.zeros((), device=device)
    for _, (name_a, points_a, _, name_b, points_b, _) in (
        generate_interface_local_points(sample_count, device).items()
    ):
        jump = (model(name_a, points_a) - model(name_b, points_b)).abs().max()
        maximum_jump = torch.maximum(maximum_jump, jump)
    return maximum_boundary_error, maximum_jump


def maximum_interface_residual_jump(model, interface_operators):
    coefficients = model.coefficient_tensors()
    jumps = []
    for name_a, operators_a, _, name_b, operators_b, _ in (
        interface_operators.values()
    ):
        residual_a = patch_laplacian(
            model, coefficients[name_a], name_a, operators_a
        )
        residual_b = patch_laplacian(
            model, coefficients[name_b], name_b, operators_b
        )
        jumps.append((residual_a - residual_b).abs().max())
    return torch.stack(jumps).max()


def maximum_interface_flux_jump(model, interface_operators):
    coefficients = model.coefficient_tensors()
    jumps = []
    for name_a, operators_a, normal_a, name_b, operators_b, normal_b in (
        interface_operators.values()
    ):
        flux_a = normal_flux(
            model, coefficients[name_a], name_a, operators_a, normal_a
        )
        flux_b = normal_flux(
            model, coefficients[name_b], name_b, operators_b, normal_b
        )
        jumps.append((flux_a + flux_b).abs().max())
    return torch.stack(jumps).max()


def load_hyperparameters(path: Path = RESULT_PATH) -> tuple[int, int]:
    if not path.exists():
        raise FileNotFoundError(
            f"No refined B-spline tuning result found at {path}. Supply that "
            "JSON file or run with --no-use-tuned."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("best_params", payload)
    return int(params["n_basis"]), int(params["degree"])


def plot_solution(model: bsplines, resolution: int = 101) -> None:
    device = model.knots_corner.device
    axis = torch.linspace(0.0, 1.0, resolution, device=device)
    xi, eta = torch.meshgrid(axis, axis, indexing="ij")
    local = torch.stack((xi.flatten(), eta.flatten()), dim=1)
    operators = precompute_operators(model, local, 0)
    coefficients = model.coefficient_tensors()
    physical_parts, prediction_parts, exact_parts = [], [], []
    with torch.no_grad():
        for name in model.patch_names:
            physical = map_patch(name, local)
            prediction = model.evaluate(name, coefficients[name], operators[name])
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
    n_basis: int = N_BASIS,
    degree: int = DEGREE,
    epochs: int = EPOCHS,
    use_tuned: bool | None = None,
) -> None:
    set_seed(SEED)
    device = select_device(device_name)
    should_load = RESULT_PATH.exists() if use_tuned is None else use_tuned
    if should_load:
        n_basis, degree = load_hyperparameters()
        print(f"Loaded tuned hyperparameters from: {RESULT_PATH}")
    print(f"Using device: {device}")
    print(f"Refined B-spline XPINN: 12 patches, rho={CORNER_SIZE}")
    print(
        f"Basis budget: n_basis={n_basis}, degree={degree}, "
        f"corner_fraction={CORNER_BASIS_FRACTION:.2f}"
    )

    model = bsplines(n_basis, degree, device)
    print(
        f"Per coordinate: corner={model.n_basis_corner}, "
        f"outer={model.n_basis_outer}; raw coefficients="
        f"{sum(parameter.numel() for parameter in model.raw_coefficients.values())} "
        f"(three-patch reference={3 * n_basis**2})"
    )
    generator = torch.Generator(device=device).manual_seed(SEED)
    interior_points = sample_local_points(
        N_INTERIOR_POINTS_PER_PATCH, device, generator
    )
    interior_operators = precompute_operators(model, interior_points, 2)
    interface_points = generate_interface_local_points(N_INTERFACE_POINTS, device)
    interface_operators = precompute_interface_operators(model, interface_points)
    loss_fn = lambda: strong_residual_loss(
        model, interior_operators, interface_operators
    )
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
        min_delta=1.0e-10,
        print_every=100,
    )

    validation_points = sample_local_points(4_000, device, generator)
    validation_operators = precompute_operators(model, validation_points, 2)
    validation_interfaces = generate_interface_local_points(501, device)
    validation_interface_operators = precompute_interface_operators(
        model, validation_interfaces
    )
    with torch.no_grad():
        relative_error = relative_l2_error(
            model, validation_points, validation_operators
        )
        residual = laplace_residual_loss(model, validation_operators)
        boundary_error, solution_jump = boundary_and_interface_errors(model)
        residual_jump = maximum_interface_residual_jump(
            model, validation_interface_operators
        )
        flux_jump = maximum_interface_flux_jump(
            model, validation_interface_operators
        )
    print(f"Relative physical L2 error: {relative_error.item():.6e}")
    print(f"Interior Laplace residual MSE: {residual.item():.6e}")
    print(f"Maximum boundary interpolation error: {boundary_error.item():.6e}")
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
        description="Solve singular L-shaped Laplace with 12 B-spline patches"
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--n-basis", type=int, default=N_BASIS)
    parser.add_argument("--degree", type=int, default=DEGREE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    tuned_group = parser.add_mutually_exclusive_group()
    tuned_group.add_argument("--use-tuned", dest="use_tuned", action="store_true")
    tuned_group.add_argument("--no-use-tuned", dest="use_tuned", action="store_false")
    parser.set_defaults(use_tuned=None)
    arguments = parser.parse_args()
    main(arguments.device, arguments.n_basis, arguments.degree, arguments.epochs, arguments.use_tuned)
