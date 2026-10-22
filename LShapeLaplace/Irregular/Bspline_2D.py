"""B-spline XPINN solver for the singular L-shaped Laplace benchmark.

The three fitted patches impose exterior Dirichlet traces and C0 solution
continuity through their coefficients. Patchwise Laplace residuals are coupled
by residual-continuity and normal-flux losses across both interfaces.
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

from LShapeLaplace.Irregular._bspline_utils import (
    basis_derivative_matrix,
    greville_abscissae,
    interpolate_boundary_trace,
    select_device,
    set_seed,
)
from training import train


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "bspline_config.json"


def load_configuration(path: Path = CONFIG_PATH) -> dict:
    """Load the complete B-spline experiment configuration."""
    if not path.exists():
        raise FileNotFoundError(f"B-spline configuration not found: {path}")
    configuration = json.loads(path.read_text(encoding="utf-8"))
    required_sections = {
        "reproducibility",
        "runtime",
        "model",
        "sampling",
        "loss_weights",
        "training",
        "output",
    }
    missing = required_sections.difference(configuration)
    if missing:
        raise ValueError(
            f"Missing configuration sections: {', '.join(sorted(missing))}"
        )
    return configuration


CONFIG = load_configuration()
SEED = int(CONFIG["reproducibility"]["seed"])
DEFAULT_DEVICE = str(CONFIG["runtime"]["device"])
LOAD_TUNED_IF_AVAILABLE = bool(
    CONFIG["runtime"]["load_tuned_if_available"]
)
N_BASIS = int(CONFIG["model"]["n_basis"])
DEGREE = int(CONFIG["model"]["degree"])
KNOT_CONCENTRATION = float(CONFIG["model"]["knot_concentration"])
CORNER_EPSILON = float(CONFIG["model"]["corner_epsilon"])
N_INTERIOR_POINTS_PER_PATCH = int(
    CONFIG["sampling"]["interior_points_per_patch"]
)
N_INTERFACE_POINTS = int(CONFIG["sampling"]["interface_points"])
N_VALIDATION_POINTS_PER_PATCH = int(
    CONFIG["sampling"]["validation_points_per_patch"]
)
N_VALIDATION_INTERFACE_POINTS = int(
    CONFIG["sampling"]["validation_interface_points"]
)
PDE_WEIGHT = float(CONFIG["loss_weights"]["pde"])
INTERFACE_RESIDUAL_WEIGHT = float(
    CONFIG["loss_weights"]["interface_residual"]
)
INTERFACE_FLUX_WEIGHT = float(CONFIG["loss_weights"]["interface_flux"])
OPTIMIZER_NAME = str(CONFIG["training"]["optimizer"])
EPOCHS = int(CONFIG["training"]["epochs"])
LEARNING_RATE = float(CONFIG["training"]["learning_rate"])
SCHEDULER_FACTOR = float(CONFIG["training"]["scheduler"]["factor"])
SCHEDULER_PATIENCE = int(CONFIG["training"]["scheduler"]["patience"])
MIN_LEARNING_RATE = float(
    CONFIG["training"]["scheduler"]["min_learning_rate"]
)
EARLY_STOPPING_PATIENCE = int(
    CONFIG["training"]["early_stopping"]["patience"]
)
MIN_DELTA = float(CONFIG["training"]["early_stopping"]["min_delta"])
PRINT_EVERY = int(CONFIG["training"]["print_every"])
RESULT_PATH = SCRIPT_DIR / CONFIG["output"]["tuned_hyperparameters"]
POSTPROCESSING_DIR = (
    SCRIPT_DIR / CONFIG["output"]["postprocessing_directory"]
)
PLOT_RESOLUTION = int(CONFIG["output"]["plot_resolution"])
SHOW_PLOTS = bool(CONFIG["output"]["show_plots"])


def exact_solution(points: torch.Tensor) -> torch.Tensor:
    """Return r^(2/3) sin(2 theta/3) with theta in [0, 2*pi)."""
    x, y = points[:, 0], points[:, 1]
    radius = torch.sqrt(x.square() + y.square())
    theta = torch.remainder(torch.atan2(y, x), 2.0 * math.pi)
    return radius.pow(2.0 / 3.0) * torch.sin(2.0 * theta / 3.0)


def map_patch(name: str, local_points: torch.Tensor) -> torch.Tensor:
    """Map coordinates in the unit square to an L-shaped patch."""
    xi, eta = local_points[:, 0], local_points[:, 1]
    if name == "upper_right":
        return torch.stack((xi, eta), dim=1)
    if name == "upper_left":
        return torch.stack((xi - 1.0, eta), dim=1)
    if name == "lower_left":
        return torch.stack((xi - 1.0, eta - 1.0), dim=1)
    raise ValueError(f"Unknown patch: {name}")


def open_corner_clustered_knots(
    n_basis: int,
    degree: int,
    concentration: float,
    corner_endpoint: int,
    device: torch.device,
) -> torch.Tensor:
    """Create open knots graded relative to a selected local endpoint.

    Exponents above one concentrate knots at ``corner_endpoint``; exponents
    below one concentrate them toward the opposite endpoint.
    """
    if degree < 2 or n_basis <= degree:
        raise ValueError("Require degree >= 2 and n_basis > degree")
    if concentration <= 0.0:
        raise ValueError("knot_concentration must be strictly positive")
    n_internal = n_basis - degree - 1
    if n_internal:
        parameter = torch.linspace(
            0.0, 1.0, n_internal + 2, device=device
        )[1:-1]
        internal = (
            parameter.pow(concentration)
            if corner_endpoint == 0
            else 1.0 - (1.0 - parameter).pow(concentration)
        )
    else:
        internal = torch.empty(0, device=device)
    return torch.cat(
        (torch.zeros(degree + 1, device=device), internal,
         torch.ones(degree + 1, device=device))
    )


class LShapeBSpline(torch.nn.Module):
    """Three compatible B-spline patches for the singular L-shape problem."""

    patch_names = ("upper_right", "upper_left", "lower_left")
    output_stem = "bspline_xpinn_irregular"
    solution_title = "B-spline XPINN singular solution"
    exact_solution_label = r"Exact singular solution: $r^{2/3}\sin(2\theta/3)$"

    def __init__(
        self,
        n_basis: int,
        degree: int,
        knot_concentration: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        if degree < 2:
            raise ValueError("The strong Laplace formulation requires degree >= 2")
        if n_basis <= degree:
            raise ValueError("n_basis must be greater than degree")
        self.n_basis = n_basis
        self.degree = degree
        self.knot_concentration = knot_concentration
        for axis in ("x", "y"):
            for endpoint in (0, 1):
                self.register_buffer(
                    f"knots_{axis}_{endpoint}",
                    open_corner_clustered_knots(
                        n_basis, degree, knot_concentration, endpoint, device
                    ),
                )
        # Register the raw coefficients for each patch as trainable parameters
        for name in self.patch_names:
            self.register_parameter(
                f"raw_{name}",
                torch.nn.Parameter(torch.zeros(n_basis, n_basis, device=device)),
            )
        #Register the fixed masks and values for each patch as buffers
        masks, values = self._build_boundary_data(device)
        for name in self.patch_names:
            self.register_buffer(f"{name}_fixed_mask", masks[name])
            self.register_buffer(f"{name}_fixed_values", values[name])

    @staticmethod
    def exact_solution(points: torch.Tensor) -> torch.Tensor:
        return exact_solution(points)

    def patch_knots(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        # Reentrant corner local ccordinates
        endpoints = {
            "upper_right": (0, 0),
            "upper_left": (1, 0),
            "lower_left": (1, 1),
        }[name]
        # Return the appropriate knot vectors for the given patch name
        return (
            getattr(self, f"knots_x_{endpoints[0]}"),
            getattr(self, f"knots_y_{endpoints[1]}"),
        )

    # Fix the exterior BCs for each patch.
    def _build_boundary_data(self, device: torch.device):
        shapes = (self.n_basis, self.n_basis)

        # Create masks and values for each patch, indicating which coefficients are fixed
        masks = {
            name: torch.zeros(*shapes, dtype=torch.bool, device=device)
            for name in self.patch_names
        }
        values = {
            name: torch.zeros(*shapes, device=device)
            for name in self.patch_names
        }
        # Assign the fixed coefficients for each patch based on the boundary conditions
        def set_edge(name: str, edge: str) -> None:
            knots_x, knots_y = self.patch_knots(name)
            if edge in ("bottom", "top"):

                # Greville abscissae are used to solve the local linear system for the 
                # boundary coefficients. They are used in spline collocation methods to 
                # evaluate the coefficients.
                local_coordinate = greville_abscissae(knots_x, self.degree)
                eta = 0.0 if edge == "bottom" else 1.0

                # form the 2D local coordinates for the boundary edge of the patch
                local = torch.stack(
                    (local_coordinate, torch.full_like(local_coordinate, eta)), 1
                )
                # Trace are the coefficients after solving the local linear system for the boundary coefficients.
                trace = interpolate_boundary_trace(
                    knots_x, self.degree, map_patch(name, local), self.exact_solution
                )
                # Assign the fixed coefficients to the appropriate row of the patch's coefficient matrix
                column = 0 if edge == "bottom" else -1
                masks[name][:, column] = True
                values[name][:, column] = trace
            else:
                # Similary assign the BC coefficients for the left and right edges of every patch
                local_coordinate = greville_abscissae(knots_y, self.degree)
                xi = 0.0 if edge == "left" else 1.0
                local = torch.stack(
                    (torch.full_like(local_coordinate, xi), local_coordinate), 1
                )
                trace = interpolate_boundary_trace(
                    knots_y, self.degree, map_patch(name, local), self.exact_solution
                )
                row = 0 if edge == "left" else -1
                masks[name][row, :] = True
                values[name][row, :] = trace

        # Assign the fixed coefficients for each patch based on the L-shape geometry
        for edge in ("bottom", "right", "top"):
            set_edge("upper_right", edge)
        for edge in ("left", "top"):
            set_edge("upper_left", edge)
        for edge in ("left", "bottom", "right"):
            set_edge("lower_left", edge)
        return masks, values

    # Impose the BC fixed coefficients on the raw trainable coefficients to get 
    # the final coefficient tensors for each patch.
    def coefficient_tensors(self) -> dict[str, torch.Tensor]:
        def impose(name: str, candidate: torch.Tensor) -> torch.Tensor:
            return torch.where(
                getattr(self, f"{name}_fixed_mask"),
                getattr(self, f"{name}_fixed_values"),
                candidate,
            )

        # Start with upper_right and then impose C0 continuity for upper_left and lower_left 
        # by using the last row/column of the previous patch.
        upper_right = impose("upper_right", self.raw_upper_right)
        upper_left = impose(
            "upper_left",
            torch.cat((self.raw_upper_left[:-1, :], upper_right[0:1, :]), 0),
        )
        lower_left = impose(
            "lower_left",
            torch.cat((self.raw_lower_left[:, :-1], upper_left[:, 0:1]), 1),
        )
        return {
            "upper_right": upper_right,
            "upper_left": upper_left,
            "lower_left": lower_left,
        }
    # Evaluate the B-spline basis functions and their derivatives 
    # at the given local points for a specific patch until the specified order.
    def operators_for_patch(
        self, name: str, local_points: torch.Tensor, max_order: int = 2
    ):
        knots_x, knots_y = self.patch_knots(name)
        return {
            "x": tuple(
                basis_derivative_matrix(
                    local_points[:, 0], knots_x, self.degree, order
                )
                for order in range(max_order + 1)
            ),
            "y": tuple(
                basis_derivative_matrix(
                    local_points[:, 1], knots_y, self.degree, order
                )
                for order in range(max_order + 1)
            ),
        }

    @staticmethod
    def evaluate(
        coefficients: torch.Tensor, operators, dx: int = 0, dy: int = 0
    ) -> torch.Tensor:
        """Evaluate a tensor-product spline or one of its derivatives."""
        return torch.einsum(
            "ni,ij,nj->n",
            operators["x"][dx],
            coefficients,
            operators["y"][dy],
        )
    # Bspline representation at given local points for a specific patch.
    def forward(self, patch_name: str, local_points: torch.Tensor) -> torch.Tensor:
        return self.evaluate(
            self.coefficient_tensors()[patch_name],
            self.operators_for_patch(patch_name, local_points, 0),
        )[:, None]


def sample_local_points(
    count: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample local coordinates uniformly in the unit square."""
    return torch.rand(count, 2, device=device, generator=generator)


def precompute_operators(
    model: LShapeBSpline,
    local_points: torch.Tensor,
    max_order: int = 2,
):
    """Precompute patch-dependent spline operators at fixed points."""
    return {
        name: model.operators_for_patch(name, local_points, max_order)
        for name in model.patch_names
    }


def patch_operators(operators, patch_name: str):
    """Select the operators belonging to one patch."""
    return operators[patch_name]


def laplace_residual_loss(
    model: LShapeBSpline, operators
) -> torch.Tensor:
    """Return the patch-averaged mean squared Laplace residual."""
    coefficients = model.coefficient_tensors()
    residual_loss = torch.zeros((), device=model.raw_upper_right.device)
    for name in model.patch_names:
        local_operators = patch_operators(operators, name)
        laplacian = (
            model.evaluate(coefficients[name], local_operators, dx=2, dy=0)
            + model.evaluate(coefficients[name], local_operators, dx=0, dy=2)
        )
        residual_loss = residual_loss + laplacian.square().mean()
    return residual_loss / len(model.patch_names)


def relative_l2_error(
    model: LShapeBSpline,
    local_points: torch.Tensor,
    operators=None,
) -> torch.Tensor:
    """Return the sampled relative L2 error over all three patches."""
    if operators is None:
        operators = precompute_operators(model, local_points, 0)
    coefficients = model.coefficient_tensors()
    squared_error = torch.zeros((), device=local_points.device)
    squared_exact = torch.zeros_like(squared_error)
    for name in model.patch_names:
        prediction = model.evaluate(
            coefficients[name], patch_operators(operators, name)
        )
        exact = model.exact_solution(map_patch(name, local_points))
        squared_error = squared_error + prediction.sub(exact).square().mean()
        squared_exact = squared_exact + exact.square().mean()
    return torch.sqrt(squared_error / squared_exact)


def boundary_and_interface_errors(
    model: LShapeBSpline, sample_count: int = 501
):
    """Return maximum exterior-boundary and C0-interface errors."""
    device = model.raw_upper_right.device
    line = torch.linspace(0.0, 1.0, sample_count, device=device)
    zero, one = torch.zeros_like(line), torch.ones_like(line)
    edges = {
        "upper_right": (
            torch.stack((line, zero), 1),
            torch.stack((one, line), 1),
            torch.stack((line, one), 1),
        ),
        "upper_left": (
            torch.stack((zero, line), 1),
            torch.stack((line, one), 1),
        ),
        "lower_left": (
            torch.stack((zero, line), 1),
            torch.stack((line, zero), 1),
            torch.stack((one, line), 1),
        ),
    }
    maximum_boundary_error = torch.zeros((), device=device)
    for name, patch_edges in edges.items():
        for local in patch_edges:
            prediction = model(name, local).squeeze(1)
            exact = model.exact_solution(map_patch(name, local))
            maximum_boundary_error = torch.maximum(
                maximum_boundary_error, (prediction - exact).abs().max()
            )
    interface_ab = torch.stack((zero, line), 1)
    interface_ba = torch.stack((one, line), 1)
    interface_bc = torch.stack((line, zero), 1)
    interface_cb = torch.stack((line, one), 1)
    jump_ab = (
        model("upper_right", interface_ab)
        - model("upper_left", interface_ba)
    ).abs().max()
    jump_bc = (
        model("upper_left", interface_bc)
        - model("lower_left", interface_cb)
    ).abs().max()
    return maximum_boundary_error, torch.maximum(jump_ab, jump_bc)


def generate_interface_local_points(
    count: int,
    device: torch.device,
    corner_epsilon: float,
) -> dict[str, tuple[str, torch.Tensor, str, torch.Tensor]]:
    """Return both local traces of each interface away from the singularity."""
    if not 0.0 < corner_epsilon < 1.0:
        raise ValueError("corner_epsilon must lie strictly between zero and one")
    vertical_line = torch.linspace(
        corner_epsilon, 1.0, count, device=device
    )
    horizontal_line = torch.linspace(
        0.0, 1.0 - corner_epsilon, count, device=device
    )
    vertical_zero = torch.zeros_like(vertical_line)
    vertical_one = torch.ones_like(vertical_line)
    horizontal_zero = torch.zeros_like(horizontal_line)
    horizontal_one = torch.ones_like(horizontal_line)
    return {
        "upper_right_upper_left": (
            "upper_right",
            torch.stack((vertical_zero, vertical_line), dim=1),
            "upper_left",
            torch.stack((vertical_one, vertical_line), dim=1),
        ),
        "upper_left_lower_left": (
            "upper_left",
            torch.stack((horizontal_line, horizontal_zero), dim=1),
            "lower_left",
            torch.stack((horizontal_line, horizontal_one), dim=1),
        ),
    }


def precompute_interface_operators(
    model: LShapeBSpline,
    interface_points: dict[
        str, tuple[str, torch.Tensor, str, torch.Tensor]
    ],
):
    """Precompute analytical operators on both sides of each interface."""
    return {
        interface_name: (
            name_a,
            precompute_operators(model, points_a, max_order=2),
            name_b,
            precompute_operators(model, points_b, max_order=2),
        )
        for interface_name, (
            name_a,
            points_a,
            name_b,
            points_b,
        ) in interface_points.items()
    }


def interface_flux_loss(
    model: LShapeBSpline, interface_operators
) -> torch.Tensor:
    """Return the mean squared outward-normal flux imbalance."""
    coefficients = model.coefficient_tensors()
    terms = []

    name_a, operators_a, name_b, operators_b = interface_operators[
        "upper_right_upper_left"
    ]
    derivative_a = model.evaluate(
        coefficients[name_a], patch_operators(operators_a, name_a), dx=1, dy=0
    )
    derivative_b = model.evaluate(
        coefficients[name_b], patch_operators(operators_b, name_b), dx=1, dy=0
    )
    terms.append((-derivative_a + derivative_b).square().mean())

    name_a, operators_a, name_b, operators_b = interface_operators[
        "upper_left_lower_left"
    ]
    derivative_a = model.evaluate(
        coefficients[name_a], patch_operators(operators_a, name_a), dx=0, dy=1
    )
    derivative_b = model.evaluate(
        coefficients[name_b], patch_operators(operators_b, name_b), dx=0, dy=1
    )
    terms.append((-derivative_a + derivative_b).square().mean())
    return torch.stack(terms).mean()


def interface_residual_loss(
    model: LShapeBSpline, interface_operators
) -> torch.Tensor:
    """Return the mean squared Laplace-residual jump."""
    coefficients = model.coefficient_tensors()
    terms = []
    for name_a, operators_a, name_b, operators_b in interface_operators.values():
        local_a = patch_operators(operators_a, name_a)
        local_b = patch_operators(operators_b, name_b)
        residual_a = (
            model.evaluate(coefficients[name_a], local_a, dx=2, dy=0)
            + model.evaluate(coefficients[name_a], local_a, dx=0, dy=2)
        )
        residual_b = (
            model.evaluate(coefficients[name_b], local_b, dx=2, dy=0)
            + model.evaluate(coefficients[name_b], local_b, dx=0, dy=2)
        )
        terms.append((residual_a - residual_b).square().mean())
    return torch.stack(terms).mean()


def strong_loss_components(
    model: LShapeBSpline, interior_operators, interface_operators
) -> dict[str, torch.Tensor]:
    """Return the three components of the strong-form objective."""
    return {
        "pde": laplace_residual_loss(model, interior_operators),
        "interface_residual": interface_residual_loss(
            model, interface_operators
        ),
        "interface_flux": interface_flux_loss(model, interface_operators),
    }


def strong_residual_loss(
    model: LShapeBSpline,
    interior_operators,
    interface_operators,
    *,
    pde_weight: float,
    interface_residual_weight: float,
    interface_flux_weight: float,
) -> torch.Tensor:
    """Return the weighted strong-form training objective."""
    terms = strong_loss_components(
        model, interior_operators, interface_operators
    )
    return (
        pde_weight * terms["pde"]
        + interface_residual_weight * terms["interface_residual"]
        + interface_flux_weight * terms["interface_flux"]
    )


def maximum_interface_residual_jump(
    model: LShapeBSpline, interface_operators
) -> torch.Tensor:
    """Return the maximum absolute Laplace-residual jump."""
    coefficients = model.coefficient_tensors()
    jumps = []
    for name_a, operators_a, name_b, operators_b in interface_operators.values():
        local_a = patch_operators(operators_a, name_a)
        local_b = patch_operators(operators_b, name_b)
        residual_a = (
            model.evaluate(coefficients[name_a], local_a, dx=2, dy=0)
            + model.evaluate(coefficients[name_a], local_a, dx=0, dy=2)
        )
        residual_b = (
            model.evaluate(coefficients[name_b], local_b, dx=2, dy=0)
            + model.evaluate(coefficients[name_b], local_b, dx=0, dy=2)
        )
        jumps.append((residual_a - residual_b).abs().max())
    return torch.stack(jumps).max()


def maximum_interface_flux_jump(
    model: LShapeBSpline, interface_operators
) -> torch.Tensor:
    """Return the maximum absolute normal-flux imbalance."""
    coefficients = model.coefficient_tensors()
    jumps = []

    name_a, operators_a, name_b, operators_b = interface_operators[
        "upper_right_upper_left"
    ]
    jumps.append(
        (
            -model.evaluate(
                coefficients[name_a],
                patch_operators(operators_a, name_a),
                dx=1,
                dy=0,
            )
            + model.evaluate(
                coefficients[name_b],
                patch_operators(operators_b, name_b),
                dx=1,
                dy=0,
            )
        ).abs().max()
    )

    name_a, operators_a, name_b, operators_b = interface_operators[
        "upper_left_lower_left"
    ]
    jumps.append(
        (
            -model.evaluate(
                coefficients[name_a],
                patch_operators(operators_a, name_a),
                dx=0,
                dy=1,
            )
            + model.evaluate(
                coefficients[name_b],
                patch_operators(operators_b, name_b),
                dx=0,
                dy=1,
            )
        ).abs().max()
    )
    return torch.stack(jumps).max()


def plot_solution(
    model: LShapeBSpline,
    postprocessing_dir: Path,
    resolution: int = 101,
    validation_relative_l2_error: float | torch.Tensor | None = None,
) -> None:
    """Save solution fields, error fields, and their reproducible plot."""
    device = model.raw_upper_right.device
    axis = torch.linspace(0.0, 1.0, resolution, device=device)
    xi, eta = torch.meshgrid(axis, axis, indexing="ij")
    local = torch.stack((xi.flatten(), eta.flatten()), dim=1)
    operators = precompute_operators(model, local, 0)
    coefficients = model.coefficient_tensors()
    physical_parts, prediction_parts, exact_parts = [], [], []
    with torch.no_grad():
        for name in model.patch_names:
            physical = map_patch(name, local)
            prediction = model.evaluate(
                coefficients[name], patch_operators(operators, name)
            )
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
    solution_data = {
        "patch_names": np.asarray(model.patch_names),
        "x": np.stack(
            [part[:, 0].reshape(resolution, resolution) for part in physical_parts]
        ),
        "y": np.stack(
            [part[:, 1].reshape(resolution, resolution) for part in physical_parts]
        ),
        "prediction": np.stack(
            [part.reshape(resolution, resolution) for part in prediction_parts]
        ),
        "exact": np.stack(
            [part.reshape(resolution, resolution) for part in exact_parts]
        ),
        "absolute_error": np.stack(
            [part.reshape(resolution, resolution) for part in error_parts]
        ),
    }
    if validation_relative_l2_error is not None:
        value = validation_relative_l2_error
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        solution_data["validation_relative_l2_error"] = np.asarray(
            value, dtype=float
        )
    np.savez_compressed(solution_data_path, **solution_data)

    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for axis_object, value_parts, title, color_map in (
        (axes[0], prediction_parts, model.solution_title, "viridis"),
        (axes[1], exact_parts, model.exact_solution_label, "viridis"),
        (axes[2], error_parts, "Absolute error", "magma"),
    ):
        combined_values = np.concatenate(value_parts)
        levels = np.linspace(combined_values.min(), combined_values.max(), 51)
        if np.isclose(levels[0], levels[-1]):
            levels = np.linspace(
                levels[0] - 1.0e-12, levels[-1] + 1.0e-12, 51
            )
        plot = None
        for physical, values in zip(physical_parts, value_parts):
            plot = axis_object.contourf(
                physical[:, 0].reshape(resolution, resolution),
                physical[:, 1].reshape(resolution, resolution),
                values.reshape(resolution, resolution),
                levels=levels,
                cmap=color_map,
            )
        figure.colorbar(plot, ax=axis_object)
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
    model: LShapeBSpline,
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
    axis.set_ylabel("B-spline XPINN loss")
    axis.set_title(f"{model.solution_title} training history")
    figure.tight_layout()
    figure_path = postprocessing_dir / f"{model.output_stem}_training.png"
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    print(f"Saved training data to: {data_path}")
    print(f"Saved training plot to: {figure_path}")


def load_hyperparameters(path: Path = RESULT_PATH) -> tuple[int, int]:
    if not path.exists():
        raise FileNotFoundError(
            f"No irregular B-spline XPINN tuning result found at {path}. "
            "Supply that JSON file or run with --no-use-tuned."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("best_params", payload)
    return (
        int(params["n_basis"]),
        int(params["degree"]),
    )


def main(
    device_name: str = DEFAULT_DEVICE,
    n_basis: int = N_BASIS,
    degree: int = DEGREE,
    knot_concentration: float = KNOT_CONCENTRATION,
    corner_epsilon: float = CORNER_EPSILON,
    epochs: int = EPOCHS,
    use_tuned: bool | None = None,
) -> None:
    set_seed(SEED)
    device = select_device(device_name)
    should_load = (
        LOAD_TUNED_IF_AVAILABLE and RESULT_PATH.exists()
        if use_tuned is None
        else use_tuned
    )
    if should_load:
        n_basis, degree = load_hyperparameters()
        print(f"Loaded tuned hyperparameters from: {RESULT_PATH}")
    print(f"Using device: {device}")
    print(f"Loaded configuration from: {CONFIG_PATH}")
    print(
        "Irregular B-spline XPINN: "
        f"n_basis={n_basis}, degree={degree}, "
        f"knot_concentration={knot_concentration:g}, "
        f"corner_epsilon={corner_epsilon:g}"
    )
    print("Exterior Dirichlet values and C0 interfaces are imposed exactly.")
    print("The soft interface terms are residual and normal-flux continuity.")

    model = LShapeBSpline(n_basis, degree, knot_concentration, device)
    generator = torch.Generator(device=device).manual_seed(SEED)
    interior_points = sample_local_points(
        N_INTERIOR_POINTS_PER_PATCH, device, generator
    )
    interior_operators = precompute_operators(
        model, interior_points, max_order=2
    )
    interface_points = generate_interface_local_points(
        N_INTERFACE_POINTS, device, corner_epsilon
    )
    interface_operators = precompute_interface_operators(
        model, interface_points
    )

    loss_fn = lambda: strong_residual_loss(
        model,
        interior_operators,
        interface_operators,
        pde_weight=PDE_WEIGHT,
        interface_residual_weight=INTERFACE_RESIDUAL_WEIGHT,
        interface_flux_weight=INTERFACE_FLUX_WEIGHT,
    )
    if OPTIMIZER_NAME.lower() != "adam":
        raise ValueError(
            f"Unsupported optimizer '{OPTIMIZER_NAME}'; expected 'Adam'"
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=MIN_LEARNING_RATE,
    )
    loss_history = train(
        model.parameters(),
        loss_fn,
        optimizer,
        epochs=epochs,
        scheduler=scheduler,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        min_delta=MIN_DELTA,
        print_every=PRINT_EVERY,
    )

    validation_points = sample_local_points(
        N_VALIDATION_POINTS_PER_PATCH, device, generator
    )
    validation_operators = precompute_operators(
        model, validation_points, max_order=2
    )
    validation_interfaces = generate_interface_local_points(
        N_VALIDATION_INTERFACE_POINTS, device, corner_epsilon
    )
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
    print(f"Relative L2 error: {relative_error.item():.6e}")
    print(f"Interior Laplace residual MSE: {residual.item():.6e}")
    print(f"Maximum boundary interpolation error: {boundary_error.item():.6e}")
    print(f"Maximum interface solution jump: {solution_jump.item():.6e}")
    print(f"Maximum interface residual jump: {residual_jump.item():.6e}")
    print(f"Maximum interface normal-flux jump: {flux_jump.item():.6e}")
    plot_solution(
        model,
        postprocessing_dir=POSTPROCESSING_DIR,
        resolution=PLOT_RESOLUTION,
        validation_relative_l2_error=relative_error,
    )
    plot_training_history(
        loss_history, model, postprocessing_dir=POSTPROCESSING_DIR
    )
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Solve the singular L-shaped Laplace problem with a B-spline XPINN"
        )
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--n-basis", type=int, default=N_BASIS)
    parser.add_argument("--degree", type=int, default=DEGREE)
    parser.add_argument(
        "--knot-concentration", type=float, default=KNOT_CONCENTRATION
    )
    parser.add_argument(
        "--corner-epsilon", type=float, default=CORNER_EPSILON
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    tuned_group = parser.add_mutually_exclusive_group()
    tuned_group.add_argument("--use-tuned", dest="use_tuned", action="store_true")
    tuned_group.add_argument(
        "--no-use-tuned", dest="use_tuned", action="store_false"
    )
    parser.set_defaults(use_tuned=None)
    arguments = parser.parse_args()
    main(
        arguments.device,
        arguments.n_basis,
        arguments.degree,
        arguments.knot_concentration,
        arguments.corner_epsilon,
        arguments.epochs,
        arguments.use_tuned,
    )
