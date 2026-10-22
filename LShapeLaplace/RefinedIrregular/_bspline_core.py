"""Internal B-spline basis, operator, loss, and plotting utilities."""

from __future__ import annotations

import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

def set_seed(seed: int) -> None:
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


def open_endpoint_clustered_knots(
    n_basis: int, degree: int, device: torch.device
) -> torch.Tensor:
    """Create open knots with cosine clustering at both patch endpoints."""
    if degree < 0 or n_basis <= degree:
        raise ValueError("n_basis must be greater than degree")
    n_internal = n_basis - degree - 1
    if n_internal:
        uniform_parameter = torch.linspace(
            0.0, 1.0, n_internal + 2, device=device
        )[1:-1]
        internal = 0.5 * (1.0 - torch.cos(math.pi * uniform_parameter))
    else:
        internal = torch.empty(0, device=device)
    return torch.cat(
        (
            torch.zeros(degree + 1, device=device),
            internal,
            torch.ones(degree + 1, device=device),
        )
    )


def basis_matrix(
    values: torch.Tensor, knots: torch.Tensor, degree: int
) -> torch.Tensor:
    """Evaluate all univariate B-spline basis functions."""
    values = values.reshape(-1)
    basis = (
        (values[:, None] >= knots[:-1])
        & (values[:, None] < knots[1:])
    ).to(values.dtype)
    at_right_end = values == knots[-1]
    if at_right_end.any():
        basis[at_right_end] = 0.0
        basis[at_right_end, -1] = 1.0
    for current_degree in range(1, degree + 1):
        column_count = knots.numel() - current_degree - 1
        updated = torch.zeros(
            values.numel(), column_count, device=values.device, dtype=values.dtype
        )
        for index in range(column_count):
            left_denominator = knots[index + current_degree] - knots[index]
            right_denominator = (
                knots[index + current_degree + 1] - knots[index + 1]
            )
            if left_denominator > 0:
                updated[:, index] += (
                    (values - knots[index]) / left_denominator
                ) * basis[:, index]
            if right_denominator > 0:
                updated[:, index] += (
                    (knots[index + current_degree + 1] - values)
                    / right_denominator
                ) * basis[:, index + 1]
        basis = updated
    if at_right_end.any():
        basis[at_right_end] = 0.0
        basis[at_right_end, -1] = 1.0
    return basis


def basis_derivative_matrix(
    values: torch.Tensor,
    knots: torch.Tensor,
    degree: int,
    derivative_order: int,
) -> torch.Tensor:
    """Evaluate analytical derivatives of every univariate basis function."""
    if derivative_order < 0:
        raise ValueError("derivative_order must be non-negative")
    n_basis = knots.numel() - degree - 1
    if derivative_order == 0:
        return basis_matrix(values, knots, degree)
    if derivative_order > degree:
        return torch.zeros(
            values.numel(), n_basis, device=values.device, dtype=values.dtype
        )
    values = values.reshape(-1)
    at_right_end = values == knots[-1]
    if at_right_end.any():
        values = values.clone()
        values[at_right_end] = torch.nextafter(knots[-1], knots[0])
    lower = basis_derivative_matrix(
        values, knots, degree - 1, derivative_order - 1
    )
    result = torch.zeros(
        values.numel(), n_basis, device=values.device, dtype=values.dtype
    )
    for index in range(n_basis):
        left_denominator = knots[index + degree] - knots[index]
        right_denominator = knots[index + degree + 1] - knots[index + 1]
        if left_denominator > 0:
            result[:, index] += degree / left_denominator * lower[:, index]
        if right_denominator > 0:
            result[:, index] -= degree / right_denominator * lower[:, index + 1]
    return result


def exact_solution(points: torch.Tensor) -> torch.Tensor:
    """Smooth harmonic field u(x,y)=exp(x) sin(y)."""
    return torch.exp(points[:, 0]) * torch.sin(points[:, 1])


def map_patch(name: str, local_points: torch.Tensor) -> torch.Tensor:
    """Map local coordinates to one of the three fitted L-shaped patches."""
    xi, eta = local_points[:, 0], local_points[:, 1]
    if name == "upper_right":
        return torch.stack((xi, eta), dim=1)
    if name == "upper_left":
        return torch.stack((xi - 1.0, eta), dim=1)
    if name == "lower_left":
        return torch.stack((xi - 1.0, eta - 1.0), dim=1)
    raise ValueError(f"Unknown patch: {name}")


def greville_abscissae(knots: torch.Tensor, degree: int) -> torch.Tensor:
    n_basis = knots.numel() - degree - 1
    return torch.stack(
        [knots[index + 1 : index + degree + 1].mean() for index in range(n_basis)]
    )


def interpolate_boundary_trace(
    knots: torch.Tensor,
    degree: int,
    physical_points: torch.Tensor,
    solution_function=exact_solution,
) -> torch.Tensor:
    greville = greville_abscissae(knots, degree)
    interpolation_matrix = basis_matrix(greville, knots, degree)
    return torch.linalg.solve(
        interpolation_matrix, solution_function(physical_points)
    )


class bsplines(torch.nn.Module):
    """Three B-spline patches with hard boundary and C0 interface constraints."""

    patch_names = ("upper_right", "upper_left", "lower_left")
    output_stem = "bspline_xpinn_regular"
    solution_title = "B-spline XPINN regular solution"
    exact_solution_label = r"Exact solution: $e^x\sin(y)$"

    def __init__(self, n_basis: int, degree: int, device: torch.device) -> None:
        super().__init__()
        if degree < 2:
            raise ValueError("The strong Laplace formulation requires degree >= 2")
        if n_basis <= degree:
            raise ValueError("n_basis must be greater than degree")
        self.n_basis = n_basis
        self.degree = degree
        self.register_buffer(
            "knots", open_endpoint_clustered_knots(n_basis, degree, device)
        )
        self.raw_upper_right = torch.nn.Parameter(
            torch.zeros(n_basis, n_basis, device=device)
        )
        self.raw_upper_left = torch.nn.Parameter(
            torch.zeros(n_basis, n_basis, device=device)
        )
        self.raw_lower_left = torch.nn.Parameter(
            torch.zeros(n_basis, n_basis, device=device)
        )
        masks, values = self._build_boundary_data(device)
        for name in self.patch_names:
            self.register_buffer(f"{name}_fixed_mask", masks[name])
            self.register_buffer(f"{name}_fixed_values", values[name])

    @staticmethod
    def exact_solution(points: torch.Tensor) -> torch.Tensor:
        return exact_solution(points)

    def _build_boundary_data(self, device: torch.device):
        n = self.n_basis
        greville = greville_abscissae(self.knots, self.degree)
        masks = {
            name: torch.zeros(n, n, dtype=torch.bool, device=device)
            for name in self.patch_names
        }
        values = {
            name: torch.zeros(n, n, device=device) for name in self.patch_names
        }

        def set_edge(name: str, edge: str) -> None:
            if edge in ("bottom", "top"):
                eta_value = 0.0 if edge == "bottom" else 1.0
                local = torch.stack(
                    (greville, torch.full_like(greville, eta_value)), dim=1
                )
                coefficients = interpolate_boundary_trace(
                    self.knots,
                    self.degree,
                    map_patch(name, local),
                    self.exact_solution,
                )
                column = 0 if edge == "bottom" else -1
                masks[name][:, column] = True
                values[name][:, column] = coefficients
            else:
                xi_value = 0.0 if edge == "left" else 1.0
                local = torch.stack(
                    (torch.full_like(greville, xi_value), greville), dim=1
                )
                coefficients = interpolate_boundary_trace(
                    self.knots,
                    self.degree,
                    map_patch(name, local),
                    self.exact_solution,
                )
                row = 0 if edge == "left" else -1
                masks[name][row, :] = True
                values[name][row, :] = coefficients

        for edge in ("bottom", "right", "top"):
            set_edge("upper_right", edge)
        for edge in ("left", "top"):
            set_edge("upper_left", edge)
        for edge in ("left", "bottom", "right"):
            set_edge("lower_left", edge)
        return masks, values

    def coefficient_tensors(self) -> dict[str, torch.Tensor]:
        """Tie interface traces and overwrite physical-boundary coefficients."""
        def impose(name: str, candidate: torch.Tensor) -> torch.Tensor:
            return torch.where(
                getattr(self, f"{name}_fixed_mask"),
                getattr(self, f"{name}_fixed_values"),
                candidate,
            )

        upper_right = impose("upper_right", self.raw_upper_right)
        upper_left_candidate = torch.cat(
            (self.raw_upper_left[:-1, :], upper_right[0:1, :]), dim=0
        )
        upper_left = impose("upper_left", upper_left_candidate)
        lower_left_candidate = torch.cat(
            (self.raw_lower_left[:, :-1], upper_left[:, 0:1]), dim=1
        )
        lower_left = impose("lower_left", lower_left_candidate)
        return {
            "upper_right": upper_right,
            "upper_left": upper_left,
            "lower_left": lower_left,
        }

    def operators(self, local_points: torch.Tensor, max_order: int = 2):
        xi, eta = local_points[:, 0], local_points[:, 1]
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

    @staticmethod
    def evaluate(
        coefficients: torch.Tensor, operators, dx: int = 0, dy: int = 0
    ) -> torch.Tensor:
        return torch.einsum(
            "ni,ij,nj->n",
            operators["x"][dx],
            coefficients,
            operators["y"][dy],
        )

    def forward(self, patch_name: str, local_points: torch.Tensor) -> torch.Tensor:
        coefficients = self.coefficient_tensors()[patch_name]
        return self.evaluate(coefficients, self.operators(local_points, 0))[:, None]


def sample_local_points(
    count: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    return torch.rand(count, 2, device=device, generator=generator)


def precompute_operators(
    model: bsplines, local_points: torch.Tensor, max_order: int = 2
):
    if hasattr(model, "operators_for_patch"):
        return {
            name: model.operators_for_patch(name, local_points, max_order)
            for name in model.patch_names
        }
    return model.operators(local_points, max_order=max_order)


def patch_operators(operators, patch_name: str):
    """Select patch-specific operators, or return a shared operator set."""
    return operators[patch_name] if patch_name in operators else operators


def laplace_residual_loss(model: bsplines, operators) -> torch.Tensor:
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
    model: bsplines, local_points: torch.Tensor, operators=None
) -> torch.Tensor:
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


def boundary_and_interface_errors(model: bsplines, sample_count: int = 501):
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


def plot_solution(
    model: bsplines,
    postprocessing_dir: Path,
    resolution: int = 101,
) -> None:
    """Save the plotted fields and all arrays needed to reproduce the plot."""
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
    for axis_object, value_parts, title, color_map in (
        (axes[0], prediction_parts, model.solution_title, "viridis"),
        (axes[1], exact_parts, model.exact_solution_label, "viridis"),
        (axes[2], error_parts, "Absolute error", "magma"),
    ):
        combined_values = np.concatenate(value_parts)
        levels = np.linspace(combined_values.min(), combined_values.max(), 51)
        if np.isclose(levels[0], levels[-1]):
            levels = np.linspace(levels[0] - 1.0e-12, levels[-1] + 1.0e-12, 51)
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
    model: bsplines,
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


def generate_interface_local_points(
    count: int, device: torch.device
) -> dict[str, tuple[str, torch.Tensor, str, torch.Tensor]]:
    """Return paired local coordinates for both internal patch interfaces."""
    line = torch.linspace(0.0, 1.0, count, device=device)
    zero, one = torch.zeros_like(line), torch.ones_like(line)
    return {
        "upper_right_upper_left": (
            "upper_right",
            torch.stack((zero, line), dim=1),
            "upper_left",
            torch.stack((one, line), dim=1),
        ),
        "upper_left_lower_left": (
            "upper_left",
            torch.stack((line, zero), dim=1),
            "lower_left",
            torch.stack((line, one), dim=1),
        ),
    }


def precompute_interface_operators(
    model: bsplines,
    interface_points: dict[
        str, tuple[str, torch.Tensor, str, torch.Tensor]
    ],
):
    """Precompute analytical B-spline operators on both sides of interfaces."""
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


def interface_flux_loss(model: bsplines, interface_operators) -> torch.Tensor:
    """Mean squared sum of outward normal derivatives across interfaces."""
    coefficients = model.coefficient_tensors()
    terms = []

    # A-B is vertical: n_A=(-1,0), n_B=(1,0).
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

    # B-C is horizontal: n_B=(0,-1), n_C=(0,1).
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
    model: bsplines, interface_operators
) -> torch.Tensor:
    """Mean squared jump of the Laplace residual across interfaces."""
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
    model: bsplines, interior_operators, interface_operators
) -> dict[str, torch.Tensor]:
    return {
        "pde": laplace_residual_loss(model, interior_operators),
        "interface_residual": interface_residual_loss(
            model, interface_operators
        ),
        "interface_flux": interface_flux_loss(model, interface_operators),
    }


def strong_residual_loss(
    model: bsplines,
    interior_operators,
    interface_operators,
    *,
    pde_weight: float,
    interface_residual_weight: float,
    interface_flux_weight: float,
) -> torch.Tensor:
    terms = strong_loss_components(
        model, interior_operators, interface_operators
    )
    return (
        pde_weight * terms["pde"]
        + interface_residual_weight * terms["interface_residual"]
        + interface_flux_weight * terms["interface_flux"]
    )


def maximum_interface_residual_jump(
    model: bsplines, interface_operators
) -> torch.Tensor:
    """Maximum absolute Laplace-residual jump on either interface."""
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
    model: bsplines, interface_operators
) -> torch.Tensor:
    """Maximum absolute normal-flux imbalance on either interface."""
    coefficients = model.coefficient_tensors()
    jumps = []

    name_a, operators_a, name_b, operators_b = interface_operators[
        "upper_right_upper_left"
    ]
    jumps.append(
        (
            -model.evaluate(
                coefficients[name_a], patch_operators(operators_a, name_a),
                dx=1, dy=0
            )
            + model.evaluate(
                coefficients[name_b], patch_operators(operators_b, name_b),
                dx=1, dy=0
            )
        )
        .abs()
        .max()
    )

    name_a, operators_a, name_b, operators_b = interface_operators[
        "upper_left_lower_left"
    ]
    jumps.append(
        (
            -model.evaluate(
                coefficients[name_a], patch_operators(operators_a, name_a),
                dx=0, dy=1
            )
            + model.evaluate(
                coefficients[name_b], patch_operators(operators_b, name_b),
                dx=0, dy=1
            )
        )
        .abs()
        .max()
    )
    return torch.stack(jumps).max()
