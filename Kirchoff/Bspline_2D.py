"""B-spline solution of a simply supported rectangular Kirchhoff plate."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training import train


CONFIG_PATH = SCRIPT_DIR / "bspline_config.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
PROBLEM = CONFIG["problem"]
PLATE_LENGTH_X = float(PROBLEM["length_x"])
PLATE_LENGTH_Y = float(PROBLEM["length_y"])
LOAD_AMPLITUDE = float(PROBLEM["load_amplitude"])
YOUNGS_MODULUS = float(PROBLEM["youngs_modulus"])
THICKNESS = float(PROBLEM["thickness"])
POISSON_RATIO = float(PROBLEM["poisson_ratio"])
FLEXURAL_RIGIDITY = (
    YOUNGS_MODULUS * THICKNESS**3 / (12.0 * (1.0 - POISSON_RATIO**2))
)

SEED = int(CONFIG["reproducibility"]["seed"])
DEFAULT_DEVICE = str(CONFIG["runtime"]["device"])
N_INTERIOR_POINTS = int(CONFIG["sampling"]["interior_points"])
N_BOUNDARY_POINTS_PER_EDGE = int(CONFIG["sampling"]["boundary_points_per_edge"])
EPOCHS = int(CONFIG["training"]["epochs"])
LEARNING_RATE = float(CONFIG["training"]["learning_rate"])
LBFGS_MAX_ITERATIONS = int(CONFIG["training"]["lbfgs_max_iterations"])
PDE_WEIGHT = 1.0
MOMENT_BC_WEIGHT = 1.0
RESULT_PATH = SCRIPT_DIR / CONFIG["output"]["tuned_hyperparameters"]
POSTPROCESSING_DIR = SCRIPT_DIR / CONFIG["output"]["postprocessing_directory"]


def set_seed(seed: int = SEED) -> None:
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


def open_uniform_knots(
    n_basis: int,
    degree: int,
    start: float,
    end: float,
    device: torch.device,
) -> torch.Tensor:
    if degree < 0 or n_basis <= degree:
        raise ValueError("n_basis must be greater than degree")
    n_internal = n_basis - degree - 1
    if n_internal:
        internal = torch.linspace(
            start, end, n_internal + 2, device=device
        )[1:-1]
    else:
        internal = torch.empty(0, device=device)
    return torch.cat(
        (
            torch.full((degree + 1,), start, device=device),
            internal,
            torch.full((degree + 1,), end, device=device),
        )
    )


def basis_matrix(
    values: torch.Tensor, knots: torch.Tensor, degree: int
) -> torch.Tensor:
    """Evaluate all one-dimensional B-spline basis functions."""
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
        n_columns = knots.numel() - current_degree - 1
        updated = torch.zeros(
            (values.numel(), n_columns),
            device=values.device,
            dtype=values.dtype,
        )
        for index in range(n_columns):
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
    """Evaluate analytical derivatives of every B-spline basis function."""
    if derivative_order < 0:
        raise ValueError("derivative_order must be non-negative")
    n_basis = knots.numel() - degree - 1
    if derivative_order == 0:
        return basis_matrix(values, knots, degree)
    if derivative_order > degree:
        return torch.zeros(
            (values.numel(), n_basis),
            device=values.device,
            dtype=values.dtype,
        )
    # At a clamped right endpoint, derivatives are left-hand limits.  Using
    # the endpoint directly in the lower-degree recursion selects a terminal
    # zero-length knot span and incorrectly produces zero derivatives.
    values = values.reshape(-1)
    at_right_end = values == knots[-1]
    if at_right_end.any():
        values = values.clone()
        values[at_right_end] = torch.nextafter(knots[-1], knots[0])
    lower = basis_derivative_matrix(
        values, knots, degree - 1, derivative_order - 1
    )
    result = torch.zeros(
        (values.numel(), n_basis),
        device=values.device,
        dtype=values.dtype,
    )
    for index in range(n_basis):
        left_denominator = knots[index + degree] - knots[index]
        right_denominator = knots[index + degree + 1] - knots[index + 1]
        if left_denominator > 0:
            result[:, index] += degree / left_denominator * lower[:, index]
        if right_denominator > 0:
            result[:, index] -= (
                degree / right_denominator * lower[:, index + 1]
            )
    return result


class bsplines(torch.nn.Module):
    """Tensor-product deflection satisfying simply supported BCs exactly."""

    def __init__(
        self,
        n_basis_x: int,
        n_basis_y: int,
        degree: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        if degree < 4:
            raise ValueError("The fourth-order plate PDE requires degree >= 4")
        if min(n_basis_x, n_basis_y) < 6:
            raise ValueError("At least six basis functions are required per axis")
        self.n_basis_x = n_basis_x
        self.n_basis_y = n_basis_y
        self.degree = degree
        self.register_buffer(
            "knots_x",
            open_uniform_knots(
                n_basis_x, degree, 0.0, PLATE_LENGTH_X, device
            ),
        )
        self.register_buffer(
            "knots_y",
            open_uniform_knots(
                n_basis_y, degree, 0.0, PLATE_LENGTH_Y, device
            ),
        )
        self.register_buffer(
            "constraint_x", self._simply_supported_transform(
                self.knots_x, degree, n_basis_x, PLATE_LENGTH_X, device
            )
        )
        self.register_buffer(
            "constraint_y", self._simply_supported_transform(
                self.knots_y, degree, n_basis_y, PLATE_LENGTH_Y, device
            )
        )
        self.independent_coefficients = torch.nn.Parameter(
            1.0e-3 * torch.randn(n_basis_x - 4, n_basis_y - 4, device=device)
        )

    @staticmethod
    def _simply_supported_transform(
        knots: torch.Tensor,
        degree: int,
        n_basis: int,
        length: float,
        device: torch.device,
    ) -> torch.Tensor:
        """Map independent coefficients to c=c''=0 at both endpoints."""
        endpoints = torch.tensor((0.0, length), device=device)
        second = basis_derivative_matrix(endpoints, knots, degree, 2)
        transform = torch.zeros(n_basis, n_basis - 4, device=device)

        # c_0=c_{n-1}=0. The second derivative at either endpoint only
        # involves the first/last three coefficients. Keep the slope-bearing
        # adjacent coefficient free and eliminate the next coefficient.
        transform[1, 0] = 1.0
        transform[2, 0] = -second[0, 1] / second[0, 2]
        transform[-2, -1] = 1.0
        transform[-3, -1] = -second[1, -2] / second[1, -3]
        for physical_index in range(3, n_basis - 3):
            transform[physical_index, physical_index - 2] = 1.0
        return transform

    def coefficient_tensor(self) -> torch.Tensor:
        """Construct coefficients with exact w=0 and normal curvature=0."""
        return (
            self.constraint_x
            @ self.independent_coefficients
            @ self.constraint_y.transpose(0, 1)
        )

    def operators(self, points: torch.Tensor, max_order: int = 4):
        x, y = points[:, 0], points[:, 1]
        return {
            "x": tuple(
                basis_derivative_matrix(x, self.knots_x, self.degree, order)
                for order in range(max_order + 1)
            ),
            "y": tuple(
                basis_derivative_matrix(y, self.knots_y, self.degree, order)
                for order in range(max_order + 1)
            ),
        }

    def evaluate(self, operators, dx: int = 0, dy: int = 0):
        return torch.einsum(
            "ni,ij,nj->n",
            operators["x"][dx],
            self.coefficient_tensor(),
            operators["y"][dy],
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.evaluate(self.operators(points, max_order=0))[:, None]


def sample_interior(
    count: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    # Reflection-complete samples prevent Monte Carlo imbalance from shifting
    # the solution away from the center of this symmetric problem.
    base_count = (count + 3) // 4
    base = torch.rand(base_count, 2, device=device, generator=generator)
    base[:, 0] *= PLATE_LENGTH_X
    base[:, 1] *= PLATE_LENGTH_Y
    reflected_x = base.clone()
    reflected_x[:, 0] = PLATE_LENGTH_X - reflected_x[:, 0]
    reflected_y = base.clone()
    reflected_y[:, 1] = PLATE_LENGTH_Y - reflected_y[:, 1]
    reflected_xy = torch.empty_like(base)
    reflected_xy[:, 0] = PLATE_LENGTH_X - base[:, 0]
    reflected_xy[:, 1] = PLATE_LENGTH_Y - base[:, 1]
    return torch.cat(
        (base, reflected_x, reflected_y, reflected_xy), dim=0
    )[:count]


def sample_boundary(
    points_per_edge: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return points on vertical and horizontal pairs of edges."""
    x_line = torch.linspace(0.0, PLATE_LENGTH_X, points_per_edge, device=device)
    y_line = torch.linspace(0.0, PLATE_LENGTH_Y, points_per_edge, device=device)
    zeros_x = torch.zeros_like(x_line)
    zeros_y = torch.zeros_like(y_line)
    max_x = torch.full_like(y_line, PLATE_LENGTH_X)
    max_y = torch.full_like(x_line, PLATE_LENGTH_Y)
    vertical = torch.cat(
        (
            torch.stack((zeros_y, y_line), dim=1),
            torch.stack((max_x, y_line), dim=1),
        ),
        dim=0,
    )
    horizontal = torch.cat(
        (
            torch.stack((x_line, zeros_x), dim=1),
            torch.stack((x_line, max_y), dim=1),
        ),
        dim=0,
    )
    return vertical, horizontal


def transverse_load(points: torch.Tensor) -> torch.Tensor:
    return (
        LOAD_AMPLITUDE
        * torch.sin(math.pi * points[:, 0] / PLATE_LENGTH_X)
        * torch.sin(math.pi * points[:, 1] / PLATE_LENGTH_Y)
    )


def exact_solution(points: torch.Tensor) -> torch.Tensor:
    wave_number_squared = (
        (math.pi / PLATE_LENGTH_X) ** 2
        + (math.pi / PLATE_LENGTH_Y) ** 2
    )
    amplitude = (
        LOAD_AMPLITUDE
        / (FLEXURAL_RIGIDITY * wave_number_squared**2)
    )
    return (
        amplitude
        * torch.sin(math.pi * points[:, 0] / PLATE_LENGTH_X)
        * torch.sin(math.pi * points[:, 1] / PLATE_LENGTH_Y)
    )


def precompute_operators(model: bsplines, points: torch.Tensor, max_order=4):
    return model.operators(points, max_order=max_order)


def pde_loss(model: bsplines, operators, points: torch.Tensor) -> torch.Tensor:
    biharmonic = (
        model.evaluate(operators, dx=4, dy=0)
        + 2.0 * model.evaluate(operators, dx=2, dy=2)
        + model.evaluate(operators, dx=0, dy=4)
    )
    residual = (
        FLEXURAL_RIGIDITY * biharmonic - transverse_load(points)
    ) / LOAD_AMPLITUDE
    return residual.square().mean()


def moment_boundary_loss(
    model: bsplines, vertical_operators, horizontal_operators
) -> torch.Tensor:
    """Enforce M_x=0 vertically and M_y=0 horizontally."""
    vertical_curvature = (
        model.evaluate(vertical_operators, dx=2, dy=0)
        + POISSON_RATIO * model.evaluate(vertical_operators, dx=0, dy=2)
    )
    horizontal_curvature = (
        POISSON_RATIO * model.evaluate(horizontal_operators, dx=2, dy=0)
        + model.evaluate(horizontal_operators, dx=0, dy=2)
    )
    deflection_scale = exact_solution(
        torch.tensor(
            [[PLATE_LENGTH_X / 2.0, PLATE_LENGTH_Y / 2.0]],
            device=vertical_curvature.device,
            dtype=vertical_curvature.dtype,
        )
    )[0]
    reference_length = math.sqrt(PLATE_LENGTH_X * PLATE_LENGTH_Y)
    scale = reference_length**2 / deflection_scale
    return 0.5 * (
        (scale * vertical_curvature).square().mean()
        + (scale * horizontal_curvature).square().mean()
    )


def total_loss(
    model: bsplines,
    interior_operators,
    interior_points: torch.Tensor,
    vertical_operators,
    horizontal_operators,
) -> torch.Tensor:
    # Both w=0 and M_nn=0 are built into the coefficient transformation.
    return PDE_WEIGHT * pde_loss(model, interior_operators, interior_points)


def refine_with_lbfgs(
    model: bsplines,
    loss_fn,
    max_iterations: int = LBFGS_MAX_ITERATIONS,
) -> float:
    """Refine the linear spline coefficients after the Adam stage."""
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=max_iterations,
        tolerance_grad=1.0e-10,
        tolerance_change=1.0e-12,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss

    optimizer.step(closure)
    return loss_fn().detach().item()


def load_hyperparameters(path: Path = RESULT_PATH) -> tuple[int, int, int]:
    if not path.exists():
        raise FileNotFoundError(
            f"No tuned hyperparameters found at {path}. Run tune_bsplines.py first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("best_params", payload)
    return (
        int(params["n_basis_x"]),
        int(params["n_basis_y"]),
        int(params["degree"]),
    )


def plot_comparison(
    model: bsplines, device: torch.device, history: list[float], resolution=None
) -> None:
    resolution = resolution or int(CONFIG["sampling"]["plot_points_per_axis"])
    x_axis = torch.linspace(0.0, PLATE_LENGTH_X, resolution, device=device)
    y_axis = torch.linspace(0.0, PLATE_LENGTH_Y, resolution, device=device)
    x_grid, y_grid = torch.meshgrid(x_axis, y_axis, indexing="ij")
    points = torch.stack((x_grid.reshape(-1), y_grid.reshape(-1)), dim=1)
    operators = precompute_operators(model, points)
    with torch.no_grad():
        prediction = model.evaluate(operators)
        exact = exact_solution(points)
        residual = (
            FLEXURAL_RIGIDITY
            * (
                model.evaluate(operators, 4, 0)
                + 2.0 * model.evaluate(operators, 2, 2)
                + model.evaluate(operators, 0, 4)
            )
            - transverse_load(points)
        ) / LOAD_AMPLITUDE
    relative_error = (
        torch.linalg.vector_norm(prediction - exact)
        / torch.linalg.vector_norm(exact)
    ).item()
    print(f"Relative L2 deflection error: {relative_error:.6e}")

    x_np, y_np = x_grid.cpu().numpy(), y_grid.cpu().numpy()
    fields = (
        ("B-spline deflection", prediction, "viridis"),
        ("Exact deflection", exact, "viridis"),
        ("Absolute error", torch.abs(prediction - exact), "magma"),
        ("Normalized |PDE residual|", torch.abs(residual), "magma"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for axis, (title, field, color_map) in zip(axes.flat, fields):
        values = field.reshape(resolution, resolution).cpu().numpy()
        contour = axis.contourf(x_np, y_np, values, levels=50, cmap=color_map)
        figure.colorbar(contour, ax=axis)
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    POSTPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    solution_path = POSTPROCESSING_DIR / "bspline_kirchhoff_hard_solution_data.npz"
    training_path = POSTPROCESSING_DIR / "bspline_kirchhoff_hard_training_data.npz"
    np.savez_compressed(
        solution_path, x=x_axis.cpu().numpy(), y=y_axis.cpu().numpy(),
        prediction=prediction.reshape(resolution, resolution).cpu().numpy(),
        reference=exact.reshape(resolution, resolution).cpu().numpy(),
        absolute_error=torch.abs(prediction-exact).reshape(resolution, resolution).cpu().numpy(),
        residual=torch.abs(residual).reshape(resolution, resolution).cpu().numpy(),
        validation_relative_l2_error=np.asarray(relative_error),
    )
    np.savez_compressed(
        training_path, epoch=np.arange(1, len(history)+1), loss=np.asarray(history)
    )
    figure.savefig(
        POSTPROCESSING_DIR / "bspline_kirchhoff_hard_solution.png",
        dpi=300, bbox_inches="tight",
    )
    training_figure, training_axis = plt.subplots(figsize=(6, 4))
    training_axis.semilogy(np.arange(1, len(history)+1), history)
    training_axis.set_xlabel("Epoch")
    training_axis.set_ylabel("Physics-informed loss")
    training_axis.set_title("B-spline Kirchhoff training")
    training_figure.tight_layout()
    training_figure.savefig(
        POSTPROCESSING_DIR / "bspline_kirchhoff_hard_training.png",
        dpi=300, bbox_inches="tight",
    )
    if bool(CONFIG["output"]["show_plots"]):
        plt.show()
    else:
        plt.close("all")


def main(
    device_name: str = DEFAULT_DEVICE,
    epochs: int | None = None,
    use_tuned: bool | None = None,
) -> None:
    set_seed()
    device = select_device(device_name)
    should_load = (
        bool(CONFIG["runtime"]["load_tuned_if_available"]) and RESULT_PATH.exists()
        if use_tuned is None else use_tuned
    )
    if should_load:
        n_basis_x, n_basis_y, degree = load_hyperparameters()
    else:
        n_basis_x = int(CONFIG["model"]["n_basis_x"])
        n_basis_y = int(CONFIG["model"]["n_basis_y"])
        degree = int(CONFIG["model"]["degree"])
    print(f"Using device: {device}")
    print(
        f"B-splines: n_basis_x={n_basis_x}, n_basis_y={n_basis_y}, "
        f"degree={degree}"
    )
    generator = torch.Generator(device=device).manual_seed(SEED)
    interior_points = sample_interior(N_INTERIOR_POINTS, device, generator)
    vertical_points, horizontal_points = sample_boundary(
        N_BOUNDARY_POINTS_PER_EDGE, device
    )
    model = bsplines(n_basis_x, n_basis_y, degree, device)
    interior_operators = precompute_operators(model, interior_points)
    vertical_operators = precompute_operators(model, vertical_points, max_order=2)
    horizontal_operators = precompute_operators(
        model, horizontal_points, max_order=2
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=50, min_lr=1.0e-5
    )
    loss_fn = lambda: total_loss(
        model,
        interior_operators,
        interior_points,
        vertical_operators,
        horizontal_operators,
    )
    history = train(
        model.parameters(),
        loss_fn,
        optimizer,
        epochs=epochs or EPOCHS,
        scheduler=scheduler,
        early_stopping_patience=500,
        min_delta=1.0e-9,
        print_every=100,
    )
    refined_loss = refine_with_lbfgs(model, loss_fn)
    print(f"Loss after L-BFGS refinement: {refined_loss:.6e}")
    with torch.no_grad():
        interior_error = pde_loss(model, interior_operators, interior_points)
        moment_error = moment_boundary_loss(
            model, vertical_operators, horizontal_operators
        )
        boundary_points = torch.cat((vertical_points, horizontal_points), dim=0)
        maximum_boundary_deflection = model(boundary_points).abs().max()
    print(f"Normalized PDE MSE: {interior_error.item():.6e}")
    print(f"Normalized moment BC MSE: {moment_error.item():.6e}")
    print(
        "Maximum boundary |w| (hard condition): "
        f"{maximum_boundary_deflection.item():.6e}"
    )
    plot_comparison(model, device, history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve a rectangular Kirchhoff plate using B-splines"
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--epochs", type=int, default=None)
    tuned = parser.add_mutually_exclusive_group()
    tuned.add_argument("--use-tuned", dest="use_tuned", action="store_true")
    tuned.add_argument("--no-use-tuned", dest="use_tuned", action="store_false")
    parser.set_defaults(use_tuned=None)
    arguments = parser.parse_args()
    main(arguments.device, arguments.epochs, arguments.use_tuned)
