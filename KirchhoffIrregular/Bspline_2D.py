"""B-spline Kirchhoff plate with simply supported exterior and free hole."""

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
import torch.nn.functional as functional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training import train


# Physical parameters from Kirchoff_plate_1.ipynb.
PLATE_LENGTH = 10.0
HOLE_CENTER = (5.0, 5.0)
HOLE_RADIUS = 1.5
LOAD_AMPLITUDE = 0.015
YOUNGS_MODULUS = 30_000.0
THICKNESS = 0.2
POISSON_RATIO = 0.2
FLEXURAL_RIGIDITY = (
    YOUNGS_MODULUS * THICKNESS**3 / (12.0 * (1.0 - POISSON_RATIO**2))
)

SEED = 42
DEFAULT_DEVICE = "gpu"
N_INTERIOR_POINTS = 20_000
N_BOUNDARY_POINTS_PER_COMPONENT = 10000
EPOCHS = 500_000
LEARNING_RATE = 1.0e-3
LBFGS_MAX_ITERATIONS = 500
PDE_WEIGHT = 100.0
MOMENT_BC_WEIGHT = 1.0
FREE_SHEAR_BC_WEIGHT = 1.0
KNOT_CONCENTRATION = 4.0
KNOT_REFINEMENT_MARGIN = 0.5
RESULT_PATH = (
    Path(__file__).resolve().parent
    / "results"
    / "bspline_hyperparameters.json"
)
FEM_RESULT_PATH = (
    Path(__file__).resolve().parent / "results" / "fem_reference.npz"
)


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


def focused_open_knots(
    n_basis: int,
    degree: int,
    start: float,
    end: float,
    focus_start: float,
    focus_end: float,
    concentration: float,
    device: torch.device,
) -> torch.Tensor:
    """Construct clamped knots with higher density in a focus interval.

    ``concentration`` is the ratio between the knot density inside and outside
    the focus interval.  A value of one recovers uniform internal knots.
    """
    if degree < 0 or n_basis <= degree:
        raise ValueError("n_basis must be greater than degree")
    if not start < focus_start < focus_end < end:
        raise ValueError("focus interval must lie strictly inside the domain")
    if concentration < 1.0:
        raise ValueError("knot concentration must be at least one")
    n_internal = n_basis - degree - 1
    left_length = focus_start - start
    focus_length = focus_end - focus_start
    right_length = end - focus_end
    total_mass = (
        left_length + concentration * focus_length + right_length
    )
    positions = []
    for index in range(1, n_internal + 1):
        mass = total_mass * index / (n_internal + 1)
        if mass <= left_length:
            position = start + mass
        elif mass <= left_length + concentration * focus_length:
            position = focus_start + (mass - left_length) / concentration
        else:
            position = focus_end + (
                mass - left_length - concentration * focus_length
            )
        positions.append(position)
    internal = torch.tensor(positions, device=device)
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
    """Evaluate all B-spline basis functions using Cox--de Boor recursion."""
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
    """Evaluate analytical derivatives of every one-dimensional basis."""
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
    # Derivatives at the clamped right endpoint are left-hand limits.
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
    """B-spline field with hard outer support and an immersed circular hole."""

    def __init__(
        self,
        n_basis_x: int,
        n_basis_y: int,
        degree: int,
        device: torch.device,
        knot_concentration: float = KNOT_CONCENTRATION,
        knot_margin: float = KNOT_REFINEMENT_MARGIN,
    ) -> None:
        super().__init__()
        if degree < 4:
            raise ValueError("Kirchhoff's fourth-order PDE requires degree >= 4")
        self.n_basis_x = n_basis_x
        self.n_basis_y = n_basis_y
        self.degree = degree
        self.knot_concentration = knot_concentration
        self.knot_margin = knot_margin
        focus_x_start = HOLE_CENTER[0] - HOLE_RADIUS - knot_margin
        focus_x_end = HOLE_CENTER[0] + HOLE_RADIUS + knot_margin
        focus_y_start = HOLE_CENTER[1] - HOLE_RADIUS - knot_margin
        focus_y_end = HOLE_CENTER[1] + HOLE_RADIUS + knot_margin
        self.register_buffer(
            "knots_x",
            focused_open_knots(
                n_basis_x,
                degree,
                0.0,
                PLATE_LENGTH,
                focus_x_start,
                focus_x_end,
                knot_concentration,
                device,
            ),
        )
        self.register_buffer(
            "knots_y",
            focused_open_knots(
                n_basis_y,
                degree,
                0.0,
                PLATE_LENGTH,
                focus_y_start,
                focus_y_end,
                knot_concentration,
                device,
            ),
        )
        self.interior_coefficients = torch.nn.Parameter(
            1.0e-3
            * torch.randn(n_basis_x - 2, n_basis_y - 2, device=device)
        )

    def coefficient_tensor(self) -> torch.Tensor:
        """Return trainable coefficients padded by an exact zero frame."""
        return functional.pad(self.interior_coefficients, (1, 1, 1, 1))

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


def inside_domain(points: torch.Tensor) -> torch.Tensor:
    dx = points[:, 0] - HOLE_CENTER[0]
    dy = points[:, 1] - HOLE_CENTER[1]
    return dx.square() + dy.square() > HOLE_RADIUS**2


def sample_interior(
    count: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample reflection-symmetric points while excluding the hole."""
    accepted = []
    remaining = count
    while remaining:
        base_count = max((remaining + 3) // 4, 64)
        base = PLATE_LENGTH * torch.rand(
            base_count, 2, device=device, generator=generator
        )
        base = base[inside_domain(base)]
        reflected_x = base.clone()
        reflected_x[:, 0] = PLATE_LENGTH - reflected_x[:, 0]
        reflected_y = base.clone()
        reflected_y[:, 1] = PLATE_LENGTH - reflected_y[:, 1]
        reflected_xy = PLATE_LENGTH - base
        candidates = torch.cat(
            (base, reflected_x, reflected_y, reflected_xy), dim=0
        )
        selected = candidates[:remaining]
        accepted.append(selected)
        remaining -= selected.shape[0]
    return torch.cat(accepted, dim=0)


def sample_boundary(
    points_per_component: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample the simply supported exterior and free circular boundary."""
    line = torch.linspace(
        0.0, PLATE_LENGTH, points_per_component, device=device
    )
    zeros = torch.zeros_like(line)
    length = torch.full_like(line, PLATE_LENGTH)
    outer_points = torch.cat(
        (
            torch.stack((line, zeros), dim=1),
            torch.stack((line, length), dim=1),
            torch.stack((zeros, line), dim=1),
            torch.stack((length, line), dim=1),
        ),
        dim=0,
    )
    outer_normals = torch.cat(
        (
            torch.tensor([0.0, -1.0], device=device).repeat(points_per_component, 1),
            torch.tensor([0.0, 1.0], device=device).repeat(points_per_component, 1),
            torch.tensor([-1.0, 0.0], device=device).repeat(points_per_component, 1),
            torch.tensor([1.0, 0.0], device=device).repeat(points_per_component, 1),
        ),
        dim=0,
    )
    angles = torch.arange(points_per_component, device=device) * (
        2.0 * math.pi / points_per_component
    )
    radial = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    center = torch.tensor(HOLE_CENTER, device=device)
    hole_points = center + HOLE_RADIUS * radial
    hole_normals = -radial
    return outer_points, outer_normals, hole_points, hole_normals


def transverse_load(points: torch.Tensor) -> torch.Tensor:
    return (
        LOAD_AMPLITUDE
        * torch.sin(math.pi * points[:, 0] / PLATE_LENGTH)
        * torch.sin(math.pi * points[:, 1] / PLATE_LENGTH)
    )


def precompute_operators(model: bsplines, points: torch.Tensor, max_order=4):
    """Precompute fixed basis values and derivatives outside training."""
    return model.operators(points, max_order=max_order)


def kirchhoff_pde_loss(
    model: bsplines, operators, points: torch.Tensor
) -> torch.Tensor:
    w_xxxx = model.evaluate(operators, dx=4, dy=0)
    w_xxyy = model.evaluate(operators, dx=2, dy=2)
    w_yyyy = model.evaluate(operators, dx=0, dy=4)
    residual = (
        FLEXURAL_RIGIDITY * (w_xxxx + 2.0 * w_xxyy + w_yyyy)
        - transverse_load(points)
    ) / LOAD_AMPLITUDE
    return residual.square().mean()


def reference_deflection_scale() -> float:
    """Characteristic deflection used only to nondimensionalize losses."""
    return (
        LOAD_AMPLITUDE
        * PLATE_LENGTH**4
        / (4.0 * math.pi**4 * FLEXURAL_RIGIDITY)
    )


def moment_curvature(
    model: bsplines, operators, normals: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return w, Hessian components, and M_nn/(-D)."""
    w = model.evaluate(operators)
    w_xx = model.evaluate(operators, dx=2, dy=0)
    w_xy = model.evaluate(operators, dx=1, dy=1)
    w_yy = model.evaluate(operators, dx=0, dy=2)
    n_x, n_y = normals[:, 0], normals[:, 1]
    w_nn = (
        n_x.square() * w_xx
        + 2.0 * n_x * n_y * w_xy
        + n_y.square() * w_yy
    )
    laplacian = w_xx + w_yy
    moment_curvature = (
        (1.0 - POISSON_RATIO) * w_nn + POISSON_RATIO * laplacian
    )
    return w, w_xx, w_xy, w_yy, moment_curvature


def simply_supported_outer_loss(
    model: bsplines, operators, normals: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized w=0 and M_nn=0 losses on the outer square."""
    w, _, _, _, normal_moment = moment_curvature(
        model, operators, normals
    )
    deflection_scale = reference_deflection_scale()
    displacement_loss = (w / deflection_scale).square().mean()
    moment_loss = (
        normal_moment * PLATE_LENGTH**2 / deflection_scale
    ).square().mean()
    return displacement_loss, moment_loss


def free_hole_loss(
    model: bsplines, operators, normals: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return M_nn=0 and effective Kirchhoff-shear losses on the hole."""
    _, w_xx, w_xy, w_yy, normal_moment = moment_curvature(
        model, operators, normals
    )
    w_xxx = model.evaluate(operators, dx=3, dy=0)
    w_xxy = model.evaluate(operators, dx=2, dy=1)
    w_xyy = model.evaluate(operators, dx=1, dy=2)
    w_yyy = model.evaluate(operators, dx=0, dy=3)
    n_x, n_y = normals[:, 0], normals[:, 1]
    t_x, t_y = -n_y, n_x
    w_nn = (
        n_x.square() * w_xx
        + 2.0 * n_x * n_y * w_xy
        + n_y.square() * w_yy
    )
    w_tt = (
        t_x.square() * w_xx
        + 2.0 * t_x * t_y * w_xy
        + t_y.square() * w_yy
    )
    w_ttn = (
        t_x.square() * n_x * w_xxx
        + (t_x.square() * n_y + 2.0 * t_x * t_y * n_x) * w_xxy
        + (t_y.square() * n_x + 2.0 * t_x * t_y * n_y) * w_xyy
        + t_y.square() * n_y * w_yyy
    )
    normal_laplacian_gradient = (
        n_x * (w_xxx + w_xyy) + n_y * (w_xxy + w_yyy)
    )
    # With n pointing out of the material and t=(-n_y,n_x), the signed
    # curvature of the inner circular boundary is +1/R.
    twisting_moment_derivative = w_ttn + (w_nn - w_tt) / HOLE_RADIUS
    effective_shear = normal_laplacian_gradient + (
        1.0 - POISSON_RATIO
    ) * twisting_moment_derivative
    deflection_scale = reference_deflection_scale()
    moment_loss = (
        normal_moment * PLATE_LENGTH**2 / deflection_scale
    ).square().mean()
    shear_loss = (
        effective_shear * PLATE_LENGTH**3 / deflection_scale
    ).square().mean()
    return moment_loss, shear_loss


def total_loss(
    model: bsplines,
    interior_operators,
    interior_points: torch.Tensor,
    outer_operators,
    outer_normals: torch.Tensor,
    hole_operators,
    hole_normals: torch.Tensor,
) -> torch.Tensor:
    pde = kirchhoff_pde_loss(model, interior_operators, interior_points)
    _, outer_moment = simply_supported_outer_loss(
        model, outer_operators, outer_normals
    )
    hole_moment, hole_shear = free_hole_loss(
        model, hole_operators, hole_normals
    )
    return (
        PDE_WEIGHT * pde
        + MOMENT_BC_WEIGHT * (outer_moment + hole_moment)
        + FREE_SHEAR_BC_WEIGHT * hole_shear
    )


def refine_with_lbfgs(
    model: bsplines,
    loss_fn,
    max_iterations: int = LBFGS_MAX_ITERATIONS,
) -> float:
    """Refine the spline coefficients after Adam using L-BFGS."""
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


def load_hyperparameters(
    path: Path = RESULT_PATH,
) -> tuple[int, int, int, float, float]:
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
        float(params.get("knot_concentration", KNOT_CONCENTRATION)),
        float(params.get("knot_margin", KNOT_REFINEMENT_MARGIN)),
    )


def load_fem_reference(path: Path = FEM_RESULT_PATH) -> dict[str, np.ndarray]:
    """Load and validate the finite-element reference saved by FEM_2D.py."""
    if not path.exists():
        raise FileNotFoundError(
            f"No FEM reference found at {path}. Run FEM_2D.py first."
        )
    with np.load(path) as archive:
        reference = {name: archive[name].copy() for name in archive.files}
    expected = {
        "plate_length": PLATE_LENGTH,
        "hole_radius": HOLE_RADIUS,
        "load_amplitude": LOAD_AMPLITUDE,
        "flexural_rigidity": FLEXURAL_RIGIDITY,
        "poisson_ratio": POISSON_RATIO,
    }
    for name, value in expected.items():
        if name not in reference or not np.isclose(reference[name], value):
            raise ValueError(
                f"FEM reference parameter '{name}' does not match the "
                "B-spline problem. Regenerate FEM_2D.py."
            )
    if not np.allclose(reference["hole_center"], HOLE_CENTER):
        raise ValueError(
            "FEM hole center does not match. Regenerate FEM_2D.py."
        )
    if (
        "hole_boundary" not in reference
        or reference["hole_boundary"].item() != "free"
    ):
        raise ValueError(
            "FEM reference does not use a free hole. Regenerate FEM_2D.py."
        )
    return reference


def plot_solution(
    model: bsplines,
    device: torch.device,
    fem_reference: dict[str, np.ndarray],
    resolution=151,
) -> None:
    axis = torch.linspace(0.0, PLATE_LENGTH, resolution, device=device)
    x_grid, y_grid = torch.meshgrid(axis, axis, indexing="ij")
    all_points = torch.stack((x_grid.reshape(-1), y_grid.reshape(-1)), dim=1)
    valid = inside_domain(all_points)
    points = all_points[valid]
    operators = precompute_operators(model, points)
    with torch.no_grad():
        w = model.evaluate(operators)
        residual = (
            FLEXURAL_RIGIDITY
            * (
                model.evaluate(operators, 4, 0)
                + 2.0 * model.evaluate(operators, 2, 2)
                + model.evaluate(operators, 0, 4)
            )
            - transverse_load(points)
        ) / LOAD_AMPLITUDE

    w_grid = torch.full((resolution * resolution,), torch.nan, device=device)
    r_grid = torch.full_like(w_grid, torch.nan)
    w_grid[valid] = w
    r_grid[valid] = residual
    x_np = x_grid.cpu().numpy()
    y_np = y_grid.cpu().numpy()
    w_np = w_grid.reshape(resolution, resolution).cpu().numpy()
    r_np = r_grid.reshape(resolution, resolution).cpu().numpy()
    fem_points = fem_reference["points"]
    fem_triangles = fem_reference["triangles"]
    fem_deflection = fem_reference["deflection"]
    fem_tensor = torch.as_tensor(
        fem_points, device=device, dtype=model.interior_coefficients.dtype
    )
    fem_operators = precompute_operators(model, fem_tensor, max_order=0)
    with torch.no_grad():
        bspline_at_fem_nodes = model.evaluate(fem_operators).cpu().numpy()
    absolute_error = np.abs(bspline_at_fem_nodes - fem_deflection)
    denominator = np.linalg.norm(fem_deflection)
    relative_error = (
        np.linalg.norm(bspline_at_fem_nodes - fem_deflection) / denominator
        if denominator > 0.0
        else float("nan")
    )
    print(f"Relative nodal L2 error versus FEM: {relative_error:.6e}")

    figure, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    first = axes[0, 0].contourf(
        x_np, y_np, w_np, levels=50, cmap="viridis"
    )
    figure.colorbar(first, ax=axes[0, 0], label="Deflection w")
    axes[0, 0].set_title("B-spline deflection")
    second = axes[0, 1].tricontourf(
        fem_points[:, 0],
        fem_points[:, 1],
        fem_triangles,
        fem_deflection,
        levels=50,
        cmap="viridis",
    )
    figure.colorbar(second, ax=axes[0, 1], label="Deflection w")
    axes[0, 1].set_title("Morley FEM deflection")
    third = axes[1, 0].tricontourf(
        fem_points[:, 0],
        fem_points[:, 1],
        fem_triangles,
        absolute_error,
        levels=50,
        cmap="magma",
    )
    figure.colorbar(third, ax=axes[1, 0], label="Absolute error")
    axes[1, 0].set_title(
        f"B-spline versus FEM (relative L2={relative_error:.3e})"
    )
    fourth = axes[1, 1].contourf(
        x_np, y_np, np.abs(r_np), levels=50, cmap="magma"
    )
    figure.colorbar(
        fourth, ax=axes[1, 1], label="Normalized |PDE residual|"
    )
    axes[1, 1].set_title("B-spline interior residual")
    for axis_object in axes.flat:
        axis_object.set_aspect("equal")
        axis_object.set_xlabel("x")
        axis_object.set_ylabel("y")
    plt.show()


def main(device_name: str = DEFAULT_DEVICE) -> None:
    set_seed()
    device = select_device(device_name)
    (
        n_basis_x,
        n_basis_y,
        degree,
        knot_concentration,
        knot_margin,
    ) = load_hyperparameters()
    fem_reference = load_fem_reference()
    print(f"Using device: {device}")
    print(
        f"B-splines: n_basis_x={n_basis_x}, n_basis_y={n_basis_y}, "
        f"degree={degree}, concentration={knot_concentration:.3f}, "
        f"margin={knot_margin:.3f}"
    )
    generator = torch.Generator(device=device).manual_seed(SEED)
    interior_points = sample_interior(N_INTERIOR_POINTS, device, generator)
    (
        outer_points,
        outer_normals,
        hole_points,
        hole_normals,
    ) = sample_boundary(N_BOUNDARY_POINTS_PER_COMPONENT, device)
    model = bsplines(
        n_basis_x,
        n_basis_y,
        degree,
        device,
        knot_concentration=knot_concentration,
        knot_margin=knot_margin,
    )
    interior_operators = precompute_operators(model, interior_points)
    outer_operators = precompute_operators(model, outer_points, max_order=2)
    hole_operators = precompute_operators(model, hole_points, max_order=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=50, min_lr=1.0e-5
    )
    loss_fn = lambda: total_loss(
        model,
        interior_operators,
        interior_points,
        outer_operators,
        outer_normals,
        hole_operators,
        hole_normals,
    )
    train(
        model.parameters(),
        loss_fn,
        optimizer,
        epochs=EPOCHS,
        scheduler=scheduler,
        early_stopping_patience=500,
        min_delta=1.0e-9,
        print_every=100,
    )
    refined_loss = refine_with_lbfgs(model, loss_fn)
    print(f"Loss after L-BFGS refinement: {refined_loss:.6e}")
    with torch.no_grad():
        pde = kirchhoff_pde_loss(model, interior_operators, interior_points)
        outer_displacement, outer_moment = simply_supported_outer_loss(
            model, outer_operators, outer_normals
        )
        hole_moment, hole_shear = free_hole_loss(
            model, hole_operators, hole_normals
        )
    print(f"Normalized PDE MSE: {pde.item():.6e}")
    print(f"Outer displacement BC MSE: {outer_displacement.item():.6e}")
    print(f"Outer moment BC MSE: {outer_moment.item():.6e}")
    print(f"Free-hole moment MSE: {hole_moment.item():.6e}")
    print(f"Free-hole effective-shear MSE: {hole_shear.item():.6e}")
    plot_solution(model, device, fem_reference)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve an irregular Kirchhoff plate with B-splines"
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    arguments = parser.parse_args()
    main(arguments.device)
