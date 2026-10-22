"""Boundary-fitted B-spline solver for an irregular Kirchhoff plate.

The physical annulus is mapped from ``(rho, theta) in [0, 1] x S1``.  The
outer edge is ``rho=1`` and the circular hole is ``rho=0``.  A clamped radial
B-spline basis and a periodic angular B-spline basis are used.  Omitting the
last radial basis function is equivalent to fixing its entire coefficient row
to zero and therefore imposes ``w=0`` exactly on the irregular outer edge.

The Kirchhoff equation is solved through its bending-energy weak form.  This
requires physical Hessians only; the simply-supported moment condition on the
outer edge and both free-edge conditions on the hole are natural conditions.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from KirchhoffIrregular.Bspline_2D import (
    FLEXURAL_RIGIDITY,
    LOAD_AMPLITUDE,
    PLATE_LENGTH,
    POISSON_RATIO,
    basis_derivative_matrix,
    open_uniform_knots,
    select_device,
)
from training import train


OUTER_CENTER = (5.0, 5.0)
OUTER_MEAN_RADIUS = 4.0
OUTER_COS3_AMPLITUDE = 0.12
OUTER_SIN2_AMPLITUDE = 0.08
HOLE_CENTER = (6.15, 4.20)
HOLE_RADIUS = 0.90

SEED = 42
DEFAULT_DEVICE = "gpu"
N_BASIS_RADIAL = 12
N_BASIS_ANGULAR = 16
DEGREE = 4
N_QUADRATURE_POINTS = 20_000
EPOCHS = 100_000
LEARNING_RATE = 1.0e-3

FEM_RESULT_PATH = (
    Path(__file__).resolve().parent
    / "results"
    / "fem_asymmetric_irregular_reference.npz"
)
TUNED_RESULT_PATH = (
    Path(__file__).resolve().parent
    / "results"
    / "asymmetric_irregular_bspline_hyperparameters.json"
)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def outer_radius(theta: torch.Tensor) -> torch.Tensor:
    return OUTER_MEAN_RADIUS * (
        1.0
        + OUTER_COS3_AMPLITUDE * torch.cos(3.0 * theta)
        + OUTER_SIN2_AMPLITUDE * torch.sin(2.0 * theta)
    )


def outer_radius_derivative(theta: torch.Tensor) -> torch.Tensor:
    return OUTER_MEAN_RADIUS * (
        -3.0 * OUTER_COS3_AMPLITUDE * torch.sin(3.0 * theta)
        + 2.0 * OUTER_SIN2_AMPLITUDE * torch.cos(2.0 * theta)
    )


def outer_radius_second_derivative(theta: torch.Tensor) -> torch.Tensor:
    return OUTER_MEAN_RADIUS * (
        -9.0 * OUTER_COS3_AMPLITUDE * torch.cos(3.0 * theta)
        - 4.0 * OUTER_SIN2_AMPLITUDE * torch.sin(2.0 * theta)
    )


def inside_outer_boundary(points: torch.Tensor) -> torch.Tensor:
    dx = points[:, 0] - OUTER_CENTER[0]
    dy = points[:, 1] - OUTER_CENTER[1]
    theta = torch.atan2(dy, dx)
    radius = torch.sqrt(dx.square() + dy.square())
    return radius < outer_radius(theta)


def outside_hole(points: torch.Tensor) -> torch.Tensor:
    dx = points[:, 0] - HOLE_CENTER[0]
    dy = points[:, 1] - HOLE_CENTER[1]
    return dx.square() + dy.square() > HOLE_RADIUS**2


def inside_domain(points: torch.Tensor) -> torch.Tensor:
    return inside_outer_boundary(points) & outside_hole(points)


def mapping_data(parameters: torch.Tensor):
    """Return physical points, Jacobians, map Hessians, and determinants."""
    rho = parameters[:, 0]
    theta = parameters[:, 1]
    cosine, sine = torch.cos(theta), torch.sin(theta)
    direction = torch.stack((cosine, sine), dim=1)
    tangent_direction = torch.stack((-sine, cosine), dim=1)

    radius = outer_radius(theta)
    radius_first = outer_radius_derivative(theta)
    radius_second = outer_radius_second_derivative(theta)
    inner = torch.tensor(
        HOLE_CENTER, device=parameters.device, dtype=parameters.dtype
    ) + HOLE_RADIUS * direction
    inner_first = HOLE_RADIUS * tangent_direction
    inner_second = -HOLE_RADIUS * direction
    outer = torch.tensor(
        OUTER_CENTER, device=parameters.device, dtype=parameters.dtype
    ) + radius[:, None] * direction
    outer_first = (
        radius_first[:, None] * direction
        + radius[:, None] * tangent_direction
    )
    outer_second = (
        (radius_second - radius)[:, None] * direction
        + 2.0 * radius_first[:, None] * tangent_direction
    )

    physical = (1.0 - rho)[:, None] * inner + rho[:, None] * outer
    derivative_rho = outer - inner
    derivative_theta = (
        (1.0 - rho)[:, None] * inner_first
        + rho[:, None] * outer_first
    )
    jacobian = torch.stack((derivative_rho, derivative_theta), dim=2)
    determinant = (
        jacobian[:, 0, 0] * jacobian[:, 1, 1]
        - jacobian[:, 0, 1] * jacobian[:, 1, 0]
    )
    if bool((determinant <= 0.0).any()):
        raise ValueError("Boundary-fitted mapping has a non-positive Jacobian")
    inverse_jacobian = torch.linalg.inv(jacobian)

    mixed = outer_first - inner_first
    theta_second = (
        (1.0 - rho)[:, None] * inner_second
        + rho[:, None] * outer_second
    )
    map_hessian = torch.zeros(
        parameters.shape[0],
        2,
        2,
        2,
        device=parameters.device,
        dtype=parameters.dtype,
    )
    map_hessian[:, :, 0, 1] = mixed
    map_hessian[:, :, 1, 0] = mixed
    map_hessian[:, :, 1, 1] = theta_second
    return physical, jacobian, inverse_jacobian, map_hessian, determinant


def map_to_physical(parameters: torch.Tensor) -> torch.Tensor:
    return mapping_data(parameters)[0]


def physical_to_parameter(
    points: torch.Tensor, iterations: int = 15
) -> torch.Tensor:
    """Invert the smooth fitted map with vectorized Newton iterations."""
    dx = points[:, 0] - HOLE_CENTER[0]
    dy = points[:, 1] - HOLE_CENTER[1]
    theta = torch.remainder(torch.atan2(dy, dx), 2.0 * math.pi)
    distance = torch.sqrt(dx.square() + dy.square())
    rho = torch.clamp((distance - HOLE_RADIUS) / 3.5, 0.0, 1.0)
    parameters = torch.stack((rho, theta), dim=1)
    for _ in range(iterations):
        physical, _, inverse, _, _ = mapping_data(parameters)
        correction = torch.einsum("nai,ni->na", inverse, physical - points)
        parameters = parameters - correction
        parameters[:, 0].clamp_(0.0, 1.0)
        parameters[:, 1] = torch.remainder(parameters[:, 1], 2.0 * math.pi)
    return parameters


def periodic_basis_derivative_matrix(
    values: torch.Tensor,
    n_basis: int,
    degree: int,
    derivative_order: int,
) -> torch.Tensor:
    """Evaluate a uniform periodic B-spline basis and its derivatives."""
    if n_basis <= degree:
        raise ValueError("n_basis_angular must be greater than degree")
    period = 2.0 * math.pi
    values = torch.remainder(values.reshape(-1), period)
    spacing = period / n_basis
    knots = spacing * torch.arange(
        -degree,
        n_basis + degree + 1,
        device=values.device,
        dtype=values.dtype,
    )
    raw = basis_derivative_matrix(values, knots, degree, derivative_order)
    periodic = torch.zeros(
        values.numel(), n_basis, device=values.device, dtype=values.dtype
    )
    for raw_index in range(raw.shape[1]):
        periodic_index = (raw_index - degree) % n_basis
        periodic[:, periodic_index] += raw[:, raw_index]
    return periodic


class bsplines(torch.nn.Module):
    """Boundary-fitted radial-periodic B-spline deflection field."""

    def __init__(
        self,
        n_basis_radial: int,
        n_basis_angular: int,
        degree: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        if degree < 2:
            raise ValueError("Kirchhoff weak form requires degree >= 2")
        if degree >= min(n_basis_radial, n_basis_angular):
            raise ValueError("degree must be smaller than both basis counts")
        self.n_basis_radial = n_basis_radial
        self.n_basis_angular = n_basis_angular
        self.degree = degree
        self.register_buffer(
            "radial_knots",
            open_uniform_knots(n_basis_radial, degree, 0.0, 1.0, device),
        )
        # The omitted last radial row is the fixed zero row at rho=1.
        self.coefficients = torch.nn.Parameter(
            torch.zeros(n_basis_radial - 1, n_basis_angular, device=device)
        )

    def parameter_operators(self, parameters: torch.Tensor, max_order=2):
        rho, theta = parameters[:, 0], parameters[:, 1]
        radial = tuple(
            basis_derivative_matrix(
                rho, self.radial_knots, self.degree, order
            )[:, :-1]
            for order in range(max_order + 1)
        )
        angular = tuple(
            periodic_basis_derivative_matrix(
                theta,
                self.n_basis_angular,
                self.degree,
                order,
            )
            for order in range(max_order + 1)
        )
        return {"radial": radial, "angular": angular}

    def basis_values(self, operators) -> torch.Tensor:
        tensor = torch.einsum(
            "ni,nj->nij", operators["radial"][0], operators["angular"][0]
        )
        return tensor.flatten(1)

    def evaluate_parameter(self, parameters: torch.Tensor) -> torch.Tensor:
        basis = self.basis_values(self.parameter_operators(parameters, 0))
        return basis @ self.coefficients.flatten()

    def forward(self, physical_points: torch.Tensor) -> torch.Tensor:
        parameters = physical_to_parameter(physical_points)
        return self.evaluate_parameter(parameters)[:, None]


def physical_basis_data(model: bsplines, parameters: torch.Tensor):
    """Precompute basis values and physical Hessians for the weak form."""
    operators = model.parameter_operators(parameters, max_order=2)
    radial, angular = operators["radial"], operators["angular"]
    values = torch.einsum("ni,nj->nij", radial[0], angular[0]).flatten(1)
    gradient_parameter = torch.stack(
        (
            torch.einsum("ni,nj->nij", radial[1], angular[0]).flatten(1),
            torch.einsum("ni,nj->nij", radial[0], angular[1]).flatten(1),
        ),
        dim=2,
    )
    hessian_parameter = torch.zeros(
        parameters.shape[0],
        values.shape[1],
        2,
        2,
        device=parameters.device,
        dtype=parameters.dtype,
    )
    hessian_parameter[:, :, 0, 0] = torch.einsum(
        "ni,nj->nij", radial[2], angular[0]
    ).flatten(1)
    mixed = torch.einsum(
        "ni,nj->nij", radial[1], angular[1]
    ).flatten(1)
    hessian_parameter[:, :, 0, 1] = mixed
    hessian_parameter[:, :, 1, 0] = mixed
    hessian_parameter[:, :, 1, 1] = torch.einsum(
        "ni,nj->nij", radial[0], angular[2]
    ).flatten(1)

    physical, _, inverse, map_hessian, determinant = mapping_data(parameters)
    gradient_physical = torch.einsum(
        "nma,nai->nmi", gradient_parameter, inverse
    )
    correction = hessian_parameter - torch.einsum(
        "nmi,niab->nmab", gradient_physical, map_hessian
    )
    hessian_physical = torch.einsum(
        "nai,nmab,nbj->nmij", inverse, correction, inverse
    )
    return physical, values, hessian_physical, determinant


def transverse_load(points: torch.Tensor) -> torch.Tensor:
    return (
        LOAD_AMPLITUDE
        * torch.sin(math.pi * points[:, 0] / PLATE_LENGTH)
        * torch.sin(math.pi * points[:, 1] / PLATE_LENGTH)
    )


def sample_parameter_domain(
    count: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    parameters = torch.rand(count, 2, device=device, generator=generator)
    parameters[:, 1] *= 2.0 * math.pi
    return parameters


def assemble_weak_system(model: bsplines, parameters: torch.Tensor):
    """Assemble K and f for the mapped Kirchhoff Galerkin equations Kc=f."""
    physical, basis, hessian, determinant = physical_basis_data(
        model, parameters
    )
    trace_hessian = hessian[:, :, 0, 0] + hessian[:, :, 1, 1]
    bending_features = torch.stack(
        (
            math.sqrt(1.0 - POISSON_RATIO) * hessian[:, :, 0, 0],
            math.sqrt(2.0 * (1.0 - POISSON_RATIO))
            * hessian[:, :, 0, 1],
            math.sqrt(1.0 - POISSON_RATIO) * hessian[:, :, 1, 1],
            math.sqrt(POISSON_RATIO) * trace_hessian,
        ),
        dim=2,
    )
    quadrature_weight = (2.0 * math.pi / parameters.shape[0]) * determinant
    stiffness = FLEXURAL_RIGIDITY * torch.einsum(
        "nma,nka,n->mk", bending_features, bending_features, quadrature_weight
    )
    force = torch.einsum(
        "nm,n,n->m", basis, transverse_load(physical), quadrature_weight
    )
    return stiffness.detach(), force.detach()


def weak_residual_loss(
    model: bsplines, stiffness: torch.Tensor, force: torch.Tensor
) -> torch.Tensor:
    residual = stiffness @ model.coefficients.flatten() - force
    scale = force.abs().max().clamp_min(torch.finfo(force.dtype).eps)
    return (residual / scale).square().mean()


def boundary_diagnostics(model: bsplines, point_count: int = 720):
    device = model.coefficients.device
    dtype = model.coefficients.dtype
    theta = torch.arange(point_count, device=device, dtype=dtype)
    theta *= 2.0 * math.pi / point_count
    coefficient_vector = model.coefficients.flatten()
    results = {}
    for name, rho, inner in (("outer", 1.0, False), ("hole", 0.0, True)):
        parameters = torch.stack((torch.full_like(theta, rho), theta), dim=1)
        _, basis, hessian_basis, _ = physical_basis_data(model, parameters)
        w = basis @ coefficient_vector
        hessian = torch.einsum("nmij,m->nij", hessian_basis, coefficient_vector)
        if inner:
            normals = -torch.stack((torch.cos(theta), torch.sin(theta)), dim=1)
        else:
            radius = outer_radius(theta)
            radius_first = outer_radius_derivative(theta)
            tangent = torch.stack(
                (
                    radius_first * torch.cos(theta) - radius * torch.sin(theta),
                    radius_first * torch.sin(theta) + radius * torch.cos(theta),
                ),
                dim=1,
            )
            normals = torch.stack((tangent[:, 1], -tangent[:, 0]), dim=1)
            normals /= torch.linalg.vector_norm(normals, dim=1, keepdim=True)
        normal_curvature = torch.einsum(
            "ni,nij,nj->n", normals, hessian, normals
        )
        laplacian = hessian[:, 0, 0] + hessian[:, 1, 1]
        moment_curvature = (
            (1.0 - POISSON_RATIO) * normal_curvature
            + POISSON_RATIO * laplacian
        )
        results[name] = (w, moment_curvature)
    return results


def load_tuned_hyperparameters(
    path: Path = TUNED_RESULT_PATH,
) -> tuple[int, int, int]:
    if not path.exists():
        raise FileNotFoundError(
            f"No tuned hyperparameters found at {path}. Run "
            "tune_bsplines_asymmetric_irregular.py first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("best_params", payload)
    required = {"n_basis_radial", "n_basis_angular", "degree"}
    missing = required.difference(params)
    if missing:
        raise ValueError(
            "The tuning JSON belongs to the former Cartesian formulation. "
            "Run tune_bsplines_asymmetric_irregular.py again for the "
            "boundary-fitted basis."
        )
    return (
        int(params["n_basis_radial"]),
        int(params["n_basis_angular"]),
        int(params["degree"]),
    )


def load_fem_reference(
    path: Path = FEM_RESULT_PATH,
) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"No FEM reference found at {path}. Run "
            "FEM_AsymmetricIrregular_2D.py first."
        )
    with np.load(path) as archive:
        reference = {name: archive[name].copy() for name in archive.files}
    expected_scalars = {
        "outer_mean_radius": OUTER_MEAN_RADIUS,
        "outer_cos3_amplitude": OUTER_COS3_AMPLITUDE,
        "outer_sin2_amplitude": OUTER_SIN2_AMPLITUDE,
        "hole_radius": HOLE_RADIUS,
        "load_amplitude": LOAD_AMPLITUDE,
        "flexural_rigidity": FLEXURAL_RIGIDITY,
        "poisson_ratio": POISSON_RATIO,
    }
    for name, value in expected_scalars.items():
        if name not in reference or not np.isclose(reference[name], value):
            raise ValueError(f"FEM parameter '{name}' does not match")
    if not np.allclose(reference["outer_center"], OUTER_CENTER):
        raise ValueError("FEM outer center does not match")
    if not np.allclose(reference["hole_center"], HOLE_CENTER):
        raise ValueError("FEM hole center does not match")
    return reference


def structured_parameter_grid(
    radial_count: int, angular_count: int, device: torch.device, dtype
):
    rho = torch.linspace(0.0, 1.0, radial_count, device=device, dtype=dtype)
    theta = torch.arange(angular_count, device=device, dtype=dtype)
    theta *= 2.0 * math.pi / angular_count
    rho_grid, theta_grid = torch.meshgrid(rho, theta, indexing="ij")
    parameters = torch.stack((rho_grid.flatten(), theta_grid.flatten()), dim=1)
    triangles = []
    for radial_index in range(radial_count - 1):
        for angular_index in range(angular_count):
            next_angle = (angular_index + 1) % angular_count
            lower_left = radial_index * angular_count + angular_index
            lower_right = radial_index * angular_count + next_angle
            upper_left = (radial_index + 1) * angular_count + angular_index
            upper_right = (radial_index + 1) * angular_count + next_angle
            triangles.append((lower_left, upper_left, upper_right))
            triangles.append((lower_left, upper_right, lower_right))
    return parameters, np.asarray(triangles, dtype=np.int32)


def plot_solution(
    model: bsplines,
    fem_reference: dict[str, np.ndarray],
    radial_count: int = 121,
    angular_count: int = 240,
) -> None:
    device = model.coefficients.device
    parameters, triangles = structured_parameter_grid(
        radial_count, angular_count, device, model.coefficients.dtype
    )
    physical = map_to_physical(parameters)
    with torch.no_grad():
        deflection = model.evaluate_parameter(parameters)
    physical_np = physical.cpu().numpy()
    deflection_np = deflection.cpu().numpy()
    triangulation = mtri.Triangulation(
        physical_np[:, 0], physical_np[:, 1], triangles
    )

    fem_points = fem_reference["points"]
    fem_triangles = fem_reference["triangles"]
    fem_deflection = fem_reference["deflection"]
    fem_tensor = torch.as_tensor(
        fem_points, device=device, dtype=model.coefficients.dtype
    )
    fem_parameters = physical_to_parameter(fem_tensor)
    with torch.no_grad():
        bspline_at_fem = model.evaluate_parameter(fem_parameters).cpu().numpy()
    absolute_error = np.abs(bspline_at_fem - fem_deflection)
    relative_error = np.linalg.norm(bspline_at_fem - fem_deflection) / np.linalg.norm(
        fem_deflection
    )
    print(f"Relative nodal L2 error versus FEM: {relative_error:.6e}")
    print(f"Maximum nodal absolute error versus FEM: {absolute_error.max():.6e}")

    figure, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    first = axes[0, 0].tricontourf(triangulation, deflection_np, 50, cmap="viridis")
    figure.colorbar(first, ax=axes[0, 0], label="Deflection w")
    axes[0, 0].set_title("Boundary-fitted B-spline")
    second = axes[0, 1].tricontourf(
        fem_points[:, 0], fem_points[:, 1], fem_triangles, fem_deflection, 50,
        cmap="viridis"
    )
    figure.colorbar(second, ax=axes[0, 1], label="Deflection w")
    axes[0, 1].set_title("Morley FEM reference")
    third = axes[1, 0].tricontourf(
        fem_points[:, 0], fem_points[:, 1], fem_triangles, absolute_error, 50,
        cmap="magma"
    )
    figure.colorbar(third, ax=axes[1, 0], label="Absolute error")
    axes[1, 0].set_title(f"Absolute error (relative L2={relative_error:.3e})")
    axes[1, 1].triplot(triangulation, linewidth=0.2)
    axes[1, 1].set_title("Boundary-fitted parameter grid")
    for axis in axes.flat:
        axis.set_aspect("equal")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    plt.show()


def main(
    device_name: str = DEFAULT_DEVICE,
    n_basis_radial: int = N_BASIS_RADIAL,
    n_basis_angular: int = N_BASIS_ANGULAR,
    degree: int = DEGREE,
    epochs: int = EPOCHS,
    use_tuned: bool | None = None,
    tuned_result_path: Path = TUNED_RESULT_PATH,
) -> None:
    set_seed()
    device = select_device(device_name)
    fem_reference = load_fem_reference()
    should_load = tuned_result_path.exists() if use_tuned is None else use_tuned
    if should_load:
        try:
            n_basis_radial, n_basis_angular, degree = load_tuned_hyperparameters(
                tuned_result_path
            )
            print(f"Loaded boundary-fitted tuning result from: {tuned_result_path}")
        except ValueError:
            if use_tuned:
                raise
            print("Existing tuning JSON is for the old formulation; using defaults.")
    print(f"Using device: {device}")
    print(
        f"Boundary-fitted B-splines: radial={n_basis_radial}, "
        f"angular={n_basis_angular}, degree={degree}"
    )

    model = bsplines(n_basis_radial, n_basis_angular, degree, device)
    generator = torch.Generator(device=device).manual_seed(SEED)
    quadrature_parameters = sample_parameter_domain(
        N_QUADRATURE_POINTS, device, generator
    )
    stiffness, force = assemble_weak_system(model, quadrature_parameters)
    loss_fn = lambda: weak_residual_loss(model, stiffness, force)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=50, min_lr=1.0e-5
    )
    train(
        model.parameters(),
        loss_fn,
        optimizer,
        epochs=epochs,
        scheduler=scheduler,
        early_stopping_patience=500,
        min_delta=1.0e-10,
        print_every=100,
    )
    with torch.no_grad():
        final_loss = loss_fn().item()
        diagnostics = boundary_diagnostics(model)
    outer_w, outer_moment = diagnostics["outer"]
    _, hole_moment = diagnostics["hole"]
    print(f"Normalized weak residual MSE: {final_loss:.6e}")
    print(f"Maximum |w| on outer boundary: {outer_w.abs().max().item():.6e}")
    print(f"Outer moment-curvature MSE: {outer_moment.square().mean().item():.6e}")
    print(f"Hole moment-curvature MSE: {hole_moment.square().mean().item():.6e}")
    plot_solution(model, fem_reference)


def run_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Boundary-fitted B-spline Kirchhoff plate solver"
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--n-basis-radial", type=int, default=N_BASIS_RADIAL)
    parser.add_argument("--n-basis-angular", type=int, default=N_BASIS_ANGULAR)
    parser.add_argument("--degree", type=int, default=DEGREE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    tuned_group = parser.add_mutually_exclusive_group()
    tuned_group.add_argument("--use-tuned", dest="use_tuned", action="store_true")
    tuned_group.add_argument(
        "--no-use-tuned", dest="use_tuned", action="store_false"
    )
    parser.set_defaults(use_tuned=None)
    parser.add_argument(
        "--tuned-result-path", type=Path, default=TUNED_RESULT_PATH
    )
    arguments = parser.parse_args()
    main(
        arguments.device,
        arguments.n_basis_radial,
        arguments.n_basis_angular,
        arguments.degree,
        arguments.epochs,
        arguments.use_tuned,
        arguments.tuned_result_path,
    )


if __name__ == "__main__":
    run_cli()
