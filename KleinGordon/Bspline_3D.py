"""Tensor-product B-spline solver for the (2+1)-D Klein--Gordon PDE.

Hard Dirichlet/initial values are interpolated at tensor-product Greville
abscissae and frozen. PDE derivatives are evaluated analytically from
B-spline derivative matrices; input-coordinate autograd is not used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from KleinGordon._common import (
    T_BOUNDS, X_BOUNDS, Y_BOUNDS, exact_solution, forcing,
    initial_value_xy, initial_velocity_xy, relative_l2_error,
    sample_constraints, sample_interior, save_results, select_device, set_seed,
)
from training import train

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "bspline_config.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
SEED = int(CONFIG["reproducibility"]["seed"])
DEFAULT_DEVICE = str(CONFIG["runtime"]["device"])
LOAD_TUNED_IF_AVAILABLE = bool(CONFIG["runtime"]["load_tuned_if_available"])
N_BASIS_SPACE = int(CONFIG["model"]["n_basis_space"])
N_BASIS_TIME = int(CONFIG["model"]["n_basis_time"])
DEGREE = int(CONFIG["model"]["degree"])
CONSTRAINT_MODE = str(CONFIG["model"].get("constraint_mode", "hard"))
ENFORCE_INITIAL_VELOCITY = bool(CONFIG["problem"].get("enforce_initial_velocity", False))
SAMPLING = CONFIG["sampling"]
TRAINING = CONFIG["training"]
LOSS_WEIGHTS = {key: float(value) for key, value in CONFIG["loss_weights"].items()}
RESULT_PATH = SCRIPT_DIR / CONFIG["output"]["tuned_hyperparameters"]
POSTPROCESSING_DIR = SCRIPT_DIR / CONFIG["output"]["postprocessing_directory"]


def clamped_uniform_knots(bounds, n_basis, degree, device):
    """Create an open, clamped, uniform knot vector."""
    if degree < 2 or n_basis <= degree:
        raise ValueError("Require degree >= 2 and n_basis > degree")
    count = n_basis - degree - 1
    internal = torch.linspace(bounds[0], bounds[1], count + 2, device=device)[1:-1]
    return torch.cat((
        torch.full((degree + 1,), bounds[0], device=device), internal,
        torch.full((degree + 1,), bounds[1], device=device),
    ))


def basis_matrix(values, knots, degree):
    """Evaluate all degree-p B-spline basis functions."""
    knots = knots.to(device=values.device, dtype=values.dtype)
    basis = ((values[:, None] >= knots[:-1]) & (values[:, None] < knots[1:])).to(values.dtype)
    for order in range(1, degree + 1):
        count = len(knots) - order - 1
        next_basis = torch.zeros(values.numel(), count, device=values.device, dtype=values.dtype)
        for index in range(count):
            left = knots[index + order] - knots[index]
            right = knots[index + order + 1] - knots[index + 1]
            if left > 0:
                next_basis[:, index] += (values - knots[index]) / left * basis[:, index]
            if right > 0:
                next_basis[:, index] += (knots[index + order + 1] - values) / right * basis[:, index + 1]
        basis = next_basis
    at_right = values == knots[-1]
    if torch.any(at_right):
        basis[at_right] = 0.0
        basis[at_right, -1] = 1.0
    return basis


def basis_and_derivatives(values, knots, degree):
    """Return B, dB/dx and d2B/dx2 using exact spline identities."""
    if degree < 2:
        raise ValueError("Second-order PDE derivatives require degree >= 2")
    knots = knots.to(device=values.device, dtype=values.dtype)
    n_basis = len(knots) - degree - 1
    basis = basis_matrix(values, knots, degree)
    lower = basis_matrix(values, knots, degree - 1)
    first = torch.zeros_like(basis)
    for index in range(n_basis):
        left = knots[index + degree] - knots[index]
        right = knots[index + degree + 1] - knots[index + 1]
        if left > 0:
            first[:, index] += degree / left * lower[:, index]
        if right > 0:
            first[:, index] -= degree / right * lower[:, index + 1]

    lowest = basis_matrix(values, knots, degree - 2)
    lower_first = torch.zeros(values.numel(), n_basis + 1, device=values.device, dtype=values.dtype)
    lower_degree = degree - 1
    for index in range(n_basis + 1):
        left = knots[index + lower_degree] - knots[index]
        right = knots[index + lower_degree + 1] - knots[index + 1]
        if left > 0:
            lower_first[:, index] += lower_degree / left * lowest[:, index]
        if right > 0:
            lower_first[:, index] -= lower_degree / right * lowest[:, index + 1]

    second = torch.zeros_like(basis)
    for index in range(n_basis):
        left = knots[index + degree] - knots[index]
        right = knots[index + degree + 1] - knots[index + 1]
        if left > 0:
            second[:, index] += degree / left * lower_first[:, index]
        if right > 0:
            second[:, index] -= degree / right * lower_first[:, index + 1]
    return basis, first, second


def greville_abscissae(knots, degree):
    """Return one Greville point per basis function."""
    n_basis = knots.numel() - degree - 1
    points = torch.stack([
        knots[index + 1:index + degree + 1].mean()
        for index in range(n_basis)
    ])
    points[0] = knots[degree]
    points[-1] = knots[-degree - 1]
    return points


def tensor_product_interpolation(first_matrix, second_matrix, targets):
    """Solve B1 C B2^T=G for a coefficient surface C."""
    intermediate = torch.linalg.solve(first_matrix, targets)
    return torch.linalg.solve(second_matrix, intermediate.T).T


class KleinGordonBSpline(torch.nn.Module):
    """Tensor spline with Greville-frozen boundary and initial coefficients."""

    def __init__(self, n_basis_space, n_basis_time, degree, constraint_mode,
                 device, enforce_initial_velocity=ENFORCE_INITIAL_VELOCITY):
        super().__init__()
        if constraint_mode not in ("soft", "hard"):
            raise ValueError("constraint_mode must be 'soft' or 'hard'")
        if constraint_mode == "hard" and enforce_initial_velocity:
            raise ValueError(
                "Greville interpolation of the time-dependent boundary and an "
                "exact initial velocity generally constrain the same second "
                "time-layer coefficients incompatibly. Use the paper setting "
                "enforce_initial_velocity=false."
            )
        self.n_basis_space = n_basis_space
        self.n_basis_time = n_basis_time
        self.degree = degree
        self.constraint_mode = constraint_mode
        self.enforce_initial_velocity = enforce_initial_velocity
        self.register_buffer("knot_x", clamped_uniform_knots(X_BOUNDS, n_basis_space, degree, device))
        self.register_buffer("knot_y", clamped_uniform_knots(Y_BOUNDS, n_basis_space, degree, device))
        self.register_buffer("knot_t", clamped_uniform_knots(T_BOUNDS, n_basis_time, degree, device))

        shape = (n_basis_space, n_basis_space, n_basis_time)
        if constraint_mode == "hard":
            fixed, fixed_mask = self._greville_boundary_coefficients()
        else:
            fixed = torch.zeros(shape, device=device)
            fixed_mask = torch.zeros(shape, dtype=torch.bool, device=device)
        free_indices = (~fixed_mask).flatten().nonzero(as_tuple=False).squeeze(1)
        self.register_buffer("fixed_coefficients", fixed)
        self.register_buffer("fixed_mask", fixed_mask)
        self.register_buffer("free_indices", free_indices)
        self.interior_coefficients = torch.nn.Parameter(
            0.01 * torch.randn(free_indices.numel(), device=device)
        )

    def _greville_boundary_coefficients(self):
        """Interpolate the four spatial faces and t=0 coefficient face."""
        xg = greville_abscissae(self.knot_x, self.degree)
        yg = greville_abscissae(self.knot_y, self.degree)
        tg = greville_abscissae(self.knot_t, self.degree)
        bx = basis_matrix(xg, self.knot_x, self.degree)
        by = basis_matrix(yg, self.knot_y, self.degree)
        bt = basis_matrix(tg, self.knot_t, self.degree)
        sums = torch.zeros(self.n_basis_space, self.n_basis_space, self.n_basis_time, device=xg.device)
        counts = torch.zeros_like(sums)

        def face(first, second, first_basis, second_basis, builder):
            grid_first, grid_second = torch.meshgrid(first, second, indexing="ij")
            points = builder(grid_first.reshape(-1), grid_second.reshape(-1))
            targets = exact_solution(points).reshape(first.numel(), second.numel())
            return tensor_product_interpolation(first_basis, second_basis, targets)

        x_left = face(yg, tg, by, bt, lambda y, t: torch.stack((torch.full_like(y, X_BOUNDS[0]), y, t), 1))
        x_right = face(yg, tg, by, bt, lambda y, t: torch.stack((torch.full_like(y, X_BOUNDS[1]), y, t), 1))
        y_bottom = face(xg, tg, bx, bt, lambda x, t: torch.stack((x, torch.full_like(x, Y_BOUNDS[0]), t), 1))
        y_top = face(xg, tg, bx, bt, lambda x, t: torch.stack((x, torch.full_like(x, Y_BOUNDS[1]), t), 1))
        grid_x, grid_y = torch.meshgrid(xg, yg, indexing="ij")
        initial_face = tensor_product_interpolation(bx, by, initial_value_xy(grid_x, grid_y))

        sums[0, :, :] += x_left; counts[0, :, :] += 1.0
        sums[-1, :, :] += x_right; counts[-1, :, :] += 1.0
        sums[:, 0, :] += y_bottom; counts[:, 0, :] += 1.0
        sums[:, -1, :] += y_top; counts[:, -1, :] += 1.0
        sums[:, :, 0] += initial_face; counts[:, :, 0] += 1.0
        mask = counts > 0
        fixed = torch.where(mask, sums / counts.clamp_min(1.0), torch.zeros_like(sums))
        return fixed.detach(), mask

    def coefficient_tensor(self):
        """Insert trainable interior values into the frozen tensor."""
        flat = self.fixed_coefficients.flatten()
        flat = flat.scatter(0, self.free_indices, self.interior_coefficients)
        return flat.reshape(self.fixed_coefficients.shape)

    def forward(self, points):
        bx = basis_matrix(points[:, 0], self.knot_x, self.degree)
        by = basis_matrix(points[:, 1], self.knot_y, self.degree)
        bt = basis_matrix(points[:, 2], self.knot_t, self.degree)
        return torch.einsum("ni,nj,nk,ijk->n", bx, by, bt, self.coefficient_tensor())


def precompute_operators(model, points):
    """Precompute all basis and derivative matrices at collocation points."""
    with torch.no_grad():
        bx, bx_x, bx_xx = basis_and_derivatives(points[:, 0], model.knot_x, model.degree)
        by, by_y, by_yy = basis_and_derivatives(points[:, 1], model.knot_y, model.degree)
        bt, bt_t, bt_tt = basis_and_derivatives(points[:, 2], model.knot_t, model.degree)
    return bx, bx_x, bx_xx, by, by_y, by_yy, bt, bt_t, bt_tt


def field_values(coefficients, operators):
    """Evaluate u and its analytical derivatives by tensor contraction."""
    bx, bx_x, bx_xx, by, by_y, by_yy, bt, bt_t, bt_tt = operators
    def contract(ax, ay, at):
        return torch.einsum("ni,nj,nk,ijk->n", ax, ay, at, coefficients)
    return (
        contract(bx, by, bt), contract(bx_x, by, bt),
        contract(bx, by_y, bt), contract(bx, by, bt_t),
        contract(bx_xx, by, bt), contract(bx, by_yy, bt),
        contract(bx, by, bt_tt),
    )


def bspline_residual(model, operators, points):
    value, _, _, _, dxx, dyy, dtt = field_values(model.coefficient_tensor(), operators)
    return dtt - dxx - dyy + value.square() - forcing(points)


def bspline_loss(model, interior_operators, interior_points,
                 constraint_operators=None, constraints=None,
                 weights=LOSS_WEIGHTS):
    pde = bspline_residual(model, interior_operators, interior_points).square().mean()
    if model.constraint_mode == "hard":
        return weights["pde"] * pde
    if constraint_operators is None or constraints is None:
        raise ValueError("Soft constraints require operators and points")
    coefficients = model.coefficient_tensor()
    initial = field_values(coefficients, constraint_operators["initial"])
    initial_points = constraints["initial"]
    value_loss = (initial[0] - initial_value_xy(initial_points[:, 0], initial_points[:, 1])).square().mean()
    if model.enforce_initial_velocity:
        velocity_loss = (initial[3] - initial_velocity_xy(initial_points[:, 0], initial_points[:, 1])).square().mean()
    else:
        velocity_loss = torch.zeros((), device=interior_points.device)
    boundary = field_values(coefficients, constraint_operators["boundary"])[0]
    boundary_loss = (boundary - exact_solution(constraints["boundary"])).square().mean()
    return (weights["pde"] * pde + weights["initial_value"] * value_loss
            + weights["initial_velocity"] * velocity_loss
            + weights["boundary"] * boundary_loss)


def bspline_constraint_errors(model, constraints, constraint_operators):
    with torch.no_grad():
        coefficients = model.coefficient_tensor()
        initial = field_values(coefficients, constraint_operators["initial"])
        initial_points = constraints["initial"]
        value_error = (initial[0] - initial_value_xy(initial_points[:, 0], initial_points[:, 1])).abs().max()
        velocity_error = (initial[3] - initial_velocity_xy(initial_points[:, 0], initial_points[:, 1])).abs().max()
        boundary = field_values(coefficients, constraint_operators["boundary"])[0]
        boundary_error = (boundary - exact_solution(constraints["boundary"])).abs().max()
    return value_error, velocity_error, boundary_error


def load_hyperparameters(path=RESULT_PATH):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    params = payload.get("best_params", payload)
    if not params:
        raise ValueError(f"No tuned B-spline parameters in {path}")
    return int(params["n_basis_space"]), int(params["n_basis_time"]), int(params["degree"])


def main(device_name=DEFAULT_DEVICE, n_basis_space=N_BASIS_SPACE,
         n_basis_time=N_BASIS_TIME, degree=DEGREE,
         constraint_mode=CONSTRAINT_MODE, epochs=int(TRAINING["epochs"]),
         use_tuned=None):
    set_seed(SEED)
    device = select_device(device_name)
    should_load = (LOAD_TUNED_IF_AVAILABLE and RESULT_PATH.exists()
                   and RESULT_PATH.stat().st_size > 3
                   if use_tuned is None else use_tuned)
    if should_load:
        n_basis_space, n_basis_time, degree = load_hyperparameters()
        print(f"Loaded tuned hyperparameters from: {RESULT_PATH}")
    print(f"Using device: {device}")
    print(f"Klein-Gordon B-spline: n_space={n_basis_space}, n_time={n_basis_time}, degree={degree}, constraints={constraint_mode}")
    model = KleinGordonBSpline(n_basis_space, n_basis_time, degree, constraint_mode, device).to(device)
    generator = torch.Generator(device=device).manual_seed(SEED)
    interior = sample_interior(int(SAMPLING["interior_points"]), device, generator)
    constraints = sample_constraints(int(SAMPLING["boundary_points_per_face"]), int(SAMPLING["initial_points"]), device, generator)
    interior_operators = precompute_operators(model, interior)
    constraint_operators = {name: precompute_operators(model, points) for name, points in constraints.items()}
    before = bspline_constraint_errors(model, constraints, constraint_operators)
    print(f"Greville-frozen errors before training: initial={before[0].item():.3e}, boundary={before[2].item():.3e}")

    optimizer = torch.optim.Adam(model.parameters(), lr=float(TRAINING["learning_rate"]))
    schedule = TRAINING["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=float(schedule["factor"]),
        patience=int(schedule["patience"]), min_lr=float(schedule["min_learning_rate"]),
    )
    stopping = TRAINING["early_stopping"]
    history = train(
        model.parameters(),
        lambda: bspline_loss(model, interior_operators, interior, constraint_operators, constraints),
        optimizer, epochs=epochs, scheduler=scheduler,
        early_stopping_patience=int(stopping["patience"]),
        min_delta=float(stopping["min_delta"]),
        print_every=int(TRAINING["print_every"]),
    )
    validation = sample_interior(int(SAMPLING["validation_points"]), device, generator)
    validation_operators = precompute_operators(model, validation)
    relative_error = relative_l2_error(model, validation)
    residual_mse = bspline_residual(model, validation_operators, validation).detach().square().mean()
    value_error, velocity_error, boundary_error = bspline_constraint_errors(model, constraints, constraint_operators)
    print(f"Relative validation L2 error: {relative_error.item():.6e}")
    print(f"Validation PDE residual MSE: {residual_mse.item():.6e}")
    print(f"Maximum initial-value error: {value_error.item():.6e}")
    velocity_label = "Maximum initial-velocity error" if ENFORCE_INITIAL_VELOCITY else "Initial-velocity diagnostic (not enforced)"
    print(f"{velocity_label}: {velocity_error.item():.6e}")
    print(f"Maximum boundary interpolation error: {boundary_error.item():.6e}")
    save_results(
        model, history, relative_error.item(), POSTPROCESSING_DIR,
        f"bspline_klein_gordon_{constraint_mode}",
        f"Klein-Gordon B-spline ({constraint_mode})",
        int(SAMPLING["plot_spatial_points"]), list(SAMPLING["plot_times"]),
        {"constraint_mode": constraint_mode, "n_basis_space": n_basis_space,
         "n_basis_time": n_basis_time, "degree": degree},
        bool(CONFIG["output"]["show_plots"]),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve (2+1)-D Klein-Gordon with tensor B-splines")
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--n-basis-space", type=int, default=N_BASIS_SPACE)
    parser.add_argument("--n-basis-time", type=int, default=N_BASIS_TIME)
    parser.add_argument("--degree", type=int, default=DEGREE)
    parser.add_argument("--constraint", choices=("soft", "hard"), default=CONSTRAINT_MODE)
    parser.add_argument("--epochs", type=int, default=int(TRAINING["epochs"]))
    tuned = parser.add_mutually_exclusive_group()
    tuned.add_argument("--use-tuned", dest="use_tuned", action="store_true")
    tuned.add_argument("--no-use-tuned", dest="use_tuned", action="store_false")
    parser.set_defaults(use_tuned=None)
    args = parser.parse_args()
    main(args.device, args.n_basis_space, args.n_basis_time, args.degree,
         args.constraint, args.epochs, args.use_tuned)
