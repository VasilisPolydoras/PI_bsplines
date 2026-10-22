"""Physics-informed tensor-product B-splines for the viscous Burgers equation."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import solve_ivp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training import train


def set_seed(seed: int) -> None:
    """Seed every random-number generator used by the solver."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(name: str) -> torch.device:
    if name == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU was requested, but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cpu")


def clamped_knots(
    bounds: tuple[float, float],
    n_basis: int,
    degree: int,
    device: torch.device,
    clustering: float,
) -> torch.Tensor:
    """Construct an open clamped knot vector with optional central grading."""
    n_knots = n_basis + degree + 1
    breakpoint_count = n_knots - 2 * degree
    coordinates = torch.linspace(-1, 1, breakpoint_count, device=device)
    if clustering > 0:
        coordinates = torch.sinh(clustering * coordinates) / np.sinh(clustering)
    midpoint = 0.5 * (bounds[0] + bounds[1])
    half_width = 0.5 * (bounds[1] - bounds[0])
    values = midpoint + half_width * coordinates
    return torch.cat(
        (
            torch.full((degree,), values[0], device=device),
            values,
            torch.full((degree,), values[-1], device=device),
        )
    )


def basis_matrix(
    values: torch.Tensor, knots: torch.Tensor, degree: int
) -> torch.Tensor:
    """Evaluate every degree-p B-spline basis function at given values."""
    knots = knots.to(device=values.device, dtype=values.dtype)
    basis = (
        (values[:, None] >= knots[:-1])
        & (values[:, None] < knots[1:])
    ).to(values.dtype)
    for p in range(1, degree + 1):
        count = len(knots) - p - 1
        next_basis = torch.zeros(
            (values.numel(), count), device=values.device, dtype=values.dtype
        )
        for i in range(count):
            left = knots[i + p] - knots[i]
            right = knots[i + p + 1] - knots[i + 1]
            if left > 0:
                next_basis[:, i] += (values - knots[i]) / left * basis[:, i]
            if right > 0:
                next_basis[:, i] += (
                    (knots[i + p + 1] - values) / right * basis[:, i + 1]
                )
        basis = next_basis
    at_right = values == knots[-1]
    if torch.any(at_right):
        basis[at_right] = 0
        basis[at_right, -1] = 1
    return basis


def basis_and_derivatives(
    values: torch.Tensor, knots: torch.Tensor, degree: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate a B-spline basis and its first two derivatives."""
    if degree < 2:
        raise ValueError("The Burgers residual requires spline degree >= 2")
    knots = knots.to(device=values.device, dtype=values.dtype)
    n_basis = len(knots) - degree - 1
    basis = basis_matrix(values, knots, degree)
    lower = basis_matrix(values, knots, degree - 1)
    first = torch.zeros_like(basis)
    for i in range(n_basis):
        left = knots[i + degree] - knots[i]
        right = knots[i + degree + 1] - knots[i + 1]
        if left > 0:
            first[:, i] += degree / left * lower[:, i]
        if right > 0:
            first[:, i] -= degree / right * lower[:, i + 1]

    lowest = basis_matrix(values, knots, degree - 2)
    lower_first = torch.zeros(
        (values.numel(), n_basis + 1),
        device=values.device,
        dtype=values.dtype,
    )
    q = degree - 1
    for i in range(n_basis + 1):
        left = knots[i + q] - knots[i]
        right = knots[i + q + 1] - knots[i + 1]
        if left > 0:
            lower_first[:, i] += q / left * lowest[:, i]
        if right > 0:
            lower_first[:, i] -= q / right * lowest[:, i + 1]

    second = torch.zeros_like(basis)
    for i in range(n_basis):
        left = knots[i + degree] - knots[i]
        right = knots[i + degree + 1] - knots[i + 1]
        if left > 0:
            second[:, i] += degree / left * lower_first[:, i]
        if right > 0:
            second[:, i] -= degree / right * lower_first[:, i + 1]
    return basis, first, second


def greville_abscissae(
    knots: torch.Tensor, degree: int
) -> torch.Tensor:
    """Return one Greville abscissa for every B-spline basis function."""
    n_basis = knots.numel() - degree - 1
    greville = torch.stack(
        [
            knots[index + 1 : index + degree + 1].mean()
            for index in range(n_basis)
        ]
    )
    # An open clamped vector has endpoint Greville points. Set them exactly
    # so CUDA float32 averaging cannot round them outside the knot interval.
    greville[0] = knots[degree]
    greville[-1] = knots[-degree - 1]
    return greville


def interpolate_boundary_trace(
    knots: torch.Tensor,
    degree: int,
    physical_points: torch.Tensor,
    boundary_function,
) -> torch.Tensor:
    """Interpolate prescribed boundary data at the Greville abscissae."""
    greville = greville_abscissae(knots, degree)
    interpolation_matrix = basis_matrix(greville, knots, degree)
    return torch.linalg.solve(
        interpolation_matrix, boundary_function(physical_points)
    )


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "bspline_config.json"


def load_configuration(path: Path = CONFIG_PATH) -> dict:
    """Load and validate the complete Burgers B-spline configuration."""
    if not path.exists():
        raise FileNotFoundError(f"B-spline configuration not found: {path}")
    configuration = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "reproducibility", "runtime", "problem", "model", "sampling",
        "training", "tuning", "output",
    }
    missing = required.difference(configuration)
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
X_BOUNDS = tuple(float(value) for value in CONFIG["problem"]["x_bounds"])
T_BOUNDS = tuple(float(value) for value in CONFIG["problem"]["time_bounds"])
VISCOSITY = float(CONFIG["problem"]["viscosity"])
N_BASIS_X = int(CONFIG["model"]["n_basis_x"])
N_BASIS_T = int(CONFIG["model"]["n_basis_t"])
DEGREE = int(CONFIG["model"]["degree"])
SPATIAL_CLUSTERING = float(CONFIG["model"]["spatial_clustering"])
N_INTERIOR_POINTS = int(CONFIG["sampling"]["interior_points"])
N_BOUNDARY_POINTS_PER_EDGE = int(
    CONFIG["sampling"]["boundary_points_per_edge"]
)
PLOT_SPATIAL_POINTS = int(CONFIG["sampling"]["plot_spatial_points"])
PLOT_TIME_POINTS = int(CONFIG["sampling"]["plot_time_points"])
REFERENCE_SPATIAL_POINTS = int(
    CONFIG["sampling"]["reference_spatial_points"]
)
OPTIMIZER_NAME = str(CONFIG["training"]["optimizer"])
INTERIOR_EPOCHS = int(CONFIG["training"]["epochs"])
INTERIOR_LEARNING_RATE = float(CONFIG["training"]["learning_rate"])
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
HYPERPARAMETER_PATH = SCRIPT_DIR / CONFIG["output"]["tuned_hyperparameters"]
POSTPROCESSING_DIR = SCRIPT_DIR / CONFIG["output"]["postprocessing_directory"]
SHOW_PLOTS = bool(CONFIG["output"]["show_plots"])
SOLUTION_DATA_PATH = POSTPROCESSING_DIR / "bspline_burgers_solution_data.npz"
SOLUTION_PLOT_PATH = POSTPROCESSING_DIR / "bspline_burgers_solution.png"
TRAINING_DATA_PATH = POSTPROCESSING_DIR / "bspline_burgers_training_data.npz"
TRAINING_PLOT_PATH = POSTPROCESSING_DIR / "bspline_burgers_training.png"


class BurgersBSpline:
    """Tensor-product B-spline approximation of u(x, t)."""

    def __init__(
        self,
        n_basis_x,
        n_basis_t,
        degree,
        device=None,
        spatial_clustering=SPATIAL_CLUSTERING,
    ):
        device = torch.device("cpu") if device is None else torch.device(device)
        self.n_basis_x = n_basis_x
        self.n_basis_t = n_basis_t
        self.degree = degree
        self.spatial_clustering = spatial_clustering
        self.knot_x = self._clamped_knots(
            X_BOUNDS, n_basis_x, device, clustering=spatial_clustering
        )
        self.knot_t = self._clamped_knots(
            T_BOUNDS, n_basis_t, device, clustering=0.0
        )
        self.coeffs = torch.rand(
            n_basis_x, n_basis_t, device=device, requires_grad=True
        )
        self.fixed_coeffs = None
        self.interior_coeffs = None
        self.set_boundary_coefficients(self._build_boundary_coefficients())

    def _clamped_knots(self, bounds, n_basis, device, clustering):
        return clamped_knots(
            bounds, n_basis, self.degree, device, clustering
        )

    def basis_matrix(self, values, knots, degree=None):
        degree = self.degree if degree is None else degree
        return basis_matrix(values, knots, degree)

    def basis_and_derivatives(self, values, knots):
        return basis_and_derivatives(values, knots, self.degree)

    def coefficient_tensor(self):
        """Combine frozen x-boundary/t=0 entries with trainable interior entries."""
        if self.interior_coeffs is None:
            return self.coeffs
        coefficients = self.fixed_coeffs.clone()
        coefficients[1:-1, 1:] = self.interior_coeffs
        return coefficients

    def _build_boundary_coefficients(self):
        """Interpolate and freeze the three prescribed Dirichlet traces."""
        x_greville = greville_abscissae(self.knot_x, self.degree)
        t_greville = greville_abscissae(self.knot_t, self.degree)
        initial_trace = interpolate_boundary_trace(
            self.knot_x, self.degree, x_greville, initial_condition
        )
        zero_trace = lambda coordinate: torch.zeros_like(coordinate)
        left_trace = interpolate_boundary_trace(
            self.knot_t, self.degree, t_greville, zero_trace
        )
        right_trace = interpolate_boundary_trace(
            self.knot_t, self.degree, t_greville, zero_trace
        )
        fixed = torch.zeros_like(self.coeffs)
        fixed[:, 0] = initial_trace
        fixed[0, :] = left_trace
        fixed[-1, :] = right_trace
        return fixed.detach()

    def set_boundary_coefficients(self, fixed_coefficients):
        """Freeze x=-1, x=1 and t=0, then initialize a fresh PDE interior."""
        if fixed_coefficients.shape != self.coeffs.shape:
            raise ValueError(
                f"Expected shape {tuple(self.coeffs.shape)}, "
                f"got {tuple(fixed_coefficients.shape)}"
            )
        self.fixed_coeffs = fixed_coefficients.detach().clone()
        time_fraction = torch.linspace(
            0,
            1,
            self.n_basis_t,
            device=fixed_coefficients.device,
            dtype=fixed_coefficients.dtype,
        )
        decay = torch.exp(-VISCOSITY * np.pi**2 * time_fraction)
        initialization = fixed_coefficients[:, 0:1] * decay[None, :]
        initialization[0, :] = fixed_coefficients[0, :]
        initialization[-1, :] = fixed_coefficients[-1, :]
        self.interior_coeffs = (
            initialization[1:-1, 1:].detach().clone().requires_grad_(True)
        )

    def forward(self, points):
        bx = self.basis_matrix(points[:, 0], self.knot_x)
        bt = self.basis_matrix(points[:, 1], self.knot_t)
        return torch.einsum("ni,ij,nj->n", bx, self.coefficient_tensor(), bt)


def initial_condition(x):
    return -torch.sin(torch.pi * x)


def sample_interior(n_points, device, generator=None):
    unit = torch.rand(n_points, 2, device=device, generator=generator)
    unit[:, 0] = X_BOUNDS[0] + (X_BOUNDS[1] - X_BOUNDS[0]) * unit[:, 0]
    unit[:, 1] = T_BOUNDS[0] + (T_BOUNDS[1] - T_BOUNDS[0]) * unit[:, 1]
    return unit


def sample_boundary(points_per_edge, device):
    x = torch.linspace(*X_BOUNDS, points_per_edge, device=device)
    t = torch.linspace(*T_BOUNDS, points_per_edge, device=device)
    initial = torch.stack((x, torch.zeros_like(x)), dim=1)
    left = torch.stack((torch.full_like(t, X_BOUNDS[0]), t), dim=1)
    right = torch.stack((torch.full_like(t, X_BOUNDS[1]), t), dim=1)
    points = torch.cat((initial, left, right), dim=0)
    targets = torch.cat(
        (initial_condition(x), torch.zeros_like(t), torch.zeros_like(t))
    )
    return points, targets


def precompute_operators(model, points):
    with torch.no_grad():
        bx, bx_x, bx_xx = model.basis_and_derivatives(
            points[:, 0], model.knot_x
        )
        bt, bt_t, _ = model.basis_and_derivatives(points[:, 1], model.knot_t)
    return bx, bx_x, bx_xx, bt, bt_t


def field_values(coefficients, operators):
    bx, bx_x, bx_xx, bt, bt_t = operators
    value = torch.einsum("ni,ij,nj->n", bx, coefficients, bt)
    dx = torch.einsum("ni,ij,nj->n", bx_x, coefficients, bt)
    dxx = torch.einsum("ni,ij,nj->n", bx_xx, coefficients, bt)
    dt = torch.einsum("ni,ij,nj->n", bx, coefficients, bt_t)
    return value, dx, dxx, dt


def boundary_loss(model, boundary_points, boundary_targets):
    return (model.forward(boundary_points) - boundary_targets).square().mean()


def interpolate_boundary_coefficients(
    n_basis_x,
    n_basis_t,
    degree,
    device,
    spatial_clustering=SPATIAL_CLUSTERING,
):
    """Return the hard boundary coefficients constructed by a model."""
    model = BurgersBSpline(
        n_basis_x,
        n_basis_t,
        degree,
        device,
        spatial_clustering=spatial_clustering,
    )
    return model.fixed_coeffs.detach().clone()


def burgers_pde_loss(model, operators, viscosity=VISCOSITY):
    u, u_x, u_xx, u_t = field_values(model.coefficient_tensor(), operators)
    residual = u_t + u * u_x - viscosity * u_xx
    return residual.square().mean()


def burgers_loss(
    model,
    operators,
    boundary_points,
    boundary_targets,
    viscosity=VISCOSITY,
):
    """Combined loss retained for comparisons with single-stage training."""
    return burgers_pde_loss(model, operators, viscosity) + boundary_loss(
        model, boundary_points, boundary_targets
    )


def numerical_reference(
    x, times, viscosity=VISCOSITY,
    spatial_points=REFERENCE_SPATIAL_POINTS,
):
    """Generate a finite-difference reference solution for plotting/errors."""
    reference_x = np.linspace(*X_BOUNDS, spatial_points)
    dx = reference_x[1] - reference_x[0]
    initial = -np.sin(np.pi * reference_x)

    def right_hand_side(_, values):
        derivative = np.zeros_like(values)
        u_x = (values[2:] - values[:-2]) / (2 * dx)
        u_xx = (values[2:] - 2 * values[1:-1] + values[:-2]) / dx**2
        derivative[1:-1] = -values[1:-1] * u_x + viscosity * u_xx
        return derivative

    solution = solve_ivp(
        right_hand_side,
        (float(times[0]), float(times[-1])),
        initial,
        t_eval=times,
        method="DOP853",
        rtol=1e-8,
        atol=1e-10,
    )
    if not solution.success:
        raise RuntimeError(f"Reference solver failed: {solution.message}")
    return np.vstack(
        [np.interp(x, reference_x, values) for values in solution.y.T]
    )


def numerical_reference_at_points(
    points,
    viscosity=VISCOSITY,
    spatial_points=REFERENCE_SPATIAL_POINTS,
):
    """Evaluate the finite-difference reference at paired (x,t) points."""
    coordinates = points.detach().cpu().numpy()
    query_x = coordinates[:, 0]
    query_times = coordinates[:, 1]
    unique_times, time_indices = np.unique(
        query_times, return_inverse=True
    )
    reference_x = np.linspace(*X_BOUNDS, spatial_points)
    dx = reference_x[1] - reference_x[0]
    initial = -np.sin(np.pi * reference_x)

    def right_hand_side(_, values):
        derivative = np.zeros_like(values)
        u_x = (values[2:] - values[:-2]) / (2 * dx)
        u_xx = (
            values[2:] - 2 * values[1:-1] + values[:-2]
        ) / dx**2
        derivative[1:-1] = -values[1:-1] * u_x + viscosity * u_xx
        return derivative

    if unique_times[-1] == 0.0:
        time_solutions = initial[:, None]
    else:
        solution = solve_ivp(
            right_hand_side,
            (0.0, float(unique_times[-1])),
            initial,
            t_eval=unique_times,
            method="DOP853",
            rtol=1e-8,
            atol=1e-10,
        )
        if not solution.success:
            raise RuntimeError(
                f"Reference solver failed: {solution.message}"
            )
        time_solutions = solution.y

    values = np.asarray(
        [
            np.interp(x_value, reference_x, time_solutions[:, time_index])
            for x_value, time_index in zip(query_x, time_indices)
        ]
    )
    return torch.as_tensor(
        values, device=points.device, dtype=points.dtype
    )


def load_best_hyperparameters(path=HYPERPARAMETER_PATH):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Hyperparameter file not found: {path}. "
            "Run Burgers/tune_bsplines.py first."
        )
    result = json.loads(path.read_text(encoding="utf-8"))
    params = result["best_params"]
    degree = int(params["degree"])
    spatial_clustering = float(
        params.get("spatial_clustering", SPATIAL_CLUSTERING)
    )
    if "n_basis_x" in params and "n_basis_t" in params:
        return (
            int(params["n_basis_x"]),
            int(params["n_basis_t"]),
            degree,
            spatial_clustering,
        )

    # Backward compatibility for results produced by the earlier isotropic tuner.
    n_basis_x = int(params["n_basis"])
    n_basis_t = max(degree + 1, min(24, n_basis_x))
    print(
        "Legacy tuning result detected: using "
        f"n_basis_x={n_basis_x}, n_basis_t={n_basis_t}. "
        "Rerun Burgers/tune_bsplines.py to tune both directions independently."
    )
    return n_basis_x, n_basis_t, degree, spatial_clustering


def main(
    device_name=DEFAULT_DEVICE,
    n_basis_x=N_BASIS_X,
    n_basis_t=N_BASIS_T,
    degree=DEGREE,
    spatial_clustering=SPATIAL_CLUSTERING,
    interior_epochs=INTERIOR_EPOCHS,
    use_tuned=None,
):
    set_seed(SEED)
    device = select_device(device_name)
    should_load = (
        LOAD_TUNED_IF_AVAILABLE and HYPERPARAMETER_PATH.exists()
        if use_tuned is None
        else use_tuned
    )
    if should_load:
        n_basis_x, n_basis_t, degree, spatial_clustering = (
            load_best_hyperparameters()
        )
        print(f"Loaded tuned hyperparameters from: {HYPERPARAMETER_PATH}")
    print(f"Using device: {device}")
    print(f"Loaded configuration from: {CONFIG_PATH}")
    print(
        "Using tuned B-splines: "
        f"n_basis_x={n_basis_x}, n_basis_t={n_basis_t}, degree={degree}, "
        f"spatial_clustering={spatial_clustering}"
    )

    interior = sample_interior(N_INTERIOR_POINTS, device)
    boundary_points, boundary_targets = sample_boundary(
        N_BOUNDARY_POINTS_PER_EDGE, device
    )

    print("Imposing initial and boundary conditions by Greville interpolation")
    model = BurgersBSpline(
        n_basis_x,
        n_basis_t,
        degree,
        device,
        spatial_clustering=spatial_clustering,
    )
    operators = precompute_operators(model, interior)
    print(
        f"Frozen boundary loss: "
        f"{boundary_loss(model, boundary_points, boundary_targets).item():.6e}"
    )

    print("Optimizing interior coefficients with PDE residual only")
    if OPTIMIZER_NAME.lower() != "adam":
        raise ValueError(
            f"Unsupported optimizer '{OPTIMIZER_NAME}'; expected 'Adam'"
        )
    optimizer = torch.optim.Adam(
        [model.interior_coeffs], lr=INTERIOR_LEARNING_RATE
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE, min_lr=MIN_LEARNING_RATE
    )
    history = train(
        [model.interior_coeffs],
        lambda: burgers_pde_loss(model, operators),
        optimizer,
        epochs=interior_epochs,
        scheduler=scheduler,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        min_delta=MIN_DELTA,
        print_every=PRINT_EVERY,
    )
    print(
        f"Boundary loss after PDE training: "
        f"{boundary_loss(model, boundary_points, boundary_targets).item():.6e}"
    )

    x = np.linspace(*X_BOUNDS, PLOT_SPATIAL_POINTS)
    times = np.linspace(*T_BOUNDS, PLOT_TIME_POINTS)
    gx, gt = np.meshgrid(x, times)
    points = torch.tensor(
        np.column_stack((gx.ravel(), gt.ravel())),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        prediction = model.forward(points).cpu().numpy().reshape(gx.shape)
    reference = numerical_reference(x, times)
    error = np.abs(prediction - reference)
    relative_error = np.linalg.norm(prediction - reference) / np.linalg.norm(reference)
    print(f"Relative L2 error against numerical reference: {relative_error:.6e}")

    POSTPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        SOLUTION_DATA_PATH,
        x=x,
        time=times,
        prediction=prediction,
        reference=reference,
        absolute_error=error,
        validation_relative_l2_error=np.asarray(relative_error, dtype=float),
        viscosity=np.asarray(VISCOSITY, dtype=float),
        n_basis_x=np.asarray(n_basis_x, dtype=int),
        n_basis_t=np.asarray(n_basis_t, dtype=int),
        degree=np.asarray(degree, dtype=int),
        spatial_clustering=np.asarray(spatial_clustering, dtype=float),
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, values, title in zip(
        axes,
        (prediction, reference, error),
        ("B-spline prediction", "Numerical reference", "Absolute error"),
    ):
        image = axis.imshow(
            values,
            extent=[*X_BOUNDS, *T_BOUNDS],
            origin="lower",
            aspect="auto",
        )
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("t")
        fig.colorbar(image, ax=axis)
    plt.tight_layout()
    fig.savefig(SOLUTION_PLOT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved solution data to: {SOLUTION_DATA_PATH}")
    print(f"Saved solution plot to: {SOLUTION_PLOT_PATH}")

    loss_values = np.asarray(history, dtype=float)
    epoch_values = np.arange(1, loss_values.size + 1)
    np.savez_compressed(
        TRAINING_DATA_PATH, epoch=epoch_values, loss=loss_values
    )
    training_figure, training_axis = plt.subplots(figsize=(6, 4))
    training_axis.plot(epoch_values, loss_values, label="Interior PDE training")
    training_axis.set_yscale("log")
    training_axis.set_xlabel("Epochs")
    training_axis.set_ylabel("Loss")
    training_axis.set_title("Viscous Burgers B-spline Loss")
    training_axis.legend()
    training_figure.tight_layout()
    training_figure.savefig(
        TRAINING_PLOT_PATH, dpi=300, bbox_inches="tight"
    )
    print(f"Saved training data to: {TRAINING_DATA_PATH}")
    print(f"Saved training plot to: {TRAINING_PLOT_PATH}")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
        plt.close(training_figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve the viscous Burgers equation with B-splines"
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--n-basis-x", type=int, default=N_BASIS_X)
    parser.add_argument("--n-basis-t", type=int, default=N_BASIS_T)
    parser.add_argument("--degree", type=int, default=DEGREE)
    parser.add_argument(
        "--spatial-clustering", type=float, default=SPATIAL_CLUSTERING
    )
    parser.add_argument("--interior-epochs", type=int, default=INTERIOR_EPOCHS)
    tuned_group = parser.add_mutually_exclusive_group()
    tuned_group.add_argument("--use-tuned", dest="use_tuned", action="store_true")
    tuned_group.add_argument(
        "--no-use-tuned", dest="use_tuned", action="store_false"
    )
    parser.set_defaults(use_tuned=None)
    args = parser.parse_args()
    main(
        args.device, args.n_basis_x, args.n_basis_t, args.degree,
        args.spatial_clustering, args.interior_epochs, args.use_tuned,
    )
