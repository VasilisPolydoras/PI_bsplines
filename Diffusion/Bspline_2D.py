"""Physics-informed tensor-product B-spline solver for diffusion."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training import train


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "bspline_config.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
SEED = int(CONFIG["reproducibility"]["seed"])
DEFAULT_DEVICE = str(CONFIG["runtime"]["device"])
LOAD_TUNED_IF_AVAILABLE = bool(CONFIG["runtime"]["load_tuned_if_available"])
X_BOUNDS = tuple(float(v) for v in CONFIG["problem"]["x_bounds"])
T_BOUNDS = tuple(float(v) for v in CONFIG["problem"]["time_bounds"])
DIFFUSIVITY = float(CONFIG["problem"]["diffusivity"])
N_BASIS = int(CONFIG["model"]["n_basis"])
DEGREE = int(CONFIG["model"]["degree"])
INTERIOR_INITIALIZATION = str(
    CONFIG["model"].get("interior_initialization", "random")
)
N_INTERIOR_POINTS = int(CONFIG["sampling"]["interior_points"])
N_INITIAL_POINTS = int(CONFIG["sampling"]["initial_points"])
N_VALIDATION_POINTS = int(CONFIG["sampling"]["validation_points"])
PLOT_SPATIAL_POINTS = int(CONFIG["sampling"]["plot_spatial_points"])
PLOT_TIME_POINTS = int(CONFIG["sampling"]["plot_time_points"])
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
POSTPROCESSING_DIR = SCRIPT_DIR / CONFIG["output"]["postprocessing_directory"]
SHOW_PLOTS = bool(CONFIG["output"]["show_plots"])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "gpu":
        raise RuntimeError("GPU requested, but CUDA is not available")
    raise ValueError("device must be 'cpu' or 'gpu'")


def open_uniform_knots(bounds, n_basis, degree, device):
    if degree < 2 or n_basis <= degree:
        raise ValueError("Require degree >= 2 and n_basis > degree")
    count = n_basis - degree - 1
    internal = (
        torch.linspace(*bounds, count + 2, device=device)[1:-1]
        if count else torch.empty(0, device=device)
    )
    return torch.cat(
        (torch.full((degree + 1,), bounds[0], device=device), internal,
         torch.full((degree + 1,), bounds[1], device=device))
    )


def basis_matrix(values, knots, degree):
    values = values.reshape(-1)
    basis = (
        (values[:, None] >= knots[:-1]) & (values[:, None] < knots[1:])
    ).to(values.dtype)
    at_right = values == knots[-1]
    if at_right.any():
        basis[at_right] = 0.0
        basis[at_right, -1] = 1.0
    for order in range(1, degree + 1):
        updated = torch.zeros(
            values.numel(), knots.numel() - order - 1,
            device=values.device, dtype=values.dtype,
        )
        for index in range(updated.shape[1]):
            left = knots[index + order] - knots[index]
            right = knots[index + order + 1] - knots[index + 1]
            if left > 0:
                updated[:, index] += (
                    (values - knots[index]) / left * basis[:, index]
                )
            if right > 0:
                updated[:, index] += (
                    (knots[index + order + 1] - values)
                    / right * basis[:, index + 1]
                )
        basis = updated
    if at_right.any():
        basis[at_right] = 0.0
        basis[at_right, -1] = 1.0
    return basis


def basis_derivative_matrix(values, knots, degree, derivative_order):
    n_basis = knots.numel() - degree - 1
    if derivative_order == 0:
        return basis_matrix(values, knots, degree)
    if derivative_order > degree:
        return torch.zeros(values.numel(), n_basis, device=values.device)
    evaluation = values.reshape(-1)
    at_right = evaluation == knots[-1]
    if at_right.any():
        evaluation = evaluation.clone()
        evaluation[at_right] = torch.nextafter(knots[-1], knots[0])
    lower = basis_derivative_matrix(
        evaluation, knots, degree - 1, derivative_order - 1
    )
    result = torch.zeros(evaluation.numel(), n_basis, device=values.device)
    for index in range(n_basis):
        left = knots[index + degree] - knots[index]
        right = knots[index + degree + 1] - knots[index + 1]
        if left > 0:
            result[:, index] += degree / left * lower[:, index]
        if right > 0:
            result[:, index] -= degree / right * lower[:, index + 1]
    return result


class DiffusionBSpline:
    """Tensor-product field with exact zero boundaries and a fitted IC trace."""

    constraint_mode = "hard"

    def __init__(
        self, n_basis, degree, device,
        interior_initialization=INTERIOR_INITIALIZATION,
    ):
        self.n_basis = n_basis
        self.degree = degree
        self.interior_initialization = interior_initialization.lower()
        if self.interior_initialization not in {"random", "coons"}:
            raise ValueError("interior_initialization must be 'random' or 'coons'")
        self.knots_x = open_uniform_knots(
            X_BOUNDS, n_basis, degree, device
        )
        self.knots_t = open_uniform_knots(
            T_BOUNDS, n_basis, degree, device
        )
        self.initial_interior_coeffs = torch.zeros(
            n_basis - 2, device=device, requires_grad=True
        )
        self.fixed_coeffs = None
        self.interior_coeffs = None

    def coefficient_tensor(self):
        if self.interior_coeffs is None:
            coefficients = torch.zeros(
                self.n_basis, self.n_basis,
                device=self.initial_interior_coeffs.device,
                dtype=self.initial_interior_coeffs.dtype,
            )
            coefficients[1:-1, 0] = self.initial_interior_coeffs
            return coefficients
        coefficients = self.fixed_coeffs.clone()
        coefficients[1:-1, 1:] = self.interior_coeffs
        return coefficients

    def trained_initial_coefficients(self):
        fixed = torch.zeros(
            self.n_basis, self.n_basis,
            device=self.initial_interior_coeffs.device,
            dtype=self.initial_interior_coeffs.dtype,
        )
        fixed[1:-1, 0] = self.initial_interior_coeffs
        return fixed.detach()

    def set_initial_coefficients(self, fixed_coefficients):
        self.fixed_coeffs = fixed_coefficients.detach().clone()
        initialized = self._initialize_full_coefficients(self.fixed_coeffs)
        self.interior_coeffs = (
            initialized[1:-1, 1:].detach().clone().requires_grad_(True)
        )

    def _initialize_full_coefficients(self, fixed):
        if self.interior_initialization == "random":
            initialized = fixed.clone()
            free = torch.empty(
                self.n_basis - 2, self.n_basis - 1,
                device=fixed.device, dtype=fixed.dtype,
            )
            torch.nn.init.xavier_normal_(free)
            initialized[1:-1, 1:] = free
            return initialized
        return self._coons_initialization(fixed)

    def _coons_initialization(self, fixed):
        """Extend the pretrained initial/side traces into the coefficient grid."""
        s = torch.linspace(
            0.0, 1.0, self.n_basis, device=fixed.device, dtype=fixed.dtype
        )
        t = torch.linspace(
            0.0, 1.0, self.n_basis, device=fixed.device, dtype=fixed.dtype
        )
        sx, ty = s[:, None], t[None, :]
        left = fixed[0, :][None, :]
        right = fixed[-1, :][None, :]
        bottom = fixed[:, 0][:, None]
        top = ((1.0 - s) * fixed[0, -1] + s * fixed[-1, -1])[:, None]
        corners = (
            (1.0 - sx) * (1.0 - ty) * fixed[0, 0]
            + (1.0 - sx) * ty * fixed[0, -1]
            + sx * (1.0 - ty) * fixed[-1, 0]
            + sx * ty * fixed[-1, -1]
        )
        initialized = (
            (1.0 - sx) * left + sx * right
            + (1.0 - ty) * bottom + ty * top - corners
        )
        initialized[0, :] = fixed[0, :]
        initialized[-1, :] = fixed[-1, :]
        initialized[:, 0] = fixed[:, 0]
        return initialized

    def operators(self, points, max_x=2, max_t=1):
        return {
            "x": tuple(
                basis_derivative_matrix(
                    points[:, 0], self.knots_x, self.degree, order
                ) for order in range(max_x + 1)
            ),
            "t": tuple(
                basis_derivative_matrix(
                    points[:, 1], self.knots_t, self.degree, order
                ) for order in range(max_t + 1)
            ),
        }

    def evaluate(self, operators, dx=0, dt=0):
        return torch.einsum(
            "ni,ij,nj->n", operators["x"][dx],
            self.coefficient_tensor(), operators["t"][dt],
        )

    def forward(self, points):
        return self.evaluate(self.operators(points, 0, 0))

    __call__ = forward


def exact_solution(points, diffusivity=DIFFUSIVITY):
    x, time_values = points[:, 0], points[:, 1]
    return torch.sin(2.0 * torch.pi * x) * torch.exp(
        -4.0 * torch.pi**2 * diffusivity * time_values
    )


def sample_interior(count, device, generator=None):
    points = torch.rand(count, 2, device=device, generator=generator)
    points[:, 0] = X_BOUNDS[0] + (X_BOUNDS[1] - X_BOUNDS[0]) * points[:, 0]
    points[:, 1] = T_BOUNDS[0] + (T_BOUNDS[1] - T_BOUNDS[0]) * points[:, 1]
    return points


def sample_initial(count, device):
    x = torch.linspace(*X_BOUNDS, count, device=device)
    initial = torch.stack((x, torch.full_like(x, T_BOUNDS[0])), dim=1)
    targets = torch.sin(2.0 * torch.pi * x)
    return initial, targets


def relative_l2_error(model, points, diffusivity=DIFFUSIVITY):
    with torch.no_grad():
        prediction = model(points)
        reference = exact_solution(points, diffusivity)
    return torch.linalg.vector_norm(prediction - reference) / torch.linalg.vector_norm(
        reference
    )


def initial_condition_loss(model, points, targets):
    return (model(points) - targets).square().mean()


def precompute_initial_operator(model, points):
    with torch.no_grad():
        return basis_matrix(points[:, 0], model.knots_x, model.degree)


def pretrain_initial_coefficients(
    model, points, targets, epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    scheduler_factor=SCHEDULER_FACTOR,
    scheduler_patience=SCHEDULER_PATIENCE,
    minimum_learning_rate=MIN_LEARNING_RATE,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    min_delta=MIN_DELTA,
    print_every=PRINT_EVERY,
):
    bx = precompute_initial_operator(model, points)
    optimizer = torch.optim.Adam(
        [model.initial_interior_coeffs], lr=learning_rate
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=scheduler_factor, patience=scheduler_patience,
        min_lr=minimum_learning_rate,
    )
    history = train(
        [model.initial_interior_coeffs],
        lambda: (
            bx[:, 1:-1] @ model.initial_interior_coeffs - targets
        ).square().mean(),
        optimizer, epochs=epochs, scheduler=scheduler,
        early_stopping_patience=early_stopping_patience,
        min_delta=min_delta, print_every=print_every,
    )
    model.set_initial_coefficients(model.trained_initial_coefficients())
    return history


def diffusion_loss(model, operators, diffusivity=DIFFUSIVITY):
    residual = (
        model.evaluate(operators, 0, 1)
        - diffusivity * model.evaluate(operators, 2, 0)
    )
    return residual.square().mean()


def load_hyperparameters(path=RESULT_PATH):
    if not path.exists():
        raise FileNotFoundError(f"No B-spline tuning result found at {path}")
    params = json.loads(path.read_text(encoding="utf-8"))["best_params"]
    return int(params["n_basis"]), int(params["degree"])


def main(
    device_name=DEFAULT_DEVICE,
    n_basis=N_BASIS,
    degree=DEGREE,
    epochs=EPOCHS,
    use_tuned=None,
    interior_initialization=INTERIOR_INITIALIZATION,
    initialization_seed=SEED,
    sampling_seed=SEED,
    validation_seed=None,
    save_outputs=True,
    verbose=True,
):
    set_seed(initialization_seed)
    device = select_device(device_name)
    should_load = (
        LOAD_TUNED_IF_AVAILABLE and RESULT_PATH.exists()
        if use_tuned is None else use_tuned
    )
    if should_load:
        n_basis, degree = load_hyperparameters()
        print(f"Loaded tuned hyperparameters from: {RESULT_PATH}")
    print(f"Using device: {device}")
    print(
        f"Diffusion B-spline: n_basis={n_basis} in x and t, degree={degree}, "
        f"initialization={interior_initialization}"
    )
    generator = torch.Generator(device=device).manual_seed(sampling_seed)
    interior = sample_interior(N_INTERIOR_POINTS, device, generator)
    initial_points, initial_targets = sample_initial(N_INITIAL_POINTS, device)
    model = DiffusionBSpline(
        n_basis, degree, device, interior_initialization
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    print("Pretraining only the inhomogeneous initial-condition coefficients")
    initial_history = pretrain_initial_coefficients(
        model, initial_points, initial_targets, epochs
    )
    frozen_initial_loss = float(
        initial_condition_loss(model, initial_points, initial_targets).detach()
    )
    print(f"Frozen initial-condition loss: {frozen_initial_loss:.6e}")
    operators = model.operators(interior)
    if OPTIMIZER_NAME.lower() != "adam":
        raise ValueError(f"Unsupported optimizer '{OPTIMIZER_NAME}'")
    optimizer = torch.optim.Adam([model.interior_coeffs], lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE,
        min_lr=MIN_LEARNING_RATE,
    )
    print("Optimizing free interior coefficients with the PDE residual")
    history = train(
        [model.interior_coeffs], lambda: diffusion_loss(model, operators),
        optimizer, epochs=epochs, scheduler=scheduler,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        min_delta=MIN_DELTA, print_every=PRINT_EVERY,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_time = time.perf_counter() - start
    validation_generator = (
        generator if validation_seed is None
        else torch.Generator(device=device).manual_seed(validation_seed)
    )
    validation = sample_interior(
        N_VALIDATION_POINTS, device, validation_generator
    )
    validation_error = float(relative_l2_error(model, validation))
    validation_residual = float(
        diffusion_loss(model, model.operators(validation)).detach()
    )
    print(f"Total training wall time: {wall_time:.3f} s")
    print(f"Relative validation L2 error: {validation_error:.6e}")
    print(f"Validation PDE residual MSE: {validation_residual:.6e}")
    result = {
        "validation_relative_l2_error": validation_error,
        "training_time_seconds": wall_time,
    }
    if not save_outputs:
        return result

    x = np.linspace(*X_BOUNDS, PLOT_SPATIAL_POINTS)
    times = np.linspace(*T_BOUNDS, PLOT_TIME_POINTS)
    grid_x, grid_t = np.meshgrid(x, times)
    plot_points = torch.tensor(
        np.column_stack((grid_x.ravel(), grid_t.ravel())),
        dtype=torch.float32, device=device,
    )
    with torch.no_grad():
        prediction = model(plot_points).cpu().numpy().reshape(grid_x.shape)
        reference = exact_solution(plot_points).cpu().numpy().reshape(grid_x.shape)
        validation_prediction = model(validation).cpu().numpy()
        validation_reference = exact_solution(validation).cpu().numpy()
    absolute_error = np.abs(prediction - reference)
    grid_error = np.linalg.norm(prediction - reference) / np.linalg.norm(reference)
    POSTPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    stem = "bspline_diffusion_hard"
    solution_path = POSTPROCESSING_DIR / f"{stem}_solution_data.npz"
    training_path = POSTPROCESSING_DIR / f"{stem}_training_data.npz"
    np.savez_compressed(
        solution_path, x=x, time=times, prediction=prediction,
        reference=reference, absolute_error=absolute_error,
        validation_relative_l2_error=validation_error,
        plot_grid_relative_l2_error=grid_error,
        validation_points=validation.cpu().numpy(),
        validation_prediction=validation_prediction,
        validation_reference=validation_reference,
        training_wall_time_seconds=wall_time,
        n_basis=n_basis, degree=degree,
        interior_initialization=np.asarray(interior_initialization),
        frozen_initial_condition_loss=frozen_initial_loss,
        boundary_constraint=np.asarray("exact_homogeneous_dirichlet"),
    )
    initial_values = np.asarray(initial_history)
    interior_values = np.asarray(history)
    np.savez_compressed(
        training_path,
        initial_epoch=np.arange(1, initial_values.size + 1),
        initial_loss=initial_values,
        interior_epoch=np.arange(1, interior_values.size + 1),
        interior_loss=interior_values,
        epoch=np.arange(1, interior_values.size + 1),
        loss=interior_values,
        training_wall_time_seconds=wall_time,
        interior_initialization=np.asarray(interior_initialization),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, values, title in zip(
        axes, (prediction, reference, absolute_error),
        ("B-spline prediction", "Exact solution", "Absolute error"),
    ):
        image = axis.imshow(
            values, extent=[*X_BOUNDS, *T_BOUNDS], origin="lower", aspect="auto"
        )
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("t")
        figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(POSTPROCESSING_DIR / f"{stem}_solution.png", dpi=300)
    training_figure, axis = plt.subplots(figsize=(6, 4))
    axis.semilogy(
        np.arange(1, initial_values.size + 1), initial_values,
        label="Initial-condition pretraining",
    )
    axis.semilogy(
        np.arange(1, interior_values.size + 1), interior_values,
        label="Interior PDE training",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.legend()
    training_figure.tight_layout()
    training_figure.savefig(POSTPROCESSING_DIR / f"{stem}_training.png", dpi=300)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")
    print(f"Saved solution data to: {solution_path}")
    print(f"Saved training data to: {training_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve diffusion with B-splines")
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--n-basis", type=int, default=N_BASIS)
    parser.add_argument("--degree", type=int, default=DEGREE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--interior-initialization", choices=("random", "coons"),
        default=INTERIOR_INITIALIZATION,
    )
    tuned = parser.add_mutually_exclusive_group()
    tuned.add_argument("--use-tuned", dest="use_tuned", action="store_true")
    tuned.add_argument("--no-use-tuned", dest="use_tuned", action="store_false")
    parser.set_defaults(use_tuned=None)
    arguments = parser.parse_args()
    main(
        arguments.device, arguments.n_basis, arguments.degree,
        arguments.epochs, arguments.use_tuned,
        arguments.interior_initialization,
    )
