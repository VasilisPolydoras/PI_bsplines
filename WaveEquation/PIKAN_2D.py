"""Physics-informed Kolmogorov-Arnold network for the wave equation."""

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

try:
    from kan import KAN
except ImportError:
    KAN = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training import train


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG = json.loads((SCRIPT_DIR / "pikan_config.json").read_text(encoding="utf-8"))
SEED = int(CONFIG["reproducibility"]["seed"])
DEFAULT_DEVICE = str(CONFIG["runtime"]["device"])
LOAD_TUNED_IF_AVAILABLE = bool(CONFIG["runtime"]["load_tuned_if_available"])
X_BOUNDS = tuple(float(value) for value in CONFIG["problem"]["x_bounds"])
T_BOUNDS = tuple(float(value) for value in CONFIG["problem"]["time_bounds"])
WAVE_SPEED = float(CONFIG["problem"]["wave_speed"])
N_LAYERS = int(CONFIG["model"]["n_layers"])
N_NEURONS = int(CONFIG["model"]["n_neurons"])
GRID = int(CONFIG["model"]["grid"])
SPLINE_ORDER = int(CONFIG["model"]["spline_order"])
CONSTRAINT_MODE = str(CONFIG["model"]["constraint_mode"])
N_INTERIOR_POINTS = int(CONFIG["sampling"]["interior_points"])
N_BOUNDARY_POINTS_PER_EDGE = int(CONFIG["sampling"]["boundary_points_per_edge"])
N_INITIAL_POINTS = int(CONFIG["sampling"]["initial_points"])
N_VALIDATION_POINTS = int(CONFIG["sampling"]["validation_points"])
OPTIMIZER_NAME = str(CONFIG["training"]["optimizer"])
EPOCHS = int(CONFIG["training"]["epochs"])
LEARNING_RATE = float(CONFIG["training"]["learning_rate"])
SCHEDULER_FACTOR = float(CONFIG["training"]["scheduler"]["factor"])
SCHEDULER_PATIENCE = int(CONFIG["training"]["scheduler"]["patience"])
MIN_LEARNING_RATE = float(CONFIG["training"]["scheduler"]["min_learning_rate"])
EARLY_STOPPING_PATIENCE = int(CONFIG["training"]["early_stopping"]["patience"])
MIN_DELTA = float(CONFIG["training"]["early_stopping"]["min_delta"])
PRINT_EVERY = int(CONFIG["training"]["print_every"])
RESULT_PATH = SCRIPT_DIR / CONFIG["output"]["tuned_hyperparameters"]


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


def exact_solution(points, wave_speed=WAVE_SPEED):
    x, time_values = points[:, 0], points[:, 1]
    return torch.sin(torch.pi * x) * torch.cos(
        torch.pi * wave_speed * time_values
    )


def sample_interior(
    count, device, x_bounds=X_BOUNDS, time_bounds=T_BOUNDS, generator=None,
):
    points = torch.rand(count, 2, device=device, generator=generator)
    points[:, 0] = x_bounds[0] + (x_bounds[1] - x_bounds[0]) * points[:, 0]
    points[:, 1] = time_bounds[0] + (time_bounds[1] - time_bounds[0]) * points[:, 1]
    return points


def sample_constraints(
    boundary_points_per_edge, initial_points, device,
    x_bounds=X_BOUNDS, time_bounds=T_BOUNDS,
):
    x = torch.linspace(*x_bounds, initial_points, device=device)
    time_values = torch.linspace(*time_bounds, boundary_points_per_edge, device=device)
    return {
        "initial": torch.stack(
            (x, torch.full_like(x, time_bounds[0])), dim=1
        ),
        "left": torch.stack(
            (torch.full_like(time_values, x_bounds[0]), time_values), dim=1
        ),
        "right": torch.stack(
            (torch.full_like(time_values, x_bounds[1]), time_values), dim=1
        ),
    }


def wave_residual(model, points, wave_speed=WAVE_SPEED):
    coordinates = points.detach().clone().requires_grad_(True)
    solution = model(coordinates)
    gradient = torch.autograd.grad(
        solution, coordinates, torch.ones_like(solution), create_graph=True
    )[0]
    u_x, u_t = gradient[:, 0], gradient[:, 1]
    u_xx = torch.autograd.grad(
        u_x, coordinates, torch.ones_like(u_x), create_graph=True
    )[0][:, 0]
    u_tt = torch.autograd.grad(
        u_t, coordinates, torch.ones_like(u_t), create_graph=True
    )[0][:, 1]
    return u_tt - wave_speed**2 * u_xx


def initial_state(model, points, create_graph=True):
    coordinates = points.detach().clone().requires_grad_(True)
    displacement = model(coordinates)
    velocity = torch.autograd.grad(
        displacement, coordinates, torch.ones_like(displacement),
        create_graph=create_graph,
    )[0][:, 1]
    return displacement, velocity


def physics_loss(model, interior, constraints):
    weights = CONFIG["loss_weights"]
    pde = wave_residual(model, interior).square().mean()
    if model.constraint_mode == "hard":
        return float(weights["pde"]) * pde
    displacement, velocity = initial_state(model, constraints["initial"])
    displacement_loss = (
        displacement - exact_solution(constraints["initial"])
    ).square().mean()
    boundary_loss = 0.5 * (
        model(constraints["left"]).square().mean()
        + model(constraints["right"]).square().mean()
    )
    return (
        float(weights["pde"]) * pde
        + float(weights["initial_displacement"]) * displacement_loss
        + float(weights["initial_velocity"]) * velocity.square().mean()
        + float(weights["boundary"]) * boundary_loss
    )


def relative_l2_error(model, points, wave_speed=WAVE_SPEED):
    with torch.no_grad():
        prediction = model(points)
        reference = exact_solution(points, wave_speed)
    return torch.linalg.vector_norm(prediction - reference) / torch.linalg.vector_norm(
        reference
    )


def constraint_errors(model, constraints):
    with torch.no_grad():
        displacement_error = (
            model(constraints["initial"])
            - exact_solution(constraints["initial"])
        ).abs().max()
        boundary_error = torch.maximum(
            model(constraints["left"]).abs().max(),
            model(constraints["right"]).abs().max(),
        )
    _, velocity = initial_state(model, constraints["initial"], create_graph=False)
    return displacement_error, velocity.detach().abs().max(), boundary_error


class WavePIKAN(torch.nn.Module):
    """KAN model with selectable soft or hard wave constraints."""

    def __init__(
        self, n_layers, n_neurons, grid, spline_order,
        constraint_mode, device, seed,
    ):
        super().__init__()
        if KAN is None:
            raise ImportError("PIKAN requires pykan (import name 'kan').")
        if constraint_mode not in ("soft", "hard"):
            raise ValueError("constraint_mode must be 'soft' or 'hard'")
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.grid = grid
        self.spline_order = spline_order
        self.constraint_mode = constraint_mode
        self.network = KAN(
            width=[2] + [n_neurons] * n_layers + [1],
            grid=grid, k=spline_order, seed=seed, device=device,
            auto_save=False, save_act=False, symbolic_enabled=False,
        )

    def forward(self, points):
        x, time_values = points[:, 0], points[:, 1]
        normalized = torch.stack(
            (
                2.0 * (x - X_BOUNDS[0]) / (X_BOUNDS[1] - X_BOUNDS[0]) - 1.0,
                2.0 * (time_values - T_BOUNDS[0])
                / (T_BOUNDS[1] - T_BOUNDS[0]) - 1.0,
            ),
            dim=1,
        )
        raw = self.network(normalized).squeeze(-1)
        if self.constraint_mode == "soft":
            return raw
        initial = torch.sin(torch.pi * x)
        normalized_time = (
            (time_values - T_BOUNDS[0]) / (T_BOUNDS[1] - T_BOUNDS[0])
        )
        distance = (x - X_BOUNDS[0]) * (X_BOUNDS[1] - x)
        return initial + normalized_time.square() * distance * raw


def pikan_loss(model, interior, constraints):
    return physics_loss(model, interior, constraints)


def load_hyperparameters(path=RESULT_PATH):
    if not path.exists():
        raise FileNotFoundError(f"No PIKAN tuning result found at {path}")
    params = json.loads(path.read_text(encoding="utf-8")).get("best_params", {})
    keys = ("n_layers", "n_neurons", "grid", "spline_order")
    return tuple(int(params[key]) for key in keys)


def save_results(model, history, validation, error, wall_time, metadata):
    sampling = CONFIG["sampling"]
    x = np.linspace(*X_BOUNDS, int(sampling["plot_spatial_points"]))
    times = np.linspace(*T_BOUNDS, int(sampling["plot_time_points"]))
    grid_x, grid_t = np.meshgrid(x, times)
    device = next(model.parameters()).device
    points = torch.tensor(
        np.column_stack((grid_x.ravel(), grid_t.ravel())),
        dtype=torch.float32, device=device,
    )
    with torch.no_grad():
        prediction = model(points).cpu().numpy().reshape(grid_x.shape)
        reference = exact_solution(points).cpu().numpy().reshape(grid_x.shape)
        validation_prediction = model(validation).cpu().numpy()
        validation_reference = exact_solution(validation).cpu().numpy()
    absolute_error = np.abs(prediction - reference)
    grid_error = np.linalg.norm(prediction - reference) / np.linalg.norm(reference)
    output = SCRIPT_DIR / CONFIG["output"]["postprocessing_directory"]
    output.mkdir(parents=True, exist_ok=True)
    stem = f"pikan_wave_{model.constraint_mode}"
    solution_path = output / f"{stem}_solution_data.npz"
    training_path = output / f"{stem}_training_data.npz"
    np.savez_compressed(
        solution_path, x=x, time=times, prediction=prediction,
        reference=reference, absolute_error=absolute_error,
        validation_relative_l2_error=error,
        plot_grid_relative_l2_error=grid_error,
        validation_points=validation.cpu().numpy(),
        validation_prediction=validation_prediction,
        validation_reference=validation_reference,
        training_wall_time_seconds=wall_time,
        constraint_mode=np.asarray(model.constraint_mode), **metadata,
    )
    losses = np.asarray(history)
    np.savez_compressed(
        training_path, epoch=np.arange(1, losses.size + 1), loss=losses,
        training_wall_time_seconds=wall_time,
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, values, title in zip(
        axes, (prediction, reference, absolute_error),
        ("PIKAN prediction", "Exact solution", "Absolute error"),
    ):
        image = axis.imshow(
            values, extent=[*X_BOUNDS, *T_BOUNDS], origin="lower", aspect="auto"
        )
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("t")
        figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output / f"{stem}_solution.png", dpi=300)
    training_figure, axis = plt.subplots(figsize=(6, 4))
    axis.semilogy(np.arange(1, losses.size + 1), losses)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    training_figure.tight_layout()
    training_figure.savefig(output / f"{stem}_training.png", dpi=300)
    if CONFIG["output"]["show_plots"]:
        plt.show()
    else:
        plt.close("all")
    print(f"Saved solution data to: {solution_path}")
    print(f"Saved training data to: {training_path}")


def main(
    device_name=DEFAULT_DEVICE,
    n_layers=N_LAYERS,
    n_neurons=N_NEURONS,
    grid=GRID,
    spline_order=SPLINE_ORDER,
    constraint_mode=CONSTRAINT_MODE,
    epochs=EPOCHS,
    use_tuned=None,
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
        n_layers, n_neurons, grid, spline_order = load_hyperparameters()
        if verbose:
            print(f"Loaded tuned hyperparameters from: {RESULT_PATH}")
    model = WavePIKAN(
        n_layers, n_neurons, grid, spline_order,
        constraint_mode, device, initialization_seed,
    ).to(device)
    generator = torch.Generator(device=device).manual_seed(sampling_seed)
    interior = sample_interior(
        N_INTERIOR_POINTS, device, X_BOUNDS, T_BOUNDS, generator
    )
    constraints = sample_constraints(
        N_BOUNDARY_POINTS_PER_EDGE, N_INITIAL_POINTS, device
    )
    if OPTIMIZER_NAME.lower() != "adam":
        raise ValueError(f"Unsupported optimizer '{OPTIMIZER_NAME}'")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE,
        min_lr=MIN_LEARNING_RATE,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    history = train(
        model.parameters(), lambda: pikan_loss(model, interior, constraints),
        optimizer, epochs=epochs, scheduler=scheduler,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        min_delta=MIN_DELTA, print_every=PRINT_EVERY if verbose else 0,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_time = time.perf_counter() - start
    validation_generator = (
        generator if validation_seed is None
        else torch.Generator(device=device).manual_seed(validation_seed)
    )
    validation = sample_interior(
        N_VALIDATION_POINTS, device, X_BOUNDS, T_BOUNDS, validation_generator
    )
    error = float(relative_l2_error(model, validation))
    residual = float(wave_residual(model, validation).detach().square().mean())
    displacement_error, velocity_error, boundary_error = constraint_errors(
        model, constraints
    )
    if verbose:
        print(f"Using device: {device}")
        print(
            f"Wave PIKAN: layers={n_layers}, neurons={n_neurons}, grid={grid}, "
            f"order={spline_order}, constraints={constraint_mode}"
        )
        print(f"Training wall time: {wall_time:.3f} s")
        print(f"Relative validation L2 error: {error:.6e}")
        print(f"Validation PDE residual MSE: {residual:.6e}")
        print(f"Maximum initial-displacement error: {displacement_error.item():.6e}")
        print(f"Maximum initial-velocity error: {velocity_error.item():.6e}")
        print(f"Maximum boundary error: {boundary_error.item():.6e}")
    if save_outputs:
        save_results(
            model, history, validation, error, wall_time,
            {
                "n_layers": n_layers, "n_neurons": n_neurons,
                "grid": grid, "spline_order": spline_order,
            },
        )
    return {
        "validation_relative_l2_error": error,
        "training_time_seconds": wall_time,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve the wave equation with a PIKAN")
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--layers", type=int, default=N_LAYERS)
    parser.add_argument("--neurons", type=int, default=N_NEURONS)
    parser.add_argument("--grid", type=int, default=GRID)
    parser.add_argument("--spline-order", type=int, default=SPLINE_ORDER)
    parser.add_argument("--constraint", choices=("soft", "hard"), default=CONSTRAINT_MODE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    tuned = parser.add_mutually_exclusive_group()
    tuned.add_argument("--use-tuned", dest="use_tuned", action="store_true")
    tuned.add_argument("--no-use-tuned", dest="use_tuned", action="store_false")
    parser.set_defaults(use_tuned=None)
    arguments = parser.parse_args()
    main(
        arguments.device, arguments.layers, arguments.neurons,
        arguments.grid, arguments.spline_order, arguments.constraint,
        arguments.epochs, arguments.use_tuned,
    )
