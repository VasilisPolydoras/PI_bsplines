"""Physics-informed neural network for the one-dimensional diffusion equation."""

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
CONFIG_PATH = SCRIPT_DIR / "pinn_config.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
SEED = int(CONFIG["reproducibility"]["seed"])
DEFAULT_DEVICE = str(CONFIG["runtime"]["device"])
LOAD_TUNED_IF_AVAILABLE = bool(CONFIG["runtime"]["load_tuned_if_available"])
X_BOUNDS = tuple(float(v) for v in CONFIG["problem"]["x_bounds"])
T_BOUNDS = tuple(float(v) for v in CONFIG["problem"]["time_bounds"])
DIFFUSIVITY = float(CONFIG["problem"]["diffusivity"])
N_LAYERS = int(CONFIG["model"]["n_layers"])
N_NEURONS = int(CONFIG["model"]["n_neurons"])
ACTIVATION = str(CONFIG["model"]["activation"])
CONSTRAINT_MODE = str(CONFIG["model"]["constraint_mode"])
N_INTERIOR_POINTS = int(CONFIG["sampling"]["interior_points"])
N_BOUNDARY_POINTS_PER_EDGE = int(CONFIG["sampling"]["boundary_points_per_edge"])
N_INITIAL_POINTS = int(CONFIG["sampling"]["initial_points"])
N_VALIDATION_POINTS = int(CONFIG["sampling"]["validation_points"])
PDE_WEIGHT = float(CONFIG["loss_weights"]["pde"])
BOUNDARY_WEIGHT = float(CONFIG["loss_weights"]["boundary"])
INITIAL_WEIGHT = float(CONFIG["loss_weights"]["initial"])
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


def exact_solution(points, diffusivity=DIFFUSIVITY):
    x, time_values = points[:, 0], points[:, 1]
    return torch.sin(2.0 * torch.pi * x) * torch.exp(
        -4.0 * torch.pi**2 * diffusivity * time_values
    )


def sample_interior(count, device, x_bounds=X_BOUNDS, time_bounds=T_BOUNDS, generator=None):
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
        "initial": torch.stack((x, torch.full_like(x, time_bounds[0])), dim=1),
        "left": torch.stack((torch.full_like(time_values, x_bounds[0]), time_values), dim=1),
        "right": torch.stack((torch.full_like(time_values, x_bounds[1]), time_values), dim=1),
    }


def diffusion_residual(model, points, diffusivity=DIFFUSIVITY):
    coordinates = points.detach().clone().requires_grad_(True)
    solution = model(coordinates)
    gradient = torch.autograd.grad(
        solution, coordinates, torch.ones_like(solution), create_graph=True
    )[0]
    u_x, u_t = gradient[:, 0], gradient[:, 1]
    u_xx = torch.autograd.grad(
        u_x, coordinates, torch.ones_like(u_x), create_graph=True
    )[0][:, 0]
    return u_t - diffusivity * u_xx


def loss_components(model, interior, constraints):
    initial = constraints["initial"]
    return {
        "pde": diffusion_residual(model, interior).square().mean(),
        "initial": (model(initial) - exact_solution(initial)).square().mean(),
        "boundary": 0.5 * (
            model(constraints["left"]).square().mean()
            + model(constraints["right"]).square().mean()
        ),
    }


def physics_loss(model, interior, constraints):
    if model.constraint_mode == "hard":
        return PDE_WEIGHT * diffusion_residual(model, interior).square().mean()
    terms = loss_components(model, interior, constraints)
    return (
        PDE_WEIGHT * terms["pde"] + INITIAL_WEIGHT * terms["initial"]
        + BOUNDARY_WEIGHT * terms["boundary"]
    )


def relative_l2_error(model, points, diffusivity=DIFFUSIVITY):
    with torch.no_grad():
        prediction, reference = model(points), exact_solution(points, diffusivity)
    return torch.linalg.vector_norm(prediction - reference) / torch.linalg.vector_norm(reference)


def constraint_errors(model, constraints, diffusivity=DIFFUSIVITY):
    with torch.no_grad():
        initial = constraints["initial"]
        initial_error = (model(initial) - exact_solution(initial, diffusivity)).abs().max()
        boundary_error = torch.maximum(
            model(constraints["left"]).abs().max(),
            model(constraints["right"]).abs().max(),
        )
    return initial_error, boundary_error


class DiffusionPINN(torch.nn.Module):
    """Tanh MLP with selectable soft or hard constraints."""

    def __init__(self, n_layers: int, n_neurons: int, constraint_mode: str):
        super().__init__()
        if constraint_mode not in ("soft", "hard"):
            raise ValueError("constraint_mode must be 'soft' or 'hard'")
        self.n_layers, self.n_neurons = n_layers, n_neurons
        self.constraint_mode = constraint_mode
        self.register_buffer("lower", torch.tensor((X_BOUNDS[0], T_BOUNDS[0])))
        self.register_buffer("upper", torch.tensor((X_BOUNDS[1], T_BOUNDS[1])))
        modules: list[torch.nn.Module] = []
        width = 2
        for _ in range(n_layers):
            linear = torch.nn.Linear(width, n_neurons)
            torch.nn.init.xavier_normal_(linear.weight)
            torch.nn.init.zeros_(linear.bias)
            modules.extend((linear, torch.nn.Tanh()))
            width = n_neurons
        output = torch.nn.Linear(width, 1)
        torch.nn.init.xavier_normal_(output.weight)
        torch.nn.init.zeros_(output.bias)
        modules.append(output)
        self.network = torch.nn.Sequential(*modules)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        normalized = 2.0 * (points - self.lower) / (self.upper - self.lower) - 1.0
        raw = self.network(normalized).squeeze(-1)
        if self.constraint_mode == "soft":
            return raw
        x, time = points[:, 0], points[:, 1]
        initial = torch.sin(2.0 * torch.pi * x)
        normalized_time = (time - T_BOUNDS[0]) / (T_BOUNDS[1] - T_BOUNDS[0])
        distance = (x - X_BOUNDS[0]) * (X_BOUNDS[1] - x)
        return initial + normalized_time * distance * raw


def pinn_loss(model, interior, constraints) -> torch.Tensor:
    return physics_loss(model, interior, constraints)


def load_hyperparameters(path: Path = RESULT_PATH) -> tuple[int, int]:
    if not path.exists():
        raise FileNotFoundError(f"No PINN tuning result found at {path}")
    params = json.loads(path.read_text(encoding="utf-8")).get("best_params", {})
    return int(params["n_layers"]), int(params["n_neurons"])


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
    stem = f"pinn_diffusion_{model.constraint_mode}"
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
        ("PINN prediction", "Exact solution", "Absolute error"),
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
    device_name: str = DEFAULT_DEVICE,
    n_layers: int = N_LAYERS,
    n_neurons: int = N_NEURONS,
    constraint_mode: str = CONSTRAINT_MODE,
    epochs: int = EPOCHS,
    use_tuned: bool | None = None,
    initialization_seed: int = SEED,
    sampling_seed: int = SEED,
    validation_seed: int | None = None,
    save_outputs: bool = True,
    verbose: bool = True,
) -> dict[str, float]:
    set_seed(initialization_seed)
    device = select_device(device_name)
    should_load = LOAD_TUNED_IF_AVAILABLE and RESULT_PATH.exists() if use_tuned is None else use_tuned
    if should_load:
        n_layers, n_neurons = load_hyperparameters()
        print(f"Loaded tuned hyperparameters from: {RESULT_PATH}")
    if ACTIVATION.lower() != "tanh":
        raise ValueError(f"Unsupported activation '{ACTIVATION}'")
    model = DiffusionPINN(n_layers, n_neurons, constraint_mode).to(device)
    generator = torch.Generator(device=device).manual_seed(sampling_seed)
    interior = sample_interior(N_INTERIOR_POINTS, device, X_BOUNDS, T_BOUNDS, generator)
    constraints = sample_constraints(
        N_BOUNDARY_POINTS_PER_EDGE, N_INITIAL_POINTS, device, X_BOUNDS, T_BOUNDS
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
        model.parameters(), lambda: pinn_loss(model, interior, constraints), optimizer,
        epochs=epochs, scheduler=scheduler,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        min_delta=MIN_DELTA, print_every=PRINT_EVERY,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_time = time.perf_counter() - start
    validation_generator = generator if validation_seed is None else torch.Generator(device=device).manual_seed(validation_seed)
    validation = sample_interior(N_VALIDATION_POINTS, device, X_BOUNDS, T_BOUNDS, validation_generator)
    error = float(relative_l2_error(model, validation, DIFFUSIVITY))
    residual = float(diffusion_residual(model, validation, DIFFUSIVITY).detach().square().mean())
    initial_error, boundary_error = constraint_errors(model, constraints, DIFFUSIVITY)
    print(f"Using device: {device}")
    print(f"Diffusion PINN: layers={n_layers}, neurons={n_neurons}, constraints={constraint_mode}")
    print(f"Training wall time: {wall_time:.3f} s")
    print(f"Relative validation L2 error: {error:.6e}")
    print(f"Validation PDE residual MSE: {residual:.6e}")
    print(f"Maximum initial-condition error: {initial_error.item():.6e}")
    print(f"Maximum boundary error: {boundary_error.item():.6e}")
    if save_outputs:
        save_results(
            model, history, validation, error, wall_time,
            {"n_layers": n_layers, "n_neurons": n_neurons},
        )
    return {"validation_relative_l2_error": error, "training_time_seconds": wall_time}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve diffusion with a PINN")
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--layers", type=int, default=N_LAYERS)
    parser.add_argument("--neurons", type=int, default=N_NEURONS)
    parser.add_argument("--constraint", choices=("soft", "hard"), default=CONSTRAINT_MODE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    tuned = parser.add_mutually_exclusive_group()
    tuned.add_argument("--use-tuned", dest="use_tuned", action="store_true")
    tuned.add_argument("--no-use-tuned", dest="use_tuned", action="store_false")
    parser.set_defaults(use_tuned=None)
    arguments = parser.parse_args()
    main(arguments.device, arguments.layers, arguments.neurons, arguments.constraint, arguments.epochs, arguments.use_tuned)
