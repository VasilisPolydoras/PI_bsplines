"""Physics-informed neural network for the viscous Burgers equation."""

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
from scipy.integrate import solve_ivp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training import train


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "pinn_config.json"


def load_configuration(path: Path = CONFIG_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"PINN configuration not found: {path}")
    configuration = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "reproducibility", "runtime", "problem", "model", "sampling",
        "loss_weights", "training", "tuning", "output",
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
N_LAYERS = int(CONFIG["model"]["n_layers"])
N_NEURONS = int(CONFIG["model"]["n_neurons"])
ACTIVATION = str(CONFIG["model"]["activation"])
CONSTRAINT_MODE = str(CONFIG["model"]["constraint_mode"])
N_INTERIOR_POINTS = int(CONFIG["sampling"]["interior_points"])
N_BOUNDARY_POINTS_PER_EDGE = int(
    CONFIG["sampling"]["boundary_points_per_edge"]
)
N_INITIAL_POINTS = int(CONFIG["sampling"]["initial_points"])
N_VALIDATION_POINTS = int(CONFIG["sampling"]["validation_points"])
PLOT_SPATIAL_POINTS = int(CONFIG["sampling"]["plot_spatial_points"])
PLOT_TIME_POINTS = int(CONFIG["sampling"]["plot_time_points"])
REFERENCE_SPATIAL_POINTS = int(
    CONFIG["sampling"]["reference_spatial_points"]
)
PDE_WEIGHT = float(CONFIG["loss_weights"]["pde"])
INITIAL_WEIGHT = float(CONFIG["loss_weights"]["initial"])
BOUNDARY_WEIGHT = float(CONFIG["loss_weights"]["boundary"])
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
    if name == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU requested, but CUDA is not available")
        return torch.device("cuda")
    raise ValueError("device must be 'cpu' or 'gpu'")


def initial_condition(x: torch.Tensor) -> torch.Tensor:
    values = -torch.sin(torch.pi * x)
    endpoints = (x == X_BOUNDS[0]) | (x == X_BOUNDS[1])
    return torch.where(endpoints, torch.zeros_like(values), values)


class BurgersPINN(torch.nn.Module):
    """Tanh PINN with selectable soft or exact boundary constraints."""

    def __init__(
        self, n_layers: int, n_neurons: int, constraint_mode: str
    ) -> None:
        super().__init__()
        if n_layers < 1 or n_neurons < 1:
            raise ValueError("n_layers and n_neurons must be positive")
        if constraint_mode not in ("soft", "hard"):
            raise ValueError("constraint_mode must be 'soft' or 'hard'")
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.constraint_mode = constraint_mode
        self.register_buffer(
            "lower_bound", torch.tensor((X_BOUNDS[0], T_BOUNDS[0]))
        )
        self.register_buffer(
            "upper_bound", torch.tensor((X_BOUNDS[1], T_BOUNDS[1]))
        )
        modules: list[torch.nn.Module] = []
        input_size = 2
        for _ in range(n_layers):
            linear = torch.nn.Linear(input_size, n_neurons)
            torch.nn.init.xavier_normal_(linear.weight)
            torch.nn.init.zeros_(linear.bias)
            modules.extend((linear, torch.nn.Tanh()))
            input_size = n_neurons
        output = torch.nn.Linear(input_size, 1)
        torch.nn.init.xavier_normal_(output.weight)
        torch.nn.init.zeros_(output.bias)
        modules.append(output)
        self.network = torch.nn.Sequential(*modules)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        normalized = 2.0 * (points - self.lower_bound) / (
            self.upper_bound - self.lower_bound
        ) - 1.0
        raw = self.network(normalized).squeeze(-1)
        if self.constraint_mode == "soft":
            return raw
        x, time = points[:, 0], points[:, 1]
        normalized_time = (time - T_BOUNDS[0]) / (
            T_BOUNDS[1] - T_BOUNDS[0]
        )
        boundary_distance = (x - X_BOUNDS[0]) * (X_BOUNDS[1] - x)
        return initial_condition(x) + normalized_time * boundary_distance * raw


def sample_interior(
    count: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    points = torch.rand(count, 2, device=device, generator=generator)
    points[:, 0] = X_BOUNDS[0] + (X_BOUNDS[1] - X_BOUNDS[0]) * points[:, 0]
    points[:, 1] = T_BOUNDS[0] + (T_BOUNDS[1] - T_BOUNDS[0]) * points[:, 1]
    return points


def sample_constraints(
    points_per_edge: int, initial_points: int, device: torch.device
) -> dict[str, torch.Tensor]:
    x = torch.linspace(*X_BOUNDS, initial_points, device=device)
    time = torch.linspace(*T_BOUNDS, points_per_edge, device=device)
    return {
        "initial": torch.stack((x, torch.full_like(x, T_BOUNDS[0])), dim=1),
        "left": torch.stack((torch.full_like(time, X_BOUNDS[0]), time), dim=1),
        "right": torch.stack((torch.full_like(time, X_BOUNDS[1]), time), dim=1),
    }


def burgers_residual(
    model: BurgersPINN, points: torch.Tensor
) -> torch.Tensor:
    coordinates = points.detach().clone().requires_grad_(True)
    solution = model(coordinates)
    gradient = torch.autograd.grad(
        solution, coordinates, torch.ones_like(solution), create_graph=True
    )[0]
    u_x, u_t = gradient[:, 0], gradient[:, 1]
    u_xx = torch.autograd.grad(
        u_x, coordinates, torch.ones_like(u_x), create_graph=True
    )[0][:, 0]
    return u_t + solution * u_x - VISCOSITY * u_xx


def loss_components(
    model: BurgersPINN,
    interior_points: torch.Tensor,
    constraint_points: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    pde = burgers_residual(model, interior_points).square().mean()
    initial_points = constraint_points["initial"]
    initial = (
        model(initial_points) - initial_condition(initial_points[:, 0])
    ).square().mean()
    boundary = 0.5 * (
        model(constraint_points["left"]).square().mean()
        + model(constraint_points["right"]).square().mean()
    )
    return {"pde": pde, "initial": initial, "boundary": boundary}


def pinn_loss(
    model: BurgersPINN,
    interior_points: torch.Tensor,
    constraint_points: dict[str, torch.Tensor],
    *,
    pde_weight: float = PDE_WEIGHT,
    initial_weight: float = INITIAL_WEIGHT,
    boundary_weight: float = BOUNDARY_WEIGHT,
) -> torch.Tensor:
    if model.constraint_mode == "hard":
        return pde_weight * burgers_residual(
            model, interior_points
        ).square().mean()
    terms = loss_components(model, interior_points, constraint_points)
    return (
        pde_weight * terms["pde"]
        + initial_weight * terms["initial"]
        + boundary_weight * terms["boundary"]
    )


def relative_l2_error(
    model: BurgersPINN,
    points: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        prediction = model(points)
        return torch.sqrt(
            (prediction - reference).square().sum()
            / reference.square().sum()
        )


def constraint_errors(
    model: BurgersPINN, points: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        initial = points["initial"]
        initial_error = (
            model(initial) - initial_condition(initial[:, 0])
        ).abs().max()
        boundary_error = torch.maximum(
            model(points["left"]).abs().max(),
            model(points["right"]).abs().max(),
        )
    return initial_error, boundary_error


def _reference_system(spatial_points: int, viscosity: float):
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

    return reference_x, initial, right_hand_side


def numerical_reference(
    x: np.ndarray,
    times: np.ndarray,
    viscosity: float = VISCOSITY,
    spatial_points: int = REFERENCE_SPATIAL_POINTS,
) -> np.ndarray:
    reference_x, initial, right_hand_side = _reference_system(
        spatial_points, viscosity
    )
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
    points: torch.Tensor,
    viscosity: float = VISCOSITY,
    spatial_points: int = REFERENCE_SPATIAL_POINTS,
) -> torch.Tensor:
    coordinates = points.detach().cpu().numpy()
    query_x, query_times = coordinates[:, 0], coordinates[:, 1]
    unique_times, time_indices = np.unique(query_times, return_inverse=True)
    reference_x, initial, right_hand_side = _reference_system(
        spatial_points, viscosity
    )
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
            raise RuntimeError(f"Reference solver failed: {solution.message}")
        time_solutions = solution.y
    values = np.asarray([
        np.interp(x_value, reference_x, time_solutions[:, time_index])
        for x_value, time_index in zip(query_x, time_indices)
    ])
    return torch.as_tensor(values, device=points.device, dtype=points.dtype)


def load_hyperparameters(path: Path = RESULT_PATH) -> tuple[int, int]:
    if not path.exists():
        raise FileNotFoundError(
            f"No PINN tuning result found at {path}. Run Burgers/tune_pinn.py."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("best_params", payload)
    return int(params["n_layers"]), int(params["n_neurons"])


def save_results(
    model: BurgersPINN,
    history: list[float],
    relative_error: float,
    training_wall_time: float,
) -> None:
    device = next(model.parameters()).device
    x = np.linspace(*X_BOUNDS, PLOT_SPATIAL_POINTS)
    times = np.linspace(*T_BOUNDS, PLOT_TIME_POINTS)
    grid_x, grid_t = np.meshgrid(x, times)
    points = torch.tensor(
        np.column_stack((grid_x.ravel(), grid_t.ravel())),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        prediction = model(points).cpu().numpy().reshape(grid_x.shape)
    reference = numerical_reference(x, times)
    error = np.abs(prediction - reference)
    stem = f"pinn_burgers_{model.constraint_mode}"
    POSTPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    solution_data = POSTPROCESSING_DIR / f"{stem}_solution_data.npz"
    solution_plot = POSTPROCESSING_DIR / f"{stem}_solution.png"
    training_data = POSTPROCESSING_DIR / f"{stem}_training_data.npz"
    training_plot = POSTPROCESSING_DIR / f"{stem}_training.png"
    np.savez_compressed(
        solution_data,
        x=x,
        time=times,
        prediction=prediction,
        reference=reference,
        absolute_error=error,
        validation_relative_l2_error=np.asarray(relative_error, dtype=float),
        constraint_mode=np.asarray(model.constraint_mode),
        viscosity=np.asarray(VISCOSITY, dtype=float),
        n_layers=np.asarray(model.n_layers, dtype=int),
        n_neurons=np.asarray(model.n_neurons, dtype=int),
        training_wall_time_seconds=np.asarray(training_wall_time, dtype=float),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, values, title in zip(
        axes,
        (prediction, reference, error),
        ("PINN prediction", "Numerical reference", "Absolute error"),
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
        figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(solution_plot, dpi=300, bbox_inches="tight")

    loss = np.asarray(history, dtype=float)
    epochs = np.arange(1, loss.size + 1)
    np.savez_compressed(
        training_data,
        epoch=epochs,
        loss=loss,
        training_wall_time_seconds=np.asarray(training_wall_time, dtype=float),
    )
    training_figure, axis = plt.subplots(figsize=(6, 4))
    axis.semilogy(epochs, loss)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("PINN loss")
    axis.set_title(f"Burgers PINN ({model.constraint_mode}) training")
    training_figure.tight_layout()
    training_figure.savefig(training_plot, dpi=300, bbox_inches="tight")
    print(f"Saved solution data to: {solution_data}")
    print(f"Saved solution plot to: {solution_plot}")
    print(f"Saved training data to: {training_data}")
    print(f"Saved training plot to: {training_plot}")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(figure)
        plt.close(training_figure)


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
    should_load = (
        LOAD_TUNED_IF_AVAILABLE and RESULT_PATH.exists()
        if use_tuned is None
        else use_tuned
    )
    if should_load:
        n_layers, n_neurons = load_hyperparameters()
        print(f"Loaded tuned hyperparameters from: {RESULT_PATH}")
    if ACTIVATION.lower() != "tanh":
        raise ValueError(f"Unsupported activation '{ACTIVATION}'; expected 'tanh'")
    print(f"Using device: {device}")
    print(f"Loaded configuration from: {CONFIG_PATH}")
    print(
        f"Burgers PINN: layers={n_layers}, neurons={n_neurons}, "
        f"constraints={constraint_mode}"
    )
    model = BurgersPINN(n_layers, n_neurons, constraint_mode).to(device)
    generator = torch.Generator(device=device).manual_seed(sampling_seed)
    interior = sample_interior(N_INTERIOR_POINTS, device, generator)
    constraints = sample_constraints(
        N_BOUNDARY_POINTS_PER_EDGE, N_INITIAL_POINTS, device
    )
    loss_fn = lambda: pinn_loss(model, interior, constraints)
    if OPTIMIZER_NAME.lower() != "adam":
        raise ValueError(f"Unsupported optimizer '{OPTIMIZER_NAME}'; expected 'Adam'")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=MIN_LEARNING_RATE,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_start = time.perf_counter()
    history = train(
        model.parameters(),
        loss_fn,
        optimizer,
        epochs=epochs,
        scheduler=scheduler,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        min_delta=MIN_DELTA,
        print_every=PRINT_EVERY,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_wall_time = time.perf_counter() - training_start
    print(f"Training wall time: {training_wall_time:.3f} s")
    validation_generator = (
        generator
        if validation_seed is None
        else torch.Generator(device=device).manual_seed(validation_seed)
    )
    validation = sample_interior(
        N_VALIDATION_POINTS, device, validation_generator
    )
    print("Computing validation reference solution")
    reference = numerical_reference_at_points(validation)
    relative_error = relative_l2_error(model, validation, reference)
    residual_mse = burgers_residual(model, validation).detach().square().mean()
    initial_error, boundary_error = constraint_errors(model, constraints)
    print(f"Relative validation L2 error: {relative_error.item():.6e}")
    print(f"Validation PDE residual MSE: {residual_mse.item():.6e}")
    print(f"Maximum initial-condition error: {initial_error.item():.6e}")
    print(f"Maximum boundary error: {boundary_error.item():.6e}")
    if save_outputs:
        save_results(model, history, relative_error.item(), training_wall_time)
    return {
        "validation_relative_l2_error": relative_error.item(),
        "training_time_seconds": training_wall_time,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve the viscous Burgers equation with a PINN"
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--layers", type=int, default=N_LAYERS)
    parser.add_argument("--neurons", type=int, default=N_NEURONS)
    parser.add_argument(
        "--constraint", choices=("soft", "hard"), default=CONSTRAINT_MODE
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
        arguments.layers,
        arguments.neurons,
        arguments.constraint,
        arguments.epochs,
        arguments.use_tuned,
    )
