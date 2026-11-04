"""Hard- or soft-constrained PINN for two-dimensional Kovasznay flow."""

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


def load_configuration(path: Path = CONFIG_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"PINN configuration not found: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "reproducibility", "runtime", "problem", "model", "sampling",
        "loss_weights", "training", "tuning", "output",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing sections: {', '.join(sorted(missing))}")
    return config


CONFIG = load_configuration()
SEED = int(CONFIG["reproducibility"]["seed"])
DEFAULT_DEVICE = str(CONFIG["runtime"]["device"])
LOAD_TUNED_IF_AVAILABLE = bool(CONFIG["runtime"]["load_tuned_if_available"])
REYNOLDS_NUMBER = float(CONFIG["problem"]["reynolds_number"])
X_BOUNDS = tuple(float(v) for v in CONFIG["problem"]["x_bounds"])
Y_BOUNDS = tuple(float(v) for v in CONFIG["problem"]["y_bounds"])
N_LAYERS = int(CONFIG["model"]["n_layers"])
N_NEURONS = int(CONFIG["model"]["n_neurons"])
ACTIVATION = str(CONFIG["model"]["activation"])
CONSTRAINT_MODE = str(CONFIG["model"]["constraint_mode"])
N_INTERIOR_POINTS = int(CONFIG["sampling"]["interior_points"])
N_BOUNDARY_POINTS = int(CONFIG["sampling"]["boundary_points_per_edge"])
N_VALIDATION_POINTS = int(CONFIG["sampling"]["validation_points"])
PLOT_X_POINTS = int(CONFIG["sampling"]["plot_x_points"])
PLOT_Y_POINTS = int(CONFIG["sampling"]["plot_y_points"])
PDE_WEIGHT = float(CONFIG["loss_weights"]["pde"])
BOUNDARY_WEIGHT = float(CONFIG["loss_weights"]["boundary"])
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
POSTPROCESSING_DIR = SCRIPT_DIR / CONFIG["output"]["postprocessing_directory"]
SHOW_PLOTS = bool(CONFIG["output"]["show_plots"])


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


def exact_solution(
    points: torch.Tensor, reynolds_number: float = REYNOLDS_NUMBER
) -> torch.Tensor:
    """Return the analytic velocity components and pressure, [u,v,p]."""
    x, y = points[:, 0], points[:, 1]
    reynolds = torch.as_tensor(
        reynolds_number, device=points.device, dtype=points.dtype
    )
    lam = reynolds / 2 - torch.sqrt(reynolds.square() / 4 + 4 * torch.pi**2)
    exponential = torch.exp(lam * x)
    u = 1 - exponential * torch.cos(2 * torch.pi * y)
    v = lam / (2 * torch.pi) * exponential * torch.sin(2 * torch.pi * y)
    pressure = 0.5 * (1 - torch.exp(2 * lam * x))
    return torch.stack((u, v, pressure), dim=1)


def normalized_coordinates(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    s = (points[:, 0] - X_BOUNDS[0]) / (X_BOUNDS[1] - X_BOUNDS[0])
    t = (points[:, 1] - Y_BOUNDS[0]) / (Y_BOUNDS[1] - Y_BOUNDS[0])
    return s, t


def coons_boundary_lifting(points: torch.Tensor) -> torch.Tensor:
    """Extend the four prescribed boundary traces into the rectangle."""
    s, t = normalized_coordinates(points)
    x = points[:, 0]
    y = points[:, 1]
    left = exact_solution(torch.stack((torch.full_like(y, X_BOUNDS[0]), y), 1))
    right = exact_solution(torch.stack((torch.full_like(y, X_BOUNDS[1]), y), 1))
    bottom = exact_solution(torch.stack((x, torch.full_like(x, Y_BOUNDS[0])), 1))
    top = exact_solution(torch.stack((x, torch.full_like(x, Y_BOUNDS[1])), 1))
    corner_points = torch.tensor(
        [
            [X_BOUNDS[0], Y_BOUNDS[0]], [X_BOUNDS[0], Y_BOUNDS[1]],
            [X_BOUNDS[1], Y_BOUNDS[0]], [X_BOUNDS[1], Y_BOUNDS[1]],
        ],
        device=points.device,
        dtype=points.dtype,
    )
    c00, c01, c10, c11 = exact_solution(corner_points)
    s_col, t_col = s[:, None], t[:, None]
    corners = (
        (1 - s_col) * (1 - t_col) * c00
        + (1 - s_col) * t_col * c01
        + s_col * (1 - t_col) * c10
        + s_col * t_col * c11
    )
    return (
        (1 - s_col) * left + s_col * right
        + (1 - t_col) * bottom + t_col * top - corners
    )


class KovasznayPINN(torch.nn.Module):
    """Tanh network returning [u,v,p] with optional exact boundary lifting."""

    def __init__(self, n_layers: int, n_neurons: int, constraint_mode: str) -> None:
        super().__init__()
        if constraint_mode not in ("soft", "hard"):
            raise ValueError("constraint_mode must be 'soft' or 'hard'")
        if min(n_layers, n_neurons) < 1:
            raise ValueError("n_layers and n_neurons must be positive")
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.constraint_mode = constraint_mode
        modules: list[torch.nn.Module] = []
        input_size = 2
        for _ in range(n_layers):
            layer = torch.nn.Linear(input_size, n_neurons)
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            modules.extend((layer, torch.nn.Tanh()))
            input_size = n_neurons
        output = torch.nn.Linear(input_size, 3)
        torch.nn.init.xavier_normal_(output.weight)
        torch.nn.init.zeros_(output.bias)
        modules.append(output)
        self.network = torch.nn.Sequential(*modules)
        self.register_buffer("lower", torch.tensor((X_BOUNDS[0], Y_BOUNDS[0])))
        self.register_buffer("upper", torch.tensor((X_BOUNDS[1], Y_BOUNDS[1])))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        normalized = 2 * (points - self.lower) / (self.upper - self.lower) - 1
        raw = self.network(normalized)
        if self.constraint_mode == "soft":
            return raw
        s, t = normalized_coordinates(points)
        distance = (s * (1 - s) * t * (1 - t))[:, None]
        return coons_boundary_lifting(points) + distance * raw


def sample_interior(
    count: int, device: torch.device, generator: torch.Generator | None = None
) -> torch.Tensor:
    points = torch.rand(count, 2, device=device, generator=generator)
    points[:, 0] = X_BOUNDS[0] + (X_BOUNDS[1] - X_BOUNDS[0]) * points[:, 0]
    points[:, 1] = Y_BOUNDS[0] + (Y_BOUNDS[1] - Y_BOUNDS[0]) * points[:, 1]
    return points


def sample_boundary(points_per_edge: int, device: torch.device) -> torch.Tensor:
    x = torch.linspace(*X_BOUNDS, points_per_edge, device=device)
    y = torch.linspace(*Y_BOUNDS, points_per_edge, device=device)
    return torch.cat(
        (
            torch.stack((x, torch.full_like(x, Y_BOUNDS[0])), 1),
            torch.stack((x, torch.full_like(x, Y_BOUNDS[1])), 1),
            torch.stack((torch.full_like(y, X_BOUNDS[0]), y), 1),
            torch.stack((torch.full_like(y, X_BOUNDS[1]), y), 1),
        )
    )


def kovasznay_residual(model: torch.nn.Module, points: torch.Tensor) -> torch.Tensor:
    coordinates = points.detach().clone().requires_grad_(True)
    fields = model(coordinates)
    u, v, pressure = fields[:, 0], fields[:, 1], fields[:, 2]

    def gradient(field: torch.Tensor) -> torch.Tensor:
        return torch.autograd.grad(
            field, coordinates, torch.ones_like(field), create_graph=True
        )[0]

    grad_u, grad_v, grad_p = gradient(u), gradient(v), gradient(pressure)
    u_x, u_y = grad_u[:, 0], grad_u[:, 1]
    v_x, v_y = grad_v[:, 0], grad_v[:, 1]
    p_x, p_y = grad_p[:, 0], grad_p[:, 1]
    u_xx = gradient(u_x)[:, 0]
    u_yy = gradient(u_y)[:, 1]
    v_xx = gradient(v_x)[:, 0]
    v_yy = gradient(v_y)[:, 1]
    viscosity = 1.0 / REYNOLDS_NUMBER
    continuity = u_x + v_y
    momentum_x = u * u_x + v * u_y + p_x - viscosity * (u_xx + u_yy)
    momentum_y = u * v_x + v * v_y + p_y - viscosity * (v_xx + v_yy)
    return torch.stack((continuity, momentum_x, momentum_y), dim=1)


def loss_components(
    model: torch.nn.Module, interior: torch.Tensor, boundary: torch.Tensor
) -> dict[str, torch.Tensor]:
    residual = kovasznay_residual(model, interior)
    boundary_error = model(boundary) - exact_solution(boundary)
    return {
        "continuity": residual[:, 0].square().mean(),
        "momentum_x": residual[:, 1].square().mean(),
        "momentum_y": residual[:, 2].square().mean(),
        "boundary": boundary_error.square().mean(),
    }


def physics_loss(
    model: torch.nn.Module,
    interior: torch.Tensor,
    boundary: torch.Tensor,
    pde_weight: float = PDE_WEIGHT,
    boundary_weight: float = BOUNDARY_WEIGHT,
) -> torch.Tensor:
    terms = loss_components(model, interior, boundary)
    pde = terms["continuity"] + terms["momentum_x"] + terms["momentum_y"]
    if model.constraint_mode == "hard":
        return pde_weight * pde
    return pde_weight * pde + boundary_weight * terms["boundary"]


def field_relative_l2_errors(
    model: torch.nn.Module, points: torch.Tensor
) -> torch.Tensor:
    with torch.no_grad():
        prediction = model(points)
        reference = exact_solution(points)
    numerator = torch.linalg.vector_norm(prediction - reference, dim=0)
    denominator = torch.linalg.vector_norm(reference, dim=0)
    return numerator / denominator


def load_hyperparameters(
    constraint_mode: str, path: Path = RESULT_PATH
) -> tuple[int, int]:
    if not path.exists():
        raise FileNotFoundError(f"No PINN tuning result found at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload = payload.get(constraint_mode, payload)
    params = payload.get("best_params", payload)
    return int(params["n_layers"]), int(params["n_neurons"])


def tuned_result_available(constraint_mode: str, path: Path = RESULT_PATH) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    entry = payload.get(constraint_mode, payload)
    return bool(entry.get("best_params")) and entry.get("status") != "not_tuned"


def save_results(
    model: KovasznayPINN, history: list[float], errors: np.ndarray,
    validation: torch.Tensor, wall_time: float,
) -> None:
    device = next(model.parameters()).device
    x = np.linspace(*X_BOUNDS, PLOT_X_POINTS)
    y = np.linspace(*Y_BOUNDS, PLOT_Y_POINTS)
    grid_x, grid_y = np.meshgrid(x, y)
    points = torch.tensor(
        np.column_stack((grid_x.ravel(), grid_y.ravel())),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        prediction = model(points).cpu().numpy().reshape(*grid_x.shape, 3)
        reference = exact_solution(points).cpu().numpy().reshape(*grid_x.shape, 3)
    absolute_error = np.abs(prediction - reference)
    stem = f"pinn_kovasznay_{model.constraint_mode}"
    POSTPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    solution_data = POSTPROCESSING_DIR / f"{stem}_solution_data.npz"
    solution_plot = POSTPROCESSING_DIR / f"{stem}_solution.png"
    training_data = POSTPROCESSING_DIR / f"{stem}_training_data.npz"
    training_plot = POSTPROCESSING_DIR / f"{stem}_training.png"
    np.savez_compressed(
        solution_data, x=x, y=y, prediction=prediction, reference=reference,
        absolute_error=absolute_error, validation_relative_l2_errors=errors,
        field_names=np.asarray(("u", "v", "p")),
        constraint_mode=np.asarray(model.constraint_mode),
        reynolds_number=np.asarray(REYNOLDS_NUMBER),
        n_layers=np.asarray(model.n_layers), n_neurons=np.asarray(model.n_neurons),
        validation_relative_l2_error=np.asarray(errors.mean()),
        validation_points=validation.detach().cpu().numpy(),
        validation_prediction=model(validation).detach().cpu().numpy(),
        validation_reference=exact_solution(validation).detach().cpu().numpy(),
        training_wall_time_seconds=np.asarray(wall_time),
    )
    figure, axes = plt.subplots(3, 3, figsize=(14, 12))
    for row, name in enumerate(("u", "v", "p")):
        for column, (values, title) in enumerate(zip(
            (prediction[:, :, row], reference[:, :, row], absolute_error[:, :, row]),
            (f"Predicted {name}", f"Exact {name}", f"Error {name}"),
        )):
            image = axes[row, column].imshow(
                values, extent=[*X_BOUNDS, *Y_BOUNDS], origin="lower", aspect="auto"
            )
            axes[row, column].set_title(title)
            axes[row, column].set_xlabel("x")
            axes[row, column].set_ylabel("y")
            figure.colorbar(image, ax=axes[row, column])
    figure.tight_layout()
    figure.savefig(solution_plot, dpi=300, bbox_inches="tight")
    loss = np.asarray(history)
    epochs = np.arange(1, loss.size + 1)
    np.savez_compressed(
        training_data, epoch=epochs, loss=loss,
        training_wall_time_seconds=np.asarray(wall_time),
    )
    training_figure, axis = plt.subplots(figsize=(6, 4))
    axis.semilogy(epochs, loss)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("PINN loss")
    axis.set_title(f"Kovasznay PINN ({model.constraint_mode}) training")
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
        LOAD_TUNED_IF_AVAILABLE and tuned_result_available(constraint_mode)
        if use_tuned is None else use_tuned
    )
    if should_load:
        n_layers, n_neurons = load_hyperparameters(constraint_mode)
        print(f"Loaded tuned hyperparameters from: {RESULT_PATH}")
    if ACTIVATION.lower() != "tanh":
        raise ValueError(f"Unsupported activation: {ACTIVATION}")
    print(f"Using device: {device}")
    print(
        f"Kovasznay PINN: layers={n_layers}, neurons={n_neurons}, "
        f"constraints={constraint_mode}"
    )
    model = KovasznayPINN(n_layers, n_neurons, constraint_mode).to(device)
    generator = torch.Generator(device=device).manual_seed(sampling_seed)
    interior = sample_interior(N_INTERIOR_POINTS, device, generator)
    boundary = sample_boundary(N_BOUNDARY_POINTS, device)
    loss_fn = lambda: physics_loss(model, interior, boundary)
    if OPTIMIZER_NAME.lower() != "adam":
        raise ValueError("Only Adam is supported")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE, min_lr=MIN_LEARNING_RATE,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    history = train(
        model.parameters(), loss_fn, optimizer, epochs=epochs,
        scheduler=scheduler, early_stopping_patience=EARLY_STOPPING_PATIENCE,
        min_delta=MIN_DELTA, print_every=PRINT_EVERY,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_time = time.perf_counter() - start
    validation_generator = generator if validation_seed is None else torch.Generator(device=device).manual_seed(validation_seed)
    validation = sample_interior(N_VALIDATION_POINTS, device, validation_generator)
    errors = field_relative_l2_errors(model, validation)
    residual_mse = kovasznay_residual(model, validation).square().mean(dim=0)
    boundary_max = (model(boundary) - exact_solution(boundary)).abs().max()
    for name, error in zip(("u", "v", "p"), errors):
        print(f"Relative validation L2 error {name}: {error.item():.6e}")
    print(
        "Validation residual MSE [continuity, x-momentum, y-momentum]: "
        f"{residual_mse.detach().cpu().numpy()}"
    )
    print(f"Maximum boundary error: {boundary_max.item():.6e}")
    print(f"Training wall time: {wall_time:.3f} s")
    result = {
        "validation_relative_l2_error": float(errors.mean()),
        "training_time_seconds": wall_time,
    }
    if save_outputs:
        save_results(model, history, errors.detach().cpu().numpy(), validation, wall_time)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve Kovasznay flow with a PINN")
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--layers", type=int, default=N_LAYERS)
    parser.add_argument("--neurons", type=int, default=N_NEURONS)
    parser.add_argument("--constraint", choices=("soft", "hard"), default=CONSTRAINT_MODE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    tuned = parser.add_mutually_exclusive_group()
    tuned.add_argument("--use-tuned", dest="use_tuned", action="store_true")
    tuned.add_argument("--no-use-tuned", dest="use_tuned", action="store_false")
    parser.set_defaults(use_tuned=None)
    args = parser.parse_args()
    main(args.device, args.layers, args.neurons, args.constraint, args.epochs, args.use_tuned)
