"""Physics-informed Kolmogorov-Arnold network for Burgers' equation."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from kan import KAN
except ImportError:  # Reported when a model is actually constructed.
    KAN = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Burgers.PINN_2D import numerical_reference, numerical_reference_at_points
from training import train


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "pikan_config.json"


def load_configuration(path: Path = CONFIG_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"PIKAN configuration not found: {path}")
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
LOAD_TUNED_IF_AVAILABLE = bool(CONFIG["runtime"]["load_tuned_if_available"])
X_BOUNDS = tuple(float(v) for v in CONFIG["problem"]["x_bounds"])
T_BOUNDS = tuple(float(v) for v in CONFIG["problem"]["time_bounds"])
VISCOSITY = float(CONFIG["problem"]["viscosity"])
N_LAYERS = int(CONFIG["model"]["n_layers"])
N_NEURONS = int(CONFIG["model"]["n_neurons"])
GRID = int(CONFIG["model"]["grid"])
SPLINE_ORDER = int(CONFIG["model"]["spline_order"])
CONSTRAINT_MODE = str(CONFIG["model"]["constraint_mode"])
N_INTERIOR_POINTS = int(CONFIG["sampling"]["interior_points"])
N_BOUNDARY_POINTS_PER_EDGE = int(CONFIG["sampling"]["boundary_points_per_edge"])
N_VALIDATION_POINTS = int(CONFIG["sampling"]["validation_points"])
PLOT_SPATIAL_POINTS = int(CONFIG["sampling"]["plot_spatial_points"])
PLOT_TIME_POINTS = int(CONFIG["sampling"]["plot_time_points"])
PDE_WEIGHT = float(CONFIG["loss_weights"]["pde"])
INITIAL_WEIGHT = float(CONFIG["loss_weights"]["initial"])
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


def initial_condition(x: torch.Tensor) -> torch.Tensor:
    values = -torch.sin(torch.pi * x)
    endpoints = (x == X_BOUNDS[0]) | (x == X_BOUNDS[1])
    return torch.where(endpoints, torch.zeros_like(values), values)


class BurgersPIKAN(torch.nn.Module):
    """PyKAN model with selectable soft or exact initial/boundary constraints."""

    def __init__(
        self,
        n_layers: int,
        n_neurons: int,
        grid: int,
        spline_order: int,
        constraint_mode: str,
        device: torch.device,
        seed: int = SEED,
    ) -> None:
        super().__init__()
        if KAN is None:
            raise ImportError(
                "PIKAN requires the 'pykan' package (import name 'kan'). "
                "Install pykan in the active Python environment first."
            )
        if constraint_mode not in ("soft", "hard"):
            raise ValueError("constraint_mode must be 'soft' or 'hard'")
        if min(n_layers, n_neurons, grid, spline_order) < 1:
            raise ValueError("All PIKAN architecture values must be positive")
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.grid = grid
        self.spline_order = spline_order
        self.constraint_mode = constraint_mode
        width = [2] + [n_neurons] * n_layers + [1]
        self.network = KAN(
            width=width,
            grid=grid,
            k=spline_order,
            seed=seed,
            device=device,
            auto_save=False,
            save_act=False,
            symbolic_enabled=False,
        )
        self.register_buffer("lower_bound", torch.tensor((X_BOUNDS[0], T_BOUNDS[0])))
        self.register_buffer("upper_bound", torch.tensor((X_BOUNDS[1], T_BOUNDS[1])))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        normalized = 2.0 * (points - self.lower_bound) / (
            self.upper_bound - self.lower_bound
        ) - 1.0
        raw = self.network(normalized).squeeze(-1)
        if self.constraint_mode == "soft":
            return raw
        x, time = points[:, 0], points[:, 1]
        time_fraction = (time - T_BOUNDS[0]) / (T_BOUNDS[1] - T_BOUNDS[0])
        boundary_distance = (x - X_BOUNDS[0]) * (X_BOUNDS[1] - x)
        return initial_condition(x) + time_fraction * boundary_distance * raw


def sample_interior(
    count: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    points = torch.rand(count, 2, device=device, generator=generator)
    points[:, 0] = X_BOUNDS[0] + (X_BOUNDS[1] - X_BOUNDS[0]) * points[:, 0]
    points[:, 1] = T_BOUNDS[0] + (T_BOUNDS[1] - T_BOUNDS[0]) * points[:, 1]
    return points


def sample_constraints(count: int, device: torch.device) -> dict[str, torch.Tensor]:
    x = torch.linspace(*X_BOUNDS, count, device=device)
    time = torch.linspace(*T_BOUNDS, count, device=device)
    return {
        "initial": torch.stack((x, torch.full_like(x, T_BOUNDS[0])), dim=1),
        "left": torch.stack((torch.full_like(time, X_BOUNDS[0]), time), dim=1),
        "right": torch.stack((torch.full_like(time, X_BOUNDS[1]), time), dim=1),
    }


def burgers_residual(model: BurgersPIKAN, points: torch.Tensor) -> torch.Tensor:
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
    model: BurgersPIKAN,
    interior: torch.Tensor,
    constraints: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    pde = burgers_residual(model, interior).square().mean()
    initial_points = constraints["initial"]
    initial = (
        model(initial_points) - initial_condition(initial_points[:, 0])
    ).square().mean()
    boundary = 0.5 * (
        model(constraints["left"]).square().mean()
        + model(constraints["right"]).square().mean()
    )
    return {"pde": pde, "initial": initial, "boundary": boundary}


def pikan_loss(
    model: BurgersPIKAN,
    interior: torch.Tensor,
    constraints: dict[str, torch.Tensor],
    pde_weight: float = PDE_WEIGHT,
    initial_weight: float = INITIAL_WEIGHT,
    boundary_weight: float = BOUNDARY_WEIGHT,
) -> torch.Tensor:
    terms = loss_components(model, interior, constraints)
    if model.constraint_mode == "hard":
        return pde_weight * terms["pde"]
    return (
        pde_weight * terms["pde"]
        + initial_weight * terms["initial"]
        + boundary_weight * terms["boundary"]
    )


def relative_l2_error(
    model: BurgersPIKAN, points: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    with torch.no_grad():
        prediction = model(points)
    return torch.linalg.vector_norm(prediction - reference) / torch.linalg.vector_norm(reference)


def constraint_errors(
    model: BurgersPIKAN, constraints: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        initial = constraints["initial"]
        initial_error = (
            model(initial) - initial_condition(initial[:, 0])
        ).abs().max()
        boundary_error = torch.maximum(
            model(constraints["left"]).abs().max(),
            model(constraints["right"]).abs().max(),
        )
    return initial_error, boundary_error


def load_hyperparameters(
    constraint_mode: str, path: Path = RESULT_PATH
) -> tuple[int, int, int, int]:
    if not path.exists():
        raise FileNotFoundError(
            f"No PIKAN tuning result found at {path}. Run Burgers/tune_pikan.py."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if constraint_mode in payload:
        payload = payload[constraint_mode]
    else:
        tuned_mode = payload.get("constraint_mode")
        if tuned_mode is not None and tuned_mode != constraint_mode:
            raise ValueError(
                f"{path.name} was tuned for '{tuned_mode}' constraints, not "
                f"'{constraint_mode}'. Retune or run with --no-use-tuned."
            )
    params = payload.get("best_params", payload)
    return (
        int(params["n_layers"]), int(params["n_neurons"]),
        int(params["grid"]), int(params["spline_order"]),
    )


def save_results(
    model: BurgersPIKAN, history: list[float], relative_error: float
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
    stem = f"pikan_burgers_{model.constraint_mode}"
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
        validation_relative_l2_error=np.asarray(relative_error),
        constraint_mode=np.asarray(model.constraint_mode),
        viscosity=np.asarray(VISCOSITY),
        n_layers=np.asarray(model.n_layers),
        n_neurons=np.asarray(model.n_neurons),
        grid=np.asarray(model.grid),
        spline_order=np.asarray(model.spline_order),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, values, title in zip(
        axes,
        (prediction, reference, error),
        ("PIKAN prediction", "Numerical reference", "Absolute error"),
    ):
        image = axis.imshow(
            values, extent=[*X_BOUNDS, *T_BOUNDS], origin="lower", aspect="auto"
        )
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("t")
        figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(solution_plot, dpi=300, bbox_inches="tight")
    loss = np.asarray(history, dtype=float)
    epochs = np.arange(1, loss.size + 1)
    np.savez_compressed(training_data, epoch=epochs, loss=loss)
    training_figure, axis = plt.subplots(figsize=(6, 4))
    axis.semilogy(epochs, loss)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("PIKAN loss")
    axis.set_title(f"Burgers PIKAN ({model.constraint_mode}) training")
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
    grid: int = GRID,
    spline_order: int = SPLINE_ORDER,
    constraint_mode: str = CONSTRAINT_MODE,
    epochs: int = EPOCHS,
    use_tuned: bool | None = None,
) -> None:
    set_seed()
    device = select_device(device_name)
    should_load = (
        LOAD_TUNED_IF_AVAILABLE and RESULT_PATH.exists()
        if use_tuned is None else use_tuned
    )
    if should_load:
        n_layers, n_neurons, grid, spline_order = load_hyperparameters(
            constraint_mode
        )
        print(f"Loaded tuned hyperparameters from: {RESULT_PATH}")
    print(f"Using device: {device}")
    print(f"Loaded configuration from: {CONFIG_PATH}")
    print(
        f"Burgers PIKAN: layers={n_layers}, neurons={n_neurons}, "
        f"grid={grid}, order={spline_order}, constraints={constraint_mode}"
    )
    model = BurgersPIKAN(
        n_layers, n_neurons, grid, spline_order, constraint_mode, device
    ).to(device)
    generator = torch.Generator(device=device).manual_seed(SEED)
    interior = sample_interior(N_INTERIOR_POINTS, device, generator)
    constraints = sample_constraints(N_BOUNDARY_POINTS_PER_EDGE, device)
    loss_fn = lambda: pikan_loss(model, interior, constraints)
    if OPTIMIZER_NAME.lower() != "adam":
        raise ValueError(f"Unsupported optimizer '{OPTIMIZER_NAME}'; expected 'Adam'")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE, min_lr=MIN_LEARNING_RATE,
    )
    history = train(
        model.parameters(), loss_fn, optimizer, epochs=epochs,
        scheduler=scheduler,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        min_delta=MIN_DELTA,
        print_every=PRINT_EVERY,
    )
    validation = sample_interior(N_VALIDATION_POINTS, device, generator)
    print("Computing validation reference solution")
    reference = numerical_reference_at_points(validation)
    relative_error = relative_l2_error(model, validation, reference)
    residual_mse = burgers_residual(model, validation).detach().square().mean()
    initial_error, boundary_error = constraint_errors(model, constraints)
    print(f"Relative validation L2 error: {relative_error.item():.6e}")
    print(f"Validation PDE residual MSE: {residual_mse.item():.6e}")
    print(f"Maximum initial-condition error: {initial_error.item():.6e}")
    print(f"Maximum boundary error: {boundary_error.item():.6e}")
    save_results(model, history, relative_error.item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve the viscous Burgers equation with a PI-KAN"
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--layers", type=int, default=N_LAYERS)
    parser.add_argument("--neurons", type=int, default=N_NEURONS)
    parser.add_argument("--grid", type=int, default=GRID)
    parser.add_argument("--spline-order", type=int, default=SPLINE_ORDER)
    parser.add_argument(
        "--constraint", choices=("soft", "hard"), default=CONSTRAINT_MODE
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    tuned = parser.add_mutually_exclusive_group()
    tuned.add_argument("--use-tuned", dest="use_tuned", action="store_true")
    tuned.add_argument("--no-use-tuned", dest="use_tuned", action="store_false")
    parser.set_defaults(use_tuned=None)
    args = parser.parse_args()
    main(
        args.device, args.layers, args.neurons, args.grid, args.spline_order,
        args.constraint, args.epochs, args.use_tuned,
    )
