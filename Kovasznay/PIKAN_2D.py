"""Hard- or soft-constrained PI-KAN for two-dimensional Kovasznay flow."""

from __future__ import annotations

import argparse
import json
import sys
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

from Kovasznay.PINN_2D import (
    REYNOLDS_NUMBER, X_BOUNDS, Y_BOUNDS, coons_boundary_lifting,
    exact_solution, field_relative_l2_errors, kovasznay_residual,
    loss_components, normalized_coordinates, physics_loss, sample_boundary,
    sample_interior, select_device, set_seed,
)
from training import train


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "pikan_config.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
SEED = int(CONFIG["reproducibility"]["seed"])
DEFAULT_DEVICE = str(CONFIG["runtime"]["device"])
LOAD_TUNED_IF_AVAILABLE = bool(CONFIG["runtime"]["load_tuned_if_available"])
N_LAYERS = int(CONFIG["model"]["n_layers"])
N_NEURONS = int(CONFIG["model"]["n_neurons"])
GRID = int(CONFIG["model"]["grid"])
SPLINE_ORDER = int(CONFIG["model"]["spline_order"])
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


def validate_problem_configuration() -> None:
    problem = CONFIG["problem"]
    expected = (
        float(problem["reynolds_number"]), tuple(problem["x_bounds"]),
        tuple(problem["y_bounds"]),
    )
    actual = (REYNOLDS_NUMBER, X_BOUNDS, Y_BOUNDS)
    if expected != actual:
        raise ValueError(
            "pikan_config.json and pinn_config.json must describe the same "
            "Kovasznay problem because they share the benchmark utilities"
        )


class KovasznayPIKAN(torch.nn.Module):
    """PyKAN returning [u,v,p] with optional exact Coons boundary lifting."""

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
            raise ValueError("All PI-KAN architecture values must be positive")
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.grid = grid
        self.spline_order = spline_order
        self.constraint_mode = constraint_mode
        self.network = KAN(
            width=[2] + [n_neurons] * n_layers + [3],
            grid=grid,
            k=spline_order,
            seed=seed,
            device=device,
            auto_save=False,
            save_act=False,
            symbolic_enabled=False,
        )
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


def load_hyperparameters(
    constraint_mode: str, path: Path = RESULT_PATH
) -> tuple[int, int, int, int]:
    if not path.exists():
        raise FileNotFoundError(f"No PI-KAN tuning result found at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload = payload.get(constraint_mode, payload)
    params = payload.get("best_params", payload)
    return (
        int(params["n_layers"]), int(params["n_neurons"]),
        int(params["grid"]), int(params["spline_order"]),
    )


def save_results(
    model: KovasznayPIKAN, history: list[float], errors: np.ndarray
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
    stem = f"pikan_kovasznay_{model.constraint_mode}"
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
        grid=np.asarray(model.grid), spline_order=np.asarray(model.spline_order),
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
    np.savez_compressed(training_data, epoch=epochs, loss=loss)
    training_figure, axis = plt.subplots(figsize=(6, 4))
    axis.semilogy(epochs, loss)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("PIKAN loss")
    axis.set_title(f"Kovasznay PIKAN ({model.constraint_mode}) training")
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
    validate_problem_configuration()
    set_seed(SEED)
    device = select_device(device_name)
    should_load = (
        LOAD_TUNED_IF_AVAILABLE and RESULT_PATH.exists()
        if use_tuned is None else use_tuned
    )
    if should_load:
        n_layers, n_neurons, grid, spline_order = load_hyperparameters(constraint_mode)
        print(f"Loaded tuned hyperparameters from: {RESULT_PATH}")
    print(f"Using device: {device}")
    print(
        f"Kovasznay PIKAN: layers={n_layers}, neurons={n_neurons}, grid={grid}, "
        f"order={spline_order}, constraints={constraint_mode}"
    )
    model = KovasznayPIKAN(
        n_layers, n_neurons, grid, spline_order, constraint_mode, device
    ).to(device)
    generator = torch.Generator(device=device).manual_seed(SEED)
    interior = sample_interior(N_INTERIOR_POINTS, device, generator)
    boundary = sample_boundary(N_BOUNDARY_POINTS, device)
    loss_fn = lambda: physics_loss(
        model, interior, boundary, PDE_WEIGHT, BOUNDARY_WEIGHT
    )
    if OPTIMIZER_NAME.lower() != "adam":
        raise ValueError("Only Adam is supported")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE, min_lr=MIN_LEARNING_RATE,
    )
    history = train(
        model.parameters(), loss_fn, optimizer, epochs=epochs,
        scheduler=scheduler, early_stopping_patience=EARLY_STOPPING_PATIENCE,
        min_delta=MIN_DELTA, print_every=PRINT_EVERY,
    )
    validation = sample_interior(N_VALIDATION_POINTS, device, generator)
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
    save_results(model, history, errors.detach().cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve Kovasznay flow with PI-KAN")
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
    args = parser.parse_args()
    main(
        args.device, args.layers, args.neurons, args.grid, args.spline_order,
        args.constraint, args.epochs, args.use_tuned,
    )
