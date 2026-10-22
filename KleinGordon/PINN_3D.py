"""PINN for the manufactured (2+1)-D nonlinear Klein--Gordon equation."""

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
    X_BOUNDS, Y_BOUNDS, T_BOUNDS, constraint_errors, exact_solution,
    hard_constraint_transform, loss_components, pde_residual, physics_loss,
    relative_l2_error, sample_constraints, sample_interior, save_results,
    select_device, set_seed,
)
from training import train

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "pinn_config.json"


def load_configuration(path: Path = CONFIG_PATH) -> dict:
    configuration = json.loads(path.read_text(encoding="utf-8"))
    required = {"reproducibility", "runtime", "problem", "model", "sampling", "loss_weights", "training", "tuning", "output"}
    missing = required.difference(configuration)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    return configuration


CONFIG = load_configuration()
SEED = int(CONFIG["reproducibility"]["seed"])
DEFAULT_DEVICE = str(CONFIG["runtime"]["device"])
LOAD_TUNED_IF_AVAILABLE = bool(CONFIG["runtime"]["load_tuned_if_available"])
N_LAYERS = int(CONFIG["model"]["n_layers"])
N_NEURONS = int(CONFIG["model"]["n_neurons"])
CONSTRAINT_MODE = str(CONFIG["model"]["constraint_mode"])
SAMPLING = CONFIG["sampling"]
TRAINING = CONFIG["training"]
LOSS_WEIGHTS = {key: float(value) for key, value in CONFIG["loss_weights"].items()}
ENFORCE_INITIAL_VELOCITY = bool(CONFIG["problem"].get("enforce_initial_velocity", False))
RESULT_PATH = SCRIPT_DIR / CONFIG["output"]["tuned_hyperparameters"]
POSTPROCESSING_DIR = SCRIPT_DIR / CONFIG["output"]["postprocessing_directory"]


class KleinGordonPINN(torch.nn.Module):
    """Fully connected tanh network with soft or exact constraints."""

    def __init__(self, n_layers: int, n_neurons: int, constraint_mode: str, enforce_initial_velocity: bool = ENFORCE_INITIAL_VELOCITY) -> None:
        super().__init__()
        if constraint_mode not in ("soft", "hard"):
            raise ValueError("constraint_mode must be 'soft' or 'hard'")
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.constraint_mode = constraint_mode
        self.enforce_initial_velocity = enforce_initial_velocity
        self.register_buffer("lower_bound", torch.tensor((X_BOUNDS[0], Y_BOUNDS[0], T_BOUNDS[0])))
        self.register_buffer("upper_bound", torch.tensor((X_BOUNDS[1], Y_BOUNDS[1], T_BOUNDS[1])))
        modules: list[torch.nn.Module] = []
        width = 3
        for _ in range(n_layers):
            layer = torch.nn.Linear(width, n_neurons)
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            modules.extend((layer, torch.nn.Tanh()))
            width = n_neurons
        output = torch.nn.Linear(width, 1)
        torch.nn.init.xavier_normal_(output.weight)
        torch.nn.init.zeros_(output.bias)
        modules.append(output)
        self.network = torch.nn.Sequential(*modules)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        normalized = 2.0 * (points - self.lower_bound) / (self.upper_bound - self.lower_bound) - 1.0
        raw = self.network(normalized).squeeze(-1)
        if self.constraint_mode == "soft":
            return raw
        return hard_constraint_transform(points, raw, self.enforce_initial_velocity)


def pinn_loss(model, interior, constraints, weights=LOSS_WEIGHTS):
    return physics_loss(model, interior, constraints, weights)


def load_hyperparameters(constraint_mode: str, path: Path = RESULT_PATH) -> tuple[int, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if constraint_mode in payload:
        payload = payload[constraint_mode]
    params = payload.get("best_params", payload)
    if not params:
        raise ValueError(f"No tuned {constraint_mode} PINN parameters in {path}")
    return int(params["n_layers"]), int(params["n_neurons"])


def main(
    device_name: str = DEFAULT_DEVICE,
    n_layers: int = N_LAYERS,
    n_neurons: int = N_NEURONS,
    constraint_mode: str = CONSTRAINT_MODE,
    epochs: int = int(TRAINING["epochs"]),
    use_tuned: bool | None = None,
) -> None:
    set_seed(SEED)
    device = select_device(device_name)
    should_load = LOAD_TUNED_IF_AVAILABLE and RESULT_PATH.exists() and RESULT_PATH.stat().st_size > 3 if use_tuned is None else use_tuned
    if should_load:
        n_layers, n_neurons = load_hyperparameters(constraint_mode)
        print(f"Loaded tuned hyperparameters from: {RESULT_PATH}")
    print(f"Using device: {device}")
    print(f"Klein-Gordon PINN: layers={n_layers}, neurons={n_neurons}, constraints={constraint_mode}, initial_velocity={ENFORCE_INITIAL_VELOCITY}")
    model = KleinGordonPINN(n_layers, n_neurons, constraint_mode).to(device)
    generator = torch.Generator(device=device).manual_seed(SEED)
    interior = sample_interior(int(SAMPLING["interior_points"]), device, generator)
    constraints = sample_constraints(int(SAMPLING["boundary_points_per_face"]), int(SAMPLING["initial_points"]), device, generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(TRAINING["learning_rate"]))
    scheduler_cfg = TRAINING["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=float(scheduler_cfg["factor"]),
        patience=int(scheduler_cfg["patience"]), min_lr=float(scheduler_cfg["min_learning_rate"]),
    )
    stopping = TRAINING["early_stopping"]
    history = train(
        model.parameters(), lambda: pinn_loss(model, interior, constraints), optimizer,
        epochs=epochs, scheduler=scheduler,
        early_stopping_patience=int(stopping["patience"]),
        min_delta=float(stopping["min_delta"]), print_every=int(TRAINING["print_every"]),
    )
    validation = sample_interior(int(SAMPLING["validation_points"]), device, generator)
    relative_error = relative_l2_error(model, validation)
    residual_mse = pde_residual(model, validation).detach().square().mean()
    value_error, velocity_error, boundary_error = constraint_errors(model, constraints)
    print(f"Relative validation L2 error: {relative_error.item():.6e}")
    print(f"Validation PDE residual MSE: {residual_mse.item():.6e}")
    print(f"Maximum initial-value error: {value_error.item():.6e}")
    velocity_label = "Maximum initial-velocity error" if ENFORCE_INITIAL_VELOCITY else "Initial-velocity diagnostic (not enforced)"
    print(f"{velocity_label}: {velocity_error.item():.6e}")
    print(f"Maximum boundary error: {boundary_error.item():.6e}")
    save_results(
        model, history, relative_error.item(), POSTPROCESSING_DIR,
        f"pinn_klein_gordon_{constraint_mode}", f"Klein-Gordon PINN ({constraint_mode})",
        int(SAMPLING["plot_spatial_points"]), list(SAMPLING["plot_times"]),
        {"constraint_mode": constraint_mode, "n_layers": n_layers, "n_neurons": n_neurons},
        bool(CONFIG["output"]["show_plots"]),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve the (2+1)-D Klein-Gordon equation with a PINN")
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--layers", type=int, default=N_LAYERS)
    parser.add_argument("--neurons", type=int, default=N_NEURONS)
    parser.add_argument("--constraint", choices=("soft", "hard"), default=CONSTRAINT_MODE)
    parser.add_argument("--epochs", type=int, default=int(TRAINING["epochs"]))
    tuned = parser.add_mutually_exclusive_group()
    tuned.add_argument("--use-tuned", dest="use_tuned", action="store_true")
    tuned.add_argument("--no-use-tuned", dest="use_tuned", action="store_false")
    parser.set_defaults(use_tuned=None)
    args = parser.parse_args()
    main(args.device, args.layers, args.neurons, args.constraint, args.epochs, args.use_tuned)
