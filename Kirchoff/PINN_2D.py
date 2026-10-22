"""PINN solver for the simply supported rectangular Kirchhoff plate."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Kirchoff._common import (
    constraint_errors, load_config, physics_loss, problem_values, relative_l2,
    sample_boundary, sample_interior, save_results, select_device, set_seed,
    train_model,
)

CONFIG_PATH = SCRIPT_DIR / "pinn_config.json"
CONFIG = load_config(CONFIG_PATH)
RESULT_PATH = SCRIPT_DIR / CONFIG["output"]["tuned_hyperparameters"]


class KirchhoffPINN(torch.nn.Module):
    """MLP with soft BCs or an exact simply-supported trial space."""

    def __init__(self, n_layers: int, n_neurons: int, constraint_mode: str, values):
        super().__init__()
        if constraint_mode not in ("soft", "hard"):
            raise ValueError("constraint_mode must be 'soft' or 'hard'")
        self.constraint_mode = constraint_mode
        self.values = values
        modules: list[torch.nn.Module] = []
        width = 2
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
        x, y = points[:, 0], points[:, 1]
        if self.constraint_mode == "hard":
            angle_x = math.pi * x / self.values["length_x"]
            angle_y = math.pi * y / self.values["length_y"]
            features = torch.stack((torch.cos(angle_x), torch.cos(angle_y)), dim=1)
            raw = self.network(features).squeeze(-1)
            return torch.sin(angle_x) * torch.sin(angle_y) * raw
        features = torch.stack((
            2.0 * x / self.values["length_x"] - 1.0,
            2.0 * y / self.values["length_y"] - 1.0,
        ), dim=1)
        return self.network(features).squeeze(-1)


def load_hyperparameters(mode: str) -> tuple[int, int]:
    payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    payload = payload.get(mode, payload)
    params = payload.get("best_params", payload)
    return int(params["n_layers"]), int(params["n_neurons"])


def main(
    device_name: str | None = None,
    constraint_mode: str | None = None,
    epochs: int | None = None,
    use_tuned: bool | None = None,
) -> None:
    seed = int(CONFIG["reproducibility"]["seed"])
    set_seed(seed)
    device = select_device(device_name or CONFIG["runtime"]["device"])
    mode = constraint_mode or CONFIG["model"]["constraint_mode"]
    should_load = (
        bool(CONFIG["runtime"]["load_tuned_if_available"]) and RESULT_PATH.exists()
        if use_tuned is None else use_tuned
    )
    if should_load:
        n_layers, n_neurons = load_hyperparameters(mode)
    else:
        n_layers = int(CONFIG["model"]["n_layers"])
        n_neurons = int(CONFIG["model"]["n_neurons"])
    values = problem_values(CONFIG)
    model = KirchhoffPINN(n_layers, n_neurons, mode, values).to(device)
    sampling = CONFIG["sampling"]
    interior = sample_interior(int(sampling["interior_points"]), values, device)
    boundaries = sample_boundary(int(sampling["boundary_points_per_edge"]), values, device)
    loss_fn = lambda: physics_loss(
        model, interior, boundaries, values, CONFIG["loss_weights"]
    )
    training = dict(CONFIG["training"])
    if epochs is not None:
        training["epochs"] = epochs
    print(f"PINN: layers={n_layers}, neurons={n_neurons}, constraints={mode}, device={device}")
    history = train_model(model, loss_fn, training)
    validation = sample_interior(int(sampling["validation_points"]), values, device)
    error = float(relative_l2(model, validation, values))
    displacement_error, moment_error = constraint_errors(model, boundaries, values)
    print(f"Validation relative L2 error: {error:.6e}")
    print(f"Boundary RMS deflection: {displacement_error:.6e}")
    print(f"Normalized boundary RMS moment: {moment_error:.6e}")
    save_results(model, history, error, CONFIG, values, "pinn")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), default=None)
    parser.add_argument("--constraint", choices=("soft", "hard"), default=None)
    parser.add_argument("--epochs", type=int, default=None)
    tuned = parser.add_mutually_exclusive_group()
    tuned.add_argument("--use-tuned", dest="use_tuned", action="store_true")
    tuned.add_argument("--no-use-tuned", dest="use_tuned", action="store_false")
    parser.set_defaults(use_tuned=None)
    args = parser.parse_args()
    main(args.device, args.constraint, args.epochs, args.use_tuned)

