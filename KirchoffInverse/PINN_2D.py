"""PINN inverse identification of Kirchhoff plate flexural rigidity."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from _common import (
    ROOT, device_from, initialize_positive_parameter, loss, observations,
    positive_parameter, problem, relative_l2, sample_boundary, sample_points,
    save_results, set_seed, train_inverse,
)

CONFIG_PATH = ROOT / "pinn_config.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
RESULT_PATH = ROOT / CONFIG["output"]["tuned_hyperparameters"]


class KirchhoffInversePINN(torch.nn.Module):
    """MLP field approximation with a jointly trainable positive rigidity."""

    def __init__(self, n_layers, n_neurons, constraint_mode, values, inverse, device):
        super().__init__()
        if constraint_mode not in ("hard", "soft"):
            raise ValueError("constraint_mode must be 'hard' or 'soft'")
        self.constraint_mode = constraint_mode
        self.values = values
        self.minimum_d = float(inverse["minimum_d"])
        self.raw_d = initialize_positive_parameter(inverse["initial_d"], self.minimum_d, device)
        layers = []
        width = 2
        for _ in range(n_layers):
            linear = torch.nn.Linear(width, n_neurons)
            torch.nn.init.xavier_normal_(linear.weight); torch.nn.init.zeros_(linear.bias)
            layers.extend((linear, torch.nn.Tanh())); width = n_neurons
        output = torch.nn.Linear(width, 1)
        torch.nn.init.xavier_normal_(output.weight); torch.nn.init.zeros_(output.bias)
        layers.append(output); self.network = torch.nn.Sequential(*layers)

    def flexural_rigidity(self):
        return positive_parameter(self.raw_d, self.minimum_d)

    def forward(self, points):
        x, y = points[:, 0], points[:, 1]
        if self.constraint_mode == "hard":
            ax = math.pi*x/self.values["length_x"]; ay = math.pi*y/self.values["length_y"]
            features = torch.stack((torch.cos(ax), torch.cos(ay)), 1)
            return torch.sin(ax)*torch.sin(ay)*self.network(features).squeeze(-1)
        features = torch.stack((2*x/self.values["length_x"]-1, 2*y/self.values["length_y"]-1), 1)
        return self.network(features).squeeze(-1)


def load_architecture(mode):
    payload = json.loads(RESULT_PATH.read_text(encoding="utf-8")); payload = payload.get(mode, payload)
    params = payload.get("best_params", payload)
    return int(params["n_layers"]), int(params["n_neurons"])


def main(device_name=None, constraint_mode=None, epochs=None, use_tuned=None):
    seed = int(CONFIG["reproducibility"]["seed"]); set_seed(seed)
    device = device_from(device_name or CONFIG["runtime"]["device"])
    mode = constraint_mode or CONFIG["model"]["constraint_mode"]
    load_tuned = (CONFIG["runtime"]["load_tuned_if_available"] and RESULT_PATH.exists()) if use_tuned is None else use_tuned
    architecture = load_architecture(mode) if load_tuned else (int(CONFIG["model"]["n_layers"]), int(CONFIG["model"]["n_neurons"]))
    values = problem(CONFIG); generator = torch.Generator(device=device).manual_seed(seed)
    sampling = CONFIG["sampling"]
    interior = sample_points(int(sampling["interior_points"]), values, device, generator)
    sensors = observations(int(sampling["sensor_points"]), float(CONFIG["inverse"]["noise_std"]), values, device, generator)
    boundaries = sample_boundary(int(sampling["boundary_points_per_edge"]), values, device)
    model = KirchhoffInversePINN(*architecture, mode, values, CONFIG["inverse"], device).to(device)
    loss_fn = lambda: loss(model, interior, *sensors, boundaries, values, CONFIG["loss_weights"])
    training = dict(CONFIG["training"])
    if epochs is not None: training["epochs"] = epochs
    print(f"PINN inverse: architecture={architecture}, constraints={mode}, device={device}")
    history, d_history = train_inverse(model, loss_fn, training)
    validation = sample_points(int(sampling["validation_points"]), values, device, generator)
    error = float(relative_l2(model, validation, values)); estimated = float(model.flexural_rigidity().detach())
    print(f"Estimated D: {estimated:.8f} (true {values['true_d']:.8f})")
    print(f"Relative D error: {abs(estimated-values['true_d'])/values['true_d']:.6e}")
    print(f"Validation relative L2 error: {error:.6e}")
    save_results(model, history, d_history, error, CONFIG, values, "pinn", sensors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), default=None)
    parser.add_argument("--constraint", choices=("soft", "hard"), default=None)
    parser.add_argument("--epochs", type=int, default=None)
    group = parser.add_mutually_exclusive_group(); group.add_argument("--use-tuned", dest="use_tuned", action="store_true"); group.add_argument("--no-use-tuned", dest="use_tuned", action="store_false")
    parser.set_defaults(use_tuned=None); args = parser.parse_args()
    main(args.device, args.constraint, args.epochs, args.use_tuned)
