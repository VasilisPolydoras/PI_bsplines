"""XPINN solver for the singular L-shaped Laplace benchmark.

The exact harmonic solution is

    u(r, theta) = r^(2/3) sin(2 theta / 3),

whose gradient is singular at the re-entrant corner (0, 0). Each subnetwork
is embedded in a patchwise trial function that satisfies the
exterior Dirichlet data exactly. The remaining XPINN loss contains interior
Laplace residuals and solution, residual, and normal-flux interface coupling.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LShapeLaplace.Irregular._xpinn_utils import (
    build_tanh_network,
    interface_diagnostics,
    laplace_residual,
    maximum_boundary_error,
    plot_solution,
    plot_training_history,
    relative_l2_error,
    select_device,
    set_seed,
    xpinn_loss,
    xpinn_loss_components,
)
from training import train


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "xpinn_config.json"

PATCH_NAMES = ("upper_right", "upper_left", "lower_left")
PATCH_BOUNDS = {
    "upper_right": ((0.0, 0.0), (1.0, 1.0)),
    "upper_left": ((-1.0, 0.0), (0.0, 1.0)),
    "lower_left": ((-1.0, -1.0), (0.0, 0.0)),
}
INTERFACE_NORMALS = {
    "upper_right_upper_left": ((-1.0, 0.0), (1.0, 0.0)),
    "upper_left_lower_left": ((0.0, -1.0), (0.0, 1.0)),
}


def load_configuration(path: Path = CONFIG_PATH) -> dict:
    """Load the complete XPINN experiment configuration."""
    if not path.exists():
        raise FileNotFoundError(f"XPINN configuration not found: {path}")
    configuration = json.loads(path.read_text(encoding="utf-8"))
    required_sections = {
        "reproducibility",
        "runtime",
        "model",
        "sampling",
        "loss_weights",
        "training",
        "output",
    }
    missing = required_sections.difference(configuration)
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
N_LAYERS = int(CONFIG["model"]["n_layers"])
N_NEURONS = int(CONFIG["model"]["n_neurons"])
ACTIVATION = str(CONFIG["model"]["activation"])
CORNER_EPSILON = float(CONFIG["model"]["corner_epsilon"])
N_INTERIOR_POINTS_PER_PATCH = int(
    CONFIG["sampling"]["interior_points_per_patch"]
)
N_INTERFACE_POINTS = int(CONFIG["sampling"]["interface_points"])
N_VALIDATION_POINTS_PER_PATCH = int(
    CONFIG["sampling"]["validation_points_per_patch"]
)
N_VALIDATION_BOUNDARY_POINTS_PER_EDGE = int(
    CONFIG["sampling"]["validation_boundary_points_per_edge"]
)
N_VALIDATION_INTERFACE_POINTS = int(
    CONFIG["sampling"]["validation_interface_points"]
)
PDE_WEIGHT = float(CONFIG["loss_weights"]["pde"])
INTERFACE_SOLUTION_WEIGHT = float(
    CONFIG["loss_weights"]["interface_solution"]
)
INTERFACE_RESIDUAL_WEIGHT = float(
    CONFIG["loss_weights"]["interface_residual"]
)
INTERFACE_FLUX_WEIGHT = float(CONFIG["loss_weights"]["interface_flux"])
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
POSTPROCESSING_DIR = (
    SCRIPT_DIR / CONFIG["output"]["postprocessing_directory"]
)
PLOT_RESOLUTION = int(CONFIG["output"]["plot_resolution"])
SHOW_PLOTS = bool(CONFIG["output"]["show_plots"])


def exact_solution(points: torch.Tensor) -> torch.Tensor:
    """Return r^(2/3) sin(2 theta/3) with theta in [0, 2*pi)."""
    x, y = points[:, 0], points[:, 1]
    radius = torch.sqrt(x.square() + y.square())
    theta = torch.remainder(torch.atan2(y, x), 2.0 * math.pi)
    return radius.pow(2.0 / 3.0) * torch.sin(2.0 * theta / 3.0)


def map_patch(name: str, local_points: torch.Tensor) -> torch.Tensor:
    """Map coordinates in [0,1]^2 to one of the three L-shaped patches."""
    xi, eta = local_points[:, 0], local_points[:, 1]
    if name == "upper_right":
        return torch.stack((xi, eta), dim=1)
    if name == "upper_left":
        return torch.stack((xi - 1.0, eta), dim=1)
    if name == "lower_left":
        return torch.stack((xi - 1.0, eta - 1.0), dim=1)
    raise ValueError(f"Unknown patch: {name}")


def sample_interior_points(
    count: int, device: torch.device,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Sample equally many uniformly distributed points in every patch."""
    return {
        name: map_patch(
            name, torch.rand(count, 2, device=device, generator=generator)
        )
        for name in PATCH_NAMES
    }


def generate_boundary_points(
    points_per_edge: int, device: torch.device
) -> dict[str, torch.Tensor]:
    """Return exterior edges only; exclude internal patch interfaces."""
    line = torch.linspace(0.0, 1.0, points_per_edge, device=device)
    zero, one = torch.zeros_like(line), torch.ones_like(line)
    local_edges = {
        "upper_right": (
            torch.stack((line, zero), dim=1),
            torch.stack((one, line), dim=1),
            torch.stack((line, one), dim=1),
        ),
        "upper_left": (
            torch.stack((zero, line), dim=1),
            torch.stack((line, one), dim=1),
        ),
        "lower_left": (
            torch.stack((zero, line), dim=1),
            torch.stack((line, zero), dim=1),
            torch.stack((one, line), dim=1),
        ),
    }
    return {
        name: torch.cat([map_patch(name, edge) for edge in edges], dim=0)
        for name, edges in local_edges.items()
    }


def generate_interface_points(
    count: int, device: torch.device, corner_epsilon: float = 0.0,
) -> dict[str, tuple[str, torch.Tensor, str, torch.Tensor]]:
    """Return paired interface points outside r < corner_epsilon."""
    if not 0.0 <= corner_epsilon < 1.0:
        raise ValueError("corner_epsilon must lie in [0, 1)")
    vertical = torch.linspace(corner_epsilon, 1.0, count, device=device)
    horizontal = torch.linspace(0.0, 1.0 - corner_epsilon, count, device=device)
    vz, vo = torch.zeros_like(vertical), torch.ones_like(vertical)
    hz, ho = torch.zeros_like(horizontal), torch.ones_like(horizontal)
    return {
        "upper_right_upper_left": (
            "upper_right", map_patch(
                "upper_right", torch.stack((vz, vertical), dim=1)
            ),
            "upper_left", map_patch(
                "upper_left", torch.stack((vo, vertical), dim=1)
            ),
        ),
        "upper_left_lower_left": (
            "upper_left", map_patch(
                "upper_left", torch.stack((horizontal, hz), dim=1)
            ),
            "lower_left", map_patch(
                "lower_left", torch.stack((horizontal, ho), dim=1)
            ),
        ),
    }


class LShapeXPINN(torch.nn.Module):
    """Three-patch XPINN with exact exterior Dirichlet constraints."""

    patch_names = PATCH_NAMES
    interface_normals = INTERFACE_NORMALS
    map_patch = staticmethod(map_patch)
    output_stem = "xpinn_irregular"
    solution_title = "XPINN singular solution"
    exact_solution_label = r"Exact singular solution: $r^{2/3}\sin(2\theta/3)$"

    def __init__(self, n_layers: int, n_neurons: int) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        networks = {}
        for name in self.patch_names:
            lower, upper = PATCH_BOUNDS[name]
            self.register_buffer(f"{name}_lower_bound", torch.tensor(lower))
            self.register_buffer(f"{name}_upper_bound", torch.tensor(upper))
            networks[name] = build_tanh_network(n_layers, n_neurons)
        self.networks = torch.nn.ModuleDict(networks)

    @staticmethod
    def exact_solution(points: torch.Tensor) -> torch.Tensor:
        return exact_solution(points)

    @staticmethod
    def _local_coordinates(
        patch_name: str, points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the local coordinates (xi, eta) in [0,1]^2."""
        x, y = points[:, 0], points[:, 1]
        if patch_name == "upper_right":
            return x, y
        if patch_name == "upper_left":
            return x + 1.0, y
        if patch_name == "lower_left":
            return x + 1.0, y + 1.0
        raise ValueError(f"Unknown patch: {patch_name}")

    def _boundary_extension(
        self, patch_name: str, points: torch.Tensor
    ) -> torch.Tensor:
        """Extend only the prescribed exterior traces into one patch."""
        xi, eta = self._local_coordinates(patch_name, points)

        if patch_name == "upper_right":
            top_points = torch.stack((xi, torch.ones_like(xi)), dim=1)
            right_points = torch.stack((torch.ones_like(eta), eta), dim=1)
            right_bottom = torch.stack(
                (torch.ones_like(eta), torch.zeros_like(eta)), dim=1
            )
            right_top = torch.stack(
                (torch.ones_like(eta), torch.ones_like(eta)), dim=1
            )
            # The exact trace is identically zero on y=0, x>=0. Writing it
            # explicitly avoids differentiating the polar formula at r=0.
            bottom = torch.zeros_like(xi)
            top = self.exact_solution(top_points)
            base = bottom + eta * (top - bottom)
            base_right = self.exact_solution(right_bottom) + eta * (
                self.exact_solution(right_top)
                - self.exact_solution(right_bottom)
            )
            return base + xi * (
                self.exact_solution(right_points) - base_right
            )

        if patch_name == "upper_left":
            left_points = torch.stack(
                (-torch.ones_like(eta), eta), dim=1
            )
            top_points = torch.stack((xi - 1.0, torch.ones_like(xi)), dim=1)
            top_left = torch.stack(
                (-torch.ones_like(eta), torch.ones_like(eta)), dim=1
            )
            return self.exact_solution(left_points) + eta * (
                self.exact_solution(top_points)
                - self.exact_solution(top_left)
            )

        bottom_points = torch.stack(
            (xi - 1.0, -torch.ones_like(xi)), dim=1
        )
        left_points = torch.stack(
            (-torch.ones_like(eta), eta - 1.0), dim=1
        )
        bottom_left = torch.stack(
            (-torch.ones_like(eta), -torch.ones_like(eta)), dim=1
        )
        bottom = self.exact_solution(bottom_points)
        base = bottom + (1.0 - xi) * (
            self.exact_solution(left_points)
            - self.exact_solution(bottom_left)
        )
        # The exact trace is identically zero on x=0, y<=0, and the base
        # already has this value at xi=1 by boundary compatibility.
        return base

    @classmethod
    def _boundary_distance(
        cls, patch_name: str, points: torch.Tensor
    ) -> torch.Tensor:
        """Return a factor vanishing precisely on exterior patch edges."""
        xi, eta = cls._local_coordinates(patch_name, points)
        if patch_name == "upper_right":
            return eta * (1.0 - xi) * (1.0 - eta)
        if patch_name == "upper_left":
            return xi * (1.0 - eta)
        if patch_name == "lower_left":
            return xi * (1.0 - xi) * eta
        raise ValueError(f"Unknown patch: {patch_name}")

    def forward(self, patch_name: str, points: torch.Tensor) -> torch.Tensor:
        lower = getattr(self, f"{patch_name}_lower_bound")
        upper = getattr(self, f"{patch_name}_upper_bound")
        normalized = 2.0 * (points - lower) / (upper - lower) - 1.0
        raw = self.networks[patch_name](normalized).squeeze(-1)
        return self._boundary_extension(
            patch_name, points
        ) + self._boundary_distance(patch_name, points) * raw


def load_hyperparameters(path: Path = RESULT_PATH) -> tuple[int, int]:
    if not path.exists():
        raise FileNotFoundError(
            f"No irregular XPINN tuning result found at {path}. Supply that "
            "JSON file or run with --no-use-tuned."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("best_params", payload)
    return int(params["n_layers"]), int(params["n_neurons"])


def main(
    device_name: str = DEFAULT_DEVICE,
    n_layers: int = N_LAYERS,
    n_neurons: int = N_NEURONS,
    corner_epsilon: float = CORNER_EPSILON,
    epochs: int = EPOCHS,
    use_tuned: bool | None = None,
    initialization_seed: int = SEED,
    sampling_seed: int = SEED,
    validation_seed: int | None = None,
    save_outputs: bool = True,
    verbose: bool = True,
) -> dict[str, float]:
    """Train once and return validation error and training wall time."""
    set_seed(initialization_seed)
    device = select_device(device_name)
    should_load = (
        LOAD_TUNED_IF_AVAILABLE and RESULT_PATH.exists()
        if use_tuned is None
        else use_tuned
    )
    if should_load:
        n_layers, n_neurons = load_hyperparameters()
        if verbose:
            print(f"Loaded tuned hyperparameters from: {RESULT_PATH}")
    if verbose:
        print(f"Using device: {device}")
        print(f"Loaded configuration from: {CONFIG_PATH}")
    if ACTIVATION.lower() != "tanh":
        raise ValueError(
            f"Unsupported activation '{ACTIVATION}'; expected 'tanh'"
        )
    if verbose:
        print(
            "Irregular XPINN subnetworks: "
            f"layers={n_layers}, neurons={n_neurons}, activation={ACTIVATION}, "
            f"corner_epsilon={corner_epsilon:g}"
        )
        print("Exterior Dirichlet values are imposed exactly by the trial functions.")

    model = LShapeXPINN(n_layers, n_neurons).to(device)
    generator = torch.Generator(device=device).manual_seed(sampling_seed)
    interior = sample_interior_points(
        N_INTERIOR_POINTS_PER_PATCH, device, generator
    )
    interfaces = generate_interface_points(
        N_INTERFACE_POINTS, device, corner_epsilon
    )

    loss_fn = lambda: xpinn_loss(
        model,
        interior,
        None,
        interfaces,
        pde_weight=PDE_WEIGHT,
        interface_solution_weight=INTERFACE_SOLUTION_WEIGHT,
        interface_residual_weight=INTERFACE_RESIDUAL_WEIGHT,
        interface_flux_weight=INTERFACE_FLUX_WEIGHT,
    )
    if OPTIMIZER_NAME.lower() != "adam":
        raise ValueError(
            f"Unsupported optimizer '{OPTIMIZER_NAME}'; expected 'Adam'"
        )
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
    loss_history = train(
        model.parameters(),
        loss_fn,
        optimizer,
        epochs=epochs,
        scheduler=scheduler,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        min_delta=MIN_DELTA,
        print_every=PRINT_EVERY if verbose else 0,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_time_seconds = time.perf_counter() - training_start

    validation_generator = (
        generator
        if validation_seed is None
        else torch.Generator(device=device).manual_seed(validation_seed)
    )
    validation = sample_interior_points(
        N_VALIDATION_POINTS_PER_PATCH, device, validation_generator
    )
    relative_error = relative_l2_error(model, validation)
    if verbose:
        validation_interfaces = generate_interface_points(
            N_VALIDATION_INTERFACE_POINTS, device, corner_epsilon
        )
        validation_boundary = generate_boundary_points(
            N_VALIDATION_BOUNDARY_POINTS_PER_EDGE, device
        )
        residual_mse = torch.stack(
            [
                laplace_residual(model, name, validation[name])
                .detach()
                .square()
                .mean()
                for name in model.patch_names
            ]
        ).mean()
        boundary_error = maximum_boundary_error(model, validation_boundary)
        solution_jump, residual_jump, flux_jump = interface_diagnostics(
            model, validation_interfaces
        )
        print(f"Relative L2 error: {relative_error.item():.6e}")
        print(f"Interior Laplace residual MSE: {residual_mse.item():.6e}")
        print(f"Maximum boundary error: {boundary_error.item():.6e}")
        print(f"Maximum interface solution jump: {solution_jump.item():.6e}")
        print(f"Maximum interface residual jump: {residual_jump.item():.6e}")
        print(f"Maximum interface normal-flux jump: {flux_jump.item():.6e}")
        print(f"Training wall time: {training_time_seconds:.6f} s")
    if save_outputs:
        plot_solution(
            model,
            postprocessing_dir=POSTPROCESSING_DIR,
            resolution=PLOT_RESOLUTION,
            validation_relative_l2_error=relative_error,
        )
        plot_training_history(
            loss_history,
            model,
            postprocessing_dir=POSTPROCESSING_DIR,
            training_time_seconds=training_time_seconds,
        )
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close("all")
    return {
        "validation_relative_l2_error": relative_error.detach().cpu().item(),
        "training_time_seconds": training_time_seconds,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve the singular L-shaped Laplace problem with XPINNs"
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--layers", type=int, default=N_LAYERS)
    parser.add_argument("--neurons", type=int, default=N_NEURONS)
    parser.add_argument(
        "--corner-epsilon", type=float, default=CORNER_EPSILON
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="Trainable-parameter initialization seed",
    )
    parser.add_argument(
        "--sampling-seed", type=int, default=SEED,
        help="Seed for the fixed interior collocation points",
    )
    parser.add_argument(
        "--validation-seed", type=int, default=None,
        help="Optional independent validation-set seed",
    )
    tuned_group = parser.add_mutually_exclusive_group()
    tuned_group.add_argument("--use-tuned", dest="use_tuned", action="store_true")
    tuned_group.add_argument(
        "--no-use-tuned", dest="use_tuned", action="store_false"
    )
    parser.set_defaults(use_tuned=None)
    arguments = parser.parse_args()
    main(
        device_name=arguments.device,
        n_layers=arguments.layers,
        n_neurons=arguments.neurons,
        corner_epsilon=arguments.corner_epsilon,
        epochs=arguments.epochs,
        use_tuned=arguments.use_tuned,
        initialization_seed=arguments.seed,
        sampling_seed=arguments.sampling_seed,
        validation_seed=arguments.validation_seed,
    )
