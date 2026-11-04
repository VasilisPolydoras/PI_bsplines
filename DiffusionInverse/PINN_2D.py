"""PINN identification of the diffusion coefficient with soft/hard constraints."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Diffusion.PINN_2D import DiffusionPINN, sample_constraints, sample_interior
from training import train

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "pinn_config.json").read_text(encoding="utf-8"))
RESULT_PATH = ROOT / CONFIG["output"]["tuned_hyperparameters"]
SEED = int(CONFIG["reproducibility"]["seed"])
DEFAULT_DEVICE = str(CONFIG["runtime"]["device"])
CONSTRAINT_MODE = str(CONFIG["model"]["constraint_mode"])
EPOCHS = int(CONFIG["training"]["epochs"])
TRUE_DIFFUSIVITY = float(CONFIG["problem"]["true_diffusivity"])


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(name):
    if name == "cpu": return torch.device("cpu")
    if name == "gpu" and torch.cuda.is_available(): return torch.device("cuda")
    if name == "gpu": raise RuntimeError("GPU requested, but CUDA is unavailable")
    raise ValueError("device must be 'cpu' or 'gpu'")


def exact_solution(points, diffusivity):
    return torch.sin(2.0 * torch.pi * points[:, 0]) * torch.exp(
        -4.0 * torch.pi**2 * diffusivity * points[:, 1]
    )


def raw_positive(value, minimum, device):
    shifted = max(value - minimum, torch.finfo(torch.float32).eps)
    return torch.nn.Parameter(torch.tensor(math.log(math.expm1(shifted)), device=device))


class InverseDiffusionPINN(DiffusionPINN):
    def __init__(self, layers, neurons, mode, inverse_config, device):
        super().__init__(layers, neurons, mode)
        self.minimum_diffusivity = float(inverse_config["minimum_diffusivity"])
        self.raw_diffusivity = raw_positive(
            float(inverse_config["initial_diffusivity"]),
            self.minimum_diffusivity, device,
        )

    def diffusivity(self):
        return self.minimum_diffusivity + functional.softplus(self.raw_diffusivity)


def sample_observations(count, noise_std, diffusivity, device, generator):
    points = sample_interior(count, device, generator=generator)
    values = exact_solution(points, diffusivity)
    if noise_std:
        values = values + noise_std * torch.randn(count, device=device, generator=generator)
    return points, values


def residual(model, points):
    coordinates = points.detach().clone().requires_grad_(True)
    solution = model(coordinates)
    gradient = torch.autograd.grad(
        solution, coordinates, torch.ones_like(solution), create_graph=True
    )[0]
    u_x, u_t = gradient[:, 0], gradient[:, 1]
    u_xx = torch.autograd.grad(
        u_x, coordinates, torch.ones_like(u_x), create_graph=True
    )[0][:, 0]
    return u_t - model.diffusivity() * u_xx


def loss_components(model, interior, sensors, sensor_values, constraints):
    components = {
        "pde": residual(model, interior).square().mean(),
        "data": (model(sensors) - sensor_values).square().mean(),
    }
    if model.constraint_mode == "hard":
        zero = components["pde"].new_zeros(())
        components.update(initial=zero, boundary=zero)
    else:
        initial = constraints["initial"]
        components["initial"] = (
            model(initial) - exact_solution(initial, TRUE_DIFFUSIVITY)
        ).square().mean()
        components["boundary"] = 0.5 * (
            model(constraints["left"]).square().mean()
            + model(constraints["right"]).square().mean()
        )
    return components


def inverse_loss(model, interior, sensors, sensor_values, constraints):
    terms = loss_components(model, interior, sensors, sensor_values, constraints)
    weights = CONFIG["loss_weights"]
    return sum(float(weights[name]) * value for name, value in terms.items())


def relative_l2_error(model, points):
    with torch.no_grad():
        prediction, reference = model(points), exact_solution(points, TRUE_DIFFUSIVITY)
    return torch.linalg.vector_norm(prediction - reference) / torch.linalg.vector_norm(reference)


def load_hyperparameters(mode, path=RESULT_PATH):
    if not path.exists():
        raise FileNotFoundError(f"No inverse PINN tuning result found at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if mode in payload:
        record = payload[mode]
    elif "best_params" in payload:
        # Backward compatibility with a single, mode-independent tuning file.
        record = payload
    elif "soft" in payload:
        # Match the forward-Diffusion behavior: use the available tuned
        # architecture for either constraint mode until hard is tuned
        # separately. A future hard record automatically takes precedence.
        record = payload["soft"]
        print(
            "No hard-specific inverse PINN tuning result is available; "
            "using the soft-tuned architecture for hard training."
        )
    else:
        available = ", ".join(sorted(payload)) or "none"
        raise KeyError(
            f"No tuned hyperparameters for constraint mode '{mode}' in "
            f"{path}. Available records: {available}."
        )
    parameters = record.get("best_params", {})
    missing = {"n_layers", "n_neurons"}.difference(parameters)
    if missing:
        raise KeyError(
            f"Missing tuned PINN parameters in {path}: "
            f"{', '.join(sorted(missing))}"
        )
    return int(parameters["n_layers"]), int(parameters["n_neurons"])


def save_results(model, history, diffusivity_history, error, wall_time, sensors):
    output = ROOT / CONFIG["output"]["postprocessing_directory"]
    output.mkdir(parents=True, exist_ok=True)
    sampling = CONFIG["sampling"]
    x = np.linspace(*CONFIG["problem"]["x_bounds"], sampling["plot_spatial_points"])
    times = np.linspace(*CONFIG["problem"]["time_bounds"], sampling["plot_time_points"])
    grid_x, grid_t = np.meshgrid(x, times)
    device = next(model.parameters()).device
    points = torch.tensor(np.column_stack((grid_x.ravel(), grid_t.ravel())), dtype=torch.float32, device=device)
    with torch.no_grad():
        prediction = model(points).cpu().numpy().reshape(grid_x.shape)
        reference = exact_solution(points, TRUE_DIFFUSIVITY).cpu().numpy().reshape(grid_x.shape)
    estimated = float(model.diffusivity().detach())
    stem = f"pinn_diffusion_inverse_{model.constraint_mode}"
    np.savez_compressed(
        output / f"{stem}_solution_data.npz", x=x, t=times,
        prediction=prediction, reference=reference, absolute_error=np.abs(prediction-reference),
        sensor_points=sensors[0].cpu().numpy(), sensor_values=sensors[1].cpu().numpy(),
        estimated_diffusivity=estimated, true_diffusivity=TRUE_DIFFUSIVITY,
        parameter_relative_error=abs(estimated-TRUE_DIFFUSIVITY)/TRUE_DIFFUSIVITY,
        validation_relative_l2_error=error, training_wall_time_seconds=wall_time,
        constraint_mode=np.asarray(model.constraint_mode),
    )
    np.savez_compressed(
        output / f"{stem}_training_data.npz", epoch=np.arange(1,len(history)+1),
        loss=np.asarray(history), estimated_diffusivity=np.asarray(diffusivity_history),
        training_wall_time_seconds=wall_time,
    )


def main(
    device_name=DEFAULT_DEVICE, constraint_mode=CONSTRAINT_MODE, epochs=EPOCHS,
    use_tuned=None, initialization_seed=SEED, sampling_seed=SEED,
    validation_seed=None, save_outputs=True, verbose=True,
):
    set_seed(initialization_seed)
    device = select_device(device_name)
    model_config, sampling = CONFIG["model"], CONFIG["sampling"]
    layers, neurons = int(model_config["n_layers"]), int(model_config["n_neurons"])
    should_load = (
        CONFIG["runtime"]["load_tuned_if_available"] and RESULT_PATH.exists()
        if use_tuned is None else use_tuned
    )
    if should_load: layers, neurons = load_hyperparameters(constraint_mode)
    generator = torch.Generator(device=device).manual_seed(sampling_seed)
    interior = sample_interior(sampling["interior_points"], device, generator=generator)
    sensors = sample_observations(
        sampling["sensor_points"], CONFIG["inverse"]["noise_std"],
        TRUE_DIFFUSIVITY, device, generator,
    )
    constraints = sample_constraints(
        sampling["boundary_points_per_edge"], sampling["initial_points"], device
    )
    model = InverseDiffusionPINN(
        layers, neurons, constraint_mode, CONFIG["inverse"], device
    ).to(device)
    training = CONFIG["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=float(training["scheduler"]["factor"]),
        patience=int(training["scheduler"]["patience"]),
        min_lr=float(training["scheduler"]["min_learning_rate"]),
    )
    diffusivity_history = []
    def loss_function():
        value = inverse_loss(model, interior, sensors[0], sensors[1], constraints)
        diffusivity_history.append(float(model.diffusivity().detach()))
        return value
    if device.type == "cuda": torch.cuda.synchronize(device)
    start = time.perf_counter()
    history = train(
        model.parameters(), loss_function, optimizer, epochs=epochs, scheduler=scheduler,
        early_stopping_patience=int(training["early_stopping"]["patience"]),
        min_delta=float(training["early_stopping"]["min_delta"]),
        print_every=int(training["print_every"] if verbose else 0),
    )
    if device.type == "cuda": torch.cuda.synchronize(device)
    wall_time = time.perf_counter() - start
    validation_generator = torch.Generator(device=device).manual_seed(
        sampling_seed if validation_seed is None else validation_seed
    )
    validation = sample_interior(sampling["validation_points"], device, generator=validation_generator)
    field_error = float(relative_l2_error(model, validation))
    estimated = float(model.diffusivity().detach())
    parameter_error = abs(estimated-TRUE_DIFFUSIVITY)/TRUE_DIFFUSIVITY
    if verbose:
        print(f"Estimated diffusivity: {estimated:.8f} (true {TRUE_DIFFUSIVITY:.8f})")
        print(f"Parameter relative error: {parameter_error:.6e}")
        print(f"Validation relative L2 error: {field_error:.6e}")
        print(f"Training wall time: {wall_time:.3f} s")
    if save_outputs: save_results(model, history, diffusivity_history, field_error, wall_time, sensors)
    return {"validation_relative_l2_error": field_error, "parameter_relative_error": parameter_error,
            "estimated_diffusivity": estimated, "training_time_seconds": wall_time}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inverse diffusion PINN")
    parser.add_argument("--device", choices=("cpu","gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--constraint", choices=("soft","hard"), default=CONSTRAINT_MODE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--no-use-tuned", action="store_true")
    args = parser.parse_args()
    main(
        args.device, args.constraint, args.epochs,
        use_tuned=False if args.no_use_tuned else None,
    )
