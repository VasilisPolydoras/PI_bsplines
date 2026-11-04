"""Physics-informed B-spline identification of the wave speed."""

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

from WaveEquation.Bspline_2D import (
    WaveBSpline,
    pretrain_initial_coefficients,
    sample_initial,
    sample_interior,
)
from training import train

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "bspline_config.json").read_text(encoding="utf-8"))
RESULT_PATH = ROOT / CONFIG["output"]["tuned_hyperparameters"]
SEED = int(CONFIG["reproducibility"]["seed"])
DEFAULT_DEVICE = str(CONFIG["runtime"]["device"])
EPOCHS = int(CONFIG["training"]["epochs"])
TRUE_WAVE_SPEED = float(CONFIG["problem"]["true_wave_speed"])


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
        raise RuntimeError("GPU requested, but CUDA is unavailable")
    raise ValueError("device must be 'cpu' or 'gpu'")


def exact_solution(points: torch.Tensor, wave_speed: float) -> torch.Tensor:
    x, time_values = points[:, 0], points[:, 1]
    return torch.sin(torch.pi * x) * torch.cos(
        torch.pi * wave_speed * time_values
    )


def sample_observations(count, noise_std, wave_speed, device, generator):
    points = sample_interior(count, device, generator)
    values = exact_solution(points, wave_speed)
    if noise_std:
        values = values + noise_std * torch.randn(
            count, device=device, generator=generator
        )
    return points, values


def raw_positive(value: float, minimum: float, device) -> torch.nn.Parameter:
    shifted = max(value - minimum, torch.finfo(torch.float32).eps)
    return torch.nn.Parameter(
        torch.tensor(math.log(math.expm1(shifted)), device=device)
    )


class InverseWaveBSpline(WaveBSpline):
    constraint_mode = "hard"

    def __init__(self, n_basis, degree, device, initialization, inverse_config):
        super().__init__(n_basis, degree, device, initialization)
        self.minimum_wave_speed = float(inverse_config["minimum_wave_speed"])
        self.raw_wave_speed = raw_positive(
            float(inverse_config["initial_wave_speed"]),
            self.minimum_wave_speed,
            device,
        )

    def wave_speed(self) -> torch.Tensor:
        return self.minimum_wave_speed + functional.softplus(
            self.raw_wave_speed
        )


def inverse_loss(model, interior_operators, sensor_operators, sensor_values):
    residual = (
        model.evaluate(interior_operators, 0, 2)
        - model.wave_speed().square()
        * model.evaluate(interior_operators, 2, 0)
    )
    data_error = model.evaluate(sensor_operators) - sensor_values
    weights = CONFIG["loss_weights"]
    return (
        float(weights["pde"]) * residual.square().mean()
        + float(weights["data"]) * data_error.square().mean()
    )


def relative_l2_error(model, points, wave_speed=TRUE_WAVE_SPEED):
    with torch.no_grad():
        prediction = model(points)
        reference = exact_solution(points, wave_speed)
    return torch.linalg.vector_norm(prediction - reference) / torch.linalg.vector_norm(
        reference
    )


def load_hyperparameters(path=RESULT_PATH):
    if not path.exists():
        raise FileNotFoundError(f"No inverse wave B-spline result found at {path}")
    parameters = json.loads(path.read_text(encoding="utf-8"))["best_params"]
    return int(parameters["n_basis"]), int(parameters["degree"])


def save_results(model, histories, validation_error, wall_time, sensors):
    output = ROOT / CONFIG["output"]["postprocessing_directory"]
    output.mkdir(parents=True, exist_ok=True)
    sampling = CONFIG["sampling"]
    x = np.linspace(*CONFIG["problem"]["x_bounds"], sampling["plot_spatial_points"])
    times = np.linspace(*CONFIG["problem"]["time_bounds"], sampling["plot_time_points"])
    grid_x, grid_t = np.meshgrid(x, times)
    device = model.interior_coeffs.device
    points = torch.tensor(
        np.column_stack((grid_x.ravel(), grid_t.ravel())),
        dtype=torch.float32, device=device,
    )
    with torch.no_grad():
        prediction = model(points).cpu().numpy().reshape(grid_x.shape)
        reference = exact_solution(
            points, TRUE_WAVE_SPEED
        ).cpu().numpy().reshape(grid_x.shape)
    estimated = float(model.wave_speed().detach())
    stem = "bspline_wave_inverse_hard"
    np.savez_compressed(
        output / f"{stem}_solution_data.npz", x=x, t=times,
        prediction=prediction, reference=reference,
        absolute_error=np.abs(prediction - reference),
        sensor_points=sensors[0].detach().cpu().numpy(),
        sensor_values=sensors[1].detach().cpu().numpy(),
        estimated_wave_speed=estimated, true_wave_speed=TRUE_WAVE_SPEED,
        parameter_relative_error=abs(estimated - TRUE_WAVE_SPEED) / TRUE_WAVE_SPEED,
        validation_relative_l2_error=validation_error,
        training_wall_time_seconds=wall_time,
    )
    initial_history, history, speed_history = histories
    np.savez_compressed(
        output / f"{stem}_training_data.npz",
        initial_epoch=np.arange(1, len(initial_history) + 1),
        initial_loss=np.asarray(initial_history),
        epoch=np.arange(1, len(history) + 1), loss=np.asarray(history),
        estimated_wave_speed=np.asarray(speed_history),
        training_wall_time_seconds=wall_time,
    )


def main(
    device_name=DEFAULT_DEVICE, n_basis=None, degree=None, epochs=EPOCHS,
    use_tuned=None, interior_initialization=None, initialization_seed=SEED,
    sampling_seed=SEED, validation_seed=None, save_outputs=True, verbose=True,
):
    set_seed(initialization_seed)
    device = select_device(device_name)
    model_config = CONFIG["model"]
    n_basis = int(model_config["n_basis"] if n_basis is None else n_basis)
    degree = int(model_config["degree"] if degree is None else degree)
    initialization = interior_initialization or model_config["interior_initialization"]
    should_load = (
        CONFIG["runtime"]["load_tuned_if_available"] and RESULT_PATH.exists()
        if use_tuned is None else use_tuned
    )
    if should_load:
        n_basis, degree = load_hyperparameters()
    generator = torch.Generator(device=device).manual_seed(sampling_seed)
    sampling = CONFIG["sampling"]
    interior = sample_interior(sampling["interior_points"], device, generator)
    sensors = sample_observations(
        sampling["sensor_points"], CONFIG["inverse"]["noise_std"],
        TRUE_WAVE_SPEED, device, generator,
    )
    initial_points, initial_targets = sample_initial(
        sampling["initial_points"], device
    )
    model = InverseWaveBSpline(
        n_basis, degree, device, initialization, CONFIG["inverse"]
    )
    training = CONFIG["training"]
    if str(training["optimizer"]).lower() != "adam":
        raise ValueError("Only the Adam optimizer is supported")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    initial_history = pretrain_initial_coefficients(
        model, initial_points, initial_targets, epochs=epochs,
        learning_rate=float(training["learning_rate"]),
        scheduler_factor=float(training["scheduler"]["factor"]),
        scheduler_patience=int(training["scheduler"]["patience"]),
        minimum_learning_rate=float(training["scheduler"]["min_learning_rate"]),
        early_stopping_patience=int(training["early_stopping"]["patience"]),
        min_delta=float(training["early_stopping"]["min_delta"]),
        print_every=int(training["print_every"] if verbose else 0),
    )
    interior_operators = model.operators(interior)
    sensor_operators = model.operators(sensors[0], max_x=0, max_t=0)
    parameters = [model.interior_coeffs, model.raw_wave_speed]
    optimizer = torch.optim.Adam(parameters, lr=float(training["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=float(training["scheduler"]["factor"]),
        patience=int(training["scheduler"]["patience"]),
        min_lr=float(training["scheduler"]["min_learning_rate"]),
    )
    speed_history = []

    def loss_function():
        value = inverse_loss(model, interior_operators, sensor_operators, sensors[1])
        speed_history.append(float(model.wave_speed().detach()))
        return value

    history = train(
        parameters, loss_function, optimizer, epochs=epochs, scheduler=scheduler,
        early_stopping_patience=int(training["early_stopping"]["patience"]),
        min_delta=float(training["early_stopping"]["min_delta"]),
        print_every=int(training["print_every"] if verbose else 0),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_time = time.perf_counter() - start
    validation_generator = torch.Generator(device=device).manual_seed(
        sampling_seed if validation_seed is None else validation_seed
    )
    validation = sample_interior(
        sampling["validation_points"], device, validation_generator
    )
    field_error = float(relative_l2_error(model, validation))
    estimated = float(model.wave_speed().detach())
    parameter_error = abs(estimated - TRUE_WAVE_SPEED) / TRUE_WAVE_SPEED
    if verbose:
        print(f"Estimated wave speed: {estimated:.8f} (true {TRUE_WAVE_SPEED:.8f})")
        print(f"Parameter relative error: {parameter_error:.6e}")
        print(f"Validation relative L2 error: {field_error:.6e}")
        print(f"Training wall time: {wall_time:.3f} s")
    if save_outputs:
        save_results(
            model, (initial_history, history, speed_history),
            field_error, wall_time, sensors,
        )
    return {
        "validation_relative_l2_error": field_error,
        "parameter_relative_error": parameter_error,
        "estimated_wave_speed": estimated,
        "training_time_seconds": wall_time,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inverse wave B-spline solver")
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--no-use-tuned", action="store_true")
    arguments = parser.parse_args()
    main(
        arguments.device, epochs=arguments.epochs,
        use_tuned=False if arguments.no_use_tuned else None,
    )
