"""Study inverse-diffusion sensitivity to the number of sensors.

The original DiffusionInverse files are imported read-only. One fixed pool of
250 observations is generated and the smaller sensor sets are nested prefixes
of that same pool. Only the number of observations changes; the PDE points,
validation points, noise level,
model, optimizer settings, initialization, and tuned hyperparameters remain
fixed. Results are saved locally without overwriting earlier experiments.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import DiffusionInverse.Bspline_2D as inverse_solver
from Diffusion.Bspline_2D import (
    pretrain_initial_coefficients,
    sample_initial,
    sample_interior,
)
from training import train


SENSOR_COUNTS = (5, 10, 20, 50, 100, 250)
BASE_OUTPUT_PATH = HERE / "diffusion_inverse_sensor_number_results.npz"


def next_output_path() -> Path:
    if not BASE_OUTPUT_PATH.exists():
        return BASE_OUTPUT_PATH
    run_number = 2
    while True:
        candidate = HERE / (
            f"diffusion_inverse_sensor_number_results_run_{run_number:02d}.npz"
        )
        if not candidate.exists():
            return candidate
        run_number += 1


def train_one_case(
    *,
    sensor_points: torch.Tensor,
    sensor_values: torch.Tensor,
    interior: torch.Tensor,
    validation: torch.Tensor,
    initial_points: torch.Tensor,
    initial_targets: torch.Tensor,
    n_basis: int,
    degree: int,
    epochs: int,
    initialization_seed: int,
    device: torch.device,
) -> dict[str, float | int]:
    """Train D and the free coefficients for one fixed sensor subset."""
    config = inverse_solver.CONFIG
    training = config["training"]
    inverse_solver.set_seed(initialization_seed)
    model = inverse_solver.InverseDiffusionBSpline(
        n_basis,
        degree,
        device,
        str(config["model"]["interior_initialization"]),
        config["inverse"],
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    pretrain_initial_coefficients(
        model,
        initial_points,
        initial_targets,
        epochs=epochs,
        learning_rate=float(training["learning_rate"]),
        scheduler_factor=float(training["scheduler"]["factor"]),
        scheduler_patience=int(training["scheduler"]["patience"]),
        minimum_learning_rate=float(training["scheduler"]["min_learning_rate"]),
        early_stopping_patience=int(training["early_stopping"]["patience"]),
        min_delta=float(training["early_stopping"]["min_delta"]),
        print_every=0,
    )
    interior_operators = model.operators(interior)
    sensor_operators = model.operators(sensor_points, max_x=0, max_t=0)
    parameters = [model.interior_coeffs, model.raw_diffusivity]
    optimizer = torch.optim.Adam(
        parameters, lr=float(training["learning_rate"])
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=float(training["scheduler"]["factor"]),
        patience=int(training["scheduler"]["patience"]),
        min_lr=float(training["scheduler"]["min_learning_rate"]),
    )
    history = train(
        parameters,
        lambda: inverse_solver.inverse_loss(
            model, interior_operators, sensor_operators, sensor_values
        ),
        optimizer,
        epochs=epochs,
        scheduler=scheduler,
        early_stopping_patience=int(training["early_stopping"]["patience"]),
        min_delta=float(training["early_stopping"]["min_delta"]),
        print_every=0,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_time = time.perf_counter() - start

    estimated = float(model.diffusivity().detach())
    true_diffusivity = inverse_solver.TRUE_DIFFUSIVITY
    parameter_error = abs(estimated - true_diffusivity) / true_diffusivity
    field_error = float(
        inverse_solver.relative_l2_error(model, validation, true_diffusivity)
    )
    with torch.no_grad():
        final_loss = float(
            inverse_solver.inverse_loss(
                model, interior_operators, sensor_operators, sensor_values
            )
        )
    return {
        "estimated_diffusivity": estimated,
        "parameter_relative_error": parameter_error,
        "validation_relative_l2_error": field_error,
        "final_training_loss": final_loss,
        "training_time_seconds": wall_time,
        "completed_epochs": len(history),
    }


def main(
    *,
    device_name: str,
    sensor_counts: tuple[int, ...],
    epochs: int,
    initialization_seed: int,
    interior_seed: int,
    sensor_seed: int,
    validation_seed: int,
) -> Path:
    if not sensor_counts or any(count < 1 for count in sensor_counts):
        raise ValueError("sensor counts must be positive")
    sensor_counts = tuple(sorted(set(sensor_counts)))
    maximum_sensor_count = max(sensor_counts)

    device = inverse_solver.select_device(device_name)
    config = inverse_solver.CONFIG
    sampling = config["sampling"]
    n_basis, degree = inverse_solver.load_hyperparameters()

    # PDE, IC, and validation samples are fixed for the complete study.
    interior_generator = torch.Generator(device=device).manual_seed(interior_seed)
    interior = sample_interior(
        int(sampling["interior_points"]), device, interior_generator
    )
    initial_points, initial_targets = sample_initial(
        int(sampling["initial_points"]), device
    )
    validation_generator = torch.Generator(device=device).manual_seed(validation_seed)
    validation = sample_interior(
        int(sampling["validation_points"]), device, validation_generator
    )

    result_shape = (len(sensor_counts),)
    estimated_diffusivities = np.empty(result_shape, dtype=float)
    parameter_errors = np.empty(result_shape, dtype=float)
    field_errors = np.empty(result_shape, dtype=float)
    final_losses = np.empty(result_shape, dtype=float)
    wall_times = np.empty(result_shape, dtype=float)
    completed_epochs = np.empty(result_shape, dtype=int)

    print(f"Using device: {device}")
    print(
        f"Tuned B-spline model: n_basis={n_basis}, degree={degree}, "
        f"initialization={config['model']['interior_initialization']}"
    )
    print(
        f"Sensor counts: {sensor_counts}; sensor_seed={sensor_seed}; "
        f"noise_std={float(config['inverse']['noise_std']):g}"
    )

    sensor_generator = torch.Generator(device=device).manual_seed(sensor_seed)
    sensor_pool, value_pool = inverse_solver.sample_observations(
        maximum_sensor_count,
        float(config["inverse"]["noise_std"]),
        inverse_solver.TRUE_DIFFUSIVITY,
        device,
        sensor_generator,
    )
    for count_index, sensor_count in enumerate(sensor_counts):
        result = train_one_case(
            sensor_points=sensor_pool[:sensor_count],
            sensor_values=value_pool[:sensor_count],
            interior=interior,
            validation=validation,
            initial_points=initial_points,
            initial_targets=initial_targets,
            n_basis=n_basis,
            degree=degree,
            epochs=epochs,
            initialization_seed=initialization_seed,
            device=device,
        )
        estimated_diffusivities[count_index] = result["estimated_diffusivity"]
        parameter_errors[count_index] = result["parameter_relative_error"]
        field_errors[count_index] = result["validation_relative_l2_error"]
        final_losses[count_index] = result["final_training_loss"]
        wall_times[count_index] = result["training_time_seconds"]
        completed_epochs[count_index] = result["completed_epochs"]
        print(
            f"[N_s={sensor_count:3d}] "
            f"D_hat={result['estimated_diffusivity']:.8f}, "
            f"D error={result['parameter_relative_error']:.6e}, "
            f"L2={result['validation_relative_l2_error']:.6e}, "
            f"epochs={result['completed_epochs']}, "
            f"time={result['training_time_seconds']:.3f} s"
        )

    output_path = next_output_path()
    np.savez_compressed(
        output_path,
        sensor_counts=np.asarray(sensor_counts, dtype=int),
        estimated_diffusivities=estimated_diffusivities,
        parameter_relative_errors=parameter_errors,
        validation_relative_l2_errors=field_errors,
        final_training_losses=final_losses,
        training_wall_time_seconds=wall_times,
        completed_epochs=completed_epochs,
        true_diffusivity=np.asarray(inverse_solver.TRUE_DIFFUSIVITY),
        n_basis=np.asarray(n_basis),
        degree=np.asarray(degree),
        noise_std=np.asarray(float(config["inverse"]["noise_std"])),
        epochs=np.asarray(epochs),
        initialization_seed=np.asarray(initialization_seed),
        interior_seed=np.asarray(interior_seed),
        sensor_seed=np.asarray(sensor_seed),
        validation_seed=np.asarray(validation_seed),
        master_sensor_locations=sensor_pool.detach().cpu().numpy(),
        master_sensor_values=value_pool.detach().cpu().numpy(),
    )
    print(f"Saved sensor-number results: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inverse-diffusion sensitivity to sensor count"
    )
    parser.add_argument(
        "--device", choices=("cpu", "gpu"), default=inverse_solver.DEFAULT_DEVICE
    )
    parser.add_argument(
        "--sensor-counts",
        type=int,
        nargs="+",
        default=SENSOR_COUNTS,
        help="Nested sensor counts (default: 5 10 20 50 100 250)",
    )
    parser.add_argument("--epochs", type=int, default=inverse_solver.EPOCHS)
    parser.add_argument(
        "--initialization-seed", type=int, default=inverse_solver.SEED
    )
    parser.add_argument("--interior-seed", type=int, default=inverse_solver.SEED)
    parser.add_argument("--sensor-seed", type=int, default=inverse_solver.SEED)
    parser.add_argument("--validation-seed", type=int, default=inverse_solver.SEED)
    arguments = parser.parse_args()
    main(
        device_name=arguments.device,
        sensor_counts=tuple(arguments.sensor_counts),
        epochs=arguments.epochs,
        initialization_seed=arguments.initialization_seed,
        interior_seed=arguments.interior_seed,
        sensor_seed=arguments.sensor_seed,
        validation_seed=arguments.validation_seed,
    )
