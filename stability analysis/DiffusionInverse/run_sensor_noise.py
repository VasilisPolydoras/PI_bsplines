"""Study inverse-diffusion sensitivity to measurement-noise magnitude.

The sensor locations and one standard-normal noise realization are fixed. Each
case changes only the absolute Gaussian-noise standard deviation. The original
DiffusionInverse implementation and its output files are not modified.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import DiffusionInverse.Bspline_2D as inverse_solver
from Diffusion.Bspline_2D import sample_initial, sample_interior
from run_sensor_number import train_one_case
from run_sensor_placement import sample_sensor_locations


NOISE_LEVELS = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05)
BASE_OUTPUT_PATH = HERE / "diffusion_inverse_sensor_noise_results.npz"


def next_output_path() -> Path:
    if not BASE_OUTPUT_PATH.exists():
        return BASE_OUTPUT_PATH
    run_number = 2
    while True:
        candidate = HERE / (
            f"diffusion_inverse_sensor_noise_results_run_{run_number:02d}.npz"
        )
        if not candidate.exists():
            return candidate
        run_number += 1


def main(
    *,
    device_name: str,
    noise_levels: tuple[float, ...],
    sensor_count: int,
    epochs: int,
    initialization_seed: int,
    interior_seed: int,
    sensor_seed: int,
    noise_seed: int,
    validation_seed: int,
) -> Path:
    if sensor_count < 1:
        raise ValueError("sensor_count must be positive")
    if not noise_levels or any(level < 0.0 for level in noise_levels):
        raise ValueError("noise levels must be nonnegative")
    noise_levels = tuple(sorted(set(noise_levels)))

    device = inverse_solver.select_device(device_name)
    config = inverse_solver.CONFIG
    sampling = config["sampling"]
    n_basis, degree = inverse_solver.load_hyperparameters()

    # All non-noise ingredients are generated once and reused in every case.
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
    sensor_points = sample_sensor_locations(
        "uniform", sensor_count, device, sensor_seed
    )
    exact_sensor_values = inverse_solver.exact_solution(
        sensor_points, inverse_solver.TRUE_DIFFUSIVITY
    )
    noise_generator = torch.Generator(device=device).manual_seed(noise_seed)
    standard_noise = torch.randn(
        sensor_count, device=device, generator=noise_generator
    )

    case_count = len(noise_levels)
    estimated = np.empty(case_count, dtype=float)
    parameter_errors = np.empty(case_count, dtype=float)
    field_errors = np.empty(case_count, dtype=float)
    final_losses = np.empty(case_count, dtype=float)
    wall_times = np.empty(case_count, dtype=float)
    completed_epochs = np.empty(case_count, dtype=int)

    print(f"Using device: {device}")
    print(
        f"Tuned B-spline model: n_basis={n_basis}, degree={degree}, "
        f"N_s={sensor_count}"
    )
    print(f"Absolute noise standard deviations: {noise_levels}")
    for index, noise_std in enumerate(noise_levels):
        sensor_values = exact_sensor_values + noise_std * standard_noise
        result = train_one_case(
            sensor_points=sensor_points,
            sensor_values=sensor_values,
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
        estimated[index] = result["estimated_diffusivity"]
        parameter_errors[index] = result["parameter_relative_error"]
        field_errors[index] = result["validation_relative_l2_error"]
        final_losses[index] = result["final_training_loss"]
        wall_times[index] = result["training_time_seconds"]
        completed_epochs[index] = result["completed_epochs"]
        print(
            f"[noise_std={noise_std:.3f}] "
            f"D_hat={result['estimated_diffusivity']:.8f}, "
            f"D error={result['parameter_relative_error']:.6e}, "
            f"L2={result['validation_relative_l2_error']:.6e}, "
            f"epochs={result['completed_epochs']}, "
            f"time={result['training_time_seconds']:.3f} s"
        )

    output_path = next_output_path()
    np.savez_compressed(
        output_path,
        noise_std=np.asarray(noise_levels, dtype=float),
        sensor_count=np.asarray(sensor_count),
        sensor_locations=sensor_points.detach().cpu().numpy(),
        exact_sensor_values=exact_sensor_values.detach().cpu().numpy(),
        standard_normal_noise=standard_noise.detach().cpu().numpy(),
        estimated_diffusivities=estimated,
        parameter_relative_errors=parameter_errors,
        validation_relative_l2_errors=field_errors,
        final_training_losses=final_losses,
        training_wall_time_seconds=wall_times,
        completed_epochs=completed_epochs,
        true_diffusivity=np.asarray(inverse_solver.TRUE_DIFFUSIVITY),
        n_basis=np.asarray(n_basis),
        degree=np.asarray(degree),
        epochs=np.asarray(epochs),
        initialization_seed=np.asarray(initialization_seed),
        interior_seed=np.asarray(interior_seed),
        sensor_seed=np.asarray(sensor_seed),
        noise_seed=np.asarray(noise_seed),
        validation_seed=np.asarray(validation_seed),
    )
    print(f"Saved sensor-noise results: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inverse-diffusion sensitivity to sensor noise"
    )
    parser.add_argument(
        "--device", choices=("cpu", "gpu"), default=inverse_solver.DEFAULT_DEVICE
    )
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=NOISE_LEVELS,
        help="Absolute Gaussian-noise standard deviations",
    )
    parser.add_argument(
        "--sensor-count",
        type=int,
        default=int(inverse_solver.CONFIG["sampling"]["sensor_points"]),
    )
    parser.add_argument("--epochs", type=int, default=inverse_solver.EPOCHS)
    parser.add_argument(
        "--initialization-seed", type=int, default=inverse_solver.SEED
    )
    parser.add_argument("--interior-seed", type=int, default=inverse_solver.SEED)
    parser.add_argument("--sensor-seed", type=int, default=inverse_solver.SEED)
    parser.add_argument("--noise-seed", type=int, default=inverse_solver.SEED)
    parser.add_argument("--validation-seed", type=int, default=inverse_solver.SEED)
    arguments = parser.parse_args()
    main(
        device_name=arguments.device,
        noise_levels=tuple(arguments.noise_levels),
        sensor_count=arguments.sensor_count,
        epochs=arguments.epochs,
        initialization_seed=arguments.initialization_seed,
        interior_seed=arguments.interior_seed,
        sensor_seed=arguments.sensor_seed,
        noise_seed=arguments.noise_seed,
        validation_seed=arguments.validation_seed,
    )
