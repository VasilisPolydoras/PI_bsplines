"""Study inverse-diffusion sensitivity to four sensor placements.

The original DiffusionInverse implementation is imported without modification.
All cases use the same sensor count, training configuration, initialization,
PDE collocation points, validation points, and noise level. Only the sensor
locations change. Results are stored locally and never overwrite earlier runs.
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


PLACEMENT_NAMES = ("uniform", "clustered", "early", "late")
PLACEMENT_LABELS = ("Uniform", "Clustered", "Early time", "Late time")
BASE_OUTPUT_PATH = HERE / "diffusion_inverse_sensor_placement_results.npz"


def next_output_path() -> Path:
    if not BASE_OUTPUT_PATH.exists():
        return BASE_OUTPUT_PATH
    run_number = 2
    while True:
        candidate = HERE / (
            f"diffusion_inverse_sensor_placement_results_run_{run_number:02d}.npz"
        )
        if not candidate.exists():
            return candidate
        run_number += 1


def sample_sensor_locations(
    placement: str,
    count: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Generate one of the four prescribed space-time sensor layouts."""
    generator = torch.Generator(device=device).manual_seed(seed)
    if placement == "clustered":
        center = torch.tensor((0.25, 0.50), device=device)
        points = center + 0.10 * torch.randn(
            count, 2, device=device, generator=generator
        )
        return points.clamp(0.0, 1.0)

    points = torch.rand(count, 2, device=device, generator=generator)
    if placement == "uniform":
        return points
    if placement == "early":
        points[:, 1] = 0.05 + 0.20 * points[:, 1]
        return points
    if placement == "late":
        points[:, 1] = 0.75 + 0.25 * points[:, 1]
        return points
    raise ValueError(f"Unknown sensor placement: {placement}")


def observations_at(
    points: torch.Tensor,
    noise_std: float,
    noise_seed: int,
) -> torch.Tensor:
    values = inverse_solver.exact_solution(
        points, inverse_solver.TRUE_DIFFUSIVITY
    )
    if noise_std > 0.0:
        generator = torch.Generator(device=points.device).manual_seed(noise_seed)
        values = values + noise_std * torch.randn(
            points.shape[0], device=points.device, generator=generator
        )
    return values


def main(
    *,
    device_name: str,
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
    device = inverse_solver.select_device(device_name)
    config = inverse_solver.CONFIG
    sampling = config["sampling"]
    noise_std = float(config["inverse"]["noise_std"])
    n_basis, degree = inverse_solver.load_hyperparameters()

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

    count = len(PLACEMENT_NAMES)
    estimated = np.empty(count, dtype=float)
    parameter_errors = np.empty(count, dtype=float)
    field_errors = np.empty(count, dtype=float)
    final_losses = np.empty(count, dtype=float)
    wall_times = np.empty(count, dtype=float)
    completed_epochs = np.empty(count, dtype=int)
    sensor_locations = np.empty((count, sensor_count, 2), dtype=float)
    sensor_values = np.empty((count, sensor_count), dtype=float)

    print(f"Using device: {device}")
    print(
        f"Tuned B-spline model: n_basis={n_basis}, degree={degree}, "
        f"N_s={sensor_count}, noise_std={noise_std:g}"
    )
    for index, (placement, label) in enumerate(
        zip(PLACEMENT_NAMES, PLACEMENT_LABELS)
    ):
        points = sample_sensor_locations(
            placement, sensor_count, device, sensor_seed
        )
        values = observations_at(points, noise_std, noise_seed)
        result = train_one_case(
            sensor_points=points,
            sensor_values=values,
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
        sensor_locations[index] = points.detach().cpu().numpy()
        sensor_values[index] = values.detach().cpu().numpy()
        print(
            f"[{label:10s}] D_hat={result['estimated_diffusivity']:.8f}, "
            f"D error={result['parameter_relative_error']:.6e}, "
            f"L2={result['validation_relative_l2_error']:.6e}, "
            f"epochs={result['completed_epochs']}, "
            f"time={result['training_time_seconds']:.3f} s"
        )

    output_path = next_output_path()
    np.savez_compressed(
        output_path,
        placement_names=np.asarray(PLACEMENT_NAMES),
        placement_labels=np.asarray(PLACEMENT_LABELS),
        sensor_count=np.asarray(sensor_count),
        sensor_locations=sensor_locations,
        sensor_values=sensor_values,
        estimated_diffusivities=estimated,
        parameter_relative_errors=parameter_errors,
        validation_relative_l2_errors=field_errors,
        final_training_losses=final_losses,
        training_wall_time_seconds=wall_times,
        completed_epochs=completed_epochs,
        true_diffusivity=np.asarray(inverse_solver.TRUE_DIFFUSIVITY),
        n_basis=np.asarray(n_basis),
        degree=np.asarray(degree),
        noise_std=np.asarray(noise_std),
        epochs=np.asarray(epochs),
        initialization_seed=np.asarray(initialization_seed),
        interior_seed=np.asarray(interior_seed),
        sensor_seed=np.asarray(sensor_seed),
        noise_seed=np.asarray(noise_seed),
        validation_seed=np.asarray(validation_seed),
        cluster_center=np.asarray((0.25, 0.50)),
        cluster_standard_deviation=np.asarray(0.10),
        early_time_interval=np.asarray((0.05, 0.25)),
        late_time_interval=np.asarray((0.75, 1.00)),
    )
    print(f"Saved sensor-placement results: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inverse-diffusion sensitivity to sensor placement"
    )
    parser.add_argument(
        "--device", choices=("cpu", "gpu"), default=inverse_solver.DEFAULT_DEVICE
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
        sensor_count=arguments.sensor_count,
        epochs=arguments.epochs,
        initialization_seed=arguments.initialization_seed,
        interior_seed=arguments.interior_seed,
        sensor_seed=arguments.sensor_seed,
        noise_seed=arguments.noise_seed,
        validation_seed=arguments.validation_seed,
    )
