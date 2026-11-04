"""Profile the diffusion coefficient without modifying DiffusionInverse.

For every fixed candidate diffusivity, only the free tensor-product B-spline
coefficients are optimized.  All candidates use the same collocation points,
sensor locations, observations, initialization seed, and validation points.
Results are saved beside this script and never overwrite an earlier run.
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
    DiffusionBSpline,
    pretrain_initial_coefficients,
    sample_initial,
    sample_interior,
)
from training import train


BASE_OUTPUT_PATH = HERE / "diffusion_inverse_identifiability_results.npz"
DEFAULT_CANDIDATES = tuple(np.arange(0.005, 0.080 + 0.0025, 0.005))


def next_output_path() -> Path:
    """Return a new result path without replacing an existing experiment."""
    if not BASE_OUTPUT_PATH.exists():
        return BASE_OUTPUT_PATH
    run_number = 2
    while True:
        candidate = HERE / (
            f"diffusion_inverse_identifiability_results_run_{run_number:02d}.npz"
        )
        if not candidate.exists():
            return candidate
        run_number += 1


def loss_components(
    model: DiffusionBSpline,
    diffusivity: float,
    interior_operators,
    sensor_operators,
    sensor_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    residual = (
        model.evaluate(interior_operators, 0, 1)
        - diffusivity * model.evaluate(interior_operators, 2, 0)
    )
    data_error = model.evaluate(sensor_operators) - sensor_values
    pde_loss = residual.square().mean()
    data_loss = data_error.square().mean()
    weights = inverse_solver.CONFIG["loss_weights"]
    total_loss = (
        float(weights["pde"]) * pde_loss
        + float(weights["data"]) * data_loss
    )
    return total_loss, pde_loss, data_loss


def main(
    device_name: str,
    diffusivities: tuple[float, ...],
    epochs: int,
    initialization_seed: int,
    sampling_seed: int,
    validation_seed: int,
) -> Path:
    if not diffusivities:
        raise ValueError("At least one candidate diffusivity is required")
    if any(value <= 0.0 for value in diffusivities):
        raise ValueError("Every candidate diffusivity must be positive")

    device = inverse_solver.select_device(device_name)
    n_basis, degree = inverse_solver.load_hyperparameters()
    config = inverse_solver.CONFIG
    sampling = config["sampling"]
    training = config["training"]
    initialization = str(config["model"]["interior_initialization"])
    true_diffusivity = inverse_solver.TRUE_DIFFUSIVITY

    # These tensors are generated once and then reused unchanged for every D_k.
    sampling_generator = torch.Generator(device=device).manual_seed(sampling_seed)
    interior = sample_interior(
        int(sampling["interior_points"]), device, sampling_generator
    )
    sensors = inverse_solver.sample_observations(
        int(sampling["sensor_points"]),
        float(config["inverse"]["noise_std"]),
        true_diffusivity,
        device,
        sampling_generator,
    )
    initial_points, initial_targets = sample_initial(
        int(sampling["initial_points"]), device
    )
    validation_generator = torch.Generator(device=device).manual_seed(validation_seed)
    validation = sample_interior(
        int(sampling["validation_points"]), device, validation_generator
    )

    total_losses = []
    pde_losses = []
    data_losses = []
    field_errors = []
    wall_times = []
    completed_epochs = []
    initial_losses = []

    print(f"Using device: {device}")
    print(
        f"Tuned B-spline model: n_basis={n_basis}, degree={degree}, "
        f"initialization={initialization}"
    )
    print(
        f"Fixed samples: N_Omega={interior.shape[0]}, "
        f"N_s={sensors[0].shape[0]}, N_val={validation.shape[0]}"
    )
    print(f"True diffusivity: D_true={true_diffusivity:.8f}")

    for index, diffusivity in enumerate(diffusivities, start=1):
        inverse_solver.set_seed(initialization_seed)
        model = DiffusionBSpline(
            n_basis, degree, device, interior_initialization=initialization
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
            minimum_learning_rate=float(
                training["scheduler"]["min_learning_rate"]
            ),
            early_stopping_patience=int(
                training["early_stopping"]["patience"]
            ),
            min_delta=float(training["early_stopping"]["min_delta"]),
            print_every=0,
        )

        interior_operators = model.operators(interior)
        sensor_operators = model.operators(sensors[0], max_x=0, max_t=0)
        optimizer = torch.optim.Adam(
            [model.interior_coeffs], lr=float(training["learning_rate"])
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=float(training["scheduler"]["factor"]),
            patience=int(training["scheduler"]["patience"]),
            min_lr=float(training["scheduler"]["min_learning_rate"]),
        )

        def objective() -> torch.Tensor:
            return loss_components(
                model,
                float(diffusivity),
                interior_operators,
                sensor_operators,
                sensors[1],
            )[0]

        history = train(
            [model.interior_coeffs],
            objective,
            optimizer,
            epochs=epochs,
            scheduler=scheduler,
            early_stopping_patience=int(
                training["early_stopping"]["patience"]
            ),
            min_delta=float(training["early_stopping"]["min_delta"]),
            print_every=0,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        wall_time = time.perf_counter() - start

        with torch.no_grad():
            total, pde, data = loss_components(
                model,
                float(diffusivity),
                interior_operators,
                sensor_operators,
                sensors[1],
            )
            field_error = inverse_solver.relative_l2_error(
                model, validation, true_diffusivity
            )
            initial_error = (model(initial_points) - initial_targets).square().mean()

        total_losses.append(float(total))
        pde_losses.append(float(pde))
        data_losses.append(float(data))
        field_errors.append(float(field_error))
        wall_times.append(wall_time)
        completed_epochs.append(len(history))
        initial_losses.append(float(initial_error))
        print(
            f"[{index:02d}/{len(diffusivities):02d}] D={diffusivity:.6f}: "
            f"profile={float(total):.6e}, PDE={float(pde):.6e}, "
            f"data={float(data):.6e}, L2={float(field_error):.6e}, "
            f"epochs={len(history)}, time={wall_time:.3f} s"
        )

    diffusivity_array = np.asarray(diffusivities, dtype=float)
    total_array = np.asarray(total_losses, dtype=float)
    minimum_index = int(np.argmin(total_array))
    output_path = next_output_path()
    np.savez_compressed(
        output_path,
        candidate_diffusivities=diffusivity_array,
        true_diffusivity=np.asarray(true_diffusivity),
        profile_total_loss=total_array,
        profile_pde_loss=np.asarray(pde_losses, dtype=float),
        profile_data_loss=np.asarray(data_losses, dtype=float),
        validation_relative_l2_errors=np.asarray(field_errors, dtype=float),
        initial_condition_losses=np.asarray(initial_losses, dtype=float),
        training_wall_time_seconds=np.asarray(wall_times, dtype=float),
        completed_epochs=np.asarray(completed_epochs, dtype=int),
        profile_minimizer=np.asarray(diffusivity_array[minimum_index]),
        n_basis=np.asarray(n_basis),
        degree=np.asarray(degree),
        interior_initialization=np.asarray(initialization),
        interior_points=np.asarray(interior.shape[0]),
        sensor_points=np.asarray(sensors[0].shape[0]),
        validation_points=np.asarray(validation.shape[0]),
        noise_std=np.asarray(float(config["inverse"]["noise_std"])),
        pde_weight=np.asarray(float(config["loss_weights"]["pde"])),
        data_weight=np.asarray(float(config["loss_weights"]["data"])),
        initialization_seed=np.asarray(initialization_seed),
        sampling_seed=np.asarray(sampling_seed),
        validation_seed=np.asarray(validation_seed),
        sensor_locations=sensors[0].detach().cpu().numpy(),
        sensor_values=sensors[1].detach().cpu().numpy(),
    )
    print(
        f"Profile minimum on tested grid: D={diffusivity_array[minimum_index]:.8f} "
        f"(D_true={true_diffusivity:.8f})"
    )
    print(f"Saved identifiability results: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fixed-D profile-loss study for inverse diffusion"
    )
    parser.add_argument(
        "--device", choices=("cpu", "gpu"), default=inverse_solver.DEFAULT_DEVICE
    )
    parser.add_argument(
        "--diffusivities",
        type=float,
        nargs="+",
        default=DEFAULT_CANDIDATES,
        help="Fixed candidate diffusivities (default: 0.005 to 0.080 by 0.005)",
    )
    parser.add_argument("--epochs", type=int, default=inverse_solver.EPOCHS)
    parser.add_argument(
        "--initialization-seed", type=int, default=inverse_solver.SEED
    )
    parser.add_argument("--sampling-seed", type=int, default=inverse_solver.SEED)
    parser.add_argument("--validation-seed", type=int, default=inverse_solver.SEED)
    arguments = parser.parse_args()
    main(
        device_name=arguments.device,
        diffusivities=tuple(arguments.diffusivities),
        epochs=arguments.epochs,
        initialization_seed=arguments.initialization_seed,
        sampling_seed=arguments.sampling_seed,
        validation_seed=arguments.validation_seed,
    )
