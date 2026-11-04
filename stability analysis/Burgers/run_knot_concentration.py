"""Isolated Burgers B-spline knot-concentration sensitivity study.

The tuned basis dimensions and degree are fixed while only the spatial knot
concentration parameter changes. The original Burgers files and their normal
postprocessing outputs are not modified. Results are saved beside this script
without overwriting earlier experiments.
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

import Burgers.Bspline_2D as burgers_solver
from training import train


BASE_OUTPUT_PATH = HERE / "burgers_knot_concentration_results.npz"


def next_output_path() -> Path:
    if not BASE_OUTPUT_PATH.exists():
        return BASE_OUTPUT_PATH
    run_number = 2
    while True:
        candidate = HERE / (
            f"burgers_knot_concentration_results_run_{run_number:02d}.npz"
        )
        if not candidate.exists():
            return candidate
        run_number += 1


def positive_knot_spans(knots: torch.Tensor) -> np.ndarray:
    unique_knots = np.unique(knots.detach().cpu().numpy())
    spans = np.diff(unique_knots)
    return spans[spans > 10.0 * np.finfo(spans.dtype).eps]


def relative_l2(prediction: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(prediction - reference)
        / torch.linalg.vector_norm(reference)
    )


def main(
    *,
    device_name: str,
    concentrations: tuple[float, ...] | None,
    epochs: int,
    local_half_width: float,
    initialization_seed: int,
    sampling_seed: int,
    validation_seed: int,
) -> Path:
    if local_half_width <= 0.0:
        raise ValueError("local_half_width must be positive")
    device = burgers_solver.select_device(device_name)
    n_basis_x, n_basis_t, degree, tuned_concentration = (
        burgers_solver.load_best_hyperparameters()
    )
    if concentrations is None:
        concentrations = tuple(
            sorted({0.0, 1.0, 2.0, 3.0, tuned_concentration, 4.0, 5.0, 6.0})
        )
    else:
        concentrations = tuple(sorted(set(concentrations)))
    if not concentrations or any(value < 0.0 for value in concentrations):
        raise ValueError("concentrations must be nonnegative")

    generator = torch.Generator(device=device).manual_seed(sampling_seed)
    interior = burgers_solver.sample_interior(
        burgers_solver.N_INTERIOR_POINTS, device, generator
    )
    initial_points, initial_targets = burgers_solver.sample_initial_condition(
        burgers_solver.N_INITIAL_POINTS, device
    )
    validation_generator = torch.Generator(device=device).manual_seed(
        validation_seed
    )
    validation = burgers_solver.sample_interior(
        burgers_solver.N_VALIDATION_POINTS, device, validation_generator
    )
    print("Computing the fixed validation reference solution")
    validation_reference = burgers_solver.numerical_reference_at_points(validation)
    local_mask = validation[:, 0].abs() <= local_half_width
    local_count = int(local_mask.sum().item())
    if local_count == 0:
        raise RuntimeError("The fixed validation set contains no local points")

    case_count = len(concentrations)
    global_errors = np.empty(case_count, dtype=float)
    local_errors = np.empty(case_count, dtype=float)
    final_pde_losses = np.empty(case_count, dtype=float)
    initial_losses = np.empty(case_count, dtype=float)
    wall_times = np.empty(case_count, dtype=float)
    completed_epochs = np.empty(case_count, dtype=int)
    minimum_spans = np.empty(case_count, dtype=float)
    maximum_spans = np.empty(case_count, dtype=float)

    print(f"Using device: {device}")
    print(
        f"Fixed tuned model: n_basis_x={n_basis_x}, n_basis_t={n_basis_t}, "
        f"degree={degree}; tuned concentration={tuned_concentration:g}"
    )
    print(
        f"Concentrations: {concentrations}; local region: "
        f"|x| <= {local_half_width:g} ({local_count} validation points)"
    )

    training = burgers_solver.CONFIG["training"]
    for index, concentration in enumerate(concentrations):
        burgers_solver.set_seed(initialization_seed)
        model = burgers_solver.BurgersBSpline(
            n_basis_x,
            n_basis_t,
            degree,
            device,
            spatial_clustering=float(concentration),
            interior_initialization=burgers_solver.INTERIOR_INITIALIZATION,
        )
        spans = positive_knot_spans(model.knot_x)
        minimum_spans[index] = float(spans.min())
        maximum_spans[index] = float(spans.max())

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        burgers_solver.pretrain_initial_coefficients(
            model,
            initial_points,
            initial_targets,
            epochs=epochs,
            learning_rate=burgers_solver.INTERIOR_LEARNING_RATE,
            scheduler_factor=burgers_solver.SCHEDULER_FACTOR,
            scheduler_patience=burgers_solver.SCHEDULER_PATIENCE,
            minimum_learning_rate=burgers_solver.MIN_LEARNING_RATE,
            early_stopping_patience=burgers_solver.EARLY_STOPPING_PATIENCE,
            min_delta=burgers_solver.MIN_DELTA,
            print_every=0,
        )
        operators = burgers_solver.precompute_operators(model, interior)
        optimizer = torch.optim.Adam(
            [model.interior_coeffs], lr=burgers_solver.INTERIOR_LEARNING_RATE
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=burgers_solver.SCHEDULER_FACTOR,
            patience=burgers_solver.SCHEDULER_PATIENCE,
            min_lr=burgers_solver.MIN_LEARNING_RATE,
        )
        history = train(
            [model.interior_coeffs],
            lambda: burgers_solver.burgers_pde_loss(model, operators),
            optimizer,
            epochs=epochs,
            scheduler=scheduler,
            early_stopping_patience=burgers_solver.EARLY_STOPPING_PATIENCE,
            min_delta=burgers_solver.MIN_DELTA,
            print_every=0,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        wall_times[index] = time.perf_counter() - start

        with torch.no_grad():
            prediction = model.forward(validation)
            global_errors[index] = relative_l2(
                prediction, validation_reference
            )
            local_errors[index] = relative_l2(
                prediction[local_mask], validation_reference[local_mask]
            )
            final_pde_losses[index] = float(
                burgers_solver.burgers_pde_loss(model, operators)
            )
            initial_losses[index] = float(
                burgers_solver.boundary_loss(
                    model, initial_points, initial_targets
                )
            )
        completed_epochs[index] = len(history)
        print(
            f"[gamma={concentration:4.1f}] h_min={minimum_spans[index]:.6e}, "
            f"global L2={global_errors[index]:.6e}, "
            f"local L2={local_errors[index]:.6e}, "
            f"epochs={completed_epochs[index]}, time={wall_times[index]:.3f} s"
        )

    output_path = next_output_path()
    np.savez_compressed(
        output_path,
        spatial_clustering=np.asarray(concentrations, dtype=float),
        minimum_knot_span=minimum_spans,
        maximum_knot_span=maximum_spans,
        validation_relative_l2_errors=global_errors,
        local_relative_l2_errors=local_errors,
        final_pde_losses=final_pde_losses,
        initial_condition_losses=initial_losses,
        training_wall_time_seconds=wall_times,
        completed_epochs=completed_epochs,
        n_basis_x=np.asarray(n_basis_x),
        n_basis_t=np.asarray(n_basis_t),
        degree=np.asarray(degree),
        tuned_spatial_clustering=np.asarray(tuned_concentration),
        local_half_width=np.asarray(local_half_width),
        local_validation_points=np.asarray(local_count),
        validation_points=np.asarray(validation.shape[0]),
        epochs=np.asarray(epochs),
        initialization_seed=np.asarray(initialization_seed),
        sampling_seed=np.asarray(sampling_seed),
        validation_seed=np.asarray(validation_seed),
    )
    print(f"Saved knot-concentration results: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Burgers B-spline spatial knot-concentration study"
    )
    parser.add_argument(
        "--device", choices=("cpu", "gpu"), default=burgers_solver.DEFAULT_DEVICE
    )
    parser.add_argument(
        "--concentrations",
        type=float,
        nargs="+",
        default=None,
        help="Spatial concentration values; default includes the tuned value",
    )
    parser.add_argument(
        "--epochs", type=int, default=burgers_solver.INTERIOR_EPOCHS
    )
    parser.add_argument("--local-half-width", type=float, default=0.1)
    parser.add_argument(
        "--initialization-seed", type=int, default=burgers_solver.SEED
    )
    parser.add_argument("--sampling-seed", type=int, default=burgers_solver.SEED)
    parser.add_argument("--validation-seed", type=int, default=burgers_solver.SEED)
    arguments = parser.parse_args()
    main(
        device_name=arguments.device,
        concentrations=(
            None
            if arguments.concentrations is None
            else tuple(arguments.concentrations)
        ),
        epochs=arguments.epochs,
        local_half_width=arguments.local_half_width,
        initialization_seed=arguments.initialization_seed,
        sampling_seed=arguments.sampling_seed,
        validation_seed=arguments.validation_seed,
    )
