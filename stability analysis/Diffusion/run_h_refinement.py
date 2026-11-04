"""Run an isolated h-refinement study using the forward Diffusion solver.

The original Diffusion source, configuration, tuned parameters, and saved
postprocessing data are never modified. Only the imported module's in-memory
collocation-count constant is changed before each independent solver call.
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import Diffusion.Bspline_2D as diffusion_solver


N_BASIS_VALUES = np.asarray((9, 11, 13, 17, 21, 29, 37), dtype=np.int64)
DEGREE = 5
DEFAULT_RHO = 10.0
BASE_OUTPUT_PATH = HERE / "diffusion_h_refinement_results.npz"


def free_coefficient_count(n_basis: int) -> int:
    """Return the independent coefficients after fixing BC and IC traces."""
    return (n_basis - 2) * (n_basis - 1)


def next_output_path() -> Path:
    """Return a new result path without replacing an earlier experiment."""
    if not BASE_OUTPUT_PATH.exists():
        return BASE_OUTPUT_PATH
    run_number = 2
    while True:
        candidate = HERE / f"diffusion_h_refinement_results_run_{run_number:02d}.npz"
        if not candidate.exists():
            return candidate
        run_number += 1


def main(
    device: str,
    rho: float,
    epochs: int,
    initialization_seed: int,
    sampling_seed: int,
    validation_seed: int,
) -> None:
    if rho <= 0.0:
        raise ValueError("rho must be positive")
    output_path = next_output_path()

    free_counts = np.asarray(
        [free_coefficient_count(int(n)) for n in N_BASIS_VALUES],
        dtype=np.int64,
    )
    collocation_counts = np.ceil(rho * free_counts).astype(np.int64)
    effective_rho = collocation_counts / free_counts
    mesh_sizes = 1.0 / (N_BASIS_VALUES - DEGREE)
    l2_errors = np.empty(N_BASIS_VALUES.size, dtype=np.float64)
    training_times = np.empty_like(l2_errors)

    print("Diffusion B-spline h-refinement study")
    print(f"degree={DEGREE}, requested rho={rho:g}, device={device}")
    print("No Diffusion files or solver postprocessing outputs will be written.")

    original_collocation_count = diffusion_solver.N_INTERIOR_POINTS
    try:
        for index, n_basis in enumerate(N_BASIS_VALUES):
            diffusion_solver.N_INTERIOR_POINTS = int(collocation_counts[index])
            print(
                f"\nCase {index + 1}/{N_BASIS_VALUES.size}: "
                f"n_basis={n_basis}, h={mesh_sizes[index]:.8f}, "
                f"N_free={free_counts[index]}, "
                f"N_Omega={collocation_counts[index]}, "
                f"rho={effective_rho[index]:.8f}"
            )
            result = diffusion_solver.main(
                device_name=device,
                n_basis=int(n_basis),
                degree=DEGREE,
                epochs=epochs,
                use_tuned=False,
                interior_initialization=diffusion_solver.INTERIOR_INITIALIZATION,
                initialization_seed=initialization_seed,
                sampling_seed=sampling_seed,
                validation_seed=validation_seed,
                save_outputs=False,
            )
            l2_errors[index] = result["validation_relative_l2_error"]
            training_times[index] = result["training_time_seconds"]
            print(
                f"Recorded: relative L2={l2_errors[index]:.8e}, "
                f"training time={training_times[index]:.3f} s"
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        diffusion_solver.N_INTERIOR_POINTS = original_collocation_count

    np.savez_compressed(
        output_path,
        n_basis=N_BASIS_VALUES,
        degree=np.asarray(DEGREE, dtype=np.int64),
        h=mesh_sizes,
        n_free=free_counts,
        interior_collocation_points=collocation_counts,
        requested_rho=np.asarray(rho, dtype=np.float64),
        effective_rho=effective_rho,
        validation_relative_l2_errors=l2_errors,
        training_times_seconds=training_times,
        epochs=np.asarray(epochs, dtype=np.int64),
        initialization_seed=np.asarray(initialization_seed, dtype=np.int64),
        sampling_seed=np.asarray(sampling_seed, dtype=np.int64),
        validation_seed=np.asarray(validation_seed, dtype=np.int64),
        interior_initialization=np.asarray(
            diffusion_solver.INTERIOR_INITIALIZATION
        ),
        source_solver=np.asarray("Diffusion/Bspline_2D.py"),
    )

    print("\nCompleted h-refinement study")
    for n_basis, h, error in zip(N_BASIS_VALUES, mesh_sizes, l2_errors):
        print(f"n_basis={n_basis:2d}, h={h:.8f}, relative L2={error:.8e}")
    print(f"Saved new result file: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run Diffusion B-spline h-refinement at a constant collocation "
            "ratio without modifying or overwriting Diffusion outputs."
        )
    )
    parser.add_argument(
        "--device", choices=("cpu", "gpu"),
        default=diffusion_solver.DEFAULT_DEVICE,
    )
    parser.add_argument(
        "--rho", type=float, default=DEFAULT_RHO,
        help="Constant N_Omega/N_free ratio (default: 10)",
    )
    parser.add_argument(
        "--epochs", type=int, default=diffusion_solver.EPOCHS,
    )
    parser.add_argument(
        "--initialization-seed", type=int, default=diffusion_solver.SEED,
    )
    parser.add_argument(
        "--sampling-seed", type=int, default=diffusion_solver.SEED,
    )
    parser.add_argument(
        "--validation-seed", type=int, default=diffusion_solver.SEED,
    )
    arguments = parser.parse_args()
    main(
        device=arguments.device,
        rho=arguments.rho,
        epochs=arguments.epochs,
        initialization_seed=arguments.initialization_seed,
        sampling_seed=arguments.sampling_seed,
        validation_seed=arguments.validation_seed,
    )
