"""Run an isolated collocation-density study for Diffusion B-splines."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import Diffusion.Bspline_2D as diffusion_solver

N_BASIS = 13
DEGREE = 5
N_FREE = (N_BASIS - 2) * (N_BASIS - 1)
MESH_SIZE = 1.0 / (N_BASIS - DEGREE)
RHO_VALUES = np.asarray((0.5, 1.0, 2.0, 4.0, 8.0, 10.0, 16.0, 32.0))
BASE_OUTPUT_PATH = HERE / "diffusion_rho_refinement_results.npz"


def next_output_path() -> Path:
    """Return a new result path without replacing an earlier experiment."""
    if not BASE_OUTPUT_PATH.exists():
        return BASE_OUTPUT_PATH
    run_number = 2
    while True:
        candidate = HERE / f"diffusion_rho_refinement_results_run_{run_number:02d}.npz"
        if not candidate.exists():
            return candidate
        run_number += 1


def main(
    device: str,
    epochs: int,
    initialization_seed: int,
    sampling_seed: int,
    validation_seed: int,
) -> None:
    output_path = next_output_path()
    collocation_counts = np.ceil(RHO_VALUES * N_FREE).astype(np.int64)
    effective_rho = collocation_counts / N_FREE
    l2_errors = np.empty(RHO_VALUES.size, dtype=np.float64)
    training_times = np.empty_like(l2_errors)

    print("Diffusion B-spline collocation-density study")
    print(
        f"n_basis={N_BASIS}, degree={DEGREE}, h={MESH_SIZE:g}, "
        f"N_free={N_FREE}, device={device}"
    )
    print(
        "The initialization, sampling, and validation seeds remain fixed; "
        "only N_Omega changes."
    )
    print("No Diffusion files or solver postprocessing outputs will be written.")

    original_collocation_count = diffusion_solver.N_INTERIOR_POINTS
    try:
        for index, rho in enumerate(RHO_VALUES):
            diffusion_solver.N_INTERIOR_POINTS = int(collocation_counts[index])
            print(
                f"\nCase {index + 1}/{RHO_VALUES.size}: "
                f"rho={rho:g}, N_Omega={collocation_counts[index]}, "
                f"effective rho={effective_rho[index]:.8f}"
            )
            result = diffusion_solver.main(
                device_name=device,
                n_basis=N_BASIS,
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
        requested_rho=RHO_VALUES,
        effective_rho=effective_rho,
        interior_collocation_points=collocation_counts,
        validation_relative_l2_errors=l2_errors,
        training_times_seconds=training_times,
        n_basis=np.asarray(N_BASIS, dtype=np.int64),
        degree=np.asarray(DEGREE, dtype=np.int64),
        n_spans=np.asarray(N_BASIS - DEGREE, dtype=np.int64),
        h=np.asarray(MESH_SIZE, dtype=np.float64),
        n_free=np.asarray(N_FREE, dtype=np.int64),
        epochs=np.asarray(epochs, dtype=np.int64),
        initialization_seed=np.asarray(initialization_seed, dtype=np.int64),
        sampling_seed=np.asarray(sampling_seed, dtype=np.int64),
        validation_seed=np.asarray(validation_seed, dtype=np.int64),
        interior_initialization=np.asarray(diffusion_solver.INTERIOR_INITIALIZATION),
        source_solver=np.asarray("Diffusion/Bspline_2D.py"),
    )

    print("\nCompleted collocation-density study")
    for rho, count, error in zip(
        effective_rho, collocation_counts, l2_errors
    ):
        print(
            f"rho={rho:.8f}, N_Omega={count:4d}, "
            f"relative L2={error:.8e}"
        )
    print(f"Saved new result file: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Vary only the Diffusion B-spline collocation density without "
            "modifying or overwriting Diffusion outputs."
        )
    )
    parser.add_argument(
        "--device", choices=("cpu", "gpu"),
        default=diffusion_solver.DEFAULT_DEVICE,
    )
    parser.add_argument("--epochs", type=int, default=diffusion_solver.EPOCHS)
    parser.add_argument(
        "--initialization-seed", type=int, default=diffusion_solver.SEED
    )
    parser.add_argument(
        "--sampling-seed", type=int, default=diffusion_solver.SEED
    )
    parser.add_argument(
        "--validation-seed", type=int, default=diffusion_solver.SEED
    )
    arguments = parser.parse_args()
    main(
        device=arguments.device,
        epochs=arguments.epochs,
        initialization_seed=arguments.initialization_seed,
        sampling_seed=arguments.sampling_seed,
        validation_seed=arguments.validation_seed,
    )
