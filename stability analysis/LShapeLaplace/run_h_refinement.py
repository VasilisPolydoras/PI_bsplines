"""Isolated h-refinement study for the singular L-shaped Laplace problem.

The spline degree and uniform-knot construction are fixed while the basis
resolution is increased. Global L2, corner-region L2, and global H1-seminorm
errors are evaluated on one fixed validation set. The original irregular
solver and its outputs are not modified.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import LShapeLaplace.Irregular.Bspline_2D as lshape_solver
from training import train


N_BASIS_VALUES = (6, 8, 10, 12, 16, 20)
BASE_OUTPUT_PATH = HERE / "lshape_h_refinement_results.npz"


def next_output_path() -> Path:
    if not BASE_OUTPUT_PATH.exists():
        return BASE_OUTPUT_PATH
    run_number = 2
    while True:
        candidate = HERE / f"lshape_h_refinement_results_run_{run_number:02d}.npz"
        if not candidate.exists():
            return candidate
        run_number += 1


def exact_gradient(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the Cartesian gradient of r^(2/3) sin(2 theta/3)."""
    alpha = 2.0 / 3.0
    x, y = points[:, 0], points[:, 1]
    radius = torch.sqrt(x.square() + y.square())
    theta = torch.remainder(torch.atan2(y, x), 2.0 * math.pi)
    factor = alpha * radius.pow(alpha - 1.0)
    ux = factor * torch.sin((alpha - 1.0) * theta)
    uy = factor * torch.cos((alpha - 1.0) * theta)
    return ux, uy


def validation_errors(
    model: lshape_solver.LShapeBSpline,
    local_points: torch.Tensor,
    operators,
    corner_radius: float,
) -> tuple[float, float, float, int]:
    coefficients = model.coefficient_tensors()
    global_error_squared = torch.zeros((), device=local_points.device)
    global_exact_squared = torch.zeros_like(global_error_squared)
    gradient_error_squared = torch.zeros_like(global_error_squared)
    gradient_exact_squared = torch.zeros_like(global_error_squared)
    corner_errors = []
    corner_exact = []
    corner_count = 0

    with torch.no_grad():
        for name in model.patch_names:
            local_operators = lshape_solver.patch_operators(operators, name)
            physical = lshape_solver.map_patch(name, local_points)
            exact = lshape_solver.exact_solution(physical)
            prediction = model.evaluate(coefficients[name], local_operators)
            predicted_x = model.evaluate(
                coefficients[name], local_operators, dx=1, dy=0
            )
            predicted_y = model.evaluate(
                coefficients[name], local_operators, dx=0, dy=1
            )
            exact_x, exact_y = exact_gradient(physical)

            global_error_squared += prediction.sub(exact).square().sum()
            global_exact_squared += exact.square().sum()
            gradient_error_squared += (
                predicted_x.sub(exact_x).square()
                + predicted_y.sub(exact_y).square()
            ).sum()
            gradient_exact_squared += (
                exact_x.square() + exact_y.square()
            ).sum()

            mask = torch.linalg.vector_norm(physical, dim=1) <= corner_radius
            if mask.any():
                corner_errors.append(prediction[mask].sub(exact[mask]))
                corner_exact.append(exact[mask])
                corner_count += int(mask.sum().item())

    if corner_count == 0:
        raise RuntimeError("The validation set contains no corner-region points")
    corner_error = torch.cat(corner_errors)
    corner_reference = torch.cat(corner_exact)
    global_l2 = torch.sqrt(global_error_squared / global_exact_squared)
    corner_l2 = torch.linalg.vector_norm(corner_error) / torch.linalg.vector_norm(
        corner_reference
    )
    h1_seminorm = torch.sqrt(gradient_error_squared / gradient_exact_squared)
    return float(global_l2), float(corner_l2), float(h1_seminorm), corner_count


def main(
    *,
    device_name: str,
    n_basis_values: tuple[int, ...],
    degree: int,
    epochs: int,
    corner_radius: float,
    initialization_seed: int,
    sampling_seed: int,
    validation_seed: int,
) -> Path:
    if degree < 2:
        raise ValueError("degree must be at least two")
    n_basis_values = tuple(sorted(set(n_basis_values)))
    if not n_basis_values or any(value <= degree for value in n_basis_values):
        raise ValueError("Every n_basis value must be greater than degree")
    if corner_radius <= 0.0:
        raise ValueError("corner_radius must be positive")

    device = lshape_solver.select_device(device_name)
    baseline_n = int(lshape_solver.N_BASIS)
    validation_generator = torch.Generator(device=device).manual_seed(
        validation_seed
    )
    validation_points = lshape_solver.sample_local_points(
        lshape_solver.N_VALIDATION_POINTS_PER_PATCH,
        device,
        validation_generator,
    )

    case_count = len(n_basis_values)
    mesh_sizes = np.empty(case_count, dtype=float)
    interior_counts = np.empty(case_count, dtype=int)
    interface_counts = np.empty(case_count, dtype=int)
    boundary_counts = np.empty(case_count, dtype=int)
    global_errors = np.empty(case_count, dtype=float)
    corner_errors = np.empty(case_count, dtype=float)
    h1_errors = np.empty(case_count, dtype=float)
    corner_validation_counts = np.empty(case_count, dtype=int)
    wall_times = np.empty(case_count, dtype=float)
    completed_epochs = np.empty(case_count, dtype=int)

    print(f"Using device: {device}")
    print(
        f"Uniform-knot L-shape refinement: degree={degree}, "
        f"n_basis={n_basis_values}, corner radius={corner_radius:g}"
    )

    for index, n_basis in enumerate(n_basis_values):
        # Approximately preserve collocation points per raw patch coefficient.
        area_scale = (n_basis / baseline_n) ** 2
        edge_scale = n_basis / baseline_n
        interior_count = max(1, int(round(
            lshape_solver.N_INTERIOR_POINTS_PER_PATCH * area_scale
        )))
        interface_count = max(2, int(round(
            lshape_solver.N_INTERFACE_POINTS * edge_scale
        )))
        boundary_count = max(2, int(round(
            lshape_solver.N_BOUNDARY_POINTS_PER_EDGE * edge_scale
        )))
        interior_counts[index] = interior_count
        interface_counts[index] = interface_count
        boundary_counts[index] = boundary_count
        mesh_sizes[index] = 1.0 / (n_basis - degree)

        lshape_solver.set_seed(initialization_seed)
        model = lshape_solver.LShapeBSpline(
            n_basis,
            degree,
            knot_concentration=1.0,
            device=device,
            interior_initialization=lshape_solver.INTERIOR_INITIALIZATION,
        )
        sampling_generator = torch.Generator(device=device).manual_seed(
            sampling_seed
        )
        interior_points = lshape_solver.sample_local_points(
            interior_count, device, sampling_generator
        )
        interior_operators = lshape_solver.precompute_operators(
            model, interior_points, max_order=2
        )
        interface_points = lshape_solver.generate_interface_local_points(
            interface_count, device, lshape_solver.CORNER_EPSILON
        )
        interface_operators = lshape_solver.precompute_interface_operators(
            model, interface_points
        )
        boundary_points = (
            lshape_solver.generate_inhomogeneous_boundary_local_points(
                boundary_count, device
            )
        )
        boundary_operators = lshape_solver.precompute_boundary_operators(
            model, boundary_points
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        lshape_solver.pretrain_boundary_coefficients(
            model,
            boundary_points,
            boundary_operators,
            epochs=epochs,
            print_every=0,
        )
        loss_fn = lambda: lshape_solver.strong_residual_loss(
            model,
            interior_operators,
            interface_operators,
            pde_weight=lshape_solver.PDE_WEIGHT,
            interface_residual_weight=lshape_solver.INTERFACE_RESIDUAL_WEIGHT,
            interface_flux_weight=lshape_solver.INTERFACE_FLUX_WEIGHT,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lshape_solver.LEARNING_RATE
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lshape_solver.SCHEDULER_FACTOR,
            patience=lshape_solver.SCHEDULER_PATIENCE,
            min_lr=lshape_solver.MIN_LEARNING_RATE,
        )
        history = train(
            model.parameters(),
            loss_fn,
            optimizer,
            epochs=epochs,
            scheduler=scheduler,
            early_stopping_patience=lshape_solver.EARLY_STOPPING_PATIENCE,
            min_delta=lshape_solver.MIN_DELTA,
            print_every=0,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        wall_times[index] = time.perf_counter() - start
        completed_epochs[index] = len(history)

        validation_operators = lshape_solver.precompute_operators(
            model, validation_points, max_order=1
        )
        (
            global_errors[index],
            corner_errors[index],
            h1_errors[index],
            corner_validation_counts[index],
        ) = validation_errors(
            model, validation_points, validation_operators, corner_radius
        )
        print(
            f"[n={n_basis:2d}, h={mesh_sizes[index]:.6f}] "
            f"global L2={global_errors[index]:.6e}, "
            f"corner L2={corner_errors[index]:.6e}, "
            f"H1={h1_errors[index]:.6e}, "
            f"N_Omega/patch={interior_count}, "
            f"epochs={completed_epochs[index]}, time={wall_times[index]:.3f} s"
        )

    output_path = next_output_path()
    np.savez_compressed(
        output_path,
        n_basis=np.asarray(n_basis_values, dtype=int),
        degree=np.asarray(degree),
        h=mesh_sizes,
        knot_concentration=np.asarray(1.0),
        validation_relative_l2_errors=global_errors,
        corner_relative_l2_errors=corner_errors,
        validation_relative_h1_seminorm_errors=h1_errors,
        corner_radius=np.asarray(corner_radius),
        corner_validation_points=corner_validation_counts,
        interior_points_per_patch=interior_counts,
        interface_points=interface_counts,
        boundary_points_per_edge=boundary_counts,
        training_wall_time_seconds=wall_times,
        completed_epochs=completed_epochs,
        epochs=np.asarray(epochs),
        initialization_seed=np.asarray(initialization_seed),
        sampling_seed=np.asarray(sampling_seed),
        validation_seed=np.asarray(validation_seed),
    )
    print(f"Saved L-shape h-refinement results: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="L-shaped singular-solution B-spline h-refinement"
    )
    parser.add_argument(
        "--device", choices=("cpu", "gpu"), default=lshape_solver.DEFAULT_DEVICE
    )
    parser.add_argument(
        "--n-basis-values", type=int, nargs="+", default=N_BASIS_VALUES
    )
    parser.add_argument("--degree", type=int, default=lshape_solver.DEGREE)
    parser.add_argument("--epochs", type=int, default=lshape_solver.EPOCHS)
    parser.add_argument("--corner-radius", type=float, default=0.2)
    parser.add_argument(
        "--initialization-seed", type=int, default=lshape_solver.SEED
    )
    parser.add_argument("--sampling-seed", type=int, default=lshape_solver.SEED)
    parser.add_argument("--validation-seed", type=int, default=lshape_solver.SEED)
    arguments = parser.parse_args()
    main(
        device_name=arguments.device,
        n_basis_values=tuple(arguments.n_basis_values),
        degree=arguments.degree,
        epochs=arguments.epochs,
        corner_radius=arguments.corner_radius,
        initialization_seed=arguments.initialization_seed,
        sampling_seed=arguments.sampling_seed,
        validation_seed=arguments.validation_seed,
    )
