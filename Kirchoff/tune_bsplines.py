"""Tune rectangular Kirchhoff-plate B-splines with Optuna TPE."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hyperparameter_optimization import run_tpe_study
from Kirchoff.Bspline_2D import (
    CONFIG,
    LEARNING_RATE,
    RESULT_PATH,
    bsplines,
    exact_solution,
    moment_boundary_loss,
    pde_loss,
    precompute_operators,
    refine_with_lbfgs,
    sample_boundary,
    sample_interior,
    select_device,
    total_loss,
)
from training import train


TUNING = CONFIG["tuning"]
SEED = int(CONFIG["reproducibility"]["seed"])
DEFAULT_DEVICE = str(CONFIG["runtime"]["device"])
N_TRIALS = int(TUNING["trials"])
TUNING_EPOCHS = int(TUNING["epochs"])
N_TRAIN_POINTS = int(TUNING["training_points"])
N_VALIDATION_POINTS = int(TUNING["validation_points"])
N_BOUNDARY_POINTS_PER_EDGE = int(TUNING["boundary_points_per_edge"])
TUNING_LBFGS_MAX_ITERATIONS = 300


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_objective(
    train_points: torch.Tensor,
    validation_points: torch.Tensor,
    vertical_points: torch.Tensor,
    horizontal_points: torch.Tensor,
    device: torch.device,
):
    def objective(trial):
        degree = trial.suggest_int(
            "degree", int(TUNING["degree"]["min"]), int(TUNING["degree"]["max"])
        )
        n_basis_x = trial.suggest_int(
            "n_basis_x", int(TUNING["n_basis_x"]["min"]), int(TUNING["n_basis_x"]["max"])
        )
        n_basis_y = trial.suggest_int(
            "n_basis_y", int(TUNING["n_basis_y"]["min"]), int(TUNING["n_basis_y"]["max"])
        )
        if degree >= min(n_basis_x, n_basis_y):
            raise ValueError("degree must be smaller than both basis counts")

        set_seed()
        model = bsplines(n_basis_x, n_basis_y, degree, device)
        train_operators = precompute_operators(model, train_points)
        validation_operators = precompute_operators(model, validation_points)
        vertical_operators = precompute_operators(
            model, vertical_points, max_order=2
        )
        horizontal_operators = precompute_operators(
            model, horizontal_points, max_order=2
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.8, patience=50, min_lr=1.0e-5
        )
        loss_fn = lambda: total_loss(
            model,
            train_operators,
            train_points,
            vertical_operators,
            horizontal_operators,
        )
        train(
            model.parameters(),
            loss_fn,
            optimizer,
            epochs=TUNING_EPOCHS,
            scheduler=scheduler,
            early_stopping_patience=300,
            min_delta=1.0e-9,
            print_every=0,
        )
        refine_with_lbfgs(
            model, loss_fn, max_iterations=TUNING_LBFGS_MAX_ITERATIONS
        )
        with torch.no_grad():
            validation_pde = pde_loss(
                model, validation_operators, validation_points
            )
            validation_moment = moment_boundary_loss(
                model, vertical_operators, horizontal_operators
            )
            predicted = model.evaluate(validation_operators)
            exact = exact_solution(validation_points)
            relative_error = (
                torch.linalg.vector_norm(predicted - exact)
                / torch.linalg.vector_norm(exact)
            ).item()
        trial.set_user_attr("validation_pde_loss", validation_pde.item())
        trial.set_user_attr("validation_moment_loss", validation_moment.item())
        trial.set_user_attr("relative_deflection_error", relative_error)
        return relative_error

    return objective


def main(device_name: str = DEFAULT_DEVICE) -> None:
    set_seed()
    device = select_device(device_name)
    print(f"Using device: {device}")
    generator = torch.Generator(device=device).manual_seed(SEED)
    train_points = sample_interior(N_TRAIN_POINTS, device, generator)
    validation_points = sample_interior(
        N_VALIDATION_POINTS, device, generator
    )
    vertical_points, horizontal_points = sample_boundary(
        N_BOUNDARY_POINTS_PER_EDGE, device
    )
    study = run_tpe_study(
        make_objective(
            train_points,
            validation_points,
            vertical_points,
            horizontal_points,
            device,
        ),
        n_trials=N_TRIALS,
        seed=SEED,
        direction="minimize",
        study_name="rectangular_kirchhoff_bspline_tuning",
        result_path=RESULT_PATH,
    )
    print(f"Best validation relative L2 error: {study.best_value:.6e}")
    print(f"Best n_basis_x: {study.best_params['n_basis_x']}")
    print(f"Best n_basis_y: {study.best_params['n_basis_y']}")
    print(f"Best degree: {study.best_params['degree']}")
    print(
        "Best-trial relative deflection error: "
        f"{study.best_trial.user_attrs['relative_deflection_error']:.6e}"
    )
    print(f"Saved result to: {RESULT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune rectangular Kirchhoff-plate B-splines"
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    arguments = parser.parse_args()
    main(arguments.device)
