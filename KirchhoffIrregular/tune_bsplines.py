"""Tune B-splines for a Kirchhoff plate with a free circular hole."""

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
from KirchhoffIrregular.Bspline_2D import (
    LEARNING_RATE,
    RESULT_PATH,
    bsplines,
    precompute_operators,
    refine_with_lbfgs,
    sample_boundary,
    sample_interior,
    select_device,
    total_loss,
)
from training import train


SEED = 42
DEFAULT_DEVICE = "gpu"
N_TRIALS = 20
TUNING_EPOCHS = 3_000
N_TRAIN_POINTS = 2_000
N_VALIDATION_POINTS = 4_000
N_BOUNDARY_POINTS_PER_COMPONENT = 160
TUNING_LBFGS_MAX_ITERATIONS = 300


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_objective(
    train_points: torch.Tensor,
    validation_points: torch.Tensor,
    train_outer_points: torch.Tensor,
    train_outer_normals: torch.Tensor,
    train_hole_points: torch.Tensor,
    train_hole_normals: torch.Tensor,
    validation_outer_points: torch.Tensor,
    validation_outer_normals: torch.Tensor,
    validation_hole_points: torch.Tensor,
    validation_hole_normals: torch.Tensor,
    device: torch.device,
):
    def objective(trial):
        degree = trial.suggest_int("degree", 4, 6)
        n_basis_x = trial.suggest_int("n_basis_x", 10, 30)
        n_basis_y = trial.suggest_int("n_basis_y", 10, 30)
        knot_concentration = trial.suggest_float(
            "knot_concentration", 1.0, 8.0
        )
        knot_margin = trial.suggest_float("knot_margin", 0.2, 1.0)
        if degree >= min(n_basis_x, n_basis_y):
            raise ValueError("degree must be smaller than both basis counts")

        set_seed()
        model = bsplines(
            n_basis_x,
            n_basis_y,
            degree,
            device,
            knot_concentration=knot_concentration,
            knot_margin=knot_margin,
        )
        train_interior_operators = precompute_operators(model, train_points)
        train_outer_operators = precompute_operators(
            model, train_outer_points, max_order=2
        )
        train_hole_operators = precompute_operators(
            model, train_hole_points, max_order=3
        )
        validation_interior_operators = precompute_operators(
            model, validation_points
        )
        validation_outer_operators = precompute_operators(
            model, validation_outer_points, max_order=2
        )
        validation_hole_operators = precompute_operators(
            model, validation_hole_points, max_order=3
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.8, patience=50, min_lr=1.0e-5
        )
        loss_fn = lambda: total_loss(
            model,
            train_interior_operators,
            train_points,
            train_outer_operators,
            train_outer_normals,
            train_hole_operators,
            train_hole_normals,
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
            training_loss = total_loss(
                model,
                train_interior_operators,
                train_points,
                train_outer_operators,
                train_outer_normals,
                train_hole_operators,
                train_hole_normals,
            ).item()
            validation_loss = total_loss(
                model,
                validation_interior_operators,
                validation_points,
                validation_outer_operators,
                validation_outer_normals,
                validation_hole_operators,
                validation_hole_normals,
            ).item()
        trial.set_user_attr("training_loss", training_loss)
        return validation_loss

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
    (
        train_outer_points,
        train_outer_normals,
        train_hole_points,
        train_hole_normals,
    ) = sample_boundary(N_BOUNDARY_POINTS_PER_COMPONENT, device)
    (
        validation_outer_points,
        validation_outer_normals,
        validation_hole_points,
        validation_hole_normals,
    ) = sample_boundary(2 * N_BOUNDARY_POINTS_PER_COMPONENT, device)
    study = run_tpe_study(
        make_objective(
            train_points,
            validation_points,
            train_outer_points,
            train_outer_normals,
            train_hole_points,
            train_hole_normals,
            validation_outer_points,
            validation_outer_normals,
            validation_hole_points,
            validation_hole_normals,
            device,
        ),
        n_trials=N_TRIALS,
        seed=SEED,
        direction="minimize",
        study_name="irregular_kirchhoff_bspline_tuning",
        result_path=RESULT_PATH,
    )
    print(f"Best validation loss: {study.best_value:.6e}")
    print(f"Best n_basis_x: {study.best_params['n_basis_x']}")
    print(f"Best n_basis_y: {study.best_params['n_basis_y']}")
    print(f"Best degree: {study.best_params['degree']}")
    print(
        "Best knot concentration: "
        f"{study.best_params['knot_concentration']:.6g}"
    )
    print(f"Best knot margin: {study.best_params['knot_margin']:.6g}")
    print(f"Saved result to: {RESULT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune B-splines for the irregular Kirchhoff plate"
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    arguments = parser.parse_args()
    main(arguments.device)
