"""Tune the Kovasznay B-spline basis count and degree with Optuna TPE."""

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
from Kovasznay.Bspline_2D import (
    CONFIG,
    DEFAULT_DEVICE,
    HYPERPARAMETER_PATH,
    INTERIOR_LEARNING_RATE,
    SEED,
    bsplines,
    exact_solution,
    kovasznay_loss,
    precompute_operators,
    sample_boundary,
    sample_interior,
    select_device,
)
from training import train


TUNING = CONFIG["tuning"]
N_TRIALS = int(TUNING["trials"])
TUNING_EPOCHS = int(TUNING["epochs"])
N_TRAIN_POINTS = int(TUNING["training_points"])
N_VALIDATION_POINTS = int(TUNING["validation_points"])
N_TRAIN_BOUNDARY = int(TUNING["training_boundary_points_per_edge"])
N_VALIDATION_BOUNDARY = int(TUNING["validation_boundary_points_per_edge"])
RESULT_PATH = HYPERPARAMETER_PATH


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_objective(
    train_points, validation_points, train_boundary, validation_boundary,
    device, tuning_epochs=TUNING_EPOCHS,
):
    def objective(trial):
        n_basis = trial.suggest_int(
            "n_basis", int(TUNING["n_basis"]["min"]),
            int(TUNING["n_basis"]["max"]),
        )
        degree = trial.suggest_int(
            "degree", int(TUNING["degree"]["min"]),
            int(TUNING["degree"]["max"]),
        )
        set_seed(SEED)
        model = bsplines(n_basis, degree, device)
        train_operators = precompute_operators(model, train_points)
        validation_operators = precompute_operators(model, validation_points)
        optimizer = torch.optim.Adam(
            [model.coeffs], lr=INTERIOR_LEARNING_RATE
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(TUNING["scheduler"]["factor"]),
            patience=int(TUNING["scheduler"]["patience"]),
            min_lr=float(TUNING["scheduler"]["min_learning_rate"]),
        )

        def loss_fn():
            return kovasznay_loss(model, train_operators, train_boundary)

        train(
            [model.coeffs], loss_fn, optimizer,
            epochs=tuning_epochs,
            scheduler=scheduler,
            early_stopping_patience=int(
                TUNING["early_stopping"]["patience"]
            ),
            min_delta=float(TUNING["early_stopping"]["min_delta"]),
            print_every=0,
        )
        with torch.no_grad():
            validation_loss = kovasznay_loss(
                model, validation_operators, validation_boundary
            ).item()
            training_loss = loss_fn().item()
            prediction = model.forward(validation_points)
            reference = exact_solution(validation_points)
            relative_errors = (
                torch.linalg.vector_norm(prediction - reference, dim=0)
                / torch.linalg.vector_norm(reference, dim=0)
            )
        trial.set_user_attr("train_loss", training_loss)
        trial.set_user_attr("validation_physics_loss", validation_loss)
        trial.set_user_attr(
            "field_relative_l2", relative_errors.cpu().tolist()
        )
        return relative_errors.mean().item()

    return objective


def main(
    device_name=DEFAULT_DEVICE,
    n_trials=N_TRIALS,
    tuning_epochs=TUNING_EPOCHS,
):
    set_seed(SEED)
    device = select_device(device_name)
    print(f"Using device: {device}")
    generator = torch.Generator(device=device).manual_seed(SEED)
    train_points = sample_interior(N_TRAIN_POINTS, device, generator)
    validation_points = sample_interior(N_VALIDATION_POINTS, device, generator)
    train_boundary = sample_boundary(N_TRAIN_BOUNDARY, device)
    validation_boundary = sample_boundary(N_VALIDATION_BOUNDARY, device)
    study = run_tpe_study(
        make_objective(
            train_points, validation_points, train_boundary,
            validation_boundary, device, tuning_epochs,
        ),
        n_trials=n_trials,
        seed=SEED,
        direction="minimize",
        study_name="kovasznay_bspline_tuning",
        result_path=RESULT_PATH,
    )
    print(f"Best mean relative validation L2 error: {study.best_value:.6e}")
    print(f"Best n_basis: {study.best_params['n_basis']}")
    print(f"Best degree: {study.best_params['degree']}")
    print(f"Saved result to: {RESULT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune Kovasznay B-splines")
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--epochs", type=int, default=TUNING_EPOCHS)
    args = parser.parse_args()
    main(args.device, args.trials, args.epochs)
