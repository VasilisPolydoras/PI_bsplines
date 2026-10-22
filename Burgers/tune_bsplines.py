"""Tune Burgers B-spline basis count and degree with Optuna TPE."""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Burgers.Bspline_2D import (
    CONFIG,
    DEFAULT_DEVICE,
    HYPERPARAMETER_PATH,
    INTERIOR_LEARNING_RATE,
    SEED,
    BurgersBSpline,
    burgers_pde_loss,
    numerical_reference_at_points,
    precompute_operators,
    sample_interior,
    select_device,
    set_seed,
)
from hyperparameter_optimization import run_tpe_study
from training import train


TUNING_CONFIG = CONFIG["tuning"]
N_TRIALS = int(TUNING_CONFIG["trials"])
TUNING_PDE_EPOCHS = int(TUNING_CONFIG["epochs"])
N_TRAIN_POINTS = int(TUNING_CONFIG["training_points"])
N_VALIDATION_POINTS = int(TUNING_CONFIG["validation_points"])
N_BASIS_X_MIN = int(TUNING_CONFIG["n_basis_x"]["min"])
N_BASIS_X_MAX = int(TUNING_CONFIG["n_basis_x"]["max"])
N_BASIS_T_MIN = int(TUNING_CONFIG["n_basis_t"]["min"])
N_BASIS_T_MAX = int(TUNING_CONFIG["n_basis_t"]["max"])
DEGREE_MIN = int(TUNING_CONFIG["degree"]["min"])
DEGREE_MAX = int(TUNING_CONFIG["degree"]["max"])
CLUSTERING_MIN = float(TUNING_CONFIG["spatial_clustering"]["min"])
CLUSTERING_MAX = float(TUNING_CONFIG["spatial_clustering"]["max"])
CLUSTERING_STEP = float(TUNING_CONFIG["spatial_clustering"]["step"])
TUNING_SCHEDULER_FACTOR = float(TUNING_CONFIG["scheduler"]["factor"])
TUNING_SCHEDULER_PATIENCE = int(TUNING_CONFIG["scheduler"]["patience"])
TUNING_MIN_LEARNING_RATE = float(
    TUNING_CONFIG["scheduler"]["min_learning_rate"]
)
TUNING_EARLY_STOPPING_PATIENCE = int(
    TUNING_CONFIG["early_stopping"]["patience"]
)
TUNING_MIN_DELTA = float(TUNING_CONFIG["early_stopping"]["min_delta"])
RESULT_PATH = HYPERPARAMETER_PATH


def make_objective(
    train_points,
    validation_points,
    validation_reference,
    device,
    tuning_epochs=TUNING_PDE_EPOCHS,
):
    def objective(trial):
        n_basis_x = trial.suggest_int(
            "n_basis_x", N_BASIS_X_MIN, N_BASIS_X_MAX
        )
        n_basis_t = trial.suggest_int(
            "n_basis_t", N_BASIS_T_MIN, N_BASIS_T_MAX
        )
        degree = trial.suggest_int("degree", DEGREE_MIN, DEGREE_MAX)
        spatial_clustering = trial.suggest_float(
            "spatial_clustering",
            CLUSTERING_MIN,
            CLUSTERING_MAX,
            step=CLUSTERING_STEP,
        )
        if degree >= min(n_basis_x, n_basis_t):
            raise ValueError("degree must be smaller than both basis counts")

        set_seed(SEED)
        model = BurgersBSpline(
            n_basis_x,
            n_basis_t,
            degree,
            device,
            spatial_clustering=spatial_clustering,
        )
        train_operators = precompute_operators(model, train_points)
        validation_operators = precompute_operators(model, validation_points)

        optimizer = torch.optim.Adam(
            [model.interior_coeffs], lr=INTERIOR_LEARNING_RATE
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=TUNING_SCHEDULER_FACTOR,
            patience=TUNING_SCHEDULER_PATIENCE,
            min_lr=TUNING_MIN_LEARNING_RATE,
        )
        train(
            [model.interior_coeffs],
            lambda: burgers_pde_loss(model, train_operators),
            optimizer,
            epochs=tuning_epochs,
            scheduler=scheduler,
            early_stopping_patience=TUNING_EARLY_STOPPING_PATIENCE,
            min_delta=TUNING_MIN_DELTA,
            print_every=0,
        )

        with torch.no_grad():
            training_loss = burgers_pde_loss(model, train_operators).item()
            validation_pde_loss = burgers_pde_loss(
                model, validation_operators
            ).item()
            validation_prediction = model.forward(validation_points)
            validation_error = torch.sqrt(
                (validation_prediction - validation_reference).square().sum()
                / validation_reference.square().sum()
            ).item()
        trial.set_user_attr("train_pde_loss", training_loss)
        trial.set_user_attr("validation_pde_loss", validation_pde_loss)
        return validation_error

    return objective


def main(
    device_name=DEFAULT_DEVICE,
    n_trials=N_TRIALS,
    tuning_epochs=TUNING_PDE_EPOCHS,
):
    set_seed(SEED)
    device = select_device(device_name)
    print(f"Using device: {device}")
    generator = torch.Generator(device=device).manual_seed(SEED)
    train_points = sample_interior(N_TRAIN_POINTS, device, generator)
    validation_points = sample_interior(
        N_VALIDATION_POINTS, device, generator
    )
    print("Computing the validation reference solution once")
    validation_reference = numerical_reference_at_points(validation_points)
    study = run_tpe_study(
        make_objective(
            train_points,
            validation_points,
            validation_reference,
            device,
            tuning_epochs,
        ),
        n_trials=n_trials,
        seed=SEED,
        direction="minimize",
        study_name="burgers_bspline_tuning",
        result_path=RESULT_PATH,
    )
    print(f"Best relative validation L2 error: {study.best_value:.6e}")
    print(f"Best n_basis_x: {study.best_params['n_basis_x']}")
    print(f"Best n_basis_t: {study.best_params['n_basis_t']}")
    print(f"Best degree: {study.best_params['degree']}")
    print(
        "Best spatial clustering: "
        f"{study.best_params['spatial_clustering']}"
    )
    print(f"Saved result to: {RESULT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune Burgers B-splines")
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--epochs", type=int, default=TUNING_PDE_EPOCHS)
    args = parser.parse_args()
    main(args.device, args.trials, args.epochs)
