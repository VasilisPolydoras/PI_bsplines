"""Tune the single-patch B-spline on an irregular Poisson domain."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hyperparameter_optimization import run_tpe_study
from IrregularPoisson.Bspline_2D import (
    DEFAULT_DEVICE,
    LEARNING_RATE,
    RESULT_PATH,
    SEED,
    bsplines,
    geometry_operator_terms,
    poisson_residual_loss,
    precompute_operators,
    relative_l2_error,
    sample_parameter_domain,
    select_device,
    set_seed,
    source_function,
)
from training import train


N_TRIALS = 20
TUNING_EPOCHS = 3_000
N_TRAIN_POINTS = 2_000
N_VALIDATION_POINTS = 5_000


def make_objective(
    training_points: torch.Tensor,
    validation_points: torch.Tensor,
    device: torch.device,
    tuning_epochs: int = TUNING_EPOCHS,
):
    training_geometry = geometry_operator_terms(training_points)
    validation_geometry = geometry_operator_terms(validation_points)
    training_source = source_function(training_points, training_geometry)

    def objective(trial):
        degree = trial.suggest_int("degree", 2, 5)
        n_basis = trial.suggest_int("n_basis", 8, 32)
        if n_basis <= degree:
            raise ValueError("n_basis must be greater than degree")
        set_seed(SEED)
        model = bsplines(n_basis, degree, device)
        training_operators = precompute_operators(model, training_points, 2)
        validation_operators = precompute_operators(model, validation_points, 2)
        loss_fn = lambda: poisson_residual_loss(
            model, training_points, training_operators,
            training_geometry, training_source,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.8, patience=100, min_lr=1.0e-5
        )
        train(
            model.parameters(),
            loss_fn,
            optimizer,
            epochs=tuning_epochs,
            scheduler=scheduler,
            early_stopping_patience=500,
            min_delta=1.0e-10,
            print_every=0,
        )
        with torch.no_grad():
            error = relative_l2_error(
                model, validation_points, validation_operators,
                validation_geometry,
            ).item()
            residual = poisson_residual_loss(
                model, training_points, training_operators,
                training_geometry, training_source,
            ).item()
        trial.set_user_attr("train_residual", residual)
        return error

    return objective


def main(
    device_name: str = DEFAULT_DEVICE,
    n_trials: int = N_TRIALS,
    tuning_epochs: int = TUNING_EPOCHS,
) -> None:
    set_seed(SEED)
    device = select_device(device_name)
    print(f"Using device: {device}")
    generator = torch.Generator(device=device).manual_seed(SEED)
    training = sample_parameter_domain(N_TRAIN_POINTS, device, generator)
    validation = sample_parameter_domain(N_VALIDATION_POINTS, device, generator)
    study = run_tpe_study(
        make_objective(training, validation, device, tuning_epochs),
        n_trials=n_trials,
        seed=SEED,
        direction="minimize",
        study_name="irregular_poisson_bspline_tuning",
        result_path=RESULT_PATH,
    )
    print(f"Best relative validation L2 error: {study.best_value:.6e}")
    print(f"Best n_basis: {study.best_params['n_basis']}")
    print(f"Best degree: {study.best_params['degree']}")
    print(f"Saved result to: {RESULT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune the single-patch irregular-domain Poisson B-spline"
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--epochs", type=int, default=TUNING_EPOCHS)
    args = parser.parse_args()
    main(args.device, args.trials, args.epochs)
