"""Tune the Poisson B-spline basis count and degree with Optuna TPE."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import optuna
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hyperparameter_optimization import run_tpe_study
from Poisson.Bspline_2D import bsplines, build_poisson_system
from training import train


SEED = 42
N_TRIALS = 100
TUNING_EPOCHS = 10000
N_TRAIN_POINTS = 5000
N_VALIDATION_POINTS = 5000
LEARNING_RATE = 1e-3
RESULT_PATH = Path(__file__).resolve().parent / "results" / "best_bspline_hyperparameters.json"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_objective(train_points, validation_points):
    """Create the Poisson-specific objective consumed by the generic tuner."""

    def objective(trial: optuna.Trial) -> float:
        n_basis = trial.suggest_int("n_basis", 8, 30)
        degree = trial.suggest_int("degree", 2, 5)

        set_seed(SEED)
        model = bsplines(n_basis=n_basis, degree=degree)
        train_operator, train_source = build_poisson_system(model, train_points)
        validation_operator, validation_source = build_poisson_system(
            model, validation_points
        )

        optimizer = torch.optim.Adam([model.coeffs], lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.8, patience=50, min_lr=1e-5
        )

        def training_loss():
            coefficients = model.coeffs[1:-1, 1:-1].reshape(-1)
            residual = train_operator @ coefficients - train_source
            return residual.square().mean()

        train(
            parameters=[model.coeffs],
            loss_fn=training_loss,
            optimizer=optimizer,
            epochs=TUNING_EPOCHS,
            scheduler=scheduler,
            early_stopping_patience=250,
            min_delta=1e-8,
            print_every=0,
        )

        with torch.no_grad():
            coefficients = model.coeffs[1:-1, 1:-1].reshape(-1)
            validation_residual = validation_operator @ coefficients - validation_source
            validation_loss = validation_residual.square().mean().item()
            train_loss = training_loss().item()

        trial.set_user_attr("train_loss", train_loss)
        return validation_loss

    return objective


def main():
    set_seed(SEED)

    # The imported Poisson implementation is CPU-based. Fixed, disjoint point
    # sets make objective values directly comparable across trials.
    generator = torch.Generator().manual_seed(SEED)
    train_points = torch.rand(N_TRAIN_POINTS, 2, generator=generator)
    validation_points = torch.rand(N_VALIDATION_POINTS, 2, generator=generator)

    study = run_tpe_study(
        make_objective(train_points, validation_points),
        n_trials=N_TRIALS,
        seed=SEED,
        direction="minimize",
        study_name="poisson_bspline_tuning",
        result_path=RESULT_PATH,
    )

    print(f"Best validation loss: {study.best_value:.6e}")
    print(f"Best n_basis: {study.best_params['n_basis']}")
    print(f"Best degree: {study.best_params['degree']}")
    print(f"Saved result to: {RESULT_PATH}")


if __name__ == "__main__":
    main()
