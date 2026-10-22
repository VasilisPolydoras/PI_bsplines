"""Tune the boundary-fitted B-spline Kirchhoff solver with Optuna."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hyperparameter_optimization import run_tpe_study
from KirchhoffIrregular.Bspline_AsymmetricIrregular_2D import (
    LEARNING_RATE,
    TUNED_RESULT_PATH,
    assemble_weak_system,
    bsplines,
    sample_parameter_domain,
    select_device,
    set_seed,
    weak_residual_loss,
)
from training import train


SEED = 42
DEFAULT_DEVICE = "gpu"
N_TRIALS = 20
TUNING_EPOCHS = 3_000
N_TRAIN_POINTS = 3_000
N_VALIDATION_POINTS = 6_000
RESULT_PATH = TUNED_RESULT_PATH


def make_objective(
    train_parameters: torch.Tensor,
    validation_parameters: torch.Tensor,
    device: torch.device,
    tuning_epochs: int = TUNING_EPOCHS,
):
    """Create the boundary-fitted training/validation objective."""

    def objective(trial):
        degree = trial.suggest_int("degree", 2, 5)
        n_basis_radial = trial.suggest_int("n_basis_radial", 8, 24)
        n_basis_angular = trial.suggest_int("n_basis_angular", 12, 36)
        if degree >= min(n_basis_radial, n_basis_angular):
            raise ValueError("degree must be smaller than both basis counts")

        set_seed(SEED)
        model = bsplines(
            n_basis_radial,
            n_basis_angular,
            degree,
            device,
        )
        train_stiffness, train_force = assemble_weak_system(
            model, train_parameters
        )
        validation_stiffness, validation_force = assemble_weak_system(
            model, validation_parameters
        )
        loss_fn = lambda: weak_residual_loss(
            model, train_stiffness, train_force
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.8, patience=50, min_lr=1.0e-5
        )
        train(
            model.parameters(),
            loss_fn,
            optimizer,
            epochs=tuning_epochs,
            scheduler=scheduler,
            early_stopping_patience=300,
            min_delta=1.0e-10,
            print_every=0,
        )
        with torch.no_grad():
            training_loss = weak_residual_loss(
                model, train_stiffness, train_force
            ).item()
            validation_loss = weak_residual_loss(
                model, validation_stiffness, validation_force
            ).item()
        trial.set_user_attr("training_loss", training_loss)
        return validation_loss

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
    train_parameters = sample_parameter_domain(
        N_TRAIN_POINTS, device, generator
    )
    validation_parameters = sample_parameter_domain(
        N_VALIDATION_POINTS, device, generator
    )
    study = run_tpe_study(
        make_objective(
            train_parameters,
            validation_parameters,
            device,
            tuning_epochs,
        ),
        n_trials=n_trials,
        seed=SEED,
        direction="minimize",
        study_name="boundary_fitted_irregular_kirchhoff_bspline_tuning",
        result_path=RESULT_PATH,
    )
    print(f"Best validation loss: {study.best_value:.6e}")
    print(f"Best radial basis count: {study.best_params['n_basis_radial']}")
    print(f"Best angular basis count: {study.best_params['n_basis_angular']}")
    print(f"Best degree: {study.best_params['degree']}")
    print(f"Saved result to: {RESULT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune the boundary-fitted irregular Kirchhoff solver"
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--epochs", type=int, default=TUNING_EPOCHS)
    arguments = parser.parse_args()
    main(arguments.device, arguments.trials, arguments.epochs)
