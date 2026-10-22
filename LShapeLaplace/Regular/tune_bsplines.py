"""Tune the B-spline XPINN for the regular L-shaped Laplace problem."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hyperparameter_optimization import run_tpe_study
from LShapeLaplace.Regular.Bspline_2D import (
    DEFAULT_DEVICE,
    LEARNING_RATE,
    RESULT_PATH,
    SEED,
    bsplines,
    generate_interface_local_points,
    precompute_interface_operators,
    precompute_operators,
    relative_l2_error,
    sample_local_points,
    select_device,
    set_seed,
    strong_loss_components,
    strong_residual_loss,
)
from training import train


N_TRIALS = 20
TUNING_EPOCHS = 3_000
N_TRAIN_POINTS = 2_000
N_VALIDATION_POINTS = 6_000
N_INTERFACE_POINTS = 500


def make_objective(
    training_points: torch.Tensor,
    validation_points: torch.Tensor,
    interface_points,
    device: torch.device,
    tuning_epochs: int = TUNING_EPOCHS,
):
    """Create an Optuna objective for B-spline basis count and degree."""
    def objective(trial):
        degree = trial.suggest_int("degree", 2, 5)
        n_basis = trial.suggest_int("n_basis", 8, 32)
        if degree >= n_basis:
            raise ValueError("degree must be smaller than n_basis")

        set_seed(SEED)
        model = bsplines(n_basis, degree, device)
        training_operators = precompute_operators(
            model, training_points, max_order=2
        )
        interface_operators = precompute_interface_operators(
            model, interface_points
        )
        validation_operators = precompute_operators(
            model, validation_points, max_order=0
        )
        loss_fn = lambda: strong_residual_loss(
            model, training_operators, interface_operators
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

        validation_error = relative_l2_error(
            model, validation_points, validation_operators
        ).item()
        components = strong_loss_components(
            model, training_operators, interface_operators
        )
        for name, value in components.items():
            trial.set_user_attr(f"train_{name}", value.detach().item())
        return validation_error

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
    training_points = sample_local_points(N_TRAIN_POINTS, device, generator)
    validation_points = sample_local_points(
        N_VALIDATION_POINTS, device, generator
    )
    interface_points = generate_interface_local_points(
        N_INTERFACE_POINTS, device
    )

    study = run_tpe_study(
        make_objective(
            training_points,
            validation_points,
            interface_points,
            device,
            tuning_epochs,
        ),
        n_trials=n_trials,
        seed=SEED,
        direction="minimize",
        study_name="l_shape_regular_bspline_xpinn_tuning",
        result_path=RESULT_PATH,
    )
    print(f"Best relative validation L2 error: {study.best_value:.6e}")
    print(f"Best n_basis: {study.best_params['n_basis']}")
    print(f"Best degree: {study.best_params['degree']}")
    print(f"Saved result to: {RESULT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune B-spline XPINNs on the regular L-shaped problem"
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--epochs", type=int, default=TUNING_EPOCHS)
    arguments = parser.parse_args()
    main(arguments.device, arguments.trials, arguments.epochs)
