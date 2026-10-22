"""Tune the budget-preserving refined B-spline XPINN.

``n_basis`` is the per-coordinate budget of each former unit-square patch. It
is split between the corner and outer intervals by the solver, so the twelve
patches collectively retain the raw coefficient count of the three-patch
model while concentrating basis functions near the singular corner.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hyperparameter_optimization import run_tpe_study
from LShapeLaplace.RefinedIrregular.Bspline_2D import (
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
N_TRAIN_POINTS = 1_000
N_VALIDATION_POINTS = 2_000
N_INTERFACE_POINTS = 250


def make_objective(
    training_points: torch.Tensor,
    validation_points: torch.Tensor,
    interface_points,
    device: torch.device,
    tuning_epochs: int = TUNING_EPOCHS,
):
    def objective(trial):
        degree = trial.suggest_int("degree", 2, 5)
        # With the fixed 60/40 split, 14 is the smallest total that supports
        # every sampled degree up to five on both interval types.
        n_basis = trial.suggest_int("n_basis", 14, 32)
        set_seed(SEED)
        model = bsplines(n_basis, degree, device)
        training_operators = precompute_operators(model, training_points, 2)
        interface_operators = precompute_interface_operators(
            model, interface_points
        )
        validation_operators = precompute_operators(model, validation_points, 0)
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
        value = relative_l2_error(
            model, validation_points, validation_operators
        ).item()
        components = strong_loss_components(
            model, training_operators, interface_operators
        )
        for name, component in components.items():
            trial.set_user_attr(f"train_{name}", component.detach().item())
        return value

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
    training = sample_local_points(N_TRAIN_POINTS, device, generator)
    validation = sample_local_points(N_VALIDATION_POINTS, device, generator)
    interfaces = generate_interface_local_points(N_INTERFACE_POINTS, device)
    study = run_tpe_study(
        make_objective(training, validation, interfaces, device, tuning_epochs),
        n_trials=n_trials,
        seed=SEED,
        direction="minimize",
        study_name="l_shape_refined_irregular_bspline_xpinn_tuning",
        result_path=RESULT_PATH,
    )
    print(f"Best relative validation L2 error: {study.best_value:.6e}")
    print(f"Best total n_basis budget: {study.best_params['n_basis']}")
    print(f"Best degree: {study.best_params['degree']}")
    print(f"Saved result to: {RESULT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune the 12-patch B-spline XPINN for the singular L-shape"
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--epochs", type=int, default=TUNING_EPOCHS)
    arguments = parser.parse_args()
    main(arguments.device, arguments.trials, arguments.epochs)
