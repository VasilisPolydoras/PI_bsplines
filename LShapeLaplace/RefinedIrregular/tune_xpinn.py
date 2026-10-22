"""Tune the twelve-patch XPINN for the singular L-shaped problem."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hyperparameter_optimization import run_tpe_study
from LShapeLaplace.RefinedIrregular.XPINN_2D import (
    DEFAULT_DEVICE,
    LEARNING_RATE,
    RESULT_PATH,
    SEED,
    XPINN,
    generate_boundary_points,
    generate_interface_points,
    relative_l2_error,
    sample_interior_points,
    select_device,
    set_seed,
    xpinn_loss,
    xpinn_loss_components,
)
from training import train


N_TRIALS = 20
TUNING_EPOCHS = 3_000
N_TRAIN_POINTS_PER_PATCH = 500
N_VALIDATION_POINTS_PER_PATCH = 1_000
N_BOUNDARY_POINTS_PER_EDGE = 100
N_INTERFACE_POINTS = 150


def make_objective(
    training_points,
    validation_points,
    boundary_points,
    interface_points,
    device: torch.device,
    tuning_epochs: int = TUNING_EPOCHS,
):
    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 2, 5)
        n_neurons = trial.suggest_int("n_neurons", 20, 100, step=20)
        set_seed(SEED)
        model = XPINN(n_layers, n_neurons).to(device)
        loss_fn = lambda: xpinn_loss(
            model, training_points, boundary_points, interface_points
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
            min_delta=1.0e-9,
            print_every=0,
        )
        value = relative_l2_error(model, validation_points).item()
        components = xpinn_loss_components(
            model, training_points, boundary_points, interface_points
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
    training = sample_interior_points(
        N_TRAIN_POINTS_PER_PATCH, device, generator
    )
    validation = sample_interior_points(
        N_VALIDATION_POINTS_PER_PATCH, device, generator
    )
    boundary = generate_boundary_points(N_BOUNDARY_POINTS_PER_EDGE, device)
    interfaces = generate_interface_points(N_INTERFACE_POINTS, device)
    study = run_tpe_study(
        make_objective(
            training, validation, boundary, interfaces, device, tuning_epochs
        ),
        n_trials=n_trials,
        seed=SEED,
        direction="minimize",
        study_name="l_shape_refined_irregular_xpinn_tuning",
        result_path=RESULT_PATH,
    )
    print(f"Best relative validation L2 error: {study.best_value:.6e}")
    print(f"Best number of layers: {study.best_params['n_layers']}")
    print(f"Best neurons per layer: {study.best_params['n_neurons']}")
    print(f"Saved result to: {RESULT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune the 12-patch XPINN for the singular L-shape"
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--epochs", type=int, default=TUNING_EPOCHS)
    arguments = parser.parse_args()
    main(arguments.device, arguments.trials, arguments.epochs)
