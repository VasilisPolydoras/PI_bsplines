"""Tune the Poisson PI-KAN architecture with Optuna TPE."""

import argparse
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
from Poisson.KAN_2D import (
    boundary_loss,
    create_kan,
    generate_boundary_points,
    poisson_residual_loss,
    select_device,
)
from training import train


SEED = 42
DEFAULT_DEVICE = "gpu"
N_TRIALS = 1
TUNING_EPOCHS = 1000
N_TRAIN_POINTS = 1000
N_VALIDATION_POINTS = 2000
N_BOUNDARY_POINTS_PER_EDGE = 50
LEARNING_RATE = 1e-3
RESULT_PATH = Path(__file__).resolve().parent / "results" / "best_kan_hyperparameters.json"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_objective(
    train_points, validation_points, train_boundary, validation_boundary, device
):
    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 1, 3)
        n_neurons = trial.suggest_int("n_neurons", 2, 10)
        grid = trial.suggest_int("grid", 3, 8)
        spline_order = trial.suggest_int("spline_order", 2, 4)

        set_seed(SEED)
        model = create_kan(
            n_layers, n_neurons, grid, spline_order, device=device, seed=SEED
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.8, patience=50, min_lr=1e-5
        )

        def training_loss():
            return (
                poisson_residual_loss(model, train_points)
                + boundary_loss(model, train_boundary)
            )

        train(
            parameters=model.parameters(),
            loss_fn=training_loss,
            optimizer=optimizer,
            epochs=TUNING_EPOCHS,
            scheduler=scheduler,
            early_stopping_patience=250,
            min_delta=1e-8,
            print_every=0,
        )

        validation_loss = (
            poisson_residual_loss(model, validation_points)
            + boundary_loss(model, validation_boundary)
        ).item()
        trial.set_user_attr("train_loss", training_loss().item())
        return validation_loss

    return objective


def main(device_name=DEFAULT_DEVICE):
    set_seed(SEED)
    device = select_device(device_name)
    print(f"Using device: {device}")
    generator = torch.Generator(device=device).manual_seed(SEED)
    train_points = torch.rand(
        N_TRAIN_POINTS, 2, generator=generator, device=device
    )
    validation_points = torch.rand(
        N_VALIDATION_POINTS, 2, generator=generator, device=device
    )
    train_boundary = generate_boundary_points(N_BOUNDARY_POINTS_PER_EDGE, device)
    validation_boundary = generate_boundary_points(
        2 * N_BOUNDARY_POINTS_PER_EDGE, device
    )

    study = run_tpe_study(
        make_objective(
            train_points,
            validation_points,
            train_boundary,
            validation_boundary,
            device,
        ),
        n_trials=N_TRIALS,
        seed=SEED,
        direction="minimize",
        study_name="poisson_kan_tuning",
        result_path=RESULT_PATH,
    )

    print(f"Best validation loss: {study.best_value:.6e}")
    for name, value in study.best_params.items():
        print(f"Best {name}: {value}")
    print(f"Saved result to: {RESULT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune the 2D Poisson PI-KAN")
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default=DEFAULT_DEVICE,
        help=f"Tuning device (default: {DEFAULT_DEVICE})",
    )
    arguments = parser.parse_args()
    main(arguments.device)
