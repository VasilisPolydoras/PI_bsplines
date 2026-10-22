"""Tune the hard- or soft-constrained Kovasznay PI-KAN with Optuna."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hyperparameter_optimization import run_tpe_study
from Kovasznay.PIKAN_2D import (
    BOUNDARY_WEIGHT, CONFIG, CONSTRAINT_MODE, DEFAULT_DEVICE, LEARNING_RATE,
    PDE_WEIGHT, RESULT_PATH, SEED, KovasznayPIKAN, field_relative_l2_errors,
    loss_components, physics_loss, sample_boundary, sample_interior,
    select_device, set_seed, validate_problem_configuration,
)
from training import train


TUNING = CONFIG["tuning"]
N_TRIALS = int(TUNING["trials"])
TUNING_EPOCHS = int(TUNING["epochs"])
N_TRAIN = int(TUNING["training_points"])
N_VALIDATION = int(TUNING["validation_points"])
N_BOUNDARY = int(TUNING["boundary_points_per_edge"])


def save_result(study, constraint_mode: str) -> None:
    payload = json.loads(RESULT_PATH.read_text()) if RESULT_PATH.exists() else {}
    payload[constraint_mode] = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_objective(training, validation, boundary, device, mode, epochs):
    def objective(trial):
        layers = trial.suggest_int(
            "n_layers", int(TUNING["n_layers"]["min"]),
            int(TUNING["n_layers"]["max"]),
        )
        neurons = trial.suggest_int(
            "n_neurons", int(TUNING["n_neurons"]["min"]),
            int(TUNING["n_neurons"]["max"]),
        )
        grid = trial.suggest_int(
            "grid", int(TUNING["grid"]["min"]), int(TUNING["grid"]["max"])
        )
        order = trial.suggest_int(
            "spline_order", int(TUNING["spline_order"]["min"]),
            int(TUNING["spline_order"]["max"]),
        )
        set_seed(SEED)
        model = KovasznayPIKAN(
            layers, neurons, grid, order, mode, device, seed=SEED
        ).to(device)
        loss_fn = lambda: physics_loss(
            model, training, boundary, PDE_WEIGHT, BOUNDARY_WEIGHT
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=float(TUNING["scheduler"]["factor"]),
            patience=int(TUNING["scheduler"]["patience"]),
            min_lr=float(TUNING["scheduler"]["min_learning_rate"]),
        )
        train(
            model.parameters(), loss_fn, optimizer, epochs=epochs,
            scheduler=scheduler,
            early_stopping_patience=int(TUNING["early_stopping"]["patience"]),
            min_delta=float(TUNING["early_stopping"]["min_delta"]),
            print_every=0,
        )
        errors = field_relative_l2_errors(model, validation)
        for name, value in loss_components(model, training, boundary).items():
            trial.set_user_attr(f"train_{name}", value.detach().item())
        trial.set_user_attr("field_relative_l2", errors.detach().cpu().tolist())
        return errors.mean().item()

    return objective


def main(
    device_name: str = DEFAULT_DEVICE,
    constraint_mode: str = CONSTRAINT_MODE,
    n_trials: int = N_TRIALS,
    epochs: int = TUNING_EPOCHS,
) -> None:
    validate_problem_configuration()
    set_seed(SEED)
    device = select_device(device_name)
    generator = torch.Generator(device=device).manual_seed(SEED)
    training = sample_interior(N_TRAIN, device, generator)
    validation = sample_interior(N_VALIDATION, device, generator)
    boundary = sample_boundary(N_BOUNDARY, device)
    print(f"Using device: {device}; constraints: {constraint_mode}")
    study = run_tpe_study(
        make_objective(training, validation, boundary, device, constraint_mode, epochs),
        n_trials=n_trials, seed=SEED, direction="minimize",
        study_name=f"kovasznay_pikan_{constraint_mode}_tuning",
    )
    save_result(study, constraint_mode)
    print(f"Best mean relative validation L2 error: {study.best_value:.6e}")
    for name, value in study.best_params.items():
        print(f"Best {name}: {value}")
    print(f"Saved result to: {RESULT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune the Kovasznay PI-KAN")
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--constraint", choices=("soft", "hard"), default=CONSTRAINT_MODE)
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--epochs", type=int, default=TUNING_EPOCHS)
    args = parser.parse_args()
    main(args.device, args.constraint, args.trials, args.epochs)
