"""Tune the Burgers PI-KAN architecture with Optuna TPE."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Burgers.PIKAN_2D import (
    BOUNDARY_WEIGHT,
    CONFIG,
    CONSTRAINT_MODE,
    DEFAULT_DEVICE,
    INITIAL_WEIGHT,
    LEARNING_RATE,
    PDE_WEIGHT,
    RESULT_PATH,
    SEED,
    BurgersPIKAN,
    loss_components,
    numerical_reference_at_points,
    pikan_loss,
    relative_l2_error,
    sample_constraints,
    sample_interior,
    select_device,
    set_seed,
)
from hyperparameter_optimization import run_tpe_study
from training import train


TUNING = CONFIG["tuning"]
N_TRIALS = int(TUNING["trials"])
TUNING_EPOCHS = int(TUNING["epochs"])
N_TRAIN_POINTS = int(TUNING["training_points"])
N_VALIDATION_POINTS = int(TUNING["validation_points"])
N_BOUNDARY_POINTS = int(TUNING["boundary_points_per_edge"])
SCHEDULER_FACTOR = float(TUNING["scheduler"]["factor"])
SCHEDULER_PATIENCE = int(TUNING["scheduler"]["patience"])
MIN_LEARNING_RATE = float(TUNING["scheduler"]["min_learning_rate"])
EARLY_STOPPING_PATIENCE = int(TUNING["early_stopping"]["patience"])
MIN_DELTA = float(TUNING["early_stopping"]["min_delta"])


def make_objective(
    training_points: torch.Tensor,
    validation_points: torch.Tensor,
    validation_reference: torch.Tensor,
    constraints: dict[str, torch.Tensor],
    device: torch.device,
    constraint_mode: str,
    epochs: int,
):
    def objective(trial):
        n_layers = trial.suggest_int(
            "n_layers", int(TUNING["n_layers"]["min"]),
            int(TUNING["n_layers"]["max"]),
        )
        n_neurons = trial.suggest_int(
            "n_neurons", int(TUNING["n_neurons"]["min"]),
            int(TUNING["n_neurons"]["max"]),
        )
        grid = trial.suggest_int(
            "grid", int(TUNING["grid"]["min"]), int(TUNING["grid"]["max"])
        )
        spline_order = trial.suggest_int(
            "spline_order", int(TUNING["spline_order"]["min"]),
            int(TUNING["spline_order"]["max"]),
        )
        set_seed(SEED)
        model = BurgersPIKAN(
            n_layers, n_neurons, grid, spline_order,
            constraint_mode, device, seed=SEED,
        ).to(device)
        loss_fn = lambda: pikan_loss(
            model, training_points, constraints,
            pde_weight=PDE_WEIGHT,
            initial_weight=INITIAL_WEIGHT,
            boundary_weight=BOUNDARY_WEIGHT,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE, min_lr=MIN_LEARNING_RATE,
        )
        train(
            model.parameters(), loss_fn, optimizer, epochs=epochs,
            scheduler=scheduler,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            min_delta=MIN_DELTA,
            print_every=0,
        )
        error = relative_l2_error(
            model, validation_points, validation_reference
        ).item()
        for name, value in loss_components(
            model, training_points, constraints
        ).items():
            trial.set_user_attr(f"train_{name}", value.detach().item())
        trial.set_user_attr("constraint_mode", constraint_mode)
        return error

    return objective


def save_result(study, constraint_mode: str) -> None:
    mode_result = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
    }
    if RESULT_PATH.exists():
        payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    else:
        payload = {}
    # Keep the tuned result for the other constraint mode.
    payload[constraint_mode] = mode_result
    RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(
    device_name: str = DEFAULT_DEVICE,
    constraint_mode: str = CONSTRAINT_MODE,
    n_trials: int = N_TRIALS,
    epochs: int = TUNING_EPOCHS,
) -> None:
    set_seed(SEED)
    device = select_device(device_name)
    print(f"Using device: {device}")
    print(f"Constraint mode: {constraint_mode}")
    generator = torch.Generator(device=device).manual_seed(SEED)
    training = sample_interior(N_TRAIN_POINTS, device, generator)
    validation = sample_interior(N_VALIDATION_POINTS, device, generator)
    constraints = sample_constraints(N_BOUNDARY_POINTS, device)
    print("Computing the validation reference solution once")
    reference = numerical_reference_at_points(validation)
    study = run_tpe_study(
        make_objective(
            training, validation, reference, constraints,
            device, constraint_mode, epochs,
        ),
        n_trials=n_trials,
        seed=SEED,
        direction="minimize",
        study_name=f"burgers_pikan_{constraint_mode}_tuning",
    )
    save_result(study, constraint_mode)
    print(f"Best relative validation L2 error: {study.best_value:.6e}")
    for name, value in study.best_params.items():
        print(f"Best {name}: {value}")
    print(f"Saved result to: {RESULT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune the Burgers PI-KAN")
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument(
        "--constraint", choices=("soft", "hard"), default=CONSTRAINT_MODE
    )
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--epochs", type=int, default=TUNING_EPOCHS)
    args = parser.parse_args()
    main(args.device, args.constraint, args.trials, args.epochs)
