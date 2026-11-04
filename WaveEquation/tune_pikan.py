"""Tune the wave-equation PIKAN with a seeded Optuna study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import optuna
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hyperparameter_optimization import run_tpe_study
from training import train
from WaveEquation.PIKAN_2D import (
    CONFIG, LEARNING_RATE, RESULT_PATH, SEED, WAVE_SPEED, WavePIKAN,
    pikan_loss, relative_l2_error, sample_constraints, sample_interior,
    select_device, set_seed,
)


def main(device_name=None, constraint_mode=None, trials=None, epochs=None) -> None:
    tuning = CONFIG["tuning"]
    mode = constraint_mode or CONFIG["model"]["constraint_mode"]
    device = select_device(device_name or CONFIG["runtime"]["device"])
    generator = torch.Generator(device=device).manual_seed(SEED)
    interior = sample_interior(
        int(tuning["training_points"]), device, generator=generator
    )
    validation = sample_interior(
        int(tuning["validation_points"]), device, generator=generator
    )
    constraints = sample_constraints(
        int(tuning["boundary_points_per_edge"]),
        int(tuning["initial_points"]), device,
    )

    def objective(trial: optuna.Trial) -> float:
        set_seed(SEED)
        model = WavePIKAN(
            trial.suggest_int(
                "n_layers", int(tuning["n_layers"]["min"]),
                int(tuning["n_layers"]["max"]),
            ),
            trial.suggest_int(
                "n_neurons", int(tuning["n_neurons"]["min"]),
                int(tuning["n_neurons"]["max"]),
                step=int(tuning["n_neurons"].get("step", 1)),
            ),
            trial.suggest_int(
                "grid", int(tuning["grid"]["min"]), int(tuning["grid"]["max"])
            ),
            trial.suggest_int(
                "spline_order", int(tuning["spline_order"]["min"]),
                int(tuning["spline_order"]["max"]),
            ), mode, device, SEED,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=float(tuning["scheduler"]["factor"]),
            patience=int(tuning["scheduler"]["patience"]),
            min_lr=float(tuning["scheduler"]["min_learning_rate"]),
        )
        stopping = tuning["early_stopping"]
        train(
            model.parameters(), lambda: pikan_loss(model, interior, constraints),
            optimizer, epochs=int(epochs or tuning["epochs"]), scheduler=scheduler,
            early_stopping_patience=int(stopping["patience"]),
            min_delta=float(stopping["min_delta"]), print_every=0,
        )
        return float(relative_l2_error(model, validation, WAVE_SPEED))

    study = run_tpe_study(
        objective, n_trials=int(trials or tuning["trials"]), seed=SEED,
        direction="minimize", result_path=RESULT_PATH,
    )
    print(f"Constraint mode: {mode}")
    print(f"Best validation relative L2 error: {study.best_value:.6e}")
    print(f"Best hyperparameters: {study.best_params}")
    print(f"Saved: {RESULT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune the wave-equation PIKAN")
    parser.add_argument("--device", choices=("cpu", "gpu"))
    parser.add_argument("--constraint", choices=("hard", "soft"))
    parser.add_argument("--trials", type=int)
    parser.add_argument("--epochs", type=int)
    arguments = parser.parse_args()
    main(arguments.device, arguments.constraint, arguments.trials, arguments.epochs)
