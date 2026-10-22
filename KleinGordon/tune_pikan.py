"""Optuna tuning for the Klein--Gordon PIKAN."""

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
from KleinGordon.PIKAN_3D import (
    CONFIG, CONSTRAINT_MODE, DEFAULT_DEVICE, LOSS_WEIGHTS, RESULT_PATH, SEED,
    KleinGordonPIKAN, pikan_loss,
)
from KleinGordon._common import relative_l2_error, sample_constraints, sample_interior, select_device, set_seed
from training import train

TUNING = CONFIG["tuning"]


def save_mode_result(study, mode: str) -> None:
    try:
        payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        payload = {}
    payload[mode] = {"best_value": study.best_value, "best_params": study.best_params, "best_trial_number": study.best_trial.number}
    RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(device_name=DEFAULT_DEVICE, constraint_mode=CONSTRAINT_MODE, n_trials=None, epochs=None) -> None:
    n_trials = int(TUNING["trials"]) if n_trials is None else n_trials
    epochs = int(TUNING["epochs"]) if epochs is None else epochs
    set_seed(SEED)
    device = select_device(device_name)
    generator = torch.Generator(device=device).manual_seed(SEED)
    training_points = sample_interior(int(TUNING["training_points"]), device, generator)
    validation_points = sample_interior(int(TUNING["validation_points"]), device, generator)
    constraints = sample_constraints(int(TUNING["boundary_points_per_face"]), int(TUNING["initial_points"]), device, generator)

    def objective(trial):
        layers = trial.suggest_int("n_layers", int(TUNING["n_layers"]["min"]), int(TUNING["n_layers"]["max"]))
        neurons = trial.suggest_int("n_neurons", int(TUNING["n_neurons"]["min"]), int(TUNING["n_neurons"]["max"]))
        grid = trial.suggest_int("grid", int(TUNING["grid"]["min"]), int(TUNING["grid"]["max"]))
        order = trial.suggest_int("spline_order", int(TUNING["spline_order"]["min"]), int(TUNING["spline_order"]["max"]))
        set_seed(SEED)
        model = KleinGordonPIKAN(layers, neurons, grid, order, constraint_mode, device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(CONFIG["training"]["learning_rate"]))
        scheduler_cfg = TUNING["scheduler"]
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=float(scheduler_cfg["factor"]), patience=int(scheduler_cfg["patience"]), min_lr=float(scheduler_cfg["min_learning_rate"]))
        stopping = TUNING["early_stopping"]
        train(model.parameters(), lambda: pikan_loss(model, training_points, constraints, LOSS_WEIGHTS), optimizer, epochs=epochs, scheduler=scheduler, early_stopping_patience=int(stopping["patience"]), min_delta=float(stopping["min_delta"]), print_every=0)
        return relative_l2_error(model, validation_points).item()

    study = run_tpe_study(objective, n_trials=n_trials, seed=SEED, direction="minimize", study_name=f"klein_gordon_pikan_{constraint_mode}")
    save_mode_result(study, constraint_mode)
    print(f"Best relative validation L2 error: {study.best_value:.6e}")
    print(f"Best parameters: {study.best_params}")
    print(f"Saved result to: {RESULT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune the Klein-Gordon PIKAN")
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--constraint", choices=("soft", "hard"), default=CONSTRAINT_MODE)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    main(args.device, args.constraint, args.trials, args.epochs)
