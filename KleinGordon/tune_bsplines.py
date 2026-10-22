"""Optuna tuning for the Klein--Gordon tensor-product B-spline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hyperparameter_optimization import run_tpe_study
from KleinGordon.Bspline_3D import (
    CONFIG, CONSTRAINT_MODE, DEFAULT_DEVICE, LOSS_WEIGHTS, RESULT_PATH, SEED,
    KleinGordonBSpline, bspline_loss, precompute_operators,
)
from KleinGordon._common import relative_l2_error, sample_constraints, sample_interior, select_device, set_seed
from training import train

TUNING = CONFIG["tuning"]


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
        n_space = trial.suggest_int("n_basis_space", int(TUNING["n_basis_space"]["min"]), int(TUNING["n_basis_space"]["max"]))
        n_time = trial.suggest_int("n_basis_time", int(TUNING["n_basis_time"]["min"]), int(TUNING["n_basis_time"]["max"]))
        degree = trial.suggest_int("degree", int(TUNING["degree"]["min"]), int(TUNING["degree"]["max"]))
        if degree >= min(n_space, n_time):
            raise ValueError("degree must be smaller than every basis count")
        set_seed(SEED)
        model = KleinGordonBSpline(n_space, n_time, degree, constraint_mode, device).to(device)
        training_operators = precompute_operators(model, training_points)
        constraint_operators = {
            name: precompute_operators(model, points)
            for name, points in constraints.items()
        }
        optimizer = torch.optim.Adam(model.parameters(), lr=float(CONFIG["training"]["learning_rate"]))
        scheduler_cfg = TUNING["scheduler"]
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=float(scheduler_cfg["factor"]), patience=int(scheduler_cfg["patience"]), min_lr=float(scheduler_cfg["min_learning_rate"]))
        stopping = TUNING["early_stopping"]
        train(
            model.parameters(),
            lambda: bspline_loss(
                model, training_operators, training_points,
                constraint_operators, constraints, LOSS_WEIGHTS,
            ),
            optimizer, epochs=epochs, scheduler=scheduler,
            early_stopping_patience=int(stopping["patience"]),
            min_delta=float(stopping["min_delta"]), print_every=0,
        )
        return relative_l2_error(model, validation_points).item()

    study = run_tpe_study(objective, n_trials=n_trials, seed=SEED, direction="minimize", study_name=f"klein_gordon_bspline_{constraint_mode}", result_path=RESULT_PATH)
    print(f"Best relative validation L2 error: {study.best_value:.6e}")
    print(f"Best parameters: {study.best_params}")
    print(f"Saved result to: {RESULT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune Klein-Gordon B-splines")
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--constraint", choices=("soft", "hard"), default=CONSTRAINT_MODE)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    main(args.device, args.constraint, args.trials, args.epochs)
