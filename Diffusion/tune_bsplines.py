"""Tune the diffusion B-spline approximation with a seeded Optuna study."""

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
from Diffusion.Bspline_2D import (
    CONFIG, DIFFUSIVITY, INTERIOR_INITIALIZATION, LEARNING_RATE,
    RESULT_PATH, SEED, DiffusionBSpline,
    diffusion_loss, pretrain_initial_coefficients, relative_l2_error,
    sample_initial, sample_interior, select_device, set_seed,
)


def main(device_name=None, trials=None, epochs=None) -> None:
    tuning = CONFIG["tuning"]
    device = select_device(device_name or CONFIG["runtime"]["device"])
    generator = torch.Generator(device=device).manual_seed(SEED)
    interior = sample_interior(
        int(tuning["training_points"]), device, generator
    )
    validation = sample_interior(
        int(tuning["validation_points"]), device, generator
    )
    initial_points, initial_targets = sample_initial(
        int(tuning["initial_points"]), device
    )

    def objective(trial: optuna.Trial) -> float:
        set_seed(SEED)
        degree = trial.suggest_int(
            "degree", int(tuning["degree"]["min"]), int(tuning["degree"]["max"])
        )
        n_basis = trial.suggest_int(
            "n_basis", max(degree + 1, int(tuning["n_basis"]["min"])),
            int(tuning["n_basis"]["max"]),
        )
        model = DiffusionBSpline(
            n_basis, degree, device, INTERIOR_INITIALIZATION
        )
        pretrain_initial_coefficients(
            model, initial_points, initial_targets,
            epochs=int(epochs or tuning["epochs"]),
            learning_rate=LEARNING_RATE,
            scheduler_factor=float(tuning["scheduler"]["factor"]),
            scheduler_patience=int(tuning["scheduler"]["patience"]),
            minimum_learning_rate=float(
                tuning["scheduler"]["min_learning_rate"]
            ),
            early_stopping_patience=int(
                tuning["early_stopping"]["patience"]
            ),
            min_delta=float(tuning["early_stopping"]["min_delta"]),
            print_every=0,
        )
        operators = model.operators(interior)
        optimizer = torch.optim.Adam([model.interior_coeffs], lr=LEARNING_RATE)
        scheduler_config = tuning["scheduler"]
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=float(scheduler_config["factor"]),
            patience=int(scheduler_config["patience"]),
            min_lr=float(scheduler_config["min_learning_rate"]),
        )
        stopping = tuning["early_stopping"]
        train(
            [model.interior_coeffs], lambda: diffusion_loss(model, operators), optimizer,
            epochs=int(epochs or tuning["epochs"]), scheduler=scheduler,
            early_stopping_patience=int(stopping["patience"]),
            min_delta=float(stopping["min_delta"]), print_every=0,
        )
        return float(relative_l2_error(model, validation, DIFFUSIVITY))

    study = run_tpe_study(
        objective, n_trials=int(trials or tuning["trials"]), seed=SEED,
        direction="minimize", result_path=RESULT_PATH,
    )
    print(f"Best validation relative L2 error: {study.best_value:.6e}")
    print(f"Best hyperparameters: {study.best_params}")
    print(f"Saved: {RESULT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune diffusion B-splines")
    parser.add_argument("--device", choices=("cpu", "gpu"))
    parser.add_argument("--trials", type=int)
    parser.add_argument("--epochs", type=int)
    arguments = parser.parse_args()
    main(arguments.device, arguments.trials, arguments.epochs)
