"""Optuna tuning for the Kirchhoff PIKAN."""

from __future__ import annotations

import argparse
import json

import optuna
import torch

from PIKAN_2D import CONFIG, RESULT_PATH, KirchhoffPIKAN
from _common import (
    physics_loss, problem_values, relative_l2, sample_boundary,
    sample_interior, select_device, set_seed, train_model,
)


def objective_factory(mode: str, device: torch.device, epochs: int):
    tuning = CONFIG["tuning"]
    values = problem_values(CONFIG)
    seed = int(CONFIG["reproducibility"]["seed"])
    generator = torch.Generator(device=device).manual_seed(seed)
    interior = sample_interior(int(tuning["interior_points"]), values, device, generator)
    validation = sample_interior(int(tuning["validation_points"]), values, device, generator)
    boundaries = sample_boundary(int(tuning["boundary_points_per_edge"]), values, device)

    def objective(trial: optuna.Trial) -> float:
        set_seed(seed)
        architecture = (
            trial.suggest_int("n_layers", int(tuning["n_layers"]["min"]), int(tuning["n_layers"]["max"])),
            trial.suggest_int("n_neurons", int(tuning["n_neurons"]["min"]), int(tuning["n_neurons"]["max"]), step=int(tuning["n_neurons"]["step"])),
            trial.suggest_int("grid", int(tuning["grid"]["min"]), int(tuning["grid"]["max"])),
            trial.suggest_int("spline_order", int(tuning["spline_order"]["min"]), int(tuning["spline_order"]["max"])),
        )
        model = KirchhoffPIKAN(*architecture, mode, values, device, seed).to(device)
        loss_fn = lambda: physics_loss(model, interior, boundaries, values, CONFIG["loss_weights"])
        training = dict(CONFIG["training"])
        training["epochs"] = epochs
        training["print_every"] = 0
        train_model(model, loss_fn, training)
        return float(relative_l2(model, validation, values))
    return objective


def save(study: optuna.Study, mode: str) -> None:
    payload = json.loads(RESULT_PATH.read_text(encoding="utf-8")) if RESULT_PATH.exists() else {}
    payload[mode] = {"best_value": study.best_value, "best_params": study.best_params}
    RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(device_name=None, mode="hard", trials=None, epochs=None) -> None:
    tuning = CONFIG["tuning"]
    device = select_device(device_name or CONFIG["runtime"]["device"])
    sampler = optuna.samplers.TPESampler(seed=int(CONFIG["reproducibility"]["seed"]))
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        objective_factory(mode, device, int(epochs or tuning["epochs"])),
        n_trials=int(trials or tuning["trials"]),
    )
    save(study, mode)
    print(f"Best validation relative L2: {study.best_value:.6e}")
    print(f"Saved tuned hyperparameters to: {RESULT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), default=None)
    parser.add_argument("--constraint", choices=("soft", "hard"), default="hard")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    main(args.device, args.constraint, args.trials, args.epochs)

