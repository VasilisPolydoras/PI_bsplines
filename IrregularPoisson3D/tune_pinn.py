"""Tune the geometry-fitted 3D Poisson PINN."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hyperparameter_optimization import run_tpe_study
from IrregularPoisson3D.PINN_3D import (
    DEFAULT_DEVICE, LEARNING_RATE, RESULT_PATH, SEED, PINN,
    geometry_operator_terms, poisson_residual_loss, relative_l2_error,
    sample_parameter_domain, select_device, set_seed, source_function,
)
from training import train

N_TRIALS = 20
TUNING_EPOCHS = 3_000
N_TRAIN_POINTS = 1_000
N_VALIDATION_POINTS = 3_000


def make_objective(training, validation, device, tuning_epochs=TUNING_EPOCHS):
    training_geometry = geometry_operator_terms(training)
    validation_geometry = geometry_operator_terms(validation)
    training_source = source_function(training, training_geometry)

    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 2, 5)
        n_neurons = trial.suggest_int("n_neurons", 20, 100, step=20)
        set_seed(SEED); model = PINN(n_layers, n_neurons).to(device)
        loss_fn = lambda: poisson_residual_loss(
            model, training, training_geometry, training_source
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=.8, patience=100, min_lr=1e-5
        )
        train(model.parameters(), loss_fn, optimizer, epochs=tuning_epochs,
              scheduler=scheduler, early_stopping_patience=500,
              min_delta=1e-9, print_every=0)
        with torch.no_grad():
            error = relative_l2_error(model, validation, validation_geometry).item()
        trial.set_user_attr("train_residual", loss_fn().detach().item())
        return error
    return objective


def main(device_name=DEFAULT_DEVICE, n_trials=N_TRIALS,
         tuning_epochs=TUNING_EPOCHS):
    set_seed(SEED); device = select_device(device_name)
    generator = torch.Generator(device=device).manual_seed(SEED)
    training = sample_parameter_domain(N_TRAIN_POINTS, device, generator)
    validation = sample_parameter_domain(N_VALIDATION_POINTS, device, generator)
    study = run_tpe_study(
        make_objective(training, validation, device, tuning_epochs),
        n_trials=n_trials, seed=SEED, direction="minimize",
        study_name="irregular_poisson_3d_pinn_tuning", result_path=RESULT_PATH,
    )
    print(f"Best relative validation L2 error: {study.best_value:.6e}")
    print(f"Best layers: {study.best_params['n_layers']}")
    print(f"Best neurons: {study.best_params['n_neurons']}")
    print(f"Saved result to: {RESULT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune the 3D Poisson PINN")
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--epochs", type=int, default=TUNING_EPOCHS)
    args = parser.parse_args(); main(args.device, args.trials, args.epochs)
