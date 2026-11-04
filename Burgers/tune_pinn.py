"""Tune the Burgers PINN architecture with Optuna TPE."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Burgers.PINN_2D import (
    BOUNDARY_WEIGHT,
    CONFIG,
    CONSTRAINT_MODE,
    DEFAULT_DEVICE,
    INITIAL_WEIGHT,
    LEARNING_RATE,
    PDE_WEIGHT,
    RESULT_PATH,
    SEED,
    BurgersPINN,
    loss_components,
    numerical_reference_at_points,
    pinn_loss,
    relative_l2_error,
    sample_constraints,
    sample_interior,
    select_device,
    set_seed,
)
from hyperparameter_optimization import run_tpe_study
from training import train


TUNING_CONFIG = CONFIG["tuning"]
N_TRIALS = int(TUNING_CONFIG["trials"])
TUNING_EPOCHS = int(TUNING_CONFIG["epochs"])
N_TRAIN_POINTS = int(TUNING_CONFIG["training_points"])
N_VALIDATION_POINTS = int(TUNING_CONFIG["validation_points"])
N_BOUNDARY_POINTS_PER_EDGE = int(
    TUNING_CONFIG["boundary_points_per_edge"]
)
N_INITIAL_POINTS = int(TUNING_CONFIG["initial_points"])
N_LAYERS_MIN = int(TUNING_CONFIG["n_layers"]["min"])
N_LAYERS_MAX = int(TUNING_CONFIG["n_layers"]["max"])
N_NEURONS_MIN = int(TUNING_CONFIG["n_neurons"]["min"])
N_NEURONS_MAX = int(TUNING_CONFIG["n_neurons"]["max"])
N_NEURONS_STEP = int(TUNING_CONFIG["n_neurons"]["step"])
SCHEDULER_FACTOR = float(TUNING_CONFIG["scheduler"]["factor"])
SCHEDULER_PATIENCE = int(TUNING_CONFIG["scheduler"]["patience"])
MIN_LEARNING_RATE = float(
    TUNING_CONFIG["scheduler"]["min_learning_rate"]
)
EARLY_STOPPING_PATIENCE = int(
    TUNING_CONFIG["early_stopping"]["patience"]
)
MIN_DELTA = float(TUNING_CONFIG["early_stopping"]["min_delta"])


def make_objective(
    training_points: torch.Tensor,
    validation_points: torch.Tensor,
    validation_reference: torch.Tensor,
    constraint_points: dict[str, torch.Tensor],
    device: torch.device,
    constraint_mode: str,
    tuning_epochs: int = TUNING_EPOCHS,
):
    def objective(trial):
        n_layers = trial.suggest_int(
            "n_layers", N_LAYERS_MIN, N_LAYERS_MAX
        )
        n_neurons = trial.suggest_int(
            "n_neurons", N_NEURONS_MIN, N_NEURONS_MAX,
            step=N_NEURONS_STEP,
        )
        trial.set_user_attr("constraint_mode", constraint_mode)
        set_seed(SEED)
        model = BurgersPINN(
            n_layers, n_neurons, constraint_mode
        ).to(device)
        loss_fn = lambda: pinn_loss(
            model,
            training_points,
            constraint_points,
            pde_weight=PDE_WEIGHT,
            initial_weight=INITIAL_WEIGHT,
            boundary_weight=BOUNDARY_WEIGHT,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE,
            min_lr=MIN_LEARNING_RATE,
        )
        train(
            model.parameters(),
            loss_fn,
            optimizer,
            epochs=tuning_epochs,
            scheduler=scheduler,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            min_delta=MIN_DELTA,
            print_every=0,
        )
        validation_error = relative_l2_error(
            model, validation_points, validation_reference
        ).item()
        components = loss_components(
            model, training_points, constraint_points
        )
        for name, value in components.items():
            trial.set_user_attr(f"train_{name}", value.detach().item())
        return validation_error

    return objective


def main(
    device_name: str = DEFAULT_DEVICE,
    constraint_mode: str = CONSTRAINT_MODE,
    n_trials: int = N_TRIALS,
    tuning_epochs: int = TUNING_EPOCHS,
) -> None:
    set_seed(SEED)
    device = select_device(device_name)
    print(f"Using device: {device}")
    print(f"Constraint mode: {constraint_mode}")
    generator = torch.Generator(device=device).manual_seed(SEED)
    training_points = sample_interior(
        N_TRAIN_POINTS, device, generator
    )
    validation_points = sample_interior(
        N_VALIDATION_POINTS, device, generator
    )
    constraint_points = sample_constraints(
        N_BOUNDARY_POINTS_PER_EDGE, N_INITIAL_POINTS, device
    )
    print("Computing the validation reference solution once")
    validation_reference = numerical_reference_at_points(validation_points)
    study = run_tpe_study(
        make_objective(
            training_points,
            validation_points,
            validation_reference,
            constraint_points,
            device,
            constraint_mode,
            tuning_epochs,
        ),
        n_trials=n_trials,
        seed=SEED,
        direction="minimize",
        study_name=f"burgers_pinn_{constraint_mode}_tuning",
        result_path=RESULT_PATH,
    )
    print(f"Best relative validation L2 error: {study.best_value:.6e}")
    print(f"Best number of layers: {study.best_params['n_layers']}")
    print(f"Best neurons per layer: {study.best_params['n_neurons']}")
    print(f"Saved result to: {RESULT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune the Burgers PINN")
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument(
        "--constraint", choices=("soft", "hard"), default=CONSTRAINT_MODE
    )
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--epochs", type=int, default=TUNING_EPOCHS)
    arguments = parser.parse_args()
    main(
        arguments.device,
        arguments.constraint,
        arguments.trials,
        arguments.epochs,
    )
