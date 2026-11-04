"""Tune XPINN depth and width for the singular L-shaped Laplace problem."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hyperparameter_optimization import run_tpe_study
from LShapeLaplace.Irregular.XPINN_2D import (
    CORNER_EPSILON,
    DEFAULT_DEVICE,
    INTERFACE_FLUX_WEIGHT,
    INTERFACE_RESIDUAL_WEIGHT,
    INTERFACE_SOLUTION_WEIGHT,
    LEARNING_RATE,
    PDE_WEIGHT,
    RESULT_PATH,
    SEED,
    LShapeXPINN,
    PATCH_NAMES,
    generate_interface_points,
    map_patch,
    relative_l2_error,
    select_device,
    set_seed,
    xpinn_loss,
    xpinn_loss_components,
)
from training import train


N_TRIALS = 50
TUNING_EPOCHS = 1000
N_TRAIN_POINTS_PER_PATCH = 2_000
N_VALIDATION_POINTS_PER_PATCH = 6_000
N_INTERFACE_POINTS = 500


def sample_common_patch_points(
    count: int,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Map one common local sample to every patch for paired comparisons."""
    local_points = torch.rand(
        count, 2, device=device, generator=generator
    )
    return {
        name: map_patch(name, local_points)
        for name in PATCH_NAMES
    }


def make_objective(
    training_points: dict[str, torch.Tensor],
    validation_points: dict[str, torch.Tensor],
    interface_points,
    device: torch.device,
    tuning_epochs: int = TUNING_EPOCHS,
):
    """Create an Optuna objective for common subnetwork depth and width."""
    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 2, 5)
        n_neurons = trial.suggest_int("n_neurons", 20, 100, step=20)

        set_seed(SEED)
        model = LShapeXPINN(n_layers, n_neurons).to(device)
        loss_fn = lambda: xpinn_loss(
            model,
            training_points,
            None,
            interface_points,
            pde_weight=PDE_WEIGHT,
            interface_solution_weight=INTERFACE_SOLUTION_WEIGHT,
            interface_residual_weight=INTERFACE_RESIDUAL_WEIGHT,
            interface_flux_weight=INTERFACE_FLUX_WEIGHT,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.8, patience=50, min_lr=1.0e-5
        )
        train(
            model.parameters(),
            loss_fn,
            optimizer,
            epochs=tuning_epochs,
            scheduler=scheduler,
            early_stopping_patience=500,
            min_delta=1.0e-9,
            print_every=0,
        )

        validation_error = relative_l2_error(model, validation_points).item()
        components = xpinn_loss_components(
            model, training_points, None, interface_points
        )
        for name, value in components.items():
            trial.set_user_attr(f"train_{name}", value.detach().item())
        return validation_error

    return objective


def main(
    device_name: str = DEFAULT_DEVICE,
    n_trials: int = N_TRIALS,
    tuning_epochs: int = TUNING_EPOCHS,
) -> None:
    set_seed(SEED)
    device = select_device(device_name)
    print(f"Using device: {device}")
    generator = torch.Generator(device=device).manual_seed(SEED)
    training_points = sample_common_patch_points(
        N_TRAIN_POINTS_PER_PATCH, device, generator
    )
    validation_points = sample_common_patch_points(
        N_VALIDATION_POINTS_PER_PATCH, device, generator
    )
    interface_points = generate_interface_points(
        N_INTERFACE_POINTS, device, CORNER_EPSILON
    )

    study = run_tpe_study(
        make_objective(
            training_points,
            validation_points,
            interface_points,
            device,
            tuning_epochs,
        ),
        n_trials=n_trials,
        seed=SEED,
        direction="minimize",
        study_name="l_shape_irregular_xpinn_tuning",
        result_path=RESULT_PATH,
    )
    print(f"Best relative validation L2 error: {study.best_value:.6e}")
    print(f"Best number of layers: {study.best_params['n_layers']}")
    print(f"Best neurons per layer: {study.best_params['n_neurons']}")
    print(f"Saved result to: {RESULT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune XPINNs for the singular L-shaped Laplace problem"
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--epochs", type=int, default=TUNING_EPOCHS)
    arguments = parser.parse_args()
    main(arguments.device, arguments.trials, arguments.epochs)
