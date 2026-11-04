"""Repeat one wave-equation solver with fixed samples and varied initialization."""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import math
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from WaveEquation.Bspline_2D import DEFAULT_DEVICE, SEED


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS = 10
DEFAULT_INITIALIZATION_SEED_START = 32


def confidence_interval_95(values: np.ndarray) -> tuple[float, float]:
    """Return a two-sided Student-t 95% confidence interval for the mean."""
    if values.size < 2:
        raise ValueError("At least two runs are required")
    try:
        from scipy.stats import t

        critical = float(t.ppf(0.975, df=values.size - 1))
    except ImportError:
        if values.size != 10:
            raise RuntimeError("SciPy is required when the number of runs is not 10")
        critical = 2.2621571628540993
    mean = float(values.mean())
    half_width = critical * float(values.std(ddof=1)) / math.sqrt(values.size)
    return mean - half_width, mean + half_width


def summarize(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "ci95": np.asarray(confidence_interval_95(values)),
    }


def solver_definition(name: str):
    if name == "bspline":
        from WaveEquation.Bspline_2D import EPOCHS, main

        return main, EPOCHS, "PI-TP-BsA", None
    if name == "pinn":
        from WaveEquation.PINN_2D import CONSTRAINT_MODE, EPOCHS, main

        return main, EPOCHS, "PINN", CONSTRAINT_MODE
    if name == "pikan":
        from WaveEquation.PIKAN_2D import CONSTRAINT_MODE, EPOCHS, main

        return main, EPOCHS, "PIKAN", CONSTRAINT_MODE
    raise ValueError("solver must be 'bspline', 'pinn', or 'pikan'")


def main(
    solver: str,
    runs: int = DEFAULT_RUNS,
    device_name: str = DEFAULT_DEVICE,
    sampling_seed: int = SEED,
    validation_seed: int = SEED,
    epochs: int | None = None,
    constraint_mode: str | None = None,
) -> None:
    if runs < 2:
        raise ValueError("runs must be at least two")
    run_solver, configured_epochs, label, configured_constraint = solver_definition(
        solver
    )
    if solver == "bspline" and constraint_mode is not None:
        raise ValueError("--constraint applies only to PINN and PIKAN")
    selected_constraint = (
        constraint_mode if constraint_mode is not None else configured_constraint
    )
    run_epochs = configured_epochs if epochs is None else epochs
    seeds = np.arange(
        DEFAULT_INITIALIZATION_SEED_START,
        DEFAULT_INITIALIZATION_SEED_START + runs,
        dtype=np.int64,
    )
    errors = np.empty(runs)
    times = np.empty(runs)
    for index, seed in enumerate(seeds):
        solver_arguments = {
            "device_name": device_name,
            "epochs": run_epochs,
            "use_tuned": None,
            "initialization_seed": int(seed),
            "sampling_seed": sampling_seed,
            "validation_seed": validation_seed,
            "save_outputs": False,
            "verbose": False,
        }
        if selected_constraint is not None:
            solver_arguments["constraint_mode"] = selected_constraint
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_solver(**solver_arguments)
        errors[index] = result["validation_relative_l2_error"]
        times[index] = result["training_time_seconds"]
        method_label = (
            label if selected_constraint is None else f"{label}-{selected_constraint}"
        )
        print(
            f"Run {index + 1:02d}/{runs} | {method_label:12s} | seed={seed} | "
            f"relative L2={errors[index]:.6e} | time={times[index]:.3f} s"
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    error_stats, time_stats = summarize(errors), summarize(times)
    output_stem = solver if selected_constraint is None else f"{solver}_{selected_constraint}"
    output = SCRIPT_DIR / "postprocessing" / f"{output_stem}_statistics.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        solver=np.asarray(solver),
        constraint_mode=np.asarray(
            "not_applicable" if selected_constraint is None else selected_constraint
        ),
        initialization_seeds=seeds,
        sampling_seed=np.asarray(sampling_seed),
        validation_seed=np.asarray(validation_seed),
        epochs=np.asarray(run_epochs),
        l2_errors=errors,
        training_times_seconds=times,
        l2_mean=error_stats["mean"],
        l2_std=error_stats["std"],
        l2_ci95=error_stats["ci95"],
        time_mean_seconds=time_stats["mean"],
        time_std_seconds=time_stats["std"],
        time_ci95_seconds=time_stats["ci95"],
    )
    print(
        f"\nRelative L2: {error_stats['mean']:.6e} +/- {error_stats['std']:.6e}; "
        f"95% CI [{error_stats['ci95'][0]:.6e}, {error_stats['ci95'][1]:.6e}]"
    )
    print(
        f"Training time: {time_stats['mean']:.3f} +/- {time_stats['std']:.3f} s; "
        f"95% CI [{time_stats['ci95'][0]:.3f}, "
        f"{time_stats['ci95'][1]:.3f}] s"
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wave-equation repeated-run statistics")
    parser.add_argument("--solver", choices=("bspline", "pinn", "pikan"), required=True)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--sampling-seed", type=int, default=SEED)
    parser.add_argument("--validation-seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--constraint", choices=("soft", "hard"),
        help=(
            "PINN/PIKAN constraint mode; when omitted, use the mode in the "
            "solver configuration file"
        ),
    )
    arguments = parser.parse_args()
    main(
        arguments.solver, arguments.runs, arguments.device,
        arguments.sampling_seed, arguments.validation_seed, arguments.epochs,
        arguments.constraint,
    )
