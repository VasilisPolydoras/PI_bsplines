"""Run initialization-seed statistics for one irregular-domain solver.

The collocation and validation sets are fixed across every run. Only the
trainable-parameter initialization seed changes. Normal solver plots and
per-run NPZ files are disabled; this script writes one compact statistical
result file instead.
"""

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LShapeLaplace.Irregular.Bspline_2D import (
    DEFAULT_DEVICE,
    EPOCHS as BSPLINE_EPOCHS,
    SEED as DEFAULT_SEED,
    main as run_bspline,
)
from LShapeLaplace.Irregular.XPINN_2D import (
    EPOCHS as XPINN_EPOCHS,
    main as run_xpinn,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "postprocessing"
DEFAULT_RUNS = 10
DEFAULT_INITIALIZATION_SEED_START = 32
DEFAULT_SAMPLING_SEED = DEFAULT_SEED
DEFAULT_VALIDATION_SEED = DEFAULT_SEED


def confidence_interval_95(values: np.ndarray) -> tuple[float, float]:
    """Return a two-sided 95% Student-t confidence interval for the mean."""
    count = values.size
    if count < 2:
        raise ValueError("At least two runs are required for a confidence interval")
    try:
        from scipy.stats import t

        critical_value = float(t.ppf(0.975, df=count - 1))
    except ImportError:
        if count != 10:
            raise RuntimeError(
                "SciPy is required to calculate a Student-t confidence "
                "interval when the number of runs is not 10"
            )
        critical_value = 2.2621571628540993
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / math.sqrt(count))
    half_width = critical_value * standard_error
    return mean - half_width, mean + half_width


def summarize(values: np.ndarray) -> dict[str, float | np.ndarray]:
    """Return the mean, sample deviation, and 95% CI of one measurement."""
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "ci95": np.asarray(confidence_interval_95(values), dtype=float),
    }


def release_cached_memory() -> None:
    """Release objects and unused CUDA cache between independent runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_quietly(function, **kwargs) -> dict[str, float]:
    """Run one solver without its routine console output."""
    with contextlib.redirect_stdout(io.StringIO()):
        return function(**kwargs)


def main(
    *,
    solver: str,
    runs: int = DEFAULT_RUNS,
    device_name: str = DEFAULT_DEVICE,
    sampling_seed: int = DEFAULT_SAMPLING_SEED,
    validation_seed: int = DEFAULT_VALIDATION_SEED,
    epochs: int | None = None,
    output_path: Path | None = None,
) -> None:
    if runs < 2:
        raise ValueError("runs must be at least two")
    if solver not in {"bspline", "xpinn"}:
        raise ValueError("solver must be either 'bspline' or 'xpinn'")
    if output_path is None:
        output_path = DEFAULT_OUTPUT_DIR / f"{solver}_statistics.npz"

    solver_epochs = (
        BSPLINE_EPOCHS if solver == "bspline" else XPINN_EPOCHS
    ) if epochs is None else epochs
    initialization_seeds = np.arange(
        DEFAULT_INITIALIZATION_SEED_START,
        DEFAULT_INITIALIZATION_SEED_START + runs,
        dtype=np.int64,
    )
    errors = np.empty(runs, dtype=float)
    times = np.empty(runs, dtype=float)

    def execute(index: int, seed: int) -> None:
        common = {
            "device_name": device_name,
            "use_tuned": True,
            "initialization_seed": seed,
            "sampling_seed": sampling_seed,
            "validation_seed": validation_seed,
            "save_outputs": False,
            "verbose": False,
        }
        if solver == "bspline":
            result = run_quietly(run_bspline, epochs=solver_epochs, **common)
            label = "B-spline"
        else:
            result = run_quietly(run_xpinn, epochs=solver_epochs, **common)
            label = "XPINN"
        errors[index] = result["validation_relative_l2_error"]
        times[index] = result["training_time_seconds"]
        print(
            f"Run {index + 1:02d}/{runs} | {label:8s} | seed={seed:2d} | "
            f"relative L2={result['validation_relative_l2_error']:.6e} | "
            f"training time={result['training_time_seconds']:.3f} s"
        )
        release_cached_memory()

    for index, seed_value in enumerate(initialization_seeds):
        seed = int(seed_value)
        execute(index, seed)

    error_stats = summarize(errors)
    time_stats = summarize(times)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        solver=np.asarray(solver),
        initialization_seeds=initialization_seeds,
        sampling_seed=np.asarray(sampling_seed, dtype=np.int64),
        validation_seed=np.asarray(validation_seed, dtype=np.int64),
        l2_errors=errors,
        training_times_seconds=times,
        l2_mean=error_stats["mean"],
        l2_std=error_stats["std"],
        l2_ci95=error_stats["ci95"],
        time_mean_seconds=time_stats["mean"],
        time_std_seconds=time_stats["std"],
        time_ci95_seconds=time_stats["ci95"],
    )

    print("\nSummary (mean +/- sample standard deviation)")
    print(
        f"Relative L2: {error_stats['mean']:.6e} +/- "
        f"{error_stats['std']:.6e}; 95% CI "
        f"[{error_stats['ci95'][0]:.6e}, {error_stats['ci95'][1]:.6e}]"
    )
    print(
        f"Training time: {time_stats['mean']:.3f} +/- "
        f"{time_stats['std']:.3f} s; 95% CI "
        f"[{time_stats['ci95'][0]:.3f}, {time_stats['ci95'][1]:.3f}] s"
    )
    print(f"Saved statistical results to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run initialization-seed statistics for one irregular-domain solver"
        )
    )
    parser.add_argument(
        "--solver",
        choices=("bspline", "xpinn"),
        required=True,
        help="Choose the solver to run",
    )
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--sampling-seed", type=int, default=DEFAULT_SAMPLING_SEED)
    parser.add_argument("--validation-seed", type=int, default=DEFAULT_VALIDATION_SEED)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional common epoch budget; defaults to each solver configuration",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output path; defaults to postprocessing/"
            "<solver>_statistics.npz"
        ),
    )
    arguments = parser.parse_args()
    main(
        solver=arguments.solver,
        runs=arguments.runs,
        device_name=arguments.device,
        sampling_seed=arguments.sampling_seed,
        validation_seed=arguments.validation_seed,
        epochs=arguments.epochs,
        output_path=arguments.output,
    )
