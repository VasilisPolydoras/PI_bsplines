"""Problem-independent Optuna helpers."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import optuna


def run_tpe_study(
    objective: Callable[[optuna.Trial], float],
    *,
    n_trials: int,
    seed: int = 42,
    direction: str = "minimize",
    study_name: str | None = None,
    storage: str | None = None,
    result_path: str | Path | None = None,
) -> optuna.Study:
    """Optimize any Optuna objective with a seeded TPE sampler.

    ``storage`` may be an Optuna URL such as ``sqlite:///study.db``. Supplying
    it together with ``study_name`` makes interrupted studies resumable.
    """
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        study_name=study_name,
        storage=storage,
        load_if_exists=storage is not None and study_name is not None,
    )
    study.optimize(objective, n_trials=n_trials)

    if result_path is not None:
        save_best_trial(study, result_path)

    return study


def save_best_trial(study: optuna.Study, result_path: str | Path) -> None:
    """Save the best objective value and hyperparameters as JSON."""
    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
    }
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
