"""Monitor NVIDIA GPU utilization and memory while running a solver.

Example
-------
python benchmark_gpu.py --label diffusion_pinn_hard --gpu 0 -- \
    python Diffusion/PINN_2D.py --constraint hard --device gpu

For an intentionally short hardware profile, add, for example,
``--max-runtime-seconds 60 --warmup-seconds 5`` before ``--``.  The solver is
then terminated after 60 seconds, while the first five seconds are retained in
the raw CSV but excluded from the reported steady-state statistics.

The script saves raw samples as CSV and a compact JSON summary.  It uses only
the Python standard library and the ``nvidia-smi`` executable.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


QUERY_FIELDS = "utilization.gpu,utilization.memory,memory.used"


def parse_sample(line: str) -> tuple[float, float, float]:
    values = [value.strip() for value in line.strip().split(",")]
    if len(values) != 3:
        raise ValueError(f"Unexpected nvidia-smi output: {line!r}")
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def query_once(nvidia_smi: str, gpu: str) -> tuple[float, float, float]:
    command = [
        nvidia_smi,
        "-i",
        gpu,
        f"--query-gpu={QUERY_FIELDS}",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    line = next((item for item in result.stdout.splitlines() if item.strip()), "")
    return parse_sample(line)


def percentile(values: Sequence[float], percentage: float) -> float:
    """Return a linearly interpolated percentile without NumPy."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute a percentile of an empty sequence")
    position = (len(ordered) - 1) * percentage / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def ensure_available(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing output: {joined}. "
            "Pass --overwrite to replace it."
        )


def prepare_isolated_run(
    command: Sequence[str], project_root: Path
) -> tuple[tempfile.TemporaryDirectory[str], list[str], Path]:
    """Copy the selected PDE code to a temporary workspace.

    Existing postprocessing directories are deliberately excluded, so a
    solver that finishes and saves normally can only write inside the
    temporary directory.
    """
    script_index = -1
    script_relative: Path | None = None
    resolved_root = project_root.resolve()
    for index, token in enumerate(command):
        candidate = Path(token)
        if candidate.suffix.lower() != ".py":
            continue
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (resolved_root / candidate).resolve()
        )
        try:
            relative = resolved.relative_to(resolved_root)
        except ValueError:
            continue
        if resolved.is_file() and len(relative.parts) >= 2:
            script_index = index
            script_relative = relative
            break
    if script_relative is None:
        raise SystemExit(
            "Safe output isolation requires a solver script located inside "
            "this repository, for example Kirchoff/PIKAN_2D.py. Pass "
            "--allow-solver-writes only if writing to the original solver "
            "directories is intentional."
        )

    temporary = tempfile.TemporaryDirectory(prefix="pi_bsplines_gpu_profile_")
    isolated_root = Path(temporary.name)

    # The PDE scripts import shared Python modules from the repository root.
    for source in resolved_root.glob("*.py"):
        shutil.copy2(source, isolated_root / source.name)

    source_package = resolved_root / script_relative.parts[0]
    destination_package = isolated_root / script_relative.parts[0]
    shutil.copytree(
        source_package,
        destination_package,
        ignore=shutil.ignore_patterns(
            "postprocessing", "__pycache__", "*.pyc", "*.pyo"
        ),
    )

    isolated_command = list(command)
    executable = shutil.which(isolated_command[0])
    if executable:
        isolated_command[0] = str(Path(executable).resolve())
    elif not Path(isolated_command[0]).is_absolute():
        local_executable = (resolved_root / isolated_command[0]).resolve()
        if local_executable.exists():
            isolated_command[0] = str(local_executable)
    isolated_command[script_index] = str(isolated_root / script_relative)
    return temporary, isolated_command, isolated_root


def monitor_reader(
    process: subprocess.Popen[str],
    samples: list[dict[str, float | str]],
    start_time: float,
) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        if not line.strip():
            continue
        try:
            gpu_util, memory_util, memory_used = parse_sample(line)
        except ValueError as error:
            print(f"Warning: {error}", file=sys.stderr)
            continue
        samples.append(
            {
                "timestamp": datetime.now().astimezone().isoformat(),
                "elapsed_s": time.perf_counter() - start_time,
                "gpu_utilization_percent": gpu_util,
                "memory_utilization_percent": memory_util,
                "memory_used_mib": memory_used,
            }
        )


def write_samples(path: Path, samples: Sequence[dict[str, float | str]]) -> None:
    fields = [
        "timestamp",
        "elapsed_s",
        "gpu_utilization_percent",
        "memory_utilization_percent",
        "memory_used_mib",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(samples)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Output-file prefix.")
    parser.add_argument("--gpu", default="0", help="Physical NVIDIA GPU index.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("gpu_benchmarks"),
        help="Directory for CSV and JSON results.",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=500,
        help="Sampling interval in milliseconds.",
    )
    parser.add_argument(
        "--baseline-samples",
        type=int,
        default=5,
        help="Number of idle memory samples collected before training.",
    )
    parser.add_argument(
        "--baseline-interval",
        type=float,
        default=0.2,
        help="Seconds between idle baseline samples.",
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        help=(
            "Stop the solver after this many seconds. Omit this option to run "
            "the solver to completion."
        ),
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=5.0,
        help=(
            "Initial seconds retained in the raw CSV but excluded from "
            "utilization and memory statistics."
        ),
    )
    parser.add_argument(
        "--minimum-samples",
        type=int,
        default=20,
        help="Warn if fewer post-warm-up samples are available.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files with the same label.",
    )
    parser.add_argument(
        "--allow-solver-writes",
        action="store_true",
        help=(
            "Run in the original repository instead of an isolated temporary "
            "copy. This may overwrite the solver's normal result files."
        ),
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Solver command, placed after --.",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    command = list(arguments.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("A solver command must be supplied after --")
    if arguments.interval_ms < 100:
        raise SystemExit("--interval-ms must be at least 100")
    if arguments.baseline_samples < 1:
        raise SystemExit("--baseline-samples must be at least 1")
    if (
        arguments.max_runtime_seconds is not None
        and arguments.max_runtime_seconds <= 0
    ):
        raise SystemExit("--max-runtime-seconds must be positive")
    if arguments.warmup_seconds < 0:
        raise SystemExit("--warmup-seconds cannot be negative")
    if arguments.minimum_samples < 1:
        raise SystemExit("--minimum-samples must be at least 1")

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise SystemExit("nvidia-smi was not found on PATH")

    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / f"{arguments.label}_gpu_samples.csv"
    summary_path = output_dir / f"{arguments.label}_gpu_summary.json"
    ensure_available((sample_path, summary_path), arguments.overwrite)

    project_root = Path(__file__).resolve().parent
    isolation: tempfile.TemporaryDirectory[str] | None = None
    run_command = command
    run_directory = project_root
    if not arguments.allow_solver_writes:
        isolation, run_command, run_directory = prepare_isolated_run(
            command, project_root
        )
        print(
            "Solver output isolation enabled: PDE result files will be "
            "written only to a temporary workspace."
        )
    else:
        print(
            "Warning: solver output isolation is disabled; normal PDE result "
            "files may be overwritten.",
            file=sys.stderr,
        )

    baseline_memory: list[float] = []
    print(f"Collecting {arguments.baseline_samples} idle baseline samples...")
    for index in range(arguments.baseline_samples):
        _, _, memory_used = query_once(nvidia_smi, arguments.gpu)
        baseline_memory.append(memory_used)
        if index + 1 < arguments.baseline_samples:
            time.sleep(arguments.baseline_interval)
    baseline_mib = statistics.median(baseline_memory)
    print(f"Baseline GPU memory: {baseline_mib:.1f} MiB")

    monitor_command = [
        nvidia_smi,
        "-i",
        arguments.gpu,
        f"--query-gpu={QUERY_FIELDS}",
        "--format=csv,noheader,nounits",
        "-lms",
        str(arguments.interval_ms),
    ]
    samples: list[dict[str, float | str]] = []
    start_time = time.perf_counter()
    monitor = subprocess.Popen(
        monitor_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    reader = threading.Thread(
        target=monitor_reader,
        args=(monitor, samples, start_time),
        daemon=True,
    )
    reader.start()

    print("Running:", " ".join(run_command))
    solver: subprocess.Popen[bytes] | None = None
    return_code = -1
    interrupted = False
    stopped_by_runtime_limit = False
    try:
        solver = subprocess.Popen(run_command, cwd=run_directory)
        try:
            return_code = solver.wait(timeout=arguments.max_runtime_seconds)
        except subprocess.TimeoutExpired:
            stopped_by_runtime_limit = True
            print(
                f"Profiling limit of {arguments.max_runtime_seconds:g} s "
                "reached; terminating the solver..."
            )
            solver.terminate()
            try:
                return_code = solver.wait(timeout=10)
            except subprocess.TimeoutExpired:
                solver.kill()
                return_code = solver.wait()
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted; terminating the solver...", file=sys.stderr)
        if solver is not None and solver.poll() is None:
            solver.terminate()
            try:
                return_code = solver.wait(timeout=10)
            except subprocess.TimeoutExpired:
                solver.kill()
                return_code = solver.wait()
    finally:
        wall_time = time.perf_counter() - start_time
        if monitor.poll() is None:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()
        reader.join(timeout=5)

    if not samples:
        raise RuntimeError("No GPU samples were collected")

    analyzed_samples = [
        item
        for item in samples
        if float(item["elapsed_s"]) >= arguments.warmup_seconds
    ]
    warmup_applied = True
    if not analyzed_samples:
        warmup_applied = False
        analyzed_samples = samples
        print(
            "Warning: the run ended before the warm-up period elapsed; "
            "all samples will be analyzed.",
            file=sys.stderr,
        )
    if len(analyzed_samples) < arguments.minimum_samples:
        print(
            f"Warning: only {len(analyzed_samples)} post-warm-up samples were "
            f"collected; at least {arguments.minimum_samples} are recommended.",
            file=sys.stderr,
        )

    gpu_utilization = [
        float(item["gpu_utilization_percent"]) for item in analyzed_samples
    ]
    memory_utilization = [
        float(item["memory_utilization_percent"]) for item in analyzed_samples
    ]
    memory_used = [
        float(item["memory_used_mib"]) for item in analyzed_samples
    ]
    peak_memory_mib = max(memory_used)
    summary = {
        "label": arguments.label,
        "gpu_index": arguments.gpu,
        "command": command,
        "solver_output_isolated": not arguments.allow_solver_writes,
        "solver_return_code": return_code,
        "interrupted": interrupted,
        "stopped_by_runtime_limit": stopped_by_runtime_limit,
        "requested_max_runtime_s": arguments.max_runtime_seconds,
        "sampling_interval_ms": arguments.interval_ms,
        "raw_sample_count": len(samples),
        "analyzed_sample_count": len(analyzed_samples),
        "requested_warmup_s": arguments.warmup_seconds,
        "warmup_exclusion_applied": warmup_applied,
        "wall_time_s": wall_time,
        "gpu_utilization_mean_percent": statistics.fmean(gpu_utilization),
        "gpu_utilization_p95_percent": percentile(gpu_utilization, 95.0),
        "gpu_utilization_max_percent": max(gpu_utilization),
        "memory_bandwidth_utilization_mean_percent": statistics.fmean(
            memory_utilization
        ),
        "baseline_memory_mib": baseline_mib,
        "memory_used_mean_mib": statistics.fmean(memory_used),
        "memory_used_peak_mib": peak_memory_mib,
        "baseline_adjusted_peak_memory_mib": max(
            0.0, peak_memory_mib - baseline_mib
        ),
    }

    write_samples(sample_path, samples)
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")

    print(f"Raw samples: {sample_path}")
    print(f"Summary:     {summary_path}")
    print(
        "GPU utilization: "
        f"mean={summary['gpu_utilization_mean_percent']:.2f}%, "
        f"p95={summary['gpu_utilization_p95_percent']:.2f}%"
    )
    print(
        "Peak GPU memory: "
        f"{peak_memory_mib:.1f} MiB total, "
        f"{summary['baseline_adjusted_peak_memory_mib']:.1f} MiB "
        "above baseline"
    )
    print(f"Measured wall time: {wall_time:.3f} s")
    if isolation is not None:
        isolation.cleanup()
    if stopped_by_runtime_limit:
        return 0
    return return_code if return_code >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
