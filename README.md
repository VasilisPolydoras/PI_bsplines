# PI-TP-BsA

This repository compares Physics-Informed Tensor-Product B-spline
Approximation (PI-TP-BsA) with soft- and hard-constrained PINNs and PIKANs on
forward and inverse PDEs. It includes hyperparameter tuning, repeated-run
statistics, manuscript plots, GPU profiling, and stability analyses.

Run all commands from the repository root. 

## Benchmarks

| Folder | Problem | Methods |
|---|---|---|
| `Diffusion/` | Forward diffusion | PI-TP-BsA, PINN, PIKAN |
| `DiffusionInverse/` | Diffusivity identification | PI-TP-BsA, PINN, PIKAN |
| `WaveEquation/` | Forward wave equation | PI-TP-BsA, PINN, PIKAN |
| `WaveEquationInverse/` | Wave-speed identification | PI-TP-BsA, PINN, PIKAN |
| `Burgers/` | Viscous Burgers equation | PI-TP-BsA, PINN, PIKAN |
| `Kovasznay/` | Steady incompressible flow | PI-TP-BsA, PINN, PIKAN |
| `Kirchoff/` | Fourth-order plate bending | PI-TP-BsA, PINN, PIKAN |
| `LShapeLaplace/Irregular/` | Multipatch L-shaped Laplace problem | PI-TP-BsA, XPINN |

## Installation

The reference environment uses Python 3.8.10.

Windows PowerShell:

```powershell
py -3.8 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements.txt
```

Linux (`pywin32` is Windows-only):

```bash
python3.8 -m venv .venv
source .venv/bin/activate
grep -v '^pywin32==' requirements.txt > /tmp/pi_bsplines_requirements.txt
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r /tmp/pi_bsplines_requirements.txt
```

Use a PyTorch build compatible with the machine's CUDA driver, or run solvers
with `--device cpu`. Verify PyTorch and PIKAN with:

```powershell
python -c "import torch; from kan import KAN; print(torch.__version__, torch.cuda.is_available())"
```

The package is named `pykan`, but its import name is `kan`.

## Standard workflow

Each benchmark contains method-specific configuration files, tuned
hyperparameter files, solvers, `run_statistics.py`, and a
`postprocessing/plot_results.py` script.

For a standard folder such as `Diffusion`, the workflow is:

```powershell
# Optional tuning
python Diffusion/tune_bsplines.py --device gpu
python Diffusion/tune_pinn.py --device gpu --constraint soft
python Diffusion/tune_pikan.py --device gpu --constraint hard

# Representative solutions
python Diffusion/Bspline_2D.py --device gpu --use-tuned
python Diffusion/PINN_2D.py --device gpu --constraint soft --use-tuned
python Diffusion/PINN_2D.py --device gpu --constraint hard --use-tuned
python Diffusion/PIKAN_2D.py --device gpu --constraint soft --use-tuned
python Diffusion/PIKAN_2D.py --device gpu --constraint hard --use-tuned

# Statistics and plots
python Diffusion/run_statistics.py --solver bspline --runs 10 --device gpu
python Diffusion/run_statistics.py --solver pinn --constraint hard --runs 10 --device gpu
python Diffusion/postprocessing/plot_results.py --solver all --show
```

Replace `Diffusion` with `WaveEquation`, `Kovasznay`, or `Kirchoff`. Inverse
solvers use the same pattern but load tuned values by default and offer
`--no-use-tuned` instead. Use `--epochs` for a short smoke test; reduced
budgets do not reproduce the reported results.

Representative runs save solution and training NPZ files under
`<folder>/postprocessing/`. Statistics runs vary initialization while keeping
sampling and validation fixed. Plotters load saved data and do not retrain.

### Folder-specific commands

Inverse diffusion:

```powershell
python DiffusionInverse/Bspline_2D.py --device gpu
python DiffusionInverse/PINN_2D.py --device gpu --constraint hard
python DiffusionInverse/PIKAN_2D.py --device gpu --constraint hard
python DiffusionInverse/postprocessing/plot_results.py --solver all --show
```

Inverse wave equation:

```powershell
python WaveEquationInverse/Bspline_2D.py --device gpu
python WaveEquationInverse/PINN_2D.py --device gpu --constraint hard
python WaveEquationInverse/PIKAN_2D.py --device gpu --constraint hard
python WaveEquationInverse/postprocessing/plot_results.py --solver all --show
```

Burgers:

```powershell
python Burgers/Bspline_2D.py --device gpu --use-tuned
python Burgers/PINN_2D.py --device gpu --constraint soft --use-tuned
python Burgers/PINN_2D.py --device gpu --constraint hard --use-tuned
python Burgers/PIKAN_2D.py --device gpu --constraint soft --use-tuned
python Burgers/PIKAN_2D.py --device gpu --constraint hard --use-tuned
python Burgers/postprocessing/plot_results.py --solver all --show
```

L-shaped Laplace problem:

```powershell
python LShapeLaplace/Irregular/tune_bsplines.py --device gpu
python LShapeLaplace/Irregular/tune_xpinn.py --device gpu
python LShapeLaplace/Irregular/Bspline_2D.py --device gpu --use-tuned
python LShapeLaplace/Irregular/XPINN_2D.py --device gpu --use-tuned
python LShapeLaplace/Irregular/run_statistics.py --solver bspline --runs 10 --device gpu
python LShapeLaplace/Irregular/run_statistics.py --solver xpinn --runs 3 --device gpu
python LShapeLaplace/Irregular/postprocessing/plot_results.py --solver all --show
```

## Statistics

Use these repetition counts to regenerate the current archives:

| Benchmark | PI-TP-BsA | PINN soft/hard | PIKAN soft/hard | XPINN |
|---|---:|---:|---:|---:|
| Diffusion | 10 | 10 / 10 | 5 / 10 | -- |
| Diffusion inverse | 5 | 5 / 5 | 3 / 3 | -- |
| Wave equation | 5 | 5 / 5 | 5 / 5 | -- |
| Wave inverse | 10 | 10 / 10 | 3 / 3 | -- |
| Burgers | 10 | 10 / 10 | 3 / 3 | -- |
| Kovasznay | 5 | 5 / 5 | 3 / 5 | -- |
| Kirchhoff | 5 | 1 / 1 | 1 / 1 | -- |
| L-shaped Laplace | 10 | -- | -- | 3 |

The normal syntax is:

```powershell
python <folder>/run_statistics.py --solver bspline --runs <N> --device gpu
python <folder>/run_statistics.py --solver pinn --constraint soft --runs <N> --device gpu
python <folder>/run_statistics.py --solver pikan --constraint hard --runs <N> --device gpu
```

Burgers is the exception: its statistics script reads `constraint_mode` from
`pinn_config.json` or `pikan_config.json`. Set that field first and use the
output name expected by the plotter:

```powershell
python Burgers/run_statistics.py --solver pinn --runs 10 --device gpu --output Burgers/postprocessing/pinn_soft_statistics.npz
python Burgers/run_statistics.py --solver pikan --runs 3 --device gpu --output Burgers/postprocessing/pikan_hard_statistics.npz
```

Change the configuration mode and output filename together for the other
constraint. Statistics overwrite an archive with the same name but do not
overwrite representative solution files. A plotter prints both the available
single-run result and repeated-run summary, for example:

```powershell
python Kovasznay/postprocessing/plot_results.py --solver bspline-hard
```

## GPU profiling

`benchmark_gpu.py` samples `nvidia-smi` while running a solver. It uses an
isolated temporary repository copy by default, so PDE solution files are not
overwritten.

```powershell
python benchmark_gpu.py --label diffusion_bspline --gpu 0 -- python Diffusion/Bspline_2D.py --device gpu

python benchmark_gpu.py --label kirchhoff_pinn_hard --gpu 0 --interval-ms 500 --max-runtime-seconds 60 --warmup-seconds 5 -- python Kirchoff/PINN_2D.py --constraint hard --device gpu
```

Outputs are `gpu_benchmarks/<label>_gpu_samples.csv` and
`gpu_benchmarks/<label>_gpu_summary.json`. Use a unique label or pass
`--overwrite`. Avoid `--allow-solver-writes` unless normal outputs may change.
A runtime-limited solver can have a nonzero return code; check
`stopped_by_runtime_limit` in the JSON summary.

## Stability and sensitivity studies

Paths contain a space and must be quoted. Runners preserve old results using
numbered NPZ filenames; plotters load the newest archive.

Forward diffusion refinement and collocation density:

```powershell
python "stability analysis/Diffusion/run_h_refinement.py" --device gpu
python "stability analysis/Diffusion/plot_h_refinement.py"
python "stability analysis/Diffusion/run_p_refinement.py" --device gpu
python "stability analysis/Diffusion/plot_p_refinement.py"
python "stability analysis/Diffusion/run_rho_refinement.py" --device gpu
python "stability analysis/Diffusion/plot_rho_refinement.py"
```

Burgers knot concentration:

```powershell
python "stability analysis/Burgers/run_knot_concentration.py" --device gpu
python "stability analysis/Burgers/plot_knot_concentration.py"
```

Inverse-diffusion identifiability, sensor count, placement, and noise:

```powershell
python "stability analysis/DiffusionInverse/run_identifiability.py" --device gpu
python "stability analysis/DiffusionInverse/plot_identifiability.py"
python "stability analysis/DiffusionInverse/run_sensor_number.py" --device gpu
python "stability analysis/DiffusionInverse/plot_sensor_number.py"
python "stability analysis/DiffusionInverse/run_sensor_placement.py" --device gpu
python "stability analysis/DiffusionInverse/plot_sensor_placement.py"
python "stability analysis/DiffusionInverse/run_sensor_noise.py" --device gpu
python "stability analysis/DiffusionInverse/plot_sensor_noise.py"
```

L-shaped-domain h-refinement:

```powershell
python "stability analysis/LShapeLaplace/run_h_refinement.py" --device gpu
python "stability analysis/LShapeLaplace/plot_h_refinement.py"
```

Use `python <script> --help` for study values, seeds, epochs, and other
overrides. On headless Linux, prefix plotting commands with `MPLBACKEND=Agg`.

## Reproducibility notes

- Run representative solvers before plotting; statistics archives do not
  contain the full solution grids.
- The configured epoch budget can differ from completed epochs because of
  early stopping.
- Re-running a representative solver overwrites its normal output files. Most
  generated `.npz`, `.csv`, `.png`, and `.pdf` files are ignored by Git.
- Use the same configurations, seeds, device, and package versions for direct
  comparisons. Small GPU differences can remain across hardware and CUDA
  versions.

