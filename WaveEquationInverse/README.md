# Inverse wave equation

This folder mirrors the structure and common numerical settings of
`DiffusionInverse` while retaining the wave-equation physics. It identifies the
positive wave speed in

```text
u_tt - c^2 u_xx = 0,
u(x, 0) = sin(pi x),  u_t(x, 0) = 0,
u(0, t) = u(1, t) = 0.
```

The folder contains B-spline, PINN, and PIKAN solvers, one JSON configuration
and one Optuna tuner per solver, consolidated postprocessing, and a selectable
repeated-run statistics script. PINN and PIKAN support soft and hard
initial/boundary constraints. The B-spline solver imposes the homogeneous
spatial boundaries and zero initial velocity exactly, pretrains and freezes the
inhomogeneous initial-displacement trace, and jointly trains the remaining
coefficients and wave speed.

Inverse-only configuration entries define the initial and minimum wave speed,
sensor count, observation noise, and data-loss weight. Sampling, optimization,
scheduling, early stopping, tuning budgets, and architecture search ranges
match `DiffusionInverse`; only PDE-specific fields and residuals differ.

```text
python WaveEquationInverse/Bspline_2D.py --device gpu
python WaveEquationInverse/PINN_2D.py --device gpu --constraint hard
python WaveEquationInverse/PIKAN_2D.py --device gpu --constraint soft
python WaveEquationInverse/run_statistics.py --solver bspline --runs 10 --device gpu
python WaveEquationInverse/postprocessing/plot_results.py --solver all --show
```
