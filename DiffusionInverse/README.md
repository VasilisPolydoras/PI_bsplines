# Inverse diffusion equation

This folder mirrors the structure and common numerical settings of `Diffusion`
for identification of the positive diffusivity in `u_t - D u_xx = 0`.

It contains B-spline, PINN, and PIKAN solvers, one JSON configuration and one
Optuna tuner per solver, consolidated postprocessing, and a selectable repeated-
run statistics script. PINN and PIKAN support soft and hard initial/boundary
constraints. The B-spline solver imposes the homogeneous spatial boundaries
exactly, pretrains and freezes the inhomogeneous initial trace, and jointly
trains the remaining coefficients and diffusivity.

Inverse-only configuration entries specify the initial/minimum diffusivity,
sensor count, observation noise, and data-loss weight. Common training,
sampling, scheduling, early-stopping, tuning-budget, and architecture-search
settings match the corresponding forward Diffusion solvers.

```text
python DiffusionInverse/Bspline_2D.py --device gpu
python DiffusionInverse/PINN_2D.py --device gpu --constraint hard
python DiffusionInverse/PIKAN_2D.py --device gpu --constraint soft
python DiffusionInverse/run_statistics.py --solver bspline --runs 10 --device gpu
python DiffusionInverse/postprocessing/plot_results.py --solver all --show
```
