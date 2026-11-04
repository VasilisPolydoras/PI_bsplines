# Forward wave equation

This folder compares PI-TP-BsA, PINN, and PIKAN approximations of

\[
u_{tt}-c^2u_{xx}=0,\qquad (x,t)\in[0,1]\times[0,1],
\]

with

\[
u(x,0)=\sin(\pi x),\qquad u_t(x,0)=0,\qquad
u(0,t)=u(1,t)=0.
\]

For `c = 1`, the exact solution is

\[
u(x,t)=\sin(\pi x)\cos(\pi t).
\]

## Solvers and constraints

- `Bspline_2D.py` uses analytical B-spline derivatives. The homogeneous
  spatial Dirichlet conditions are imposed exactly by fixing the endpoint
  coefficient rows. The displacement trace is pretrained from sampled initial
  data, and the adjacent temporal coefficient column is then linked to the
  first one so that `u_t(x,0)=0` holds exactly. Only the remaining interior
  coefficients are trained with the strong-form residual.
- `PINN_2D.py` and `PIKAN_2D.py` support `soft` and `hard` constraints. In hard
  mode they use `sin(pi*x) + t^2*x*(1-x)*raw_model(x,t)`, which satisfies both
  initial conditions and both spatial boundary conditions exactly.
- Each solver reads its own JSON configuration and can load the corresponding
  tuned hyperparameter JSON file.

The full-training and Optuna-tuning budgets mirror the corresponding files in
the `Diffusion` folder. Training time includes B-spline initial-condition
pretraining.

## Commands

Run a solver:

```powershell
python WaveEquation/Bspline_2D.py
python WaveEquation/PINN_2D.py --constraint soft
python WaveEquation/PIKAN_2D.py --constraint hard
```

Tune a solver:

```powershell
python WaveEquation/tune_bsplines.py
python WaveEquation/tune_pinn.py --constraint soft
python WaveEquation/tune_pikan.py --constraint hard
```

Collect repeated-run statistics while fixing training and validation samples
and varying only trainable-parameter initialization:

```powershell
python WaveEquation/run_statistics.py --solver bspline
python WaveEquation/run_statistics.py --solver pinn --constraint soft
python WaveEquation/run_statistics.py --solver pikan --constraint hard
```

Recreate PDF figures from saved NPZ data without retraining:

```powershell
python WaveEquation/postprocessing/plot_results.py --solver all
python WaveEquation/postprocessing/plot_results.py --solver all --show
```
