# Kirchhoff plate inverse problem

This independent folder identifies the unknown flexural rigidity

\[
D=\frac{E h^3}{12(1-\nu^2)}
\]

from displacement observations while simultaneously solving

\[
D\left(w_{,xxxx}+2w_{,xxyy}+w_{,yyyy}\right)=p(x,y).
\]

The manufactured plate is square and simply supported: `w = 0` and
`M_nn = 0` on all four edges. The true value from the configured material is
`D = 20.833333...`; optimization starts from `D = 10`. Positivity is exact
through `D = D_min + softplus(raw_D)`.

## Solvers

- `Bspline_2D.py`: analytical spline derivatives and an exact coefficient
  transform imposing both simply-supported conditions.
- `PINN_2D.py`: fourth derivatives by autograd; `hard` and `soft` constraints.
- `PIKAN_2D.py`: the same inverse loss and constraint choices using pykan.

All methods use the same normalized PDE and sensor losses. `noise_std` in each
JSON controls additive observation noise. Unlike the source notebook, where
noise was generated but the clean values were used in the loss, these scripts
actually train against the configured noisy observations.

## Run

```powershell
python KirchoffInverse/Bspline_2D.py --device cpu --no-use-tuned
python KirchoffInverse/PINN_2D.py --device cpu --constraint hard --no-use-tuned
python KirchoffInverse/PIKAN_2D.py --device cpu --constraint hard --no-use-tuned
```

Tune with `tune_bsplines.py`, `tune_pinn.py`, or `tune_pikan.py`. Each tuner
minimizes the sum of validation field relative L2 error and relative rigidity
error and writes its adjacent `*_hyper_tuned.json` file.

Recreate saved figures without retraining:

```powershell
python KirchoffInverse/postprocessing/plot_results.py --solver all --show
```
