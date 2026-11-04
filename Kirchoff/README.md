# Simply supported Kirchhoff plate

This folder compares PI-TP-BsA, PINN, and PIKAN approximations of

\[
D\left(w_{xxxx}+2w_{xxyy}+w_{yyyy}\right)=q(x,y)
\]

on `[0,10] x [0,10]`, with sinusoidal loading and simply supported conditions
`w=0` and `M_nn=0` on every edge.

- `Bspline_2D.py` uses analytical derivatives through fourth order and a
  coefficient transform that imposes both boundary conditions exactly.
- `PINN_2D.py` and `PIKAN_2D.py` provide soft and hard constraints. Their hard
  sine/cosine trial construction satisfies deflection and moment conditions.
- Every solver has an independent JSON configuration and Optuna tuner.
- `run_statistics.py` varies only initialization seeds while fixing training
  and validation samples, then saves errors, timings, dispersion, and 95% CIs.
- `postprocessing/plot_results.py` recreates split PDF comparisons and the
  combined loss-evolution figure from NPZ files.

The common training, sampling, tuning, output, and plotting settings mirror
the corresponding Diffusion configuration. Plate-specific exceptions are the
B-spline degree range `4--6` and PIKAN spline order `4--6`, required for stable
fourth derivatives.

```powershell
python Kirchoff/Bspline_2D.py
python Kirchoff/PINN_2D.py --constraint soft
python Kirchoff/PIKAN_2D.py --constraint hard
python Kirchoff/tune_bsplines.py
python Kirchoff/run_statistics.py --solver pinn --constraint hard
python Kirchoff/postprocessing/plot_results.py --solver all --show
```
