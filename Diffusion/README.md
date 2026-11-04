# Forward diffusion equation

This folder solves

\[
u_t-\alpha u_{xx}=0,\qquad (x,t)\in[0,1]\times[0,1],
\]

with \(\alpha=0.03\), \(u(x,0)=\sin(2\pi x)\), and
\(u(0,t)=u(1,t)=0\). The reference solution is
\(u(x,t)=\sin(2\pi x)e^{-4\pi^2\alpha t}\).

The layout follows the Burgers example:

- `Bspline_2D.py`, `PINN_2D.py`, and `PIKAN_2D.py` are the three solvers.
- Each solver has an independent JSON configuration and Optuna tuning script.
- PINN and PIKAN support `hard` and `soft` initial/boundary constraints.
- `run_statistics.py` repeats one selected solver over initialization seeds.
- `postprocessing/plot_results.py` recreates figures from saved NPZ files.

The B-spline solver imposes the homogeneous spatial Dirichlet conditions
exactly by fixing the first and last spatial coefficient rows to zero. It fits
only the inhomogeneous initial trace using `initial_points` samples and then
freezes its first temporal coefficient column. The free interior coefficients
are initialized using either `random` or `coons`, selected by
`model.interior_initialization` in `bspline_config.json`. The second training
stage minimizes only the interior PDE residual. B-spline derivatives are
analytical, whereas PINN and PIKAN use automatic differentiation.

Example commands:

```powershell
python Diffusion/Bspline_2D.py --device cpu
python Diffusion/Bspline_2D.py --device cpu --interior-initialization random
python Diffusion/PINN_2D.py --device cpu --constraint hard
python Diffusion/PIKAN_2D.py --device cpu --constraint soft

python Diffusion/tune_bsplines.py --device cpu
python Diffusion/tune_pinn.py --device cpu --constraint hard
python Diffusion/tune_pikan.py --device cpu --constraint hard

python Diffusion/run_statistics.py --solver bspline --device cpu
python Diffusion/postprocessing/plot_results.py --solver all --show
```
