# (2+1)-D nonlinear Klein–Gordon benchmark

This folder implements the manufactured benchmark used in arXiv:2408.13101:

\[
u_{tt}-u_{xx}-u_{yy}+u^2=f,
\qquad (x,y,t)\in[-1,1]^2\times[0,10],
\]

with

\[
u^*(x,y,t)=(x+y)\cos(2t)+xy\sin(2t),
\qquad f=(u^*)^2-4u^*.
\]

The Dirichlet data are the trace of \(u^*\). The authors' released notebook
imposes \(u(x,y,0)=x+y\) but does not impose an initial velocity. The updated
hard B-spline formulation also imposes \(u_t(x,y,0)=2xy\), yielding complete
Cauchy data. It sequentially pretrains the time-dependent spatial boundary
surfaces, the initial-value surface, and the second temporal coefficient
surface associated with the initial velocity. PINN and PIKAN can impose the
same condition by enabling `enforce_initial_velocity` and assigning its loss
weight a nonzero value in their respective configurations.

Solvers:

- `Bspline_3D.py`: tensor-product B-spline representation. In hard mode its
  four spatial boundary faces, initial-value face, and initial-velocity-related
  second temporal face are optimized sequentially and frozen. The remaining
  coefficients are trained using the PDE residual. PDE derivatives use
  analytical B-spline derivative matrices rather than coordinate autograd.
- `PINN_3D.py`: fully connected tanh PINN with architecture selected from its
  JSON configuration or loaded tuned parameters.
- `PIKAN_3D.py`: pykan-based physics-informed KAN.

All three accept `--constraint soft` or `--constraint hard`. PINN and PIKAN use
an analytical hard lifting; B-splines use their native frozen-coefficient
construction. Each corresponding `tune_*.py` script minimizes validation
relative L2 error. Solver runs save NPZ
data under `postprocessing/`; run
`python postprocessing/plot_results.py` to recreate publication-ready PDF
figures without training. The plotting script creates separate PI-TP-BsA,
PINN, and PIKAN comparison PDFs plus a common loss-evolution PDF.
The solution figure displays prediction, exact solution, and absolute error as
colored x-y slices stacked inside their physical three-dimensional x-y-t boxes.

Repeated-run statistics use fixed training/validation samples while varying
only trainable-parameter initialization. For example:

```bash
python KleinGordon/run_statistics.py --solver bspline --device gpu
python KleinGordon/run_statistics.py --solver pinn --constraint soft --device gpu
python KleinGordon/run_statistics.py --solver pikan --constraint hard --device gpu
```

Each statistics run stores the individual relative L2 errors and wall times,
their means and sample standard deviations, and Student-t 95% confidence
intervals in `postprocessing/<solver>[_<constraint>]_statistics.npz`.
