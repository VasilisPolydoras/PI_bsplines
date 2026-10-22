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
imposes \(u(x,y,0)=x+y\) but does not impose an initial velocity. Therefore
`enforce_initial_velocity` is `false` by default. For PINN and PIKAN, setting it
to `true` also imposes \(u_t(x,y,0)=2xy\), yielding complete Cauchy data; set
the corresponding loss weight to `1.0` too. The hard Greville B-spline mode
keeps the paper setting: independently interpolating the time-dependent spatial
traces and fixing the initial derivative would generally prescribe conflicting
values to the second time-coefficient layer. Its soft mode can include the
initial-velocity loss.

Solvers:

- `Bspline_3D.py`: tensor-product B-spline representation. In hard mode its
  four spatial boundary faces and its initial coefficient face are computed by
  tensor-product Greville interpolation and frozen. The PDE derivatives use
  analytical B-spline derivative matrices rather than coordinate autograd.
- `PINN_3D.py`: fully connected tanh PINN; the default 5-by-128 architecture
  follows the benchmark PINN reported by the paper.
- `PIKAN_3D.py`: pykan-based physics-informed KAN.

All three accept `--constraint soft` or `--constraint hard`. PINN and PIKAN use
an analytical hard lifting; B-splines use their native frozen-coefficient
construction. Each corresponding `tune_*.py` script minimizes validation
relative L2 error. Solver runs save NPZ
data and PNG figures under `postprocessing/`; run
`python postprocessing/plot_results.py` to recreate figures without training.
The solution figure displays prediction, exact solution, and absolute error as
colored x-y slices stacked inside their physical three-dimensional x-y-t boxes.
