# Two-dimensional Kovasznay-flow benchmark

This folder compares PI-TP-BsA, PINN, and PIKAN approximations of the steady
incompressible Navier--Stokes equations using the analytic Kovasznay solution.
All solvers use the same rectangular domain, Reynolds number, interior
collocation budget, boundary samples, validation samples, optimizer,
scheduler, and early-stopping settings.

The B-spline solver first trains and freezes its exterior coefficient edges on
the nonhomogeneous Dirichlet data. It then optimizes only the free interior
coefficients with analytical B-spline derivatives. PINN and PIKAN support both
soft boundary penalties and exact hard constraints constructed from a Coons
boundary lifting and a distance function.

Hyperparameter optimization minimizes the mean of the validation relative L2
errors of the velocity components and pressure. Run the tuners with:

```bash
python Kovasznay/tune_bsplines.py --device gpu
python Kovasznay/tune_pinn.py --constraint soft --device gpu
python Kovasznay/tune_pikan.py --constraint soft --device gpu
```

Repeated-run statistics keep collocation and validation samples fixed while
varying trainable-parameter initialization:

```bash
python Kovasznay/run_statistics.py --solver bspline --device gpu
python Kovasznay/run_statistics.py --solver pinn --constraint soft --device gpu
python Kovasznay/run_statistics.py --solver pikan --constraint hard --device gpu
```

Recreate all comparison and loss-evolution PDFs without retraining using:

```bash
python Kovasznay/postprocessing/plot_results.py
```
