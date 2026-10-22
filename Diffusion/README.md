# Forward diffusion equation

Solves `u_t - alpha u_xx = 0` on `(x,t) in [0,1]^2`, with `alpha=0.03`,
`u(x,0)=sin(2*pi*x)`, and `u(0,t)=u(1,t)=0`. The exact solution is
`sin(2*pi*x) exp(-4*pi^2 alpha t)`.

`Bspline_2D.py` fixes the spatial edge coefficients to zero and obtains the
initial coefficient column by Greville interpolation. PINN and PIKAN expose
`--constraint hard|soft`; the hard trial is `sin(2*pi*x)+t*x*(1-x)*N(x,t)`.

Each solver has a JSON config, Optuna tuner, and saves NPZ/PNG output under
`postprocessing`. Replot with `python Diffusion/postprocessing/plot_results.py --show`.
