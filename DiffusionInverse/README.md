# Inverse diffusion equation

Identifies the positive diffusivity `alpha` while learning the solution of
`u_t-alpha*u_xx=0`. Synthetic observations use the exact solution with true
`alpha=0.03`; the default observation noise standard deviation is `0.05`, as
in `diffusion_inverse_gpu.ipynb`. Set `noise_std` to zero for clean data.

The B-spline method hard-imposes the spatial boundaries and constructs its
initial coefficient trace by Greville interpolation. PINN and PIKAN support
hard and soft constraints. Positivity is enforced as
`alpha=alpha_min+softplus(raw_alpha)`.

Tuners minimize field relative L2 error plus diffusivity relative error.
Results and parameter histories are stored in `postprocessing`. Replot with
`python DiffusionInverse/postprocessing/plot_results.py --show`.
