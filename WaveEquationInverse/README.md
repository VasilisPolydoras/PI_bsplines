# Inverse wave equation

Identifies the positive physical wave speed `c` in
`u_tt-c^2*u_xx=0`. Synthetic data use `c_true=1` and the exact field
`sin(pi*x) cos(pi*c*t)`. The default observation noise is `0.05`.

For B-splines, the Greville-interpolated displacement is stored in the first
time coefficient row. The second row is set equal to the first, which exactly
imposes `u_t(x,0)=0` for the clamped time basis. Spatial boundary columns are
zero. Thus none of these conditions needs a penalty.

PINN and PIKAN offer hard and soft constraints. Positivity is enforced with
`c=c_min+softplus(raw_c)`. Tuners minimize validation field error plus relative
wave-speed error, and postprocessing archives include the complete `c` history.
