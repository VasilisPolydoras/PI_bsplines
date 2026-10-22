# Forward wave equation

Solves `u_tt-c^2*u_xx=0` on `[0,1] x [0,1]`, with spatial Dirichlet data,
`u(x,0)=sin(pi*x)`, and Neumann-in-time initial data `u_t(x,0)=0`.

The B-spline initial trace is obtained by Greville interpolation. For a
clamped time basis, `u_t(x,0)` is proportional to the difference between the
second and first coefficient rows, so `C[1,:]=C[0,:]` imposes zero initial
velocity exactly before training. Spatial endpoint coefficient columns vanish.

PINN and PIKAN support `hard` and `soft`; their hard trial is
`sin(pi*x)+t^2*x*(1-x)*N(x,t)`. Each method has a JSON config and tuner.
