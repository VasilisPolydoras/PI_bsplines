# Simply supported Kirchhoff plate

This folder compares three physics-informed representations for

\[
D\left(w_{xxxx}+2w_{xxyy}+w_{yyyy}\right)=q(x,y)
\]

on `[0,10] x [0,10]`, with sinusoidal loading and simply supported edges
`w = 0`, `M_nn = 0`.

## Solvers

- `Bspline_2D.py`: hard `w=0` and `w_nn=0` through a coefficient transform.
- `PINN_2D.py`: `--constraint soft` uses boundary losses; `hard` uses a
  sine/cosine trial space that satisfies both conditions exactly.
- `PIKAN_2D.py`: the same soft/hard formulations using PyKAN.

Each solver reads its own JSON configuration. Tuned parameters are written by
`tune_bsplines.py`, `tune_pinn.py`, and `tune_pikan.py`.

Examples:

```powershell
python Kirchoff/Bspline_2D.py --device cpu
python Kirchoff/PINN_2D.py --device cpu --constraint hard
python Kirchoff/PINN_2D.py --device cpu --constraint soft
python Kirchoff/PIKAN_2D.py --device cpu --constraint hard
```

Recreate saved plots without training:

```powershell
python Kirchoff/postprocessing/plot_results.py --solver all --show
```

