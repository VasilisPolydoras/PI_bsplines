"""Recreate wave plots from saved NPZ data."""
import argparse,sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent));from _common import plot_solution,plot_training
SOLVERS={"bspline-hard":("bspline_wave_hard","B-spline"),"pinn-hard":("pinn_wave_hard","PINN hard"),"pinn-soft":("pinn_wave_soft","PINN soft"),"pikan-hard":("pikan_wave_hard","PIKAN hard"),"pikan-soft":("pikan_wave_soft","PIKAN soft")}
def main(name,show):
 for key in (SOLVERS if name=="all" else (name,)):
  stem,title=SOLVERS[key];sp=HERE/f"{stem}_solution_data.npz";tp=HERE/f"{stem}_training_data.npz"
  if not(sp.exists() and tp.exists()):print(f"Skipped {key}: data missing");continue
  plot_solution(sp,HERE/f"{stem}_solution.png",title);plot_training(tp,HERE/f"{stem}_training.png",title)
  with np.load(sp) as d:print(f"{key} validation L2: {float(d['validation_relative_l2_error']):.6e}")
 plt.show() if show else plt.close("all")
if __name__=="__main__":p=argparse.ArgumentParser();p.add_argument("--solver",choices=("all",*SOLVERS),default="all");p.add_argument("--show",action="store_true");a=p.parse_args();main(a.solver,a.show)
