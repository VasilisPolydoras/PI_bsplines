"""Recreate forward diffusion plots without retraining."""
import argparse,sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent));from _common import plot_solution,plot_training
SOLVERS={"bspline-hard":("bspline_diffusion_hard","B-spline"),"pinn-hard":("pinn_diffusion_hard","PINN hard"),"pinn-soft":("pinn_diffusion_soft","PINN soft"),"pikan-hard":("pikan_diffusion_hard","PIKAN hard"),"pikan-soft":("pikan_diffusion_soft","PIKAN soft")}
def main(name,show):
 for key in (SOLVERS if name=="all" else (name,)):
  stem,title=SOLVERS[key];s=HERE/f"{stem}_solution_data.npz";h=HERE/f"{stem}_training_data.npz"
  if not(s.exists() and h.exists()):print(f"Skipped {key}: data missing");continue
  plot_solution(s,HERE/f"{stem}_solution.png",title);plot_training(h,HERE/f"{stem}_training.png",title)
  with np.load(s) as d:print(f"{key} validation L2: {float(d['validation_relative_l2_error']):.6e}")
 plt.show() if show else plt.close("all")
if __name__=="__main__":p=argparse.ArgumentParser();p.add_argument("--solver",choices=("all",*SOLVERS),default="all");p.add_argument("--show",action="store_true");a=p.parse_args();main(a.solver,a.show)
