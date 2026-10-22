"""Recreate inverse Kirchhoff plots from saved NPZ data without retraining."""
import argparse,sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
from _common import plot_solution,plot_training

SOLVERS={
 "bspline-hard":("bspline_kirchhoff_inverse_hard","B-spline (hard)"),
 "pinn-hard":("pinn_kirchhoff_inverse_hard","PINN (hard)"),
 "pinn-soft":("pinn_kirchhoff_inverse_soft","PINN (soft)"),
 "pikan-hard":("pikan_kirchhoff_inverse_hard","PIKAN (hard)"),
 "pikan-soft":("pikan_kirchhoff_inverse_soft","PIKAN (soft)"),
}
def reproduce(name):
    stem,title=SOLVERS[name];solution=HERE/f"{stem}_solution_data.npz";training=HERE/f"{stem}_training_data.npz"
    if not solution.exists() or not training.exists():raise FileNotFoundError(f"Run {name} first; saved NPZ data are missing")
    with np.load(solution) as data:true_d=float(data["true_d"]);print(f"{name}: D={float(data['estimated_d']):.8f}, relative D error={float(data['parameter_relative_error']):.6e}, validation L2={float(data['validation_relative_l2_error']):.6e}")
    plot_solution(solution,HERE/f"{stem}_solution.png",title);plot_training(training,HERE/f"{stem}_training.png",title,true_d)
def main(solver,show):
    for name in (SOLVERS if solver=="all" else (solver,)):
        try:reproduce(name)
        except FileNotFoundError as error:
            if solver!="all":raise
            print(f"Skipped {name}: {error}")
    plt.show() if show else plt.close("all")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--solver",choices=("all",*SOLVERS),default="all");p.add_argument("--show",action="store_true");a=p.parse_args();main(a.solver,a.show)
