"""Recreate publication-ready Kirchhoff comparisons without retraining."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
HERE=Path(__file__).resolve().parent; BASE_FONT_SIZE=18; TITLE_SIZE=22; LABEL_SIZE=20; TICK_SIZE=17; LEGEND_SIZE=16
plt.rcParams.update({"font.size":BASE_FONT_SIZE})
SOLVERS={"bspline-hard":("bspline_kirchhoff_hard","PI-TP-BsA"),"pinn-hard":("pinn_kirchhoff_hard","PINN hard"),"pinn-soft":("pinn_kirchhoff_soft","PINN soft"),"pikan-hard":("pikan_kirchhoff_hard","PIKAN hard"),"pikan-soft":("pikan_kirchhoff_soft","PIKAN soft")}

def load_solution(path):
    with np.load(path,allow_pickle=False) as d:
        return {
            "x":d["x"],"y":d["y"],"prediction":d["prediction"],
            "absolute_error":d["absolute_error"],
            "l2":float(d["validation_relative_l2_error"]),
            "time":float(d["training_wall_time_seconds"])
            if "training_wall_time_seconds" in d.files else float("nan"),
        }

def plot_comparison(keys,output_path):
    records={}
    for key in keys:
        path=HERE/f"{SOLVERS[key][0]}_solution_data.npz"
        if path.exists():
            try: records[key]=load_solution(path)
            except KeyError: print(f"Legacy data lack timing/validation fields: {path}")
        else: print(f"Missing solution data for {key}: {path}")
    if not records: raise FileNotFoundError("No compatible saved Kirchhoff data were found")
    fig,axes=plt.subplots(len(keys),2,figsize=(17,5.2*len(keys)),squeeze=False,sharex=True,sharey=True,constrained_layout=True)
    for row,key in enumerate(keys):
        pa,ea=axes[row]
        if key not in records:
            for axis in (pa,ea): axis.text(.5,.5,"Saved data unavailable",ha="center",va="center",transform=axis.transAxes); axis.set_axis_off()
            continue
        r=records[key]; x=np.asarray(r["x"]); y=np.asarray(r["y"]); extent=[x[0],x[-1],y[0],y[-1]]; si=pa.imshow(r["prediction"],extent=extent,origin="lower",aspect="equal",cmap="viridis"); ei=ea.imshow(r["absolute_error"],extent=extent,origin="lower",aspect="equal",cmap="magma",vmin=0)
        for axis in (pa,ea): axis.set_xlabel("$x$",fontsize=LABEL_SIZE); axis.set_ylabel("$y$",fontsize=LABEL_SIZE); axis.tick_params(labelsize=TICK_SIZE)
        pa.annotate(SOLVERS[key][1],xy=(-.23,.5),xycoords="axes fraction",ha="center",va="center",rotation=90,fontsize=TITLE_SIZE,fontweight="bold")
        for axis,image in ((pa,si),(ea,ei)):
            bar=fig.colorbar(image,ax=axis,pad=.02,fraction=.05); bar.ax.tick_params(labelsize=TICK_SIZE); bar.ax.yaxis.get_offset_text().set_fontsize(TICK_SIZE)
        ea.text(.97,.95,rf"relative $L^2={r['l2']:.2e}$",ha="right",va="top",transform=ea.transAxes,fontsize=TICK_SIZE,bbox={"facecolor":"white","alpha":.8,"edgecolor":"none"})
    axes[0,0].set_title("Predicted solution",fontsize=TITLE_SIZE,pad=12); axes[0,1].set_title("Absolute error",fontsize=TITLE_SIZE,pad=12); fig.savefig(output_path,format="pdf",dpi=650,bbox_inches="tight")

def plot_training(keys,output_path):
    fig,axes=plt.subplots(len(keys),1,figsize=(13,4.2*len(keys)),squeeze=False,constrained_layout=True)
    for row,key in enumerate(keys):
        axis=axes[row,0]; path=HERE/f"{SOLVERS[key][0]}_training_data.npz"
        if not path.exists(): axis.text(.5,.5,"Saved training data unavailable",ha="center",va="center",transform=axis.transAxes); axis.set_axis_off(); continue
        with np.load(path,allow_pickle=False) as d: epoch,loss=d["epoch"],d["loss"]
        label="Total soft-constraint loss" if key.endswith("soft") else "PDE-residual training loss"; axis.semilogy(epoch,loss,label=label,linewidth=2); axis.set_title(SOLVERS[key][1],fontsize=TITLE_SIZE); axis.set_xlabel("Epoch",fontsize=LABEL_SIZE); axis.set_ylabel("Loss",fontsize=LABEL_SIZE); axis.tick_params(labelsize=TICK_SIZE); axis.grid(True,which="both",alpha=.25); axis.legend(fontsize=LEGEND_SIZE)
    fig.savefig(output_path,format="pdf",dpi=650,bbox_inches="tight")

def main(name,show):
    if name=="all":
        preferred=("bspline-hard","pinn-soft","pinn-hard","pikan-soft","pikan-hard")
        selected=tuple(
            key for key in preferred
            if (HERE/f"{SOLVERS[key][0]}_solution_data.npz").exists()
        )
        groups={"kirchhoff_method_comparison.pdf":selected}
    else:
        selected=(name,)
        groups={f"{SOLVERS[name][0]}_comparison.pdf":selected}
    for filename,keys in groups.items():
        try: plot_comparison(keys,HERE/filename); print(f"Saved comparison figure: {HERE/filename}")
        except FileNotFoundError as e: print(e)
    loss=HERE/("kirchhoff_loss_evolution.pdf" if name=="all" else f"{SOLVERS[name][0]}_loss_evolution.pdf"); plot_training(selected,loss); print(f"Saved loss-evolution figure: {loss}")
    for key in selected:
        path=HERE/f"{SOLVERS[key][0]}_solution_data.npz"
        if path.exists():
            try:
                r=load_solution(path)
                timing=(f"{r['time']:.3f} s" if np.isfinite(r["time"])
                        else "unavailable in legacy data")
                print(f"{key}: validation relative L2={r['l2']:.6e}, time={timing}")
            except KeyError: pass
    if show: plt.show()
    else: plt.close("all")
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--solver",choices=("all",*SOLVERS),default="all"); p.add_argument("--show",action="store_true"); a=p.parse_args(); main(a.solver,a.show)
