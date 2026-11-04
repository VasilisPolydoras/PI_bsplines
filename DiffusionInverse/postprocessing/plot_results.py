"""Recreate inverse-diffusion comparison figures from saved NPZ data."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
HERE=Path(__file__).resolve().parent
SOLVERS={
 "bspline-hard":("bspline_diffusion_inverse_hard","PI-TP-BsA"),
 "pinn-soft":("pinn_diffusion_inverse_soft","PINN soft"),
 "pinn-hard":("pinn_diffusion_inverse_hard","PINN hard"),
 "pikan-soft":("pikan_diffusion_inverse_soft","PIKAN soft"),
 "pikan-hard":("pikan_diffusion_inverse_hard","PIKAN hard"),
}
def load(key):
 stem,label=SOLVERS[key];solution=HERE/f"{stem}_solution_data.npz";training=HERE/f"{stem}_training_data.npz"
 if not solution.exists() or not training.exists():return None
 return label,np.load(solution),np.load(training)
def comparison(keys):
 rows=[(key,load(key)) for key in keys];rows=[row for row in rows if row[1] is not None]
 if not rows:print("No saved inverse-diffusion data found");return
 figure,axes=plt.subplots(len(rows),2,figsize=(13,4.1*len(rows)),squeeze=False,constrained_layout=True)
 for row,(key,(label,data,_)) in enumerate(rows):
  gx,gt=np.meshgrid(data["x"],data["t"])
  for column,(values,title,cmap) in enumerate(((data["prediction"],"Predicted solution","viridis"),(data["absolute_error"],"Absolute error","magma"))):
   image=axes[row,column].contourf(gx,gt,values,40,cmap=cmap);bar=figure.colorbar(image,ax=axes[row,column]);bar.ax.tick_params(labelsize=12);axes[row,column].tick_params(labelsize=12);axes[row,column].set_xlabel("x",fontsize=14);axes[row,column].set_ylabel("t",fontsize=14)
   if row==0:axes[row,column].set_title(title,fontsize=16)
  estimated=float(data["estimated_diffusivity"]);axes[row,0].text(-0.18,0.5,f"{label}\n$D={estimated:.5f}$",transform=axes[row,0].transAxes,rotation=90,va="center",ha="center",fontsize=14)
  print(f"{key}: D={estimated:.8f}, parameter error={float(data['parameter_relative_error']):.6e}, validation L2={float(data['validation_relative_l2_error']):.6e}")
 figure.savefig(HERE/"diffusion_inverse_method_comparison.pdf",dpi=900,bbox_inches="tight")
 for _,(_,data,training) in rows:data.close();training.close()
def histories(keys):
 rows=[(key,load(key)) for key in keys];rows=[row for row in rows if row[1] is not None]
 if not rows:return
 figure,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
 true_value=None
 for key,(label,data,training) in rows:
  axes[0].semilogy(training["epoch"],training["loss"],label=label,linewidth=2);axes[1].plot(training["epoch"],training["estimated_diffusivity"],label=label,linewidth=2);true_value=float(data["true_diffusivity"]);data.close();training.close()
 axes[1].axhline(true_value,color="black",linestyle="--",label="Exact diffusivity");axes[0].set(xlabel="Epoch",ylabel="Loss",title="Loss evolution");axes[1].set(xlabel="Epoch",ylabel="Diffusivity",title="Parameter identification");
 for axis in axes:axis.tick_params(labelsize=12);axis.xaxis.label.set_size(14);axis.yaxis.label.set_size(14);axis.title.set_size(16);axis.legend(fontsize=11)
 figure.savefig(HERE/"diffusion_inverse_training_evolution.pdf",dpi=900,bbox_inches="tight")
def main(solver,show):
 keys=list(SOLVERS) if solver=="all" else [solver];comparison(keys);histories(keys);plt.show() if show else plt.close("all")
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--solver",choices=("all",*SOLVERS),default="all");parser.add_argument("--show",action="store_true");args=parser.parse_args();main(args.solver,args.show)
