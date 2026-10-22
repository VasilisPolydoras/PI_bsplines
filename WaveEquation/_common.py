"""Shared utilities for the forward one-dimensional wave benchmark."""
from __future__ import annotations
import json,math,random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
def set_seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def device_from(n):
 if n=="cpu":return torch.device("cpu")
 if n=="gpu" and torch.cuda.is_available():return torch.device("cuda")
 if n=="gpu":raise RuntimeError("GPU requested, but CUDA is unavailable")
 raise ValueError("device must be cpu or gpu")
def exact(z,c):return torch.sin(math.pi*z[:,0])*torch.cos(math.pi*c*z[:,1])
def sample(n,d,g=None):return torch.rand(n,2,device=d,generator=g)
def boundary(n,d,g=None):
 t=torch.rand(n,device=d,generator=g);return torch.cat((torch.stack((torch.zeros_like(t),t),1),torch.stack((torch.ones_like(t),t),1)))
def initial(n,d):
 x=torch.linspace(0,1,n,device=d);return torch.stack((x,torch.zeros_like(x)),1)
def derivative(y,x,axis,order=1):
 out=y
 for _ in range(order):out=torch.autograd.grad(out,x,torch.ones_like(out),create_graph=True)[0][:,axis]
 return out
def residual(model,z,c):
 q=z.detach().clone().requires_grad_(True);u=model(q);return derivative(u,q,1,2)-c**2*derivative(u,q,0,2)
def loss(model,inside,bc,ic,c,w):
 pde=residual(model,inside,c).square().mean()
 if model.constraint_mode=="hard":return float(w["pde"])*pde
 q=ic.detach().clone().requires_grad_(True);u=model(q);velocity=derivative(u,q,1);return float(w["pde"])*pde+float(w["boundary"])*model(bc).square().mean()+float(w["initial_displacement"])*(u-exact(q,c)).square().mean()+float(w["initial_velocity"])*velocity.square().mean()
def relative_l2(m,z,c):
 with torch.no_grad():p=m(z);r=exact(z,c)
 return torch.linalg.vector_norm(p-r)/torch.linalg.vector_norm(r)
def train(model,fn,cfg):
 o=torch.optim.Adam(model.parameters(),lr=float(cfg["learning_rate"]));s=torch.optim.lr_scheduler.ReduceLROnPlateau(o,factor=float(cfg["scheduler"]["factor"]),patience=int(cfg["scheduler"]["patience"]),min_lr=float(cfg["scheduler"]["min_learning_rate"]));best=float("inf");stale=0;state=None;h=[]
 for e in range(int(cfg["epochs"])):
  o.zero_grad();v=fn();n=float(v.detach());v.backward();o.step();s.step(n);h.append(n)
  if n<best-float(cfg["early_stopping"]["min_delta"]):best=n;stale=0;state={k:x.detach().clone() for k,x in model.state_dict().items()}
  else:stale+=1
  if int(cfg["print_every"]) and (e+1)%int(cfg["print_every"])==0:print(f"Epoch [{e+1}/{cfg['epochs']}], loss={n:.6e}")
  if stale>=int(cfg["early_stopping"]["patience"]):break
 if state:model.load_state_dict(state)
 return h
def save(model,h,error,config,c,method):
 post=ROOT/config["output"]["postprocessing_directory"];post.mkdir(parents=True,exist_ok=True);n=int(config["sampling"]["plot_points_per_axis"]);x=np.linspace(0,1,n);t=np.linspace(0,1,n);gx,gt=np.meshgrid(x,t,indexing="xy");d=next(model.parameters()).device;z=torch.tensor(np.column_stack((gx.ravel(),gt.ravel())),dtype=torch.float32,device=d)
 with torch.no_grad():p=model(z).cpu().numpy().reshape(gx.shape);r=exact(z,c).cpu().numpy().reshape(gx.shape)
 stem=f"{method}_wave_{model.constraint_mode}";np.savez_compressed(post/f"{stem}_solution_data.npz",x=x,t=t,prediction=p,reference=r,absolute_error=np.abs(p-r),validation_relative_l2_error=error);np.savez_compressed(post/f"{stem}_training_data.npz",epoch=np.arange(1,len(h)+1),loss=h);plot_solution(post/f"{stem}_solution_data.npz",post/f"{stem}_solution.png",method.upper());plot_training(post/f"{stem}_training_data.npz",post/f"{stem}_training.png",method.upper());plt.show() if config["output"]["show_plots"] else plt.close("all")
def plot_solution(path,out,title):
 with np.load(path) as d:x,t=d["x"],d["t"];f=(d["prediction"],d["reference"],d["absolute_error"])
 gx,gt=np.meshgrid(x,t,indexing="xy");fig,axes=plt.subplots(1,3,figsize=(16,4),constrained_layout=True)
 for ax,z,l in zip(axes,f,("Prediction","Exact","Absolute error")):im=ax.contourf(gx,gt,z,40,cmap="viridis");fig.colorbar(im,ax=ax);ax.set_title(l);ax.set_xlabel("x");ax.set_ylabel("t")
 fig.suptitle(title);fig.savefig(out,dpi=300,bbox_inches="tight")
def plot_training(path,out,title):
 with np.load(path) as d:e,l=d["epoch"],d["loss"]
 fig,ax=plt.subplots();ax.semilogy(e,l);ax.set_xlabel("Epoch");ax.set_ylabel("Loss");ax.set_title(title);fig.tight_layout();fig.savefig(out,dpi=300,bbox_inches="tight")
