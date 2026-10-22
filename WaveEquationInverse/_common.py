"""Utilities for inverse wave-speed identification."""
from __future__ import annotations
import math,random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parent
def set_seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def device_from(n):
 if n=="cpu":return torch.device("cpu")
 if n=="gpu" and torch.cuda.is_available():return torch.device("cuda")
 if n=="gpu":raise RuntimeError("GPU requested, but CUDA is unavailable")
 raise ValueError("device must be cpu or gpu")
def exact(z,c):return torch.sin(math.pi*z[:,0])*torch.cos(math.pi*c*z[:,1])
def sample(n,d,g=None):return torch.rand(n,2,device=d,generator=g)
def observations(n,noise,c,d,g=None):
 z=sample(n,d,g);return z,exact(z,c)+noise*torch.randn(n,device=d,generator=g)
def boundary(n,d,g=None):
 t=torch.rand(n,device=d,generator=g);return torch.cat((torch.stack((torch.zeros_like(t),t),1),torch.stack((torch.ones_like(t),t),1)))
def initial(n,d):
 x=torch.linspace(0,1,n,device=d);return torch.stack((x,torch.zeros_like(x)),1)
def raw_positive(v,m,d):return torch.nn.Parameter(torch.tensor(math.log(math.expm1(max(v-m,torch.finfo(torch.float32).eps))),device=d))
def positive(r,m):return float(m)+F.softplus(r)
def derivative(y,x,axis,order=1):
 out=y
 for _ in range(order):out=torch.autograd.grad(out,x,torch.ones_like(out),create_graph=True)[0][:,axis]
 return out
def residual(model,z):
 q=z.detach().clone().requires_grad_(True);u=model(q);return derivative(u,q,1,2)-model.wave_speed().square()*derivative(u,q,0,2)
def loss(model,inside,sensors,values,bc,ic,true_c,w):
 total=float(w["pde"])*residual(model,inside).square().mean()+float(w["data"])*(model(sensors)-values).square().mean()
 if model.constraint_mode=="soft":
  q=ic.detach().clone().requires_grad_(True);u=model(q);total+=float(w["boundary"])*model(bc).square().mean()+float(w["initial_displacement"])*(u-exact(q,true_c)).square().mean()+float(w["initial_velocity"])*derivative(u,q,1).square().mean()
 return total
def relative_l2(m,z,c):
 with torch.no_grad():p=m(z);r=exact(z,c)
 return torch.linalg.vector_norm(p-r)/torch.linalg.vector_norm(r)
def train(model,fn,cfg):
 o=torch.optim.Adam(model.parameters(),lr=float(cfg["learning_rate"]));s=torch.optim.lr_scheduler.ReduceLROnPlateau(o,factor=float(cfg["scheduler"]["factor"]),patience=int(cfg["scheduler"]["patience"]),min_lr=float(cfg["scheduler"]["min_learning_rate"]));best=float("inf");stale=0;state=None;h=[];ch=[]
 for e in range(int(cfg["epochs"])):
  o.zero_grad();v=fn();n=float(v.detach());v.backward();o.step();s.step(n);h.append(n);ch.append(float(model.wave_speed().detach()))
  if n<best-float(cfg["early_stopping"]["min_delta"]):best=n;stale=0;state={k:x.detach().clone() for k,x in model.state_dict().items()}
  else:stale+=1
  if int(cfg["print_every"]) and (e+1)%int(cfg["print_every"])==0:print(f"Epoch [{e+1}/{cfg['epochs']}], loss={n:.6e}, c={ch[-1]:.6f}")
  if stale>=int(cfg["early_stopping"]["patience"]):break
 if state:model.load_state_dict(state)
 return h,ch
def save(model,h,ch,error,config,true_c,method,sensors):
 post=ROOT/config["output"]["postprocessing_directory"];post.mkdir(parents=True,exist_ok=True);n=int(config["sampling"]["plot_points_per_axis"]);x=np.linspace(0,1,n);t=np.linspace(0,1,n);gx,gt=np.meshgrid(x,t,indexing="xy");d=next(model.parameters()).device;z=torch.tensor(np.column_stack((gx.ravel(),gt.ravel())),dtype=torch.float32,device=d)
 with torch.no_grad():p=model(z).cpu().numpy().reshape(gx.shape);r=exact(z,true_c).cpu().numpy().reshape(gx.shape)
 est=float(model.wave_speed().detach());stem=f"{method}_wave_inverse_{model.constraint_mode}";np.savez_compressed(post/f"{stem}_solution_data.npz",x=x,t=t,prediction=p,reference=r,absolute_error=np.abs(p-r),sensor_points=sensors[0].cpu().numpy(),sensor_values=sensors[1].cpu().numpy(),estimated_wave_speed=est,true_wave_speed=true_c,parameter_relative_error=abs(est-true_c)/true_c,validation_relative_l2_error=error);np.savez_compressed(post/f"{stem}_training_data.npz",epoch=np.arange(1,len(h)+1),loss=h,estimated_wave_speed=ch);plot_solution(post/f"{stem}_solution_data.npz",post/f"{stem}_solution.png",method.upper());plot_training(post/f"{stem}_training_data.npz",post/f"{stem}_training.png",method.upper(),true_c);plt.show() if config["output"]["show_plots"] else plt.close("all")
def plot_solution(path,out,title):
 with np.load(path) as d:x,t=d["x"],d["t"];f=(d["prediction"],d["reference"],d["absolute_error"]);est=float(d["estimated_wave_speed"]);true=float(d["true_wave_speed"])
 gx,gt=np.meshgrid(x,t,indexing="xy");fig,axes=plt.subplots(1,3,figsize=(16,4),constrained_layout=True)
 for ax,z,l in zip(axes,f,("Prediction","Exact","Absolute error")):im=ax.contourf(gx,gt,z,40,cmap="viridis");fig.colorbar(im,ax=ax);ax.set_title(l);ax.set_xlabel("x");ax.set_ylabel("t")
 fig.suptitle(f"{title}: c={est:.6f} (true {true:.6f})");fig.savefig(out,dpi=300,bbox_inches="tight")
def plot_training(path,out,title,true):
 with np.load(path) as d:e,l,c=d["epoch"],d["loss"],d["estimated_wave_speed"]
 fig,axes=plt.subplots(1,2,figsize=(11,4));axes[0].semilogy(e,l);axes[1].plot(e,c);axes[1].axhline(true,color="k",ls="--");axes[0].set_title("Loss");axes[1].set_title("Identified wave speed");fig.suptitle(title);fig.tight_layout();fig.savefig(out,dpi=300,bbox_inches="tight")
