"""Shared utilities for inverse diffusion identification."""
from __future__ import annotations
import json,math,random
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
def exact(z,a):return torch.sin(2*math.pi*z[:,0])*torch.exp(-4*math.pi**2*a*z[:,1])
def sample(n,d,g=None):return torch.rand(n,2,device=d,generator=g)
def observations(n,noise,a,d,g=None):
 z=sample(n,d,g);return z,exact(z,a)+noise*torch.randn(n,device=d,generator=g)
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
 q=z.detach().clone().requires_grad_(True);u=model(q);return derivative(u,q,1)-model.diffusivity()*derivative(u,q,0,2)
def loss(model,inside,sensors,values,bc,ic,true_a,weights):
 pde=residual(model,inside).square().mean();data=(model(sensors)-values).square().mean();total=float(weights["pde"])*pde+float(weights["data"])*data
 if model.constraint_mode=="soft":total+=float(weights["boundary"])*model(bc).square().mean()+float(weights["initial"])*(model(ic)-exact(ic,true_a)).square().mean()
 return total
def relative_l2(m,z,a):
 with torch.no_grad():p=m(z);r=exact(z,a)
 return torch.linalg.vector_norm(p-r)/torch.linalg.vector_norm(r)
def train(model,fn,c):
 o=torch.optim.Adam(model.parameters(),lr=float(c["learning_rate"]));s=torch.optim.lr_scheduler.ReduceLROnPlateau(o,factor=float(c["scheduler"]["factor"]),patience=int(c["scheduler"]["patience"]),min_lr=float(c["scheduler"]["min_learning_rate"]));best=float("inf");stale=0;state=None;h=[];ah=[]
 for e in range(int(c["epochs"])):
  o.zero_grad();v=fn();n=float(v.detach());v.backward();o.step();s.step(n);h.append(n);ah.append(float(model.diffusivity().detach()))
  if n<best-float(c["early_stopping"]["min_delta"]):best=n;stale=0;state={k:x.detach().clone() for k,x in model.state_dict().items()}
  else:stale+=1
  if int(c["print_every"]) and (e+1)%int(c["print_every"])==0:print(f"Epoch [{e+1}/{c['epochs']}], loss={n:.6e}, alpha={ah[-1]:.6f}")
  if stale>=int(c["early_stopping"]["patience"]):break
 if state:model.load_state_dict(state)
 return h,ah
def save(model,h,ah,error,config,true_a,method,sensors):
 post=ROOT/config["output"]["postprocessing_directory"];post.mkdir(parents=True,exist_ok=True);n=int(config["sampling"]["plot_points_per_axis"]);x=np.linspace(0,1,n);t=np.linspace(0,1,n);gx,gt=np.meshgrid(x,t,indexing="xy");d=next(model.parameters()).device;z=torch.tensor(np.column_stack((gx.ravel(),gt.ravel())),dtype=torch.float32,device=d)
 with torch.no_grad():p=model(z).cpu().numpy().reshape(gx.shape);r=exact(z,true_a).cpu().numpy().reshape(gx.shape)
 est=float(model.diffusivity().detach());stem=f"{method}_diffusion_inverse_{model.constraint_mode}";np.savez_compressed(post/f"{stem}_solution_data.npz",x=x,t=t,prediction=p,reference=r,absolute_error=np.abs(p-r),sensor_points=sensors[0].cpu().numpy(),sensor_values=sensors[1].cpu().numpy(),estimated_diffusivity=est,true_diffusivity=true_a,parameter_relative_error=abs(est-true_a)/true_a,validation_relative_l2_error=error);np.savez_compressed(post/f"{stem}_training_data.npz",epoch=np.arange(1,len(h)+1),loss=h,estimated_diffusivity=ah);plot_solution(post/f"{stem}_solution_data.npz",post/f"{stem}_solution.png",method.upper());plot_training(post/f"{stem}_training_data.npz",post/f"{stem}_training.png",method.upper(),true_a);plt.show() if config["output"]["show_plots"] else plt.close("all")
def plot_solution(path,out,title):
 with np.load(path) as d:x,t=d["x"],d["t"];f=(d["prediction"],d["reference"],d["absolute_error"]);est=float(d["estimated_diffusivity"]);true=float(d["true_diffusivity"])
 gx,gt=np.meshgrid(x,t,indexing="xy");fig,axes=plt.subplots(1,3,figsize=(16,4),constrained_layout=True)
 for ax,z,l in zip(axes,f,("Prediction","Exact","Absolute error")):im=ax.contourf(gx,gt,z,40,cmap="viridis");fig.colorbar(im,ax=ax);ax.set_title(l);ax.set_xlabel("x");ax.set_ylabel("t")
 fig.suptitle(f"{title}: alpha={est:.6f} (true {true:.6f})");fig.savefig(out,dpi=300,bbox_inches="tight")
def plot_training(path,out,title,true):
 with np.load(path) as d:e,l,a=d["epoch"],d["loss"],d["estimated_diffusivity"]
 fig,axes=plt.subplots(1,2,figsize=(11,4));axes[0].semilogy(e,l);axes[1].plot(e,a);axes[1].axhline(true,color="k",ls="--");axes[0].set_title("Loss");axes[1].set_title("Identified diffusivity");fig.suptitle(title);fig.tight_layout();fig.savefig(out,dpi=300,bbox_inches="tight")
